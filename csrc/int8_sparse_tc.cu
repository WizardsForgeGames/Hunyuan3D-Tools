/*
 * Hand-rolled Sparse INT8 Tensor Core GEMM via PTX mma.sp
 *
 * Bypasses cuSPARSELt entirely. Directly uses the mma.sp PTX instruction
 * to access Sparse Tensor Cores for 2x INT8 throughput.
 *
 * Instruction: mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
 *   - A (sparse weight): 16x32 with 2:4 sparsity, compressed to 16x16
 *   - B (dense activation): 32x8  (col-major)
 *   - C/D (accumulator): 16x8 int32
 *   - E (metadata): 2-bit indices packed per-thread
 *   - spsel: immediate 0 or 1 (selects which thread-pair provides metadata)
 *
 * Register counts per thread (32 threads/warp):
 *   A: 2 regs (8 int8 compressed)
 *   B: 2 regs (8 int8 dense)
 *   C/D: 4 regs (4 int32)
 *   E: 1 reg (uint32 metadata)
 *
 * Thread mapping (groupID = lane>>2, tid = lane&3):
 *   A[0]: row=groupID,   k_comp=[tid*4..+3]
 *   A[1]: row=groupID+8, k_comp=[tid*4..+3]
 *   B[0]: n=groupID,     k_dense=[tid*4..+3]       (first K-half)
 *   B[1]: n=groupID,     k_dense=[16+tid*4..+3]    (second K-half)
 *   C[0]: (groupID,   tid*2)
 *   C[1]: (groupID,   tid*2+1)
 *   C[2]: (groupID+8, tid*2)
 *   C[3]: (groupID+8, tid*2+1)
 *
 * Metadata layout (E per thread, empirically verified on RTX 4090):
 *   Lane groupID*4+0 (tid=0): all 8 nibbles for row groupID
 *   Lane groupID*4+1 (tid=1): all 8 nibbles for row groupID+8
 *   Lane groupID*4+2 (tid=2): for spsel=1 (second K-range)
 *   Lane groupID*4+3 (tid=3): for spsel=1 (second K-range)
 *   Packing: E = nib_g0 | (nib_g1<<4) | ... | (nib_g7<<28)
 *
 * Compression format:
 *   For each group of 4 along K: keep 2 largest magnitude, zero 2 smallest
 *   nibble = idx0 | (idx1 << 2), idx0 < idx1
 *   Compressed values: non-zeros packed contiguously, K halved
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

// ===============================================================================
//  Error checking
// ===============================================================================

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("CUDA error: ") + cudaGetErrorString(err) +         \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));            \
        }                                                                       \
    } while (0)

// ===============================================================================
//  cp.async infrastructure (SM80+, Ampere and later)
//
//  Copies 16 bytes from global to shared memory without going through registers.
//  The LSU handles the copy while Tensor Cores can compute simultaneously.
// ===============================================================================

__device__ __forceinline__ void cp_async_16(void* dst_smem, const void* src_gmem) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_addr), "l"(src_gmem)
    );
}

// cp.async.cg: caches in L2 only and bypasses L1 (it sets no eviction priority).
// Used for streamed data (activations/B tiles) that won't be re-read from L1.
__device__ __forceinline__ void cp_async_16_stream(void* dst_smem, const void* src_gmem) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_addr), "l"(src_gmem)
    );
}

__device__ __forceinline__ void cp_async_8(void* dst_smem, const void* src_gmem) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n"
        :: "r"(dst_addr), "l"(src_gmem)
    );
}

__device__ __forceinline__ void cp_async_16_u32(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_16_stream_u32(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_8_u32(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ===============================================================================
//  XOR bank-conflict-free smem swizzle (inspired by Marlin/CUTLASS)
//
//  Problem: consecutive warp groups reading from smem rows with stride
//  that is a multiple of 32 banks causes bank conflicts.
//  Fix: XOR the 16-byte chunk index with bits from the row index.
//  This redistributes accesses across different bank sets.
//
//  A tile (32 bytes/row = 2 chunks): XOR with (row>>2)&1
//  B tile (64 bytes/row = 4 chunks): XOR with (row>>1)&3
// ===============================================================================

// Swizzle column offset for A smem (32 bytes/row, 2 x 16-byte chunks)
__device__ __forceinline__ int a_swiz(int row, int byte_col) {
    return (((byte_col >> 4) ^ ((row >> 2) & 1)) << 4) | (byte_col & 15);
}

// Swizzle column offset for B smem (64 bytes/row, 4 x 16-byte chunks)
__device__ __forceinline__ int b_swiz(int row, int byte_col) {
    return (((byte_col >> 4) ^ ((row >> 1) & 3)) << 4) | (byte_col & 15);
}

// ===============================================================================
//  Stage 1: Prune + compress kernel
//  Dense int8 (N, K) -> compressed (N, K/2) + raw metadata (N, K/4) uint8
// ===============================================================================

/*
 * Prune to 2:4 pattern and compress.
 * One thread per group of 4 elements.
 *
 * Output metadata is RAW sequential nibbles (not hardware-reordered).
 * metadata_raw: (N * K/4) uint8, one nibble per group stored in low 4 bits.
 */
__global__ void prune_compress_kernel(
    const int8_t* __restrict__ weight,       // (N, K)
    int8_t* __restrict__ compressed,          // (N, K/2)
    uint8_t* __restrict__ metadata_raw,       // (N * K/4) -- one byte per group
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups_per_row = K / 4;
    const int total_groups = N * groups_per_row;
    if (idx >= total_groups) return;

    const int row = idx / groups_per_row;
    const int group = idx % groups_per_row;

    // Load 4 elements
    const int base = row * K + group * 4;
    int8_t vals[4];
    vals[0] = weight[base + 0];
    vals[1] = weight[base + 1];
    vals[2] = weight[base + 2];
    vals[3] = weight[base + 3];

    // Find 2 smallest magnitude positions (to prune)
    int abs_vals[4];
    abs_vals[0] = abs((int)vals[0]);
    abs_vals[1] = abs((int)vals[1]);
    abs_vals[2] = abs((int)vals[2]);
    abs_vals[3] = abs((int)vals[3]);

    int min1_pos = 0, min2_pos = 1;
    if (abs_vals[min1_pos] > abs_vals[min2_pos]) {
        int tmp = min1_pos; min1_pos = min2_pos; min2_pos = tmp;
    }
    for (int i = 2; i < 4; i++) {
        if (abs_vals[i] < abs_vals[min1_pos]) {
            min2_pos = min1_pos;
            min1_pos = i;
        } else if (abs_vals[i] < abs_vals[min2_pos]) {
            min2_pos = i;
        }
    }

    // Kept positions (sorted ascending by iteration order)
    int keep[2];
    int ki = 0;
    for (int i = 0; i < 4; i++) {
        if (i != min1_pos && i != min2_pos) {
            keep[ki++] = i;
        }
    }

    // Store compressed values
    const int comp_base = row * (K / 2) + group * 2;
    compressed[comp_base + 0] = vals[keep[0]];
    compressed[comp_base + 1] = vals[keep[1]];

    // Store raw metadata nibble: idx0 | (idx1 << 2)
    uint8_t nibble = (uint8_t)(keep[0] | (keep[1] << 2));
    metadata_raw[idx] = nibble;
}

// ===============================================================================
//  Stage 2: Activation quantization (reused from int8_gemm_tc.cu)
// ===============================================================================

__global__ void _sp_fused_quantize_kernel(
    const __half* __restrict__ input,   // (M, K) fp16
    int8_t* __restrict__ output,        // (M, K) int8
    float* __restrict__ scales,         // (M,) fp32
    const int K
) {
    const int row = blockIdx.x;
    const __half* in_row = input + (size_t)row * K;
    int8_t* out_row = output + (size_t)row * K;

    float thread_max = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = __half2float(in_row[k]);
        thread_max = fmaxf(thread_max, fabsf(val));
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));

    __shared__ float warp_maxes[32];
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) warp_maxes[warp_id] = thread_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + warpSize - 1) / warpSize)
                    ? warp_maxes[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        if (lane_id == 0) warp_maxes[0] = val;
    }
    __syncthreads();

    const float scale = fmaxf(warp_maxes[0], 1e-8f) / 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    const float inv_scale = 1.0f / scale;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = __half2float(in_row[k]);
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        out_row[k] = static_cast<int8_t>(q);
    }
}


// Single-pass variant: the row stays in registers as 16-byte vectors (VPT per thread),
// so the fp16 activation is read from memory once instead of twice, with 16-byte loads
// and 8-byte stores. Same arithmetic as _sp_fused_quantize_kernel (bit-identical output).
// Requires K % 8 == 0 and K <= VPT * 8 * blockDim.x. STREAM: evict-first loads, used when
// the input is larger than L2 (it cannot stay resident, and should not flush the int8
// output the GEMM reads next); smaller inputs use plain loads so sibling layers fed the
// same activation (e.g. Q/K/V projections) can re-hit it in L2.
template <int VPT, bool STREAM>
__global__ void _sp_quantize_vec_kernel(
    const __half* __restrict__ input,   // (M, K) fp16
    int8_t* __restrict__ output,        // (M, K) int8
    float* __restrict__ scales,         // (M,) fp32
    const int K
) {
    const int row  = blockIdx.x;
    const int nvec = K / 8;
    const uint4* in_row = reinterpret_cast<const uint4*>(input + (size_t)row * K);
    uint2* out_row = reinterpret_cast<uint2*>(output + (size_t)row * K);

    uint4 v[VPT];
    float thread_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        const int idx = threadIdx.x + i * blockDim.x;
        if (idx < nvec) {
            v[i] = STREAM ? __ldcs(in_row + idx) : in_row[idx];
            const __half2* h = reinterpret_cast<const __half2*>(&v[i]);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __half22float2(h[j]);
                thread_max = fmaxf(thread_max, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    __shared__ float warp_maxes[32];
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) warp_maxes[warp_id] = thread_max;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? warp_maxes[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        if (lane_id == 0) warp_maxes[0] = val;
    }
    __syncthreads();

    const float scale = fmaxf(warp_maxes[0], 1e-8f) / 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    const float inv_scale = 1.0f / scale;

    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        const int idx = threadIdx.x + i * blockDim.x;
        if (idx < nvec) {
            const __half2* h = reinterpret_cast<const __half2*>(&v[i]);
            uint32_t packed[2] = {0u, 0u};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __half22float2(h[j]);
                const int q0 = max(-128, min(127, __float2int_rn(f.x * inv_scale)));
                const int q1 = max(-128, min(127, __float2int_rn(f.y * inv_scale)));
                packed[j >> 1] |= ((uint32_t)(q0 & 0xFF) << (16 * (j & 1)))
                                | ((uint32_t)(q1 & 0xFF) << (16 * (j & 1) + 8));
            }
            out_row[idx] = make_uint2(packed[0], packed[1]);
        }
    }
}

// ===============================================================================
//  Stage 3: Minimal mma.sp test kernel (single warp, single tile)
// ===============================================================================

/*
 * Single-warp test: one m16n8k32 sparse MMA with spsel=0.
 *
 * A_comp:  (16, 16) int8  -- compressed from (16, 32) with 2:4
 * B_dense: (8, 32) int8   -- row-major = col-major (32, 8) for MMA
 * meta:    (32,) uint32   -- per-lane metadata (hardware format)
 * C_out:   (16, 8) int32  -- output
 */
__global__ void test_mma_sp_kernel(
    const int8_t* __restrict__ A_comp,
    const int8_t* __restrict__ B_dense,
    const uint32_t* __restrict__ meta,
    int32_t* __restrict__ C_out
) {
    const int lane = threadIdx.x;
    if (lane >= 32) return;

    const int groupID = lane >> 2;       // 0-7
    const int tid = lane & 3;            // 0-3
    const int K_COMP = 16;
    const int K_DENSE = 32;

    // -- Load A fragment (compressed, row-major 16x16) --
    uint32_t a_frag[2];
    memcpy(&a_frag[0], A_comp + groupID * K_COMP + tid * 4, 4);
    memcpy(&a_frag[1], A_comp + (groupID + 8) * K_COMP + tid * 4, 4);

    // -- Load B fragment (dense, stored as (8,32) row-major = (32,8) col-major) --
    // MMA B fragment mapping for m16n8k32 INT8:
    //   groupID indexes N column (0..7), tid indexes K groups
    //   b[0]: B_rm[groupID, tid*4 .. tid*4+3]     (K range 0..15)
    //   b[1]: B_rm[groupID, 16+tid*4 .. 16+tid*4+3] (K range 16..31)
    uint32_t b_frag[2];
    memcpy(&b_frag[0], B_dense + groupID * K_DENSE + tid * 4, 4);
    memcpy(&b_frag[1], B_dense + groupID * K_DENSE + 16 + tid * 4, 4);

    // -- Load per-lane metadata --
    uint32_t e_meta = meta[lane];

    // -- Accumulator --
    int32_t c_frag[4] = {0, 0, 0, 0};

    // -- Sparse MMA: m16n8k32 INT8 with spsel=0 --
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "{%8, %9, %10, %11}, "
        "%12, 0x0;\n"
        : "=r"(c_frag[0]), "=r"(c_frag[1]), "=r"(c_frag[2]), "=r"(c_frag[3])
        : "r"(a_frag[0]), "r"(a_frag[1]),
          "r"(b_frag[0]), "r"(b_frag[1]),
          "r"(c_frag[0]), "r"(c_frag[1]), "r"(c_frag[2]), "r"(c_frag[3]),
          "r"(e_meta)
    );

    // -- Store output (16x8 row-major) --
    // C fragment: c[0]=(groupID, tid*2), c[1]=(groupID, tid*2+1),
    //             c[2]=(groupID+8, tid*2), c[3]=(groupID+8, tid*2+1)
    const int col0 = tid * 2;
    const int col1 = col0 + 1;
    C_out[groupID * 8 + col0]       = c_frag[0];
    C_out[groupID * 8 + col1]       = c_frag[1];
    C_out[(groupID + 8) * 8 + col0] = c_frag[2];
    C_out[(groupID + 8) * 8 + col1] = c_frag[3];
}

// ===============================================================================
//  Stage 4: Pack raw metadata nibbles into per-row uint32 chunks
//  meta_raw (N, K/4) uint8 -> meta_packed (N, K/32) uint32
//  Each uint32 packs 8 consecutive nibbles (one 32-element dense K-chunk)
// ===============================================================================

__global__ void pack_metadata_kernel(
    const uint8_t* __restrict__ meta_raw,   // (N * K/4) sequential nibbles
    uint32_t* __restrict__ meta_packed,     // (N * K/32) packed uint32
    const int N, const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups_per_row = K / 4;
    const int chunks_per_row = K / 32;
    const int total = N * chunks_per_row;
    if (idx >= total) return;

    const int row = idx / chunks_per_row;
    const int chunk = idx % chunks_per_row;

    uint32_t packed = 0;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        uint8_t nib = meta_raw[row * groups_per_row + chunk * 8 + g];
        packed |= ((uint32_t)(nib & 0xF)) << (g * 4);
    }
    meta_packed[idx] = packed;
}

// ===============================================================================
//  Stage 5: Tiled Sparse INT8 GEMM kernel (3-stage pipelined, templatized)
//
//  Computes: output(M, N_out) = dequant( W_comp(N_out, K/2) x x_int8(M, K)^T )
//
//  W_comp is the sparse operand A (row-major, 2:4 compressed along K).
//  x_int8 is the dense operand B (treated as col-major KxM = row-major MxK).
//  The MMA produces C(N_out, M). Epilogue transposes to output(M, N_out).
//
//  Template parameters:
//    BLOCK_WR_T: 128 (standard) or 256 (tall tile, higher arithmetic intensity)
//    BLOCK_BC_T: 64 (narrow) or 128 (wide) batch columns
//  3-stage cp.async pipeline: two tiles prefetched ahead of compute for better
//  latency hiding. Persistent CTAs (one per SM) walk tiles in M-fastest order
//  (N-fastest once the int8 activation exceeds L2); tile shape and split-K come from
//  the cost model (sp_plan).
//
//  FQ = true (sparse_gemm_fq_kernel, 128x64 only): fused activation quantize. B is not
//  pipelined: each CTA quantizes its 64-token block (full K <= 1280) from fp16 into a
//  resident smem block once, runs every weight tile against it (N-fastest), and
//  prefetches its next block into L2 meanwhile. Removes the int8 activation's DRAM
//  round trip on large-M, narrow-N layers (see sp_use_fused).
//
//  Grid: (ceil(N_out/BLOCK_WR_T), ceil(M/BLOCK_BC_T))
// ===============================================================================

// Fixed geometry constants (BLOCK_WR is now a template parameter)
constexpr int SP_BLOCK_K  = 64;     // dense K elements per iteration (2 x mma K=32)
constexpr int SP_K_COMP   = 32;     // compressed K per iteration (SP_BLOCK_K / 2)
constexpr int SP_WARPS_WR = 4;
constexpr int SP_WARPS_BC = 2;
constexpr int SP_THREADS  = SP_WARPS_WR * SP_WARPS_BC * 32;  // 256
constexpr int SP_STAGE_PAD = 9;     // epilogue staging pad for bank conflicts


// Sparsity metadata staging (2 uint32 per weight row per 64-deep K step):
//   META_STEPS = 1: 8 bytes per row in every pipeline stage. 8 useful bytes per 32-byte
//                   L2 sector, but the cheapest loop; best for the small 128x64 tile, which
//                   is barrier/latency-bound rather than L2-bound.
//   META_STEPS = 4: 32 bytes (4 K steps) per row into a 2-slot ring every 4th K step: full
//                   sectors, ~25% less L2 traffic for the large, L2-bound tiles.
// FQ (fused quantize, see sp_fq_base_bytes) 128x64: 2-step chunks (16 B/row, 2 stages ahead
// need META_STEPS >= 2); its K loop streams only weights, so metadata sectors matter more.
template <int BW, int BC, bool FQ = false>
constexpr int sp_meta_steps() { return (BC == 64) ? (FQ ? 2 : 1) : 4; }

// Pipeline depth of the occupancy-1 kernel per tile (A+B per stage: 16 / 12 / 8 KB).
// Total smem is padded to just over half an SM anyway (one CTA per SM, see launch).
template <int BW, int BC>
constexpr int sp_stages() {
    return (BW == 256) ? 3 : (BC == 128) ? 3 : 3;
}

// Fused-quantize (FQ) path layout: [A+metadata pipeline | epilogue staging] (the larger of
// the two), then the resident int8 activation block (BC rows x full K, stored as K steps in
// the pipeline's B-stage layout), then BC fp32 per-token scales.
template <int BW, int BC>
constexpr int sp_fq_base_bytes() {
    constexpr int MS   = sp_meta_steps<BW, BC, true>();
    constexpr int PIPE = sp_stages<BW, BC>() * (BW * SP_K_COMP + (MS == 1 ? BW * 8 : 0))
                       + (MS == 1 ? 0 : 2 * BW * MS * 8);
    constexpr int STG  = BC * BW * 2;
    return PIPE > STG ? PIPE : STG;
}
// Register-staged row quantize: each lane holds FQ_VPT 16-byte vectors of a row
// (K <= FQ_VPT * 256 = 1280).
template <int BC>
constexpr int sp_fq_vpt() { return 5; }

template <int BW, int BC, int STAGES>
constexpr int sp_smem_bytes() {
    constexpr int MS = sp_meta_steps<BW, BC>();
    constexpr int PER_STAGE = BW * SP_K_COMP + BC * SP_BLOCK_K + (MS == 1 ? BW * 8 : 0);
    return STAGES * PER_STAGE + (MS == 1 ? 0 : 2 * BW * MS * 8);
}

// ===============================================================================
//  Split-K finalization: sum fp32 workspace partials -> fp16 output + bias
// ===============================================================================

__global__ void splitk_reduce_kernel(
    const float*  __restrict__ workspace,  // (M, N_out) fp32 accumulated partials
    const __half* __restrict__ bias,       // (N_out,) fp16 or nullptr
    __half*       __restrict__ output,     // (M, N_out) fp16
    const int N_out, const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float val = workspace[idx];
    if (bias != nullptr)
        val += __half2float(bias[idx % N_out]);
    output[idx] = __float2half(val);
}

// NF: N-fastest tile order (FQ always; the two-kernel GEMM when the int8 activation
// exceeds L2). A template parameter so the M-fastest kernels compile exactly as before.
template <int BLOCK_WR_T, int BLOCK_BC_T, int DB_STAGES_T, bool FQ = false, bool NF = FQ>
__device__ __forceinline__ void sparse_gemm_impl(
    const int8_t*   __restrict__ W_comp,     // (N_out, K/2) compressed weight
    const int8_t*   __restrict__ x_int8,     // (M, K) quantized activation
    const uint32_t* __restrict__ w_meta,     // (N_out, K/32) packed metadata
    const float*    __restrict__ act_scale,  // (M,) fp32
    const __half*   __restrict__ w_scale,    // (N_out,) fp16
    const __half*   __restrict__ bias,       // (N_out,) fp16 or nullptr
    __half*         __restrict__ output,     // (M, N_out) fp16
    float*          __restrict__ splitk_ws,  // split-K fp32 workspace, or nullptr
    const int M, const int K, const int N_out,
    const int split_k,             // 1 = normal, >1 = split-K mode
    const int k_iters_per_split,   // K iterations per split (ceiling)
    const __half* __restrict__ x_fp16,  // FQ: (M, K) fp16 activation, quantized in-kernel
    const int x_stream                  // FQ: evict-first loads of x_fp16 (input > L2)
) {
    // -- Derived constants from template parameters --
    constexpr int WARP_WR = BLOCK_WR_T / SP_WARPS_WR;       // 32 or 64
    constexpr int WARP_BC = BLOCK_BC_T / SP_WARPS_BC;        // 32 or 64
    constexpr int TILES_M = WARP_WR / 16;                    // 2 or 4 MMA M-tiles per warp
    constexpr int TILES_N = WARP_BC / 8;                     // 4 or 8
    constexpr int DB_STAGES = DB_STAGES_T;

    // Pipelined smem: DB_STAGES x (A_tile + B_tile), then a 2-slot metadata ring
    constexpr int SMEM_A_STAGE = BLOCK_WR_T * SP_K_COMP;             // 4096 or 8192
    constexpr int SMEM_B_STAGE = FQ ? 0 : BLOCK_BC_T * SP_BLOCK_K;   // 4096 or 8192 (FQ: resident)
    // Sparsity metadata: see sp_meta_steps (per-stage slot, or a 2-slot 4-step ring)
    constexpr int META_STEPS   = sp_meta_steps<BLOCK_WR_T, BLOCK_BC_T, FQ>();
    // Ring slot swizzle: rows 16/META_STEPS apart share banks; XOR their 8-byte step slot
    constexpr int MQ_SHIFT     = (META_STEPS == 2) ? 3 : 2;
    constexpr int SMEM_M_STAGE = (META_STEPS == 1) ? BLOCK_WR_T * 8 : 0;
    constexpr int SMEM_PER_STAGE = SMEM_A_STAGE + SMEM_B_STAGE + SMEM_M_STAGE;
    constexpr int SMEM_M_SLOT  = (META_STEPS == 1) ? 0 : BLOCK_WR_T * META_STEPS * 8;
    constexpr int SMEM_M_OFF   = DB_STAGES * SMEM_PER_STAGE;
    constexpr int SMEM_DB      = SMEM_M_OFF + 2 * SMEM_M_SLOT;
    static_assert(FQ || SMEM_DB == sp_smem_bytes<BLOCK_WR_T, BLOCK_BC_T, DB_STAGES>(), "smem layout");
    // FQ: resident activation block offset (after the pipeline / epilogue staging region)
    constexpr int SMEM_FQ_B = FQ ? sp_fq_base_bytes<BLOCK_WR_T, BLOCK_BC_T>() : 0;
    static_assert(!FQ || SMEM_FQ_B >= SMEM_DB, "FQ layout");
    static_assert(META_STEPS == 1 || META_STEPS >= DB_STAGES - 1,
                  "2-slot metadata ring needs META_STEPS >= DB_STAGES - 1");
    // Epilogue staging (reuses smem after K-loop)
    constexpr int SMEM_EPILOGUE = 8 * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
    constexpr int SMEM_BYTES = (SMEM_DB > SMEM_EPILOGUE) ? SMEM_DB : SMEM_EPILOGUE;

    // -- Thread-local constants (invariant across tiles) --
    const int t    = threadIdx.x;
    const int warp = t / 32;
    const int lane = t % 32;
    const int groupID = lane >> 2;
    const int tid     = lane & 3;

    const int warp_wr = (warp / SP_WARPS_BC) * WARP_WR;   // 0, WARP_WR, 2*WARP_WR, ...
    const int warp_bc = (warp % SP_WARPS_BC) * WARP_BC;    // 0, WARP_BC

    const int K_comp     = K / 2;
    const int meta_cols  = K / 32;
    const int K_iters_total = K / SP_BLOCK_K;

    // -- Persistent kernel tile grid --
    // 1D grid with block-partitioned assignment: contiguous tiles per CTA
    // for maximum L2 cache reuse of weight (A) tiles.
    // When split_k > 1, the tile space is (spatial_tiles x split_k) -- each
    // (spatial_tile, k_split) pair is a separate work item.
    const int tiles_x = (N_out + BLOCK_WR_T - 1) / BLOCK_WR_T;
    const int tiles_y = (M     + BLOCK_BC_T - 1) / BLOCK_BC_T;
    const int spatial_tiles = tiles_x * tiles_y;
    const int total_tiles = spatial_tiles * split_k;

    // -- Shared memory (persists across tile iterations) --
    // Dynamic smem allows > 48KB (needed for 4-stage 256x128 = 64KB)
    extern __shared__ int8_t smem_raw[];

    const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));

    // ==================================================================
    //  Persistent tile loop: each CTA processes multiple tiles
    //  Block-partitioned: CTA i gets tiles [i*tpc, (i+1)*tpc)
    //  to keep weight tiles hot in L2 across consecutive M-tiles.
    // ==================================================================
    const int tiles_per_cta = (total_tiles + (int)gridDim.x - 1) / (int)gridDim.x;
    const int tile_start = blockIdx.x * tiles_per_cta;
    const int tile_end   = min(tile_start + tiles_per_cta, total_tiles);

    [[maybe_unused]] int res_m = -1;   // FQ: M-tile whose activation block is resident in smem

    for (int tile_id = tile_start; tile_id < tile_end; tile_id++) {

    // -- Decode tile_id -> (spatial_id, k_split_id) --
    // K-splits for the same spatial tile are grouped together so the
    // persistent kernel's block-partition keeps weight tiles in L2.
    const int k_split_id = tile_id / spatial_tiles;
    const int spatial_id = tile_id % spatial_tiles;

    // -- Tile order: spatial_id -> (tile_n, tile_m) --
    // M-fastest: consecutive tiles of a CTA (block-partitioned) share the weight tile,
    // and co-running CTAs mostly share weight rows. Measured faster than N-fastest and
    // than banded/grouped orders on the headline shapes.
    // N-fastest (FQ always; two-kernel path when the int8 activation exceeds L2): the CTA
    // runs every weight tile against one activation block before moving on, so the
    // activation is read from DRAM once instead of once per weight-tile column.
    const int tile_n = NF ? spatial_id % tiles_x : spatial_id / tiles_y;
    const int tile_m = NF ? spatial_id / tiles_x : spatial_id % tiles_y;

    // -- K-range for this split --
    const int k_start_iter  = k_split_id * k_iters_per_split;
    const int k_end_iter    = min(k_start_iter + k_iters_per_split, K_iters_total);
    const int local_K_iters = k_end_iter - k_start_iter;
    if (local_K_iters <= 0) { __syncthreads(); continue; }

    const int block_wr = tile_n * BLOCK_WR_T;
    const int block_bc = tile_m * BLOCK_BC_T;

    // ================================================================
    //  FQ: quantize this M-tile's activation rows into the resident smem block
    //  (once per M-tile; consecutive tiles of a CTA share it, N-fastest order).
    //  Each warp quantizes BLOCK_BC_T/8 rows, two at a time: a lane holds up to
    //  FQ_VPT 16-byte vectors of each row in registers, the warp reduces the row
    //  absmax with shuffles, then stores 8 int8 per vector in the B-stage layout of
    //  K step k/64. Arithmetic matches _sp_quantize_vec_kernel (bit-identical).
    //  The previous tile's trailing __syncthreads() protects the overwrite.
    // ================================================================
    if constexpr (FQ) {
        if (tile_m != res_m) {
            res_m = tile_m;
            constexpr int FQ_VPT = sp_fq_vpt<BLOCK_BC_T>();
            constexpr int ROWS_PER_WARP = BLOCK_BC_T / (SP_THREADS / 32);
            const int nvec = K / 8;
            float* s_scale = reinterpret_cast<float*>(smem_raw + SMEM_FQ_B + BLOCK_BC_T * K);
            #pragma unroll 1
            for (int rr = 0; rr < ROWS_PER_WARP; rr += 2) {
                uint4 v[2][FQ_VPT];
                float amax[2];
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    const int r = warp * ROWS_PER_WARP + rr + u;
                    const uint4* src = reinterpret_cast<const uint4*>(
                        x_fp16 + (size_t)min(block_bc + r, M - 1) * K);
                    amax[u] = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < FQ_VPT; i++) {
                        const int idx = lane + 32 * i;
                        if (idx < nvec) {
                            v[u][i] = x_stream ? __ldcs(src + idx) : src[idx];
                            const __half2* h = reinterpret_cast<const __half2*>(&v[u][i]);
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                const float2 f = __half22float2(h[j]);
                                amax[u] = fmaxf(amax[u], fmaxf(fabsf(f.x), fabsf(f.y)));
                            }
                        }
                    }
                }
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        amax[u] = fmaxf(amax[u], __shfl_xor_sync(0xffffffff, amax[u], o));
                    const int r = warp * ROWS_PER_WARP + rr + u;
                    const float scale = fmaxf(amax[u], 1e-8f) / 127.0f;
                    const float inv_scale = 1.0f / scale;
                    if (lane == 0) s_scale[r] = scale;
                    #pragma unroll
                    for (int i = 0; i < FQ_VPT; i++) {
                        const int idx = lane + 32 * i;
                        if (idx < nvec) {
                            const __half2* h = reinterpret_cast<const __half2*>(&v[u][i]);
                            uint32_t packed[2] = {0u, 0u};
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                const float2 f = __half22float2(h[j]);
                                const int q0 = max(-128, min(127, __float2int_rn(f.x * inv_scale)));
                                const int q1 = max(-128, min(127, __float2int_rn(f.y * inv_scale)));
                                packed[j >> 1] |= ((uint32_t)(q0 & 0xFF) << (16 * (j & 1)))
                                                | ((uint32_t)(q1 & 0xFF) << (16 * (j & 1) + 8));
                            }
                            const int k = idx * 8;
                            const uint32_t a = smem_base + SMEM_FQ_B
                                + (uint32_t)((k / SP_BLOCK_K) * (BLOCK_BC_T * SP_BLOCK_K)
                                             + r * SP_BLOCK_K + b_swiz(r, k % SP_BLOCK_K));
                            asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
                                         :: "r"(a), "r"(packed[0]), "r"(packed[1]));
                        }
                    }
                }
            }
            __syncthreads();
            // Prefetch the CTA's next M-tile (contiguous rows of x) into L2 now, so its DRAM
            // read overlaps this M-tile's K loops and its quantize phase above hits L2.
            const int nm = tile_m + 1;
            if (nm < tiles_y && nm * tiles_x < tile_end) {
                const char* pf_base = reinterpret_cast<const char*>(x_fp16 + (size_t)nm * BLOCK_BC_T * K);
                const int pf_lines = min(BLOCK_BC_T, M - nm * BLOCK_BC_T) * (K / 64);   // 128-byte lines
                for (int l = t; l < pf_lines; l += SP_THREADS)
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(pf_base + (size_t)l * 128));
                // Measured on 1280x320, M=147456: no prefetch 0.774 ms, this 0.689; issuing it
                // before the quantize phase 0.793, two M-tiles ahead 0.779, evict_last 0.721.
            }
        }
    }

    // -- Accumulators: TILES_M M-tiles x TILES_N N-tiles x 4 int32 --
    int32_t acc[TILES_M][TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < TILES_M; mi++)
        #pragma unroll
        for (int nj = 0; nj < TILES_N; nj++) {
            acc[mi][nj][0] = 0; acc[mi][nj][1] = 0;
            acc[mi][nj][2] = 0; acc[mi][nj][3] = 0;
        }

    // ================================================================
    //  3-stage pipelined K-loop with cp.async
    //
    //  Prologue: async load tiles into buf[0] and buf[1].
    //  Main loop: while computing from buf[ki%3], async load into
    //             buf[(ki+2)%3]. Two tiles always prefetched ahead.
    //  Epilogue staging reuses smem after K-loop completes.
    // ================================================================

    // -- Per-thread cp.async assignments, fixed for the whole tile --
    // Each thread copies A_ITERS 16-byte A chunks, B_ITERS 16-byte B chunks and (if
    // t < BLOCK_WR_T) one 8-byte metadata pair per K step. Source pointers and swizzled
    // smem offsets are computed once here; per K step only the K offset changes.
    // Rows past N_out / M are clamped to the last valid row instead of zero-filled:
    // they only feed output rows/columns the epilogue never stores, and clamped
    // metadata is still a valid 2:4 pattern.
    constexpr int TOTAL_A_16 = BLOCK_WR_T * SP_K_COMP / 16;
    constexpr int TOTAL_B_16 = BLOCK_BC_T * SP_BLOCK_K / 16;
    constexpr int A_CPR = SP_K_COMP / 16;                  // 16B chunks per A row (2)
    constexpr int B_CPR = SP_BLOCK_K / 16;                 // 16B chunks per B row (4)
    constexpr int A_ITERS = TOTAL_A_16 / SP_THREADS;
    constexpr int B_ITERS = TOTAL_B_16 / SP_THREADS;
    static_assert(TOTAL_A_16 % SP_THREADS == 0 && TOTAL_B_16 % SP_THREADS == 0,
                  "tile loads must divide evenly across the CTA");
    static_assert(BLOCK_WR_T <= SP_THREADS, "one metadata row per thread");

    const int8_t* a_src[A_ITERS];
    uint32_t      a_dst[A_ITERS];
    #pragma unroll
    for (int it = 0; it < A_ITERS; it++) {
        const int row = t / A_CPR + it * (SP_THREADS / A_CPR);
        const int col = (t % A_CPR) * 16;
        a_src[it] = W_comp + (size_t)min(block_wr + row, N_out - 1) * K_comp + col;
        a_dst[it] = row * SP_K_COMP + a_swiz(row, col);
    }
    [[maybe_unused]] const int8_t* b_src[B_ITERS];   // unused by FQ (B is resident)
    [[maybe_unused]] uint32_t      b_dst[B_ITERS];
    #pragma unroll
    for (int it = 0; it < B_ITERS; it++) {
        const int row = t / B_CPR + it * (SP_THREADS / B_CPR);
        const int col = (t % B_CPR) * 16;
        b_src[it] = x_int8 + (size_t)min(block_bc + row, M - 1) * K + col;
        b_dst[it] = SMEM_A_STAGE + row * SP_BLOCK_K + b_swiz(row, col);
    }
    // Metadata chunk: a row's META_STEPS x 8 bytes are copied by META_STEPS consecutive
    // threads (one K step each), so each row is one coalesced request. Within the row the
    // 8-byte step slot is XOR-swizzled by (row >> 2) & 3 so compute-side reads (16 rows x
    // 2 words per warp) are bank-conflict-free.
    constexpr int M_COPIES = (BLOCK_WR_T * META_STEPS + SP_THREADS - 1) / SP_THREADS;
    static_assert(META_STEPS == 1 || BLOCK_WR_T * META_STEPS % SP_THREADS == 0,
                  "metadata copies must divide evenly");
    const bool m_loader = (t < BLOCK_WR_T * META_STEPS);
    const int  m_j = t % META_STEPS;                      // K step within the chunk
    const uint32_t* m_src[M_COPIES];
    uint32_t        m_dst[M_COPIES];
    #pragma unroll
    for (int it = 0; it < M_COPIES; it++) {
        const int row = t / META_STEPS + it * (SP_THREADS / META_STEPS);
        m_src[it] = w_meta + (size_t)min(block_wr + row, N_out - 1) * meta_cols + 2 * m_j;
        m_dst[it] = (META_STEPS == 1)
            ? (uint32_t)(SMEM_A_STAGE + SMEM_B_STAGE + row * 8)            // stage-relative
            : (uint32_t)(row * (META_STEPS * 8)
                         + ((m_j ^ ((row >> MQ_SHIFT) & (META_STEPS - 1))) << 3));
    }

    // Async load of absolute K step K_ABS into the stage at smem byte offset STAGE_OFF.
    #define LOAD_TILE(STAGE_OFF, K_ABS) do {                                      \
        const uint32_t _so = smem_base + (STAGE_OFF);                             \
        _Pragma("unroll")                                                          \
        for (int _it = 0; _it < A_ITERS; _it++)                                   \
            cp_async_16_u32(_so + a_dst[_it], a_src[_it] + (K_ABS) * SP_K_COMP);  \
        if constexpr (!FQ) {                                                       \
            _Pragma("unroll")                                                      \
            for (int _it = 0; _it < B_ITERS; _it++)                               \
                cp_async_16_stream_u32(_so + b_dst[_it], b_src[_it] + (K_ABS) * SP_BLOCK_K); \
        }                                                                          \
        if constexpr (META_STEPS == 1) {                                           \
            if (m_loader)                                                          \
                cp_async_8_u32(_so + m_dst[0], m_src[0] + (K_ABS) * 2);            \
        } else if (((K_ABS) % META_STEPS) == 0 || (K_ABS) == k_start_iter) {       \
            /* metadata chunk: at chunk boundaries and at the tile's first K step */\
            const int _c = (K_ABS) / META_STEPS;                                   \
            if (_c * META_STEPS + m_j < K_iters_total) {                           \
                const uint32_t _ms = smem_base + SMEM_M_OFF + (_c & 1) * SMEM_M_SLOT;\
                _Pragma("unroll")                                                  \
                for (int _it = 0; _it < M_COPIES; _it++)                           \
                    cp_async_8_u32(_ms + m_dst[_it], m_src[_it] + _c * (META_STEPS * 2));\
            }                                                                      \
        }                                                                          \
        cp_async_commit();                                                         \
    } while (0)

    // -- Prologue: async load first (DB_STAGES-1) tiles --
    // Fill all but one pipeline stage before entering the K-loop.
    // Offsets are absolute (k_start_iter + s) to handle split-K.
    #pragma unroll
    for (int s = 0; s < DB_STAGES - 1; s++)
        if (s < local_K_iters)
            LOAD_TILE(s * SMEM_PER_STAGE, k_start_iter + s);

    // -- Per-lane ldmatrix / metadata offsets within a stage --
    // A (x4): lane L addresses row warp_wr + (L & 15), K_comp half (L >> 4); +16 rows per
    //   M-tile keeps the swizzle phase, so M-tile mi is +mi*16*SP_K_COMP.
    // B (x4): lane L addresses row warp_bc + (L & 7), 16-byte K chunk (L >> 3); +8 rows per
    //   N-tile keeps the swizzle phase (+nj*8*SP_BLOCK_K).
    // Metadata: lane (g, tid) reads row warp_wr + mi*16 + g + 8*(tid&1), word tid>>1 of the
    //   current step's (swizzled) 8-byte slot; the swizzle phase q is the same for every mi.
    const uint32_t a_ldm_off = [&] {
        const int row = warp_wr + (lane & 15);
        return (uint32_t)(row * SP_K_COMP + a_swiz(row, (lane >> 4) << 4));
    }();
    const uint32_t b_ldm_off = [&] {
        const int row = warp_bc + (lane & 7);
        return (uint32_t)((FQ ? SMEM_FQ_B : SMEM_A_STAGE) + row * SP_BLOCK_K
                          + b_swiz(row, (lane >> 3) << 4));
    }();
    const int      m_row     = warp_wr + groupID + ((tid & 1) << 3);
    const int      m_q       = (m_row >> MQ_SHIFT) & (META_STEPS - 1);
    const uint32_t m_ldm_off = (META_STEPS == 1 ? SMEM_A_STAGE + SMEM_B_STAGE : SMEM_M_OFF)
                             + (uint32_t)(m_row * (META_STEPS * 8) + (tid >> 1) * 4);

    // -- Main K-loop (over local K range) --
    //
    //  Single-sync pipeline: one __syncthreads() per K-iteration instead of two.
    //  The sync serves double duty: (1) ensures previous iteration's compute is
    //  done (so the ahead buffer can be safely overwritten), and (2) ensures the
    //  current buffer's cp.async data is visible to all warps.
    //
    //  LOAD_TILE is placed AFTER the sync (but before compute) so the buffer it
    //  writes to -- buf[(ki+DB_STAGES-1)%DB_STAGES] -- is guaranteed free: its
    //  last consumer was compute at iteration ki-(DB_STAGES-1), which completed
    //  before the sync at iteration ki-(DB_STAGES-2) at the latest.
    //
    uint32_t comp_off  = 0;                                   // stage being computed
    uint32_t ahead_off = (DB_STAGES - 1) * SMEM_PER_STAGE;    // stage being loaded
    for (int ki = 0; ki < local_K_iters; ki++) {
        const int  k_ahead   = ki + (DB_STAGES - 1);          // local index
        const bool has_ahead = (k_ahead < local_K_iters);

        // -- Wait for CURRENT tile --
        // At most (DB_STAGES-2) commit groups may still be in flight.
        if (has_ahead)
            cp_async_wait_group<DB_STAGES - 2>();
        else
            cp_async_wait_group<0>();
        __syncthreads();

        // -- Initiate load (DB_STAGES-1) tiles ahead --
        if (has_ahead)
            LOAD_TILE(ahead_off, k_start_iter + k_ahead);

        const uint32_t cur = smem_base + comp_off;
        // B fragments: the current pipeline stage, or (FQ) K step k of the resident block
        const uint32_t bcur = FQ ? smem_base + (uint32_t)((k_start_iter + ki) * (BLOCK_BC_T * SP_BLOCK_K))
                                 : cur;

        // FQ: warps whose weight rows are all past N_out (narrow N, last tile) skip the math
        if (!FQ || block_wr + warp_wr < N_out) {

        // -- Metadata from the shared-memory ring (staged by LOAD_TILE) --
        // One conflict-free LDS per M-tile.
        uint32_t m_cur;
        if constexpr (META_STEPS == 1) {
            m_cur = cur + m_ldm_off;
        } else {
            const int k_abs = k_start_iter + ki;
            m_cur = smem_base + m_ldm_off
                  + ((k_abs / META_STEPS) & 1) * SMEM_M_SLOT
                  + (((k_abs % META_STEPS) ^ m_q) << 3);
        }
        uint32_t e_meta[TILES_M];
        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++)
            asm volatile("ld.shared.u32 %0, [%1];\n"
                         : "=r"(e_meta[mi]) : "r"(m_cur + mi * 16 * (META_STEPS * 8)));

        // -- Per-warp MMA computation --
        // All TILES_M A fragments are loaded once per K step and held in registers; the
        // N-tile loop is outermost so each B fragment is loaded exactly once and reused
        // across the warp's TILES_M M-tiles: TILES_M + TILES_N ldmatrix.x4 per K step.
        uint32_t a_frag0[TILES_M][2], a_frag1[TILES_M][2];
        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++) {
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[mi][0]), "=r"(a_frag0[mi][1]),
                  "=r"(a_frag1[mi][0]), "=r"(a_frag1[mi][1])
                : "r"(cur + a_ldm_off + mi * 16 * SP_K_COMP)
            );
        }

        #pragma unroll
        for (int nj = 0; nj < TILES_N; nj++) {
            uint32_t b_frag0[2], b_frag1[2];
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag0[0]), "=r"(b_frag0[1]), "=r"(b_frag1[0]), "=r"(b_frag1[1])
                : "r"(bcur + b_ldm_off + nj * 8 * SP_BLOCK_K)
            );

            #pragma unroll
            for (int mi = 0; mi < TILES_M; mi++) {
                // One m16n8k64 sparse MMA covers the whole 64-deep K step: A is the
                // 16x32 compressed tile {a_frag0, a_frag1}, B the 64x8 tile {b_frag0,
                // b_frag1}. Metadata: lanes tid=0/1 hold rows g/g+8 for K[0:31], lanes
                // tid=2/3 hold rows g/g+8 for K[32:63] (same E as the two k32 MMAs).
                asm volatile(
                    "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9, %10, %11}, "
                    "{%12, %13, %14, %15}, "
                    "%16, 0x0;\n"
                    : "=r"(acc[mi][nj][0]), "=r"(acc[mi][nj][1]),
                      "=r"(acc[mi][nj][2]), "=r"(acc[mi][nj][3])
                    : "r"(a_frag0[mi][0]), "r"(a_frag0[mi][1]),
                      "r"(a_frag1[mi][0]), "r"(a_frag1[mi][1]),
                      "r"(b_frag0[0]), "r"(b_frag0[1]),
                      "r"(b_frag1[0]), "r"(b_frag1[1]),
                      "r"(acc[mi][nj][0]), "r"(acc[mi][nj][1]),
                      "r"(acc[mi][nj][2]), "r"(acc[mi][nj][3]),
                      "r"(e_meta[mi])
                );
            }
        }
        }   // warp-live

        // Rotate stage offsets
        comp_off  = (comp_off  + SMEM_PER_STAGE == DB_STAGES * SMEM_PER_STAGE) ? 0 : comp_off  + SMEM_PER_STAGE;
        ahead_off = (ahead_off + SMEM_PER_STAGE == DB_STAGES * SMEM_PER_STAGE) ? 0 : ahead_off + SMEM_PER_STAGE;
    }

    // ================================================================
    //  Epilogue: dequant + bias in registers, transposed store
    //
    //  MMA output is C(N_out, M) in fragment layout: acc[mi][nj] holds
    //  (wr = m0+g, bc = n0+2tid / n0+2tid+1) and (wr = m0+g+8, same bc).
    //  Output is (M, N_out) row-major, so consecutive wr must be adjacent.
    //  Each 8x8 block is packed to half2 (row wr, cols bc, bc+1) and transposed
    //  in registers with movmatrix: afterwards the lane holds (row bc = n0+g,
    //  cols wr = m0+2tid, m0+2tid+1) -> one aligned 4-byte store per block.
    //  No shared memory, so no extra CTA barrier before the next tile.
    // ================================================================
    {
        // Per-token activation scale: global, or (FQ) the block's smem copy
        auto act_sc = [&](int bc) -> float {
            if constexpr (FQ)
                return reinterpret_cast<const float*>(smem_raw + SMEM_FQ_B + BLOCK_BC_T * K)[bc - block_bc];
            else
                return act_scale[min(bc, M - 1)];
        };
        float ws_r[TILES_M][2], bias_r[TILES_M][2];
        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++) {
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int wr = min(block_wr + warp_wr + mi * 16 + groupID + h * 8, N_out - 1);
                ws_r[mi][h]   = __half2float(w_scale[wr]);
                bias_r[mi][h] = (bias != nullptr && splitk_ws == nullptr)
                              ? __half2float(bias[wr]) : 0.0f;
            }
        }

        if (splitk_ws != nullptr) {
            // Split-K: fp32 partials accumulated with atomics (bias in reduce kernel)
            #pragma unroll
            for (int nj = 0; nj < TILES_N; nj++) {
                const int bc_frag = block_bc + warp_bc + nj * 8 + tid * 2;   // fragment columns
                const float as0 = act_sc(bc_frag);
                const float as1 = act_sc(bc_frag + 1);
                #pragma unroll
                for (int mi = 0; mi < TILES_M; mi++) {
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int wr = block_wr + warp_wr + mi * 16 + groupID + h * 8;
                        if (wr < N_out) {
                            if (bc_frag < M)
                                atomicAdd(&splitk_ws[(size_t)bc_frag * N_out + wr],
                                          (float)acc[mi][nj][2 * h] * (as0 * ws_r[mi][h]));
                            if (bc_frag + 1 < M)
                                atomicAdd(&splitk_ws[(size_t)(bc_frag + 1) * N_out + wr],
                                          (float)acc[mi][nj][2 * h + 1] * (as1 * ws_r[mi][h]));
                        }
                    }
                }
            }
        } else {
            // Stage the warp's WARP_BC x WARP_WR fp16 tile in shared memory as output rows
            // (bc-major), then write 16-byte vectors: a warp store covers whole 64/128-byte
            // row segments instead of 8 rows x 16 bytes. The pipeline buffers are free once
            // every warp has finished its last K step (barrier below).
            constexpr int ROW_B = WARP_WR * 2;              // staged row pitch (bytes)
            constexpr int CPR   = WARP_WR / 8;              // 16-byte chunks per row (8 or 4)
            static_assert(CPR == 8 || CPR == 4, "staging swizzle assumes 4 or 8 chunks per row");
            static_assert(SP_WARPS_WR * SP_WARPS_BC * WARP_BC * ROW_B <= (FQ ? SMEM_FQ_B : SMEM_DB),
                          "epilogue staging must fit in the pipeline buffers");
            // XOR swizzle of the 16-byte chunk: conflict-free for the 8-row fragment stores
            // and for the 16-byte row reads.
            auto swz = [](int r) { return (CPR == 8) ? (r & 7) : ((r >> 1) & 3); };
            const uint32_t stg = smem_base + (uint32_t)(warp * WARP_BC * ROW_B);
            __syncthreads();

            #pragma unroll
            for (int nj = 0; nj < TILES_N; nj++) {
                const int bc_frag = block_bc + warp_bc + nj * 8 + tid * 2;   // fragment columns
                const float as0 = act_sc(bc_frag);
                const float as1 = act_sc(bc_frag + 1);
                const int r = nj * 8 + groupID;                              // row after transpose
                #pragma unroll
                for (int mi = 0; mi < TILES_M; mi++) {
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        float v0 = (float)acc[mi][nj][2 * h]     * (as0 * ws_r[mi][h]) + bias_r[mi][h];
                        float v1 = (float)acc[mi][nj][2 * h + 1] * (as1 * ws_r[mi][h]) + bias_r[mi][h];
                        __half2 hv = __floats2half2_rn(v0, v1);
                        uint32_t packed = *reinterpret_cast<uint32_t*>(&hv);
                        uint32_t trans;
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                                     : "=r"(trans) : "r"(packed));
                        const int c = mi * 2 + h;                            // 16-byte chunk
                        asm volatile("st.shared.u32 [%0], %1;\n" ::
                            "r"(stg + r * ROW_B + ((c ^ swz(r)) << 4) + tid * 4), "r"(trans));
                    }
                }
            }
            __syncwarp();

            #pragma unroll
            for (int i = lane; i < WARP_BC * CPR; i += 32) {
                const int r = i / CPR, c = i % CPR;
                uint32_t v0, v1, v2, v3;
                asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                             : "r"(stg + r * ROW_B + ((c ^ swz(r)) << 4)));
                const int bc = block_bc + warp_bc + r;
                const int wr = block_wr + warp_wr + c * 8;
                // N_out % 16 == 0, so an 8-column chunk is entirely in or out of range
                if (bc < M && wr < N_out)
                    *reinterpret_cast<uint4*>(output + (size_t)bc * N_out + wr) =
                        make_uint4(v0, v1, v2, v3);
            }
        }
    }

    // Next tile's prologue overwrites pipeline stages (and the epilogue staging) that
    // other warps may still be reading.
    __syncthreads();
    } // end persistent tile loop

    #undef LOAD_TILE
}

// -- Thin kernel wrappers: same body, different launch_bounds --

#define SPARSE_GEMM_PARAMS                                          \
    const int8_t*   __restrict__ W_comp,                            \
    const int8_t*   __restrict__ x_int8,                            \
    const uint32_t* __restrict__ w_meta,                            \
    const float*    __restrict__ act_scale,                         \
    const __half*   __restrict__ w_scale,                           \
    const __half*   __restrict__ bias,                              \
    __half*         __restrict__ output,                            \
    float*          __restrict__ splitk_ws,                         \
    const int M, const int K, const int N_out,                     \
    const int split_k, const int k_iters_per_split,                \
    const __half* __restrict__ x_fp16, const int x_stream

#define SPARSE_GEMM_ARGS \
    W_comp, x_int8, w_meta, act_scale, w_scale, bias, output, \
    splitk_ws, M, K, N_out, split_k, k_iters_per_split, x_fp16, x_stream

// Occupancy-1 kernel (original behavior, deep pipeline).
// Cap smem at 48KB to stay in 64KB config tier -> preserve 64KB L1 cache.
template <int BLOCK_WR_T, int BLOCK_BC_T, bool NF = false>
__global__ void sparse_gemm_kernel(SPARSE_GEMM_PARAMS) {
    constexpr int STAGES = sp_stages<BLOCK_WR_T, BLOCK_BC_T>();
    sparse_gemm_impl<BLOCK_WR_T, BLOCK_BC_T, STAGES, false, NF>(SPARSE_GEMM_ARGS);
}

// Fused-quantize kernel: reads the fp16 activation, quantizes each M-tile's rows into a
// resident smem block, runs every weight tile against it (see sparse_int8_linear_ex).
template <int BLOCK_WR_T, int BLOCK_BC_T>
__global__ void sparse_gemm_fq_kernel(SPARSE_GEMM_PARAMS) {
    constexpr int STAGES = sp_stages<BLOCK_WR_T, BLOCK_BC_T>();
    sparse_gemm_impl<BLOCK_WR_T, BLOCK_BC_T, STAGES, true>(SPARSE_GEMM_ARGS);
}

// Occupancy-2 kernel: __launch_bounds__(256, 2) forces <=128 regs/thread.
// Uses fewer pipeline stages to keep smem <= 32KB/block (2 blocks x 32KB = 64KB config).
template <int BLOCK_WR_T, int BLOCK_BC_T>
__global__ void __launch_bounds__(SP_THREADS, 2)
sparse_gemm_occ2_kernel(SPARSE_GEMM_PARAMS) {
    constexpr int PER_STAGE = BLOCK_WR_T * SP_K_COMP + BLOCK_BC_T * SP_BLOCK_K;
    constexpr int STAGES = (PER_STAGE * 4 <= 32 * 1024) ? 4
                         : (PER_STAGE * 3 <= 32 * 1024) ? 3 : 2;
    sparse_gemm_impl<BLOCK_WR_T, BLOCK_BC_T, STAGES>(SPARSE_GEMM_ARGS);
}

// ===============================================================================
//  Warp-Specialized Sparse GEMM Kernel
//
//  Eliminates __syncthreads() from the K-loop by splitting warps into:
//    - 2 producer warps (warps 0-1): cp.async loads from global -> shared memory
//    - 6 consumer warps (warps 2-7): MMA computation + epilogue
//
//  Producer/consumer communication via shared memory flags + named barriers.
//  Producers load next tile while consumers compute current tile -> overlap.
//  This targets the 47% Wait stall measured by Nsight Compute at occupancy 1.
// ===============================================================================

constexpr int WS_NUM_PROD_WARPS = 2;
constexpr int WS_NUM_CONS_WARPS = 6;
constexpr int WS_PROD_THREADS = WS_NUM_PROD_WARPS * 32;   // 64
constexpr int WS_CONS_THREADS = WS_NUM_CONS_WARPS * 32;    // 192
constexpr int WS_CONS_WARPS_WR = 3;  // 3 warp rows along weight (N_out)
constexpr int WS_CONS_WARPS_BC = 2;  // 2 warp cols along batch (M)

template <int BLOCK_WR_T, int BLOCK_BC_T>
__global__ void sparse_gemm_warpspec_kernel(
    const int8_t*   __restrict__ W_comp,     // (N_out, K/2) compressed weight
    const int8_t*   __restrict__ x_int8,     // (M, K) quantized activation
    const uint32_t* __restrict__ w_meta,     // (N_out, K/32) packed metadata
    const float*    __restrict__ act_scale,  // (M,) fp32
    const __half*   __restrict__ w_scale,    // (N_out,) fp16
    const __half*   __restrict__ bias,       // (N_out,) fp16 or nullptr
    __half*         __restrict__ output,     // (M, N_out) fp16
    float*          __restrict__ splitk_ws,  // split-K fp32 workspace, or nullptr
    const int M, const int K, const int N_out,
    const int split_k,
    const int k_iters_per_split
) {
    // -- Consumer warp-grid derived constants --
    constexpr int WARP_WR = BLOCK_WR_T / WS_CONS_WARPS_WR;    // 64
    constexpr int WARP_BC = BLOCK_BC_T / WS_CONS_WARPS_BC;     // 64
    constexpr int TILES_M = WARP_WR / 16;                      // 4
    constexpr int TILES_N = WARP_BC / 8;                        // 8

    // Pipeline staging
    constexpr int SMEM_PER_STAGE_RAW = BLOCK_WR_T * SP_K_COMP + BLOCK_BC_T * SP_BLOCK_K;
    constexpr int DB_STAGES = (SMEM_PER_STAGE_RAW * 4 <= 48 * 1024) ? 4 : 3;
    constexpr int SMEM_A_STAGE = BLOCK_WR_T * SP_K_COMP;
    constexpr int SMEM_B_STAGE = BLOCK_BC_T * SP_BLOCK_K;
    constexpr int SMEM_PER_STAGE = SMEM_A_STAGE + SMEM_B_STAGE;
    constexpr int SMEM_DB = DB_STAGES * SMEM_PER_STAGE;
    constexpr int WS_SMEM_EPILOGUE = WS_NUM_CONS_WARPS * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
    constexpr int SMEM_BYTES = (SMEM_DB > WS_SMEM_EPILOGUE) ? SMEM_DB : WS_SMEM_EPILOGUE;

    // Load chunk counts
    constexpr int TOTAL_A_16 = BLOCK_WR_T * SP_K_COMP / 16;
    constexpr int TOTAL_B_16 = BLOCK_BC_T * SP_BLOCK_K / 16;

    // -- Thread identity --
    const int t        = threadIdx.x;
    const int warp     = t / 32;
    const int lane     = t % 32;
    const int groupID  = lane >> 2;
    const int tid      = lane & 3;

    // -- Warp roles --
    const bool is_producer = (warp < WS_NUM_PROD_WARPS);
    const int cons_warp    = warp - WS_NUM_PROD_WARPS;  // 0-5 for consumers
    const int c_warp_wr    = (cons_warp / WS_CONS_WARPS_BC) * WARP_WR;
    const int c_warp_bc    = (cons_warp % WS_CONS_WARPS_BC) * WARP_BC;

    // -- Dimension constants --
    const int K_comp        = K / 2;
    const int meta_cols     = K / 32;
    const int K_iters_total = K / SP_BLOCK_K;

    // -- Persistent kernel tile grid --
    const int tiles_x       = (N_out + BLOCK_WR_T - 1) / BLOCK_WR_T;
    const int tiles_y       = (M     + BLOCK_BC_T - 1) / BLOCK_BC_T;
    const int spatial_tiles = tiles_x * tiles_y;
    const int total_tiles   = spatial_tiles * split_k;
    constexpr int SWIZZLE_W = 4;

    // -- Shared memory (dynamic for large tiles, shared with old kernel) --
    extern __shared__ int8_t smem_raw[];
    __shared__ volatile int ws_data_ready[DB_STAGES];
    __shared__ volatile int ws_buf_consumed[DB_STAGES];

    #define SMEM_A(stage) reinterpret_cast<int8_t (*)[SP_K_COMP]>( \
        smem_raw + (stage) * SMEM_PER_STAGE)
    #define SMEM_B(stage) reinterpret_cast<int8_t (*)[SP_BLOCK_K]>( \
        smem_raw + (stage) * SMEM_PER_STAGE + SMEM_A_STAGE)

    // -- Producer load macro (64 threads stride) --
    #define LOAD_TILE_PROD(STAGE, K_DENSE_OFF) do {                                \
        const int _k_comp_off = (K_DENSE_OFF) / 2;                                \
        auto _sA = SMEM_A(STAGE);                                                 \
        auto _sB = SMEM_B(STAGE);                                                 \
        for (int _i = t; _i < TOTAL_A_16; _i += WS_PROD_THREADS) {               \
            const int _row  = _i / (SP_K_COMP / 16);                              \
            const int _col4 = _i % (SP_K_COMP / 16);                              \
            const int _gr = block_wr + _row;                                       \
            const int _sc = a_swiz(_row, _col4 * 16);                             \
            if (_gr < N_out)                                                       \
                cp_async_16(&_sA[_row][_sc],                                       \
                    W_comp + (size_t)_gr * K_comp + _k_comp_off + _col4*16);       \
            else                                                                   \
                *reinterpret_cast<int4*>(&_sA[_row][_sc]) = make_int4(0,0,0,0);   \
        }                                                                          \
        for (int _i = t; _i < TOTAL_B_16; _i += WS_PROD_THREADS) {               \
            const int _row  = _i / (SP_BLOCK_K / 16);                             \
            const int _col4 = _i % (SP_BLOCK_K / 16);                             \
            const int _gr = block_bc + _row;                                       \
            const int _sc = b_swiz(_row, _col4 * 16);                             \
            if (_gr < M)                                                           \
                cp_async_16_stream(&_sB[_row][_sc],                                \
                    x_int8 + (size_t)_gr * K + (K_DENSE_OFF) + _col4 * 16);       \
            else                                                                   \
                *reinterpret_cast<int4*>(&_sB[_row][_sc]) = make_int4(0,0,0,0);   \
        }                                                                          \
        cp_async_commit();                                                         \
    } while (0)

    // ==================================================================
    //  Persistent tile loop
    // ==================================================================
    const int tiles_per_cta = (total_tiles + (int)gridDim.x - 1) / (int)gridDim.x;
    const int tile_start    = blockIdx.x * tiles_per_cta;
    const int tile_end      = min(tile_start + tiles_per_cta, total_tiles);

    for (int tile_id = tile_start; tile_id < tile_end; tile_id++) {

    // -- Decode tile_id -> (spatial_id, k_split_id) --
    const int k_split_id = tile_id / spatial_tiles;
    const int spatial_id = tile_id % spatial_tiles;

    int tile_n, tile_m;
    if (tiles_x >= 2 * SWIZZLE_W) {
        const int group_tiles = SWIZZLE_W * tiles_y;
        const int group_id    = spatial_id / group_tiles;
        const int within      = spatial_id % group_tiles;
        tile_n = (within / tiles_y) + group_id * SWIZZLE_W;
        tile_m = within % tiles_y;
        if (tile_n >= tiles_x) continue;
    } else {
        tile_n = spatial_id % tiles_x;
        tile_m = spatial_id / tiles_x;
    }

    const int k_start_iter  = k_split_id * k_iters_per_split;
    const int k_end_iter    = min(k_start_iter + k_iters_per_split, K_iters_total);
    const int local_K_iters = k_end_iter - k_start_iter;
    if (local_K_iters <= 0) { __syncthreads(); continue; }

    const int block_wr = tile_n * BLOCK_WR_T;
    const int block_bc = tile_m * BLOCK_BC_T;

    // -- Initialize producer/consumer flags --
    if (t < DB_STAGES) {
        ws_data_ready[t]   = 0;
        ws_buf_consumed[t] = DB_STAGES;  // all buffers start available
    }
    __syncthreads();  // one-time per tile (outside K-loop)

    // ==============================================================
    //  PRODUCER WARPS (warps 0-1): load tiles via cp.async
    // ==============================================================
    if (is_producer) {

        for (int ki = 0; ki < local_K_iters; ki++) {
            const int stage       = ki % DB_STAGES;
            const int k_dense_off = (k_start_iter + ki) * SP_BLOCK_K;

            // Wait for consumers to release this buffer (skip first DB_STAGES)
            if (ki >= DB_STAGES) {
                if (lane == 0 && warp == 0)
                    while (ws_buf_consumed[stage] < ki) {}
                asm volatile("bar.sync 1, %0;" :: "r"(WS_PROD_THREADS));
            }

            // Load A + B tiles into smem stage
            LOAD_TILE_PROD(stage, k_dense_off);

            // Wait for all cp.async to land
            cp_async_wait_group<0>();
            asm volatile("bar.sync 1, %0;" :: "r"(WS_PROD_THREADS));

            // Signal: data is ready in this stage
            __threadfence_block();
            if (t == 0) ws_data_ready[stage] = ki + 1;
        }
        // Producers idle until tile-end sync

    }

    // ==============================================================
    //  CONSUMER WARPS (warps 2-7): MMA computation + epilogue
    // ==============================================================
    if (!is_producer) {

        // -- Accumulators --
        int32_t acc[TILES_M][TILES_N][4];
        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++)
            #pragma unroll
            for (int nj = 0; nj < TILES_N; nj++) {
                acc[mi][nj][0] = 0; acc[mi][nj][1] = 0;
                acc[mi][nj][2] = 0; acc[mi][nj][3] = 0;
            }

        // -- K-loop (no __syncthreads -- flag-based sync) --
        for (int ki = 0; ki < local_K_iters; ki++) {
            const int k_abs   = k_start_iter + ki;
            const int k_dense = k_abs * SP_BLOCK_K;
            const int stage   = ki % DB_STAGES;

            // Wait for producers to signal data ready
            if (lane == 0 && cons_warp == 0)
                while (ws_data_ready[stage] < ki + 1) {}
            asm volatile("bar.sync 2, %0;" :: "r"(WS_CONS_THREADS));
            __threadfence_block();

            // -- Tile pointers --
            auto cA = SMEM_A(stage);
            auto cB = SMEM_B(stage);

            // -- Metadata prefetch --
            const int k_meta0 = k_dense / 32;
            const int k_meta1 = k_meta0 + 1;

            uint32_t e_meta_prefetch[TILES_M];
            #pragma unroll
            for (int mi = 0; mi < TILES_M; mi++) {
                const int m_local  = c_warp_wr + mi * 16;
                const int m_global = block_wr + m_local;
                uint32_t e;
                if (tid == 0) {
                    e = (m_global + groupID < N_out)
                        ? w_meta[(size_t)(m_global + groupID) * meta_cols + k_meta0]
                        : 0x44444444u;
                } else if (tid == 1) {
                    e = (m_global + groupID + 8 < N_out)
                        ? w_meta[(size_t)(m_global + groupID + 8) * meta_cols + k_meta0]
                        : 0x44444444u;
                } else if (tid == 2) {
                    e = (m_global + groupID < N_out)
                        ? w_meta[(size_t)(m_global + groupID) * meta_cols + k_meta1]
                        : 0x44444444u;
                } else {
                    e = (m_global + groupID + 8 < N_out)
                        ? w_meta[(size_t)(m_global + groupID + 8) * meta_cols + k_meta1]
                        : 0x44444444u;
                }
                e_meta_prefetch[mi] = e;
            }

            // -- Per-warp MMA computation --
            #pragma unroll
            for (int mi = 0; mi < TILES_M; mi++) {
                const int m_local = c_warp_wr + mi * 16;
                const uint32_t e_meta = e_meta_prefetch[mi];

                const int a_row0 = m_local + groupID;
                const int a_row1 = m_local + groupID + 8;
                uint32_t a_frag0[2], a_frag1[2];
                a_frag0[0] = *reinterpret_cast<const uint32_t*>(
                    &cA[a_row0][a_swiz(a_row0, tid * 4)]);
                a_frag0[1] = *reinterpret_cast<const uint32_t*>(
                    &cA[a_row1][a_swiz(a_row1, tid * 4)]);
                a_frag1[0] = *reinterpret_cast<const uint32_t*>(
                    &cA[a_row0][a_swiz(a_row0, 16 + tid * 4)]);
                a_frag1[1] = *reinterpret_cast<const uint32_t*>(
                    &cA[a_row1][a_swiz(a_row1, 16 + tid * 4)]);

                #pragma unroll
                for (int nj = 0; nj < TILES_N; nj++) {
                    const int n_local = c_warp_bc + nj * 8 + groupID;

                    uint32_t b_frag0[2];
                    b_frag0[0] = *reinterpret_cast<const uint32_t*>(
                        &cB[n_local][b_swiz(n_local, tid * 4)]);
                    b_frag0[1] = *reinterpret_cast<const uint32_t*>(
                        &cB[n_local][b_swiz(n_local, 16 + tid * 4)]);

                    // MMA 1: spsel=0, K[0:31]
                    asm volatile(
                        "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5}, "
                        "{%6, %7}, "
                        "{%8, %9, %10, %11}, "
                        "%12, 0x0;\n"
                        : "=r"(acc[mi][nj][0]), "=r"(acc[mi][nj][1]),
                          "=r"(acc[mi][nj][2]), "=r"(acc[mi][nj][3])
                        : "r"(a_frag0[0]), "r"(a_frag0[1]),
                          "r"(b_frag0[0]), "r"(b_frag0[1]),
                          "r"(acc[mi][nj][0]), "r"(acc[mi][nj][1]),
                          "r"(acc[mi][nj][2]), "r"(acc[mi][nj][3]),
                          "r"(e_meta)
                    );

                    uint32_t b_frag1[2];
                    b_frag1[0] = *reinterpret_cast<const uint32_t*>(
                        &cB[n_local][b_swiz(n_local, 32 + tid * 4)]);
                    b_frag1[1] = *reinterpret_cast<const uint32_t*>(
                        &cB[n_local][b_swiz(n_local, 48 + tid * 4)]);

                    // MMA 2: spsel=1, K[32:63]
                    asm volatile(
                        "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5}, "
                        "{%6, %7}, "
                        "{%8, %9, %10, %11}, "
                        "%12, 0x1;\n"
                        : "=r"(acc[mi][nj][0]), "=r"(acc[mi][nj][1]),
                          "=r"(acc[mi][nj][2]), "=r"(acc[mi][nj][3])
                        : "r"(a_frag1[0]), "r"(a_frag1[1]),
                          "r"(b_frag1[0]), "r"(b_frag1[1]),
                          "r"(acc[mi][nj][0]), "r"(acc[mi][nj][1]),
                          "r"(acc[mi][nj][2]), "r"(acc[mi][nj][3]),
                          "r"(e_meta)
                    );
                }
            }

            // Signal: buffer consumed (producers can reuse this stage)
            asm volatile("bar.sync 2, %0;" :: "r"(WS_CONS_THREADS));
            __threadfence_block();
            if (t == WS_PROD_THREADS)   // first consumer thread
                ws_buf_consumed[stage] = ki + DB_STAGES;
        }

        // -- Epilogue: dequant + bias + transposed write --
        int32_t (*smem_stage)[16][SP_STAGE_PAD] =
            reinterpret_cast<int32_t (*)[16][SP_STAGE_PAD]>(smem_raw);

        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++) {
            #pragma unroll
            for (int nj = 0; nj < TILES_N; nj++) {
                smem_stage[cons_warp][groupID    ][tid * 2]     = acc[mi][nj][0];
                smem_stage[cons_warp][groupID    ][tid * 2 + 1] = acc[mi][nj][1];
                smem_stage[cons_warp][groupID + 8][tid * 2]     = acc[mi][nj][2];
                smem_stage[cons_warp][groupID + 8][tid * 2 + 1] = acc[mi][nj][3];
                __syncwarp();

                const int tile_wr = block_wr + c_warp_wr + mi * 16;
                const int tile_bc = block_bc + c_warp_bc + nj * 8;

                #pragma unroll
                for (int i = lane; i < 128; i += 32) {
                    const int bc_local = i / 16;
                    const int wr_local = i % 16;
                    const int wr = tile_wr + wr_local;
                    const int bc = tile_bc + bc_local;

                    if (wr < N_out && bc < M) {
                        float val = static_cast<float>(smem_stage[cons_warp][wr_local][bc_local]);
                        val *= act_scale[bc] * __half2float(w_scale[wr]);
                        if (splitk_ws != nullptr) {
                            atomicAdd(&splitk_ws[(size_t)bc * N_out + wr], val);
                        } else {
                            if (bias != nullptr)
                                val += __half2float(bias[wr]);
                            output[(size_t)bc * N_out + wr] = __float2half(val);
                        }
                    }
                }
                __syncwarp();
            }
        }

    } // end consumer

    __syncthreads();  // ensure epilogue writes complete before next tile
    } // end persistent tile loop

    #undef LOAD_TILE_PROD
    #undef SMEM_A
    #undef SMEM_B
}

// ===============================================================================
//  Shared memory size helpers (host-side, matches kernel-side constants)
// ===============================================================================

template <int BW, int BC>
constexpr int sparse_gemm_smem_bytes() {
    constexpr int STAGES = sp_stages<BW, BC>();
    constexpr int SMEM_DB = sp_smem_bytes<BW, BC, STAGES>();
    constexpr int SMEM_EPI = 8 * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
    return (SMEM_DB > SMEM_EPI) ? SMEM_DB : SMEM_EPI;
}

template <int BW, int BC>
constexpr int warpspec_smem_bytes() {
    constexpr int PER_STAGE = BW * SP_K_COMP + BC * SP_BLOCK_K;
    constexpr int STAGES = (PER_STAGE * 4 <= 100 * 1024) ? 4 : 3;
    constexpr int SMEM_DB = STAGES * PER_STAGE;
    constexpr int SMEM_EPI = WS_NUM_CONS_WARPS * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
    return (SMEM_DB > SMEM_EPI) ? SMEM_DB : SMEM_EPI;
}

template <int BW, int BC>
constexpr int occ2_smem_bytes() {
    constexpr int PER_STAGE = BW * SP_K_COMP + BC * SP_BLOCK_K;
    // Fewer stages to keep per-block smem <= 32KB (2 blocks x 32KB = 64KB config)
    constexpr int STAGES = (PER_STAGE * 4 <= 32 * 1024) ? 4
                         : (PER_STAGE * 3 <= 32 * 1024) ? 3 : 2;
    constexpr int SMEM_DB = sp_smem_bytes<BW, BC, STAGES>();
    constexpr int SMEM_EPI = 8 * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
    return (SMEM_DB > SMEM_EPI) ? SMEM_DB : SMEM_EPI;
}

// ===============================================================================
//  C++ wrappers
// ===============================================================================

std::vector<torch::Tensor> sparse_prune_compress(torch::Tensor weight_int8) {
    TORCH_CHECK(weight_int8.is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(weight_int8.dtype() == torch::kInt8, "Weight must be int8");
    TORCH_CHECK(weight_int8.dim() == 2, "Weight must be 2D (N, K)");
    TORCH_CHECK(weight_int8.is_contiguous(), "Weight must be contiguous");

    const int N = weight_int8.size(0);
    const int K = weight_int8.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4");

    auto compressed = torch::zeros({N, K / 2},
        torch::dtype(torch::kInt8).device(weight_int8.device()));
    auto meta_raw = torch::zeros({N * (K / 4)},
        torch::dtype(torch::kUInt8).device(weight_int8.device()));

    const int total_groups = N * (K / 4);
    const int threads = 256;
    const int blocks = (total_groups + threads - 1) / threads;

    prune_compress_kernel<<<blocks, threads>>>(
        weight_int8.data_ptr<int8_t>(),
        compressed.data_ptr<int8_t>(),
        meta_raw.data_ptr<uint8_t>(),
        N, K
    );

    return {compressed, meta_raw};
}

torch::Tensor test_mma_sp(
    torch::Tensor A_comp,
    torch::Tensor B_dense,
    torch::Tensor meta
) {
    TORCH_CHECK(A_comp.is_cuda() && A_comp.dtype() == torch::kInt8);
    TORCH_CHECK(A_comp.size(0) == 16 && A_comp.size(1) == 16,
                "A_comp must be (16, 16)");
    TORCH_CHECK(B_dense.is_cuda() && B_dense.dtype() == torch::kInt8);
    TORCH_CHECK(B_dense.size(0) == 8 && B_dense.size(1) == 32,
                "B_dense must be (8, 32) row-major");
    TORCH_CHECK(meta.is_cuda() && meta.dtype() == torch::kInt32);
    TORCH_CHECK(meta.numel() == 32, "meta must have 32 elements (per-lane)");

    auto C = torch::zeros({16, 8},
        torch::dtype(torch::kInt32).device(A_comp.device()));

    test_mma_sp_kernel<<<1, 32>>>(
        A_comp.contiguous().data_ptr<int8_t>(),
        B_dense.contiguous().data_ptr<int8_t>(),
        reinterpret_cast<const uint32_t*>(meta.contiguous().data_ptr<int32_t>()),
        C.data_ptr<int32_t>()
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return C;
}

std::vector<torch::Tensor> sparse_prune_and_pack(torch::Tensor weight_int8) {
    /*
     * Full weight preparation: prune + compress + pack metadata.
     * Returns {compressed (N, K/2) int8, meta_packed (N, K/32) int32}.
     */
    TORCH_CHECK(weight_int8.is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(weight_int8.dtype() == torch::kInt8, "Weight must be int8");
    TORCH_CHECK(weight_int8.dim() == 2, "Weight must be 2D (N, K)");
    TORCH_CHECK(weight_int8.is_contiguous(), "Weight must be contiguous");

    const int N = weight_int8.size(0);
    const int K = weight_int8.size(1);
    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");

    // Step 1: Prune + compress -> compressed (N, K/2) + meta_raw (N*K/4) uint8
    auto compressed = torch::zeros({N, K / 2},
        torch::dtype(torch::kInt8).device(weight_int8.device()));
    auto meta_raw = torch::zeros({N * (K / 4)},
        torch::dtype(torch::kUInt8).device(weight_int8.device()));

    {
        const int total_groups = N * (K / 4);
        const int threads = 256;
        const int blocks = (total_groups + threads - 1) / threads;
        prune_compress_kernel<<<blocks, threads>>>(
            weight_int8.data_ptr<int8_t>(),
            compressed.data_ptr<int8_t>(),
            meta_raw.data_ptr<uint8_t>(),
            N, K
        );
    }

    // Step 2: Pack metadata -> meta_packed (N, K/32) int32
    const int chunks_per_row = K / 32;
    auto meta_packed = torch::empty({N, chunks_per_row},
        torch::dtype(torch::kInt32).device(weight_int8.device()));

    {
        const int total = N * chunks_per_row;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        pack_metadata_kernel<<<blocks, threads>>>(
            meta_raw.data_ptr<uint8_t>(),
            reinterpret_cast<uint32_t*>(meta_packed.data_ptr<int32_t>()),
            N, K
        );
    }

    return {compressed, meta_packed};
}

// Quantize activation (M, K) fp16 -> int8 (M, K) + per-token fp32 scales (M,).
std::vector<torch::Tensor> sparse_quantize_act(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16, "x must be fp16 CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (M, K)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    const int M = x.size(0);
    const int K = x.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    // Step 1: Quantize activation (M, K) fp16 -> int8 + scales
    auto x_int8    = torch::empty({M, K}, torch::dtype(torch::kInt8).device(x.device()));
    auto act_scale = torch::empty({M},    torch::dtype(torch::kFloat32).device(x.device()));
    const __half* x_ptr = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const bool vec_ok = (K % 8 == 0) && (K <= 4 * 8 * 256)
                     && (reinterpret_cast<uintptr_t>(x_ptr) % 16 == 0);
    if (vec_ok) {
        const int nvec    = K / 8;
        const int threads = min(256, ((nvec + 31) / 32) * 32);
        const int vpt     = (nvec + threads - 1) / threads;          // 1..4
        const bool big = (size_t)M * K * sizeof(__half)
                       > (size_t)at::cuda::getCurrentDeviceProperties()->l2CacheSize;
        #define LAUNCH_QVEC(V) do {                                                    \
            if (big)                                                                   \
                _sp_quantize_vec_kernel<V, true><<<M, threads, 0, stream>>>(           \
                    x_ptr, x_int8.data_ptr<int8_t>(), act_scale.data_ptr<float>(), K); \
            else                                                                       \
                _sp_quantize_vec_kernel<V, false><<<M, threads, 0, stream>>>(          \
                    x_ptr, x_int8.data_ptr<int8_t>(), act_scale.data_ptr<float>(), K); \
        } while (0)
        if (vpt == 1)      LAUNCH_QVEC(1);
        else if (vpt == 2) LAUNCH_QVEC(2);
        else if (vpt == 3) LAUNCH_QVEC(3);
        else               LAUNCH_QVEC(4);
        #undef LAUNCH_QVEC
    } else {
        const int threads = min(256, ((K + 31) / 32) * 32);
        _sp_fused_quantize_kernel<<<M, threads, 0, stream>>>(
            x_ptr, x_int8.data_ptr<int8_t>(), act_scale.data_ptr<float>(), K);
    }

    return {x_int8, act_scale};
}

// Dispatch: pick (tile, split-K) by a wave-aware cost model fitted on an RTX 4090
// sweep (K 1280..6144 x N 768..6144 x M 64..8192, every tile x split):
//   t(tile, s) = waves * (a_tile * ceil(ksteps / s) + b_tile)
//              + [s > 1] * (c0 + (c1 + c2 * s) * M*N/1e6)
//   waves      = ceil(spatial_tiles(tile) * s / SMs)   (one persistent CTA per SM)
// a_tile: time per 64-deep K step of one tile, b_tile: per-tile fixed cost (pipeline
// fill, epilogue), c0/c1/c2: split-K launch, workspace memset + reduce, atomics.
// Splits above 6 lost on every measured shape (atomic contention on the M x N
// partials outgrows the extra parallelism), so they are not candidates.
// Wave quantization dominates small and mid shapes, which fixed M/N thresholds miss.
// The model is compute-side only (every sweep shape keeps its int8 activation in L2);
// DRAM effects are handled by the tile order (sparse_int8_gemm) and by the fused path
// (sp_use_fused).
static const int    kSpTileBW[3] = {256, 128, 128};
static const int    kSpTileBC[3] = {128, 128, 64};
static const double kSpA[3] = {8.1730e-04, 4.0900e-04, 3.1130e-04};   // ms per K step per tile
static const double kSpB[3] = {3.5480e-03, 2.2260e-03, 1.2530e-03};   // ms per tile

struct SpPlan { int tile; int split_k; double cost; };

static SpPlan sp_plan(int M, int K, int N_out, int sms, int force_tile, int force_split) {
    constexpr double kSplitC0 = 2.9200e-03;                      // ms per split-K launch
    constexpr double kSplitC1 = 1.1810e-03;                      // ms per 1e6 outputs
    constexpr double kSplitC2 = 2.4990e-03;                      // ms per 1e6 outputs per split
    static const int kSplits[] = {1, 2, 3, 4, 6};
    const int k_iters_total = K / SP_BLOCK_K;
    SpPlan plan{-1, 1, 1e30};
    for (int t = 0; t < 3; t++) {
        if (force_tile >= 0 && t != force_tile) continue;
        const long long tiles = (long long)((N_out + kSpTileBW[t] - 1) / kSpTileBW[t])
                              * ((M + kSpTileBC[t] - 1) / kSpTileBC[t]);
        for (int s : kSplits) {
            if (force_split >= 1 && s != force_split) continue;
            if (force_split < 1 && s > 1 && s > k_iters_total / 2) continue;
            const long long waves = (tiles * s + sms - 1) / sms;
            const int ksteps = (k_iters_total + s - 1) / s;
            double cost = (double)waves * (kSpA[t] * ksteps + kSpB[t]);
            if (s > 1) cost += kSplitC0 + (kSplitC1 + kSplitC2 * s) * ((double)M * N_out * 1e-6);
            if (cost < plan.cost) plan = {t, s, cost};
        }
    }
    if (force_split >= 1 && plan.tile < 0)   // forced split outside the candidate list
        plan = {(force_tile >= 0) ? force_tile : 1, force_split, 0.0};
    return plan;
}

// Sparse GEMM + dequant + bias on pre-quantized activations.
//   force_tile: -1 = heuristic, 0 = 256x128, 1 = 128x128, 2 = 128x64
//   force_split: -1 = heuristic, >= 1 = that split-K factor
torch::Tensor sparse_int8_gemm(
    torch::Tensor x_int8,      // (M, K) int8 activation
    torch::Tensor act_scale,   // (M,) fp32 per-token scale
    torch::Tensor w_comp,      // (N_out, K/2) int8 compressed weight
    torch::Tensor w_meta,      // (N_out, K/32) int32 packed metadata
    torch::Tensor w_scale,     // (N_out,) fp16 weight scale
    torch::Tensor bias,        // (N_out,) fp16 or empty tensor
    int64_t force_tile,
    int64_t force_split,
    int64_t force_order
) {
    TORCH_CHECK(x_int8.is_cuda() && x_int8.dtype() == torch::kInt8, "x_int8 must be int8 CUDA");
    TORCH_CHECK(x_int8.dim() == 2 && x_int8.is_contiguous(), "x_int8 must be 2D contiguous");
    TORCH_CHECK(act_scale.is_cuda() && act_scale.dtype() == torch::kFloat32, "act_scale must be fp32 CUDA");
    TORCH_CHECK(w_comp.is_cuda() && w_comp.dtype() == torch::kInt8, "w_comp must be int8 CUDA");
    TORCH_CHECK(w_comp.is_contiguous(), "w_comp must be contiguous");
    TORCH_CHECK(w_meta.is_cuda() && w_meta.dtype() == torch::kInt32, "w_meta must be int32 CUDA");
    TORCH_CHECK(w_meta.is_contiguous(), "w_meta must be contiguous");

    const int M     = x_int8.size(0);
    const int K     = x_int8.size(1);
    const int N_out = w_comp.size(0);
    TORCH_CHECK(act_scale.numel() == M, "act_scale must have M elements");
    TORCH_CHECK(w_comp.size(1) == K / 2, "w_comp cols must be K/2");
    TORCH_CHECK(w_meta.size(0) == N_out && w_meta.size(1) == K / 32,
                "w_meta shape must be (N_out, K/32)");
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64 (BLOCK_K=64), got ", K);
    TORCH_CHECK(N_out % 16 == 0, "N_out must be multiple of 16, got ", N_out);

    auto stream = at::cuda::getCurrentCUDAStream();
    const auto& x = x_int8;  // device / options source for allocations below

    // Step 2: Sparse GEMM + dequant + bias -> output (M, N_out) fp16
    auto output = torch::empty({M, N_out},
        torch::dtype(torch::kFloat16).device(x.device()));

    const __half* bias_ptr = (bias.numel() > 0)
        ? reinterpret_cast<const __half*>(bias.data_ptr<at::Half>())
        : nullptr;

    const int SP_MAX_CTAS = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int k_iters_total = K / SP_BLOCK_K;
    const SpPlan plan = sp_plan(M, K, N_out, SP_MAX_CTAS, (int)force_tile, (int)force_split);
    const int tile = plan.tile, split_k = plan.split_k;
    const int BW = kSpTileBW[tile], BC = kSpTileBC[tile];
    const int spatial_tiles = ((N_out + BW - 1) / BW) * ((M + BC - 1) / BC);

    const int k_iters_per_split = (k_iters_total + split_k - 1) / split_k;

    // Tile order: M-fastest keeps a weight tile hot across consecutive M-tiles, but every
    // weight-tile column then streams the whole int8 activation; once that exceeds L2 it is
    // re-read from DRAM per column (3x for N_out = 320 on 128-row tiles). N-fastest reads it once.
    const int n_fastest = (force_order >= 0) ? (int)force_order
        : ((long long)M * K > (long long)at::cuda::getCurrentDeviceProperties()->l2CacheSize) ? 1 : 0;

    // Allocate split-K workspace if needed
    torch::Tensor splitk_ws;
    float* splitk_ws_ptr = nullptr;
    if (split_k > 1) {
        splitk_ws = torch::zeros({M, N_out},
            torch::dtype(torch::kFloat32).device(x.device()));
        splitk_ws_ptr = splitk_ws.data_ptr<float>();
    }

    #define LAUNCH_SPARSE_GEMM(BWv, BCv, NFv) do {                                      \
        /* One persistent CTA per SM. When registers and smem would allow two, the */\
        /* block scheduler can place two of the #SMs CTAs on one SM and leave     */\
        /* another idle (measured: 128x64 at 96 regs ran 12-19% slower), and two  */\
        /* co-resident CTAs were slower even with a 2x grid. Requesting just over */\
        /* half the SM's shared memory guarantees one CTA per SM on any GPU.      */\
        constexpr int _smem_k = sparse_gemm_smem_bytes<(BWv), (BCv)>();            \
        static int _smem = 0;                                                      \
        if (_smem == 0) {                                                          \
            const auto* _p = at::cuda::getCurrentDeviceProperties();               \
            const int _half = (int)_p->sharedMemPerMultiprocessor / 2              \
                            - (int)_p->reservedSharedMemPerBlock + 128;            \
            _smem = (_smem_k > _half) ? _smem_k : _half;                           \
            if (_smem > (int)_p->sharedMemPerBlockOptin) _smem = _smem_k;          \
            if (_smem > 48 * 1024)                                                 \
                cudaFuncSetAttribute(sparse_gemm_kernel<(BWv), (BCv), (NFv)>,      \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, _smem);           \
        }                                                                          \
        const int _total = spatial_tiles * split_k;                                \
        const int _nctas = (_total < SP_MAX_CTAS) ? _total : SP_MAX_CTAS;          \
        sparse_gemm_kernel<(BWv), (BCv), (NFv)><<<_nctas, SP_THREADS, _smem, stream>>>( \
            w_comp.data_ptr<int8_t>(),                                             \
            x_int8.data_ptr<int8_t>(),                                             \
            reinterpret_cast<const uint32_t*>(w_meta.data_ptr<int32_t>()),         \
            act_scale.data_ptr<float>(),                                           \
            reinterpret_cast<const __half*>(w_scale.data_ptr<at::Half>()),         \
            bias_ptr,                                                              \
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),                \
            splitk_ws_ptr,                                                         \
            M, K, N_out,                                                           \
            split_k, k_iters_per_split, nullptr, 0                                 \
        );                                                                         \
    } while (0)

    #define LAUNCH_SPARSE_GEMM_WS(BWv, BCv) do {                                           \
        constexpr int _smem = warpspec_smem_bytes<(BWv), (BCv)>();                          \
        if (_smem > 48 * 1024) {                                                            \
            cudaFuncSetAttribute(sparse_gemm_warpspec_kernel<(BWv), (BCv)>,                 \
                cudaFuncAttributeMaxDynamicSharedMemorySize, _smem);                         \
        }                                                                                   \
        const int _total = spatial_tiles * split_k;                                          \
        const int _nctas = (_total < SP_MAX_CTAS) ? _total : SP_MAX_CTAS;                  \
        sparse_gemm_warpspec_kernel<(BWv), (BCv)><<<_nctas, SP_THREADS, _smem, stream>>>(  \
            w_comp.data_ptr<int8_t>(),                                                       \
            x_int8.data_ptr<int8_t>(),                                                       \
            reinterpret_cast<const uint32_t*>(w_meta.data_ptr<int32_t>()),                   \
            act_scale.data_ptr<float>(),                                                     \
            reinterpret_cast<const __half*>(w_scale.data_ptr<at::Half>()),                   \
            bias_ptr,                                                                        \
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),                          \
            splitk_ws_ptr,                                                                   \
            M, K, N_out,                                                                     \
            split_k, k_iters_per_split                                                       \
        );                                                                                   \
    } while (0)

    #define LAUNCH_SPARSE_GEMM_OCC2(BWv, BCv) do {                                          \
        constexpr int _smem = occ2_smem_bytes<(BWv), (BCv)>();                              \
        if (_smem > 48 * 1024) {                                                            \
            cudaFuncSetAttribute(sparse_gemm_occ2_kernel<(BWv), (BCv)>,                     \
                cudaFuncAttributeMaxDynamicSharedMemorySize, _smem);                         \
        }                                                                                   \
        const int _total = spatial_tiles * split_k;                                          \
        const int _nctas = (_total < 2 * SP_MAX_CTAS) ? _total : 2 * SP_MAX_CTAS;         \
        sparse_gemm_occ2_kernel<(BWv), (BCv)><<<_nctas, SP_THREADS, _smem, stream>>>(      \
            w_comp.data_ptr<int8_t>(),                                                       \
            x_int8.data_ptr<int8_t>(),                                                       \
            reinterpret_cast<const uint32_t*>(w_meta.data_ptr<int32_t>()),                   \
            act_scale.data_ptr<float>(),                                                     \
            reinterpret_cast<const __half*>(w_scale.data_ptr<at::Half>()),                   \
            bias_ptr,                                                                        \
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),                          \
            splitk_ws_ptr,                                                                   \
            M, K, N_out,                                                                     \
            split_k, k_iters_per_split, nullptr, 0                                           \
        );                                                                                   \
    } while (0)

    if (BW == 256) {
        if (n_fastest) LAUNCH_SPARSE_GEMM(256, 128, true);
        else           LAUNCH_SPARSE_GEMM(256, 128, false);
    } else if (BC == 128) {
        if (n_fastest) LAUNCH_SPARSE_GEMM(128, 128, true);
        else           LAUNCH_SPARSE_GEMM(128, 128, false);
    } else {
        if (n_fastest) LAUNCH_SPARSE_GEMM(128, 64, true);
        else           LAUNCH_SPARSE_GEMM(128, 64, false);
    }

    #undef LAUNCH_SPARSE_GEMM
    #undef LAUNCH_SPARSE_GEMM_WS
    #undef LAUNCH_SPARSE_GEMM_OCC2

    // Split-K finalization: reduce fp32 workspace -> fp16 output + bias
    if (split_k > 1) {
        const int total_elements = M * N_out;
        const int reduce_threads = 256;
        const int reduce_blocks = (total_elements + reduce_threads - 1) / reduce_threads;
        splitk_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
            splitk_ws_ptr,
            bias_ptr,
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            N_out, total_elements
        );
    }

    return output;
}

// Fused quantize + sparse GEMM: 128-row weight tiles x 64-token activation blocks; each CTA
// quantizes a block's fp16 rows into shared memory (full K resident) and runs every weight
// tile against it, so the int8 activation never round-trips through DRAM.
constexpr int SP_FQ_BW = 128, SP_FQ_BC = 64;

static int sp_fq_smem_bytes(int K) {
    return sp_fq_base_bytes<SP_FQ_BW, SP_FQ_BC>() + SP_FQ_BC * K + SP_FQ_BC * (int)sizeof(float);
}

// Fused path possible: the resident block fits (K <= 1280 on a 99 KB/block GPU).
static bool sp_fq_fits(int K) {
    return K <= sp_fq_vpt<SP_FQ_BC>() * 256
        && sp_fq_smem_bytes(K) <= (int)at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
}

// Path choice for the fp16-input layer. The two-kernel path writes an int8 copy of x and
// the GEMM reads it back: nearly free while x and the copy (3*M*K bytes) stay in L2, 2*M*K
// bytes of DRAM traffic once they do not (measured cut-over: between 83% and 94% of the
// 72 MiB L2, which also holds weights and output; 90% used). The fused path saves that
// traffic but runs the small 128x64 tile with the weights re-streamed from L2 per 64
// tokens. Take it when the DRAM time saved (at ~0.92 TB/s, RTX 4090 copy bandwidth)
// exceeds the cost-model penalty of the 128x64 tile vs the best two-kernel plan. Matches
// the faster path on all 56 measured narrow/Hunyuan shapes (K 320..1280, N 256..10240,
// M 768..147456). Narrow N_out at huge M takes it (1280->320 at M=147456); wide N_out
// (640x5120, 1024x1280 at M=37k) and every shape whose activation fits L2 (all M <= 8192
// headline shapes) stay on the two-kernel path.
static bool sp_use_fused(int M, int K, int N_out) {
    const auto* p = at::cuda::getCurrentDeviceProperties();
    if (!sp_fq_fits(K)) return false;
    if (3.0 * M * K <= 0.9 * (double)p->l2CacheSize) return false;
    const int sms = p->multiProcessorCount;
    const double fused = sp_plan(M, K, N_out, sms, 2, 1).cost;     // 128x64, no split
    const double best  = sp_plan(M, K, N_out, sms, -1, -1).cost;
    constexpr double kDramBytesPerMs = 0.92e9;
    return fused - best < 2.0 * M * K / kDramBytesPerMs;
}

static torch::Tensor sparse_fq_launch(
    const torch::Tensor& x, const torch::Tensor& w_comp, const torch::Tensor& w_meta,
    const torch::Tensor& w_scale, const torch::Tensor& bias
) {
    constexpr int BW = SP_FQ_BW, BC = SP_FQ_BC;
    const int M = x.size(0), K = x.size(1), N_out = w_comp.size(0);
    TORCH_CHECK(sp_fq_fits(K), "fused sparse path: K=", K, " does not fit shared memory");
    const auto* p = at::cuda::getCurrentDeviceProperties();
    const int smem = sp_fq_smem_bytes(K);
    static int smem_set = 0;   // max dynamic smem attribute set so far (raised on demand)
    if (smem > smem_set) {
        CHECK_CUDA(cudaFuncSetAttribute(sparse_gemm_fq_kernel<BW, BC>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        smem_set = smem;
    }
    auto output = torch::empty({M, N_out}, torch::dtype(torch::kFloat16).device(x.device()));
    if (M == 0) return output;
    const __half* bias_ptr = (bias.numel() > 0)
        ? reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()) : nullptr;
    const int tiles = ((N_out + BW - 1) / BW) * ((M + BC - 1) / BC);
    const int nctas = tiles < p->multiProcessorCount ? tiles : p->multiProcessorCount;
    // Evict-first loads when x exceeds L2: it streams, and L2 is kept for the prefetched rows
    const int x_stream = ((size_t)M * K * sizeof(__half) > (size_t)p->l2CacheSize) ? 1 : 0;
    sparse_gemm_fq_kernel<BW, BC><<<nctas, SP_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        w_comp.data_ptr<int8_t>(), nullptr,
        reinterpret_cast<const uint32_t*>(w_meta.data_ptr<int32_t>()),
        nullptr,
        reinterpret_cast<const __half*>(w_scale.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        nullptr, M, K, N_out, 1, K / SP_BLOCK_K,
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_stream);
    CHECK_CUDA(cudaGetLastError());
    return output;
}

// Full sparse INT8 linear with an explicit path: -1 = auto (sp_use_fused), 0 = quantize
// kernel + sparse GEMM, 1 = fused quantize + GEMM. Outputs are bit-identical across paths.
torch::Tensor sparse_int8_linear_ex(
    torch::Tensor x, torch::Tensor w_comp, torch::Tensor w_meta, torch::Tensor w_scale,
    torch::Tensor bias, int64_t path
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16, "x must be fp16 CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (M, K)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(1) == 2 * w_comp.size(1), "x cols must be 2 * w_comp cols");
    TORCH_CHECK(x.size(1) % 64 == 0, "K must be multiple of 64, got ", x.size(1));
    TORCH_CHECK(w_comp.size(0) % 16 == 0, "N_out must be multiple of 16, got ", w_comp.size(0));
    TORCH_CHECK(w_comp.is_contiguous() && w_meta.is_contiguous(), "weights must be contiguous");
    const int M = x.size(0), K = x.size(1), N_out = w_comp.size(0);
    // The fused kernel reads x with 16-byte vector loads (rows are 128-byte multiples)
    const bool x_aligned = reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
    TORCH_CHECK(path != 1 || x_aligned, "fused sparse path needs a 16-byte aligned x");
    if (path == 1 || (path < 0 && M > 0 && x_aligned && sp_use_fused(M, K, N_out)))
        return sparse_fq_launch(x, w_comp, w_meta, w_scale, bias);
    auto q = sparse_quantize_act(x);
    return sparse_int8_gemm(q[0], q[1], w_comp, w_meta, w_scale, bias, -1, -1, -1);
}

torch::Tensor sparse_int8_linear(
    torch::Tensor x,           // (M, K) fp16 activation
    torch::Tensor w_comp,      // (N_out, K/2) int8 compressed weight
    torch::Tensor w_meta,      // (N_out, K/32) int32 packed metadata
    torch::Tensor w_scale,     // (N_out,) fp16 weight scale
    torch::Tensor bias         // (N_out,) fp16 or empty tensor
) {
    /*
     * Full sparse INT8 linear: quantize activation -> sparse GEMM -> dequant + bias
     * (one fused kernel when that saves DRAM traffic, see sp_use_fused).
     * Returns output (M, N_out) fp16.
     */
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16, "x must be fp16 CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (M, K)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(1) == 2 * w_comp.size(1), "x cols must be 2 * w_comp cols");
    return sparse_int8_linear_ex(x, w_comp, w_meta, w_scale, bias, -1);
}

// ===============================================================================
//  Python bindings
// ===============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_prune_compress", &sparse_prune_compress,
          "Prune to 2:4 + compress: (N,K) int8 -> compressed (N,K/2) + raw metadata");
    m.def("sparse_prune_and_pack", &sparse_prune_and_pack,
          "Prune + compress + pack metadata: (N,K) int8 -> compressed + packed meta");
    m.def("sparse_int8_linear", &sparse_int8_linear,
          "Full sparse INT8 linear: quant -> sparse GEMM -> dequant+bias");
    m.def("sparse_quantize_act", &sparse_quantize_act,
          "Per-token activation quantize: (M,K) fp16 -> int8 (M,K) + fp32 scales (M,)");
    m.def("sparse_int8_gemm", &sparse_int8_gemm,
          "Sparse GEMM + dequant + bias on pre-quantized activations",
          py::arg("x_int8"), py::arg("act_scale"), py::arg("w_comp"), py::arg("w_meta"),
          py::arg("w_scale"), py::arg("bias"), py::arg("force_tile") = -1,
          py::arg("force_split") = -1, py::arg("force_order") = -1);
    m.def("sparse_int8_linear_ex", &sparse_int8_linear_ex,
          "Sparse INT8 linear with an explicit path (-1 auto, 0 quantize + GEMM, 1 fused)",
          py::arg("x"), py::arg("w_comp"), py::arg("w_meta"), py::arg("w_scale"), py::arg("bias"),
          py::arg("path") = -1);
    m.def("sparse_int8_linear_path", [](int64_t M, int64_t K, int64_t N_out) -> int64_t {
              return sp_use_fused((int)M, (int)K, (int)N_out) ? 1 : 0; },
          "Path sparse_int8_linear takes for a shape: 0 quantize + GEMM, 1 fused");
    m.def("test_mma_sp", &test_mma_sp,
          "Minimal mma.sp test: one 16x8 tile via m16n8k32 sparse INT8");
}
