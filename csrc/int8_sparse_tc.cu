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
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════════
//  Error checking
// ═══════════════════════════════════════════════════════════════════════════════

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("CUDA error: ") + cudaGetErrorString(err) +         \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));            \
        }                                                                       \
    } while (0)

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 1: Prune + compress kernel
//  Dense int8 (N, K) -> compressed (N, K/2) + raw metadata (N, K/4) uint8
// ═══════════════════════════════════════════════════════════════════════════════

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
    uint8_t* __restrict__ metadata_raw,       // (N * K/4) — one byte per group
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

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 2: Activation quantization (reused from int8_gemm_tc.cu)
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 3: Minimal mma.sp test kernel (single warp, single tile)
// ═══════════════════════════════════════════════════════════════════════════════

/*
 * Single-warp test: one m16n8k32 sparse MMA with spsel=0.
 *
 * A_comp:  (16, 16) int8  — compressed from (16, 32) with 2:4
 * B_dense: (8, 32) int8   — row-major = col-major (32, 8) for MMA
 * meta:    (32,) uint32   — per-lane metadata (hardware format)
 * C_out:   (16, 8) int32  — output
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

    // ── Load A fragment (compressed, row-major 16×16) ──
    uint32_t a_frag[2];
    memcpy(&a_frag[0], A_comp + groupID * K_COMP + tid * 4, 4);
    memcpy(&a_frag[1], A_comp + (groupID + 8) * K_COMP + tid * 4, 4);

    // ── Load B fragment (dense, stored as (8,32) row-major = (32,8) col-major) ──
    // MMA B fragment mapping for m16n8k32 INT8:
    //   groupID indexes N column (0..7), tid indexes K groups
    //   b[0]: B_rm[groupID, tid*4 .. tid*4+3]     (K range 0..15)
    //   b[1]: B_rm[groupID, 16+tid*4 .. 16+tid*4+3] (K range 16..31)
    uint32_t b_frag[2];
    memcpy(&b_frag[0], B_dense + groupID * K_DENSE + tid * 4, 4);
    memcpy(&b_frag[1], B_dense + groupID * K_DENSE + 16 + tid * 4, 4);

    // ── Load per-lane metadata ──
    uint32_t e_meta = meta[lane];

    // ── Accumulator ──
    int32_t c_frag[4] = {0, 0, 0, 0};

    // ── Sparse MMA: m16n8k32 INT8 with spsel=0 ──
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

    // ── Store output (16×8 row-major) ──
    // C fragment: c[0]=(groupID, tid*2), c[1]=(groupID, tid*2+1),
    //             c[2]=(groupID+8, tid*2), c[3]=(groupID+8, tid*2+1)
    const int col0 = tid * 2;
    const int col1 = col0 + 1;
    C_out[groupID * 8 + col0]       = c_frag[0];
    C_out[groupID * 8 + col1]       = c_frag[1];
    C_out[(groupID + 8) * 8 + col0] = c_frag[2];
    C_out[(groupID + 8) * 8 + col1] = c_frag[3];
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 4: Pack raw metadata nibbles into per-row uint32 chunks
//  meta_raw (N, K/4) uint8 -> meta_packed (N, K/32) uint32
//  Each uint32 packs 8 consecutive nibbles (one 32-element dense K-chunk)
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 5: Tiled Sparse INT8 GEMM kernel
//
//  Computes: output(M, N_out) = dequant( W_comp(N_out, K/2) × x_int8(M, K)^T )
//
//  W_comp is the sparse operand A (row-major, 2:4 compressed along K).
//  x_int8 is the dense operand B (treated as col-major K×M = row-major M×K).
//  The MMA produces C(N_out, M). Epilogue transposes to output(M, N_out).
//
//  Block: 128 weight-rows × 64 batch-cols × 32 dense-K per iteration
//  8 warps (4×2), warp tile: 32 WR × 32 BC = 2 M-tiles × 4 N-tiles
//  Grid: (ceil(N_out/128), ceil(M/64))
// ═══════════════════════════════════════════════════════════════════════════════

// Block/warp geometry constants
constexpr int SP_BLOCK_WR = 128;    // weight rows (output N_out dim) per block
constexpr int SP_BLOCK_BC = 64;     // batch cols (output M dim) per block
constexpr int SP_BLOCK_K  = 64;     // dense K elements per iteration (2 × mma K=32)
constexpr int SP_K_COMP   = 32;     // compressed K per iteration (SP_BLOCK_K / 2)
constexpr int SP_WARPS_WR = 4;
constexpr int SP_WARPS_BC = 2;
constexpr int SP_WARP_WR  = SP_BLOCK_WR / SP_WARPS_WR;   // 32
constexpr int SP_WARP_BC  = SP_BLOCK_BC / SP_WARPS_BC;    // 32
constexpr int SP_TILES_M  = SP_WARP_WR / 16;              // 2 MMA M-tiles per warp
constexpr int SP_TILES_N  = SP_WARP_BC / 8;               // 4 MMA N-tiles per warp
constexpr int SP_THREADS  = SP_WARPS_WR * SP_WARPS_BC * 32;  // 256

// Shared memory: max of GEMM tiles and epilogue staging
// GEMM:    A=128×32 + B=64×64 = 4096 + 4096 = 8192 bytes
// Stage:   8 warps × 16×9 int32 = 4608 bytes  (padded stride=9 for bank conflicts)
constexpr int SP_SMEM_GEMM  = SP_BLOCK_WR * SP_K_COMP + SP_BLOCK_BC * SP_BLOCK_K;
constexpr int SP_STAGE_PAD   = 9;
constexpr int SP_SMEM_STAGE  = 8 * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
constexpr int SP_SMEM_BYTES  = (SP_SMEM_GEMM > SP_SMEM_STAGE) ? SP_SMEM_GEMM : SP_SMEM_STAGE;

__global__ void sparse_gemm_kernel(
    const int8_t*   __restrict__ W_comp,     // (N_out, K/2) compressed weight
    const int8_t*   __restrict__ x_int8,     // (M, K) quantized activation
    const uint32_t* __restrict__ w_meta,     // (N_out, K/32) packed metadata
    const float*    __restrict__ act_scale,  // (M,) fp32
    const __half*   __restrict__ w_scale,    // (N_out,) fp16
    const __half*   __restrict__ bias,       // (N_out,) fp16 or nullptr
    __half*         __restrict__ output,     // (M, N_out) fp16
    const int M, const int K, const int N_out
) {
    const int block_wr = blockIdx.x * SP_BLOCK_WR;
    const int block_bc = blockIdx.y * SP_BLOCK_BC;

    const int t    = threadIdx.x;
    const int warp = t / 32;
    const int lane = t % 32;
    const int groupID = lane >> 2;
    const int tid     = lane & 3;

    const int warp_wr = (warp / SP_WARPS_BC) * SP_WARP_WR;   // 0, 32, 64, 96
    const int warp_bc = (warp % SP_WARPS_BC) * SP_WARP_BC;    // 0, 32

    const int K_comp     = K / 2;
    const int meta_cols  = K / 32;   // metadata chunks per row

    // ── Shared memory (union of GEMM tiles and epilogue staging) ──
    __shared__ int8_t smem_buf[SP_SMEM_BYTES];

    int8_t (*smem_A)[SP_K_COMP]  = reinterpret_cast<int8_t (*)[SP_K_COMP]>(smem_buf);
    int8_t (*smem_B)[SP_BLOCK_K] = reinterpret_cast<int8_t (*)[SP_BLOCK_K]>(
                                       smem_buf + SP_BLOCK_WR * SP_K_COMP);

    // ── Accumulators: 2 M-tiles × 4 N-tiles × 4 int32 per fragment ──
    int32_t acc[SP_TILES_M][SP_TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < SP_TILES_M; mi++)
        #pragma unroll
        for (int nj = 0; nj < SP_TILES_N; nj++) {
            acc[mi][nj][0] = 0; acc[mi][nj][1] = 0;
            acc[mi][nj][2] = 0; acc[mi][nj][3] = 0;
        }

    // ════════════════════════════════════════════════════════════════
    //  K-loop: iterate over dense K in chunks of 64
    //  Each iteration does TWO mma.sp calls: spsel=0 for K[0:31],
    //  spsel=1 for K[32:63], halving the K-loop overhead.
    // ════════════════════════════════════════════════════════════════
    for (int k_dense = 0; k_dense < K; k_dense += SP_BLOCK_K) {
        const int k_comp_base = k_dense / 2;

        // ── Load A_comp tile (128×32 = 4096 bytes = 256 int4 loads) ──
        {
            constexpr int TOTAL_A = SP_BLOCK_WR * SP_K_COMP / 16;  // 256
            for (int i = t; i < TOTAL_A; i += SP_THREADS) {
                const int row  = i / (SP_K_COMP / 16);   // 0..127
                const int col4 = i % (SP_K_COMP / 16);   // 0..1
                const int gr = block_wr + row;
                int4* dst = reinterpret_cast<int4*>(&smem_A[row][col4 * 16]);
                if (gr < N_out)
                    *dst = *reinterpret_cast<const int4*>(
                        W_comp + (size_t)gr * K_comp + k_comp_base + col4 * 16);
                else
                    *dst = make_int4(0, 0, 0, 0);
            }
        }

        // ── Load B tile (64×64 = 4096 bytes = 256 int4 loads) ──
        {
            constexpr int TOTAL_B = SP_BLOCK_BC * SP_BLOCK_K / 16;  // 256
            for (int i = t; i < TOTAL_B; i += SP_THREADS) {
                const int row  = i / (SP_BLOCK_K / 16);   // 0..63
                const int col4 = i % (SP_BLOCK_K / 16);   // 0..3
                const int gr = block_bc + row;
                int4* dst = reinterpret_cast<int4*>(&smem_B[row][col4 * 16]);
                if (gr < M)
                    *dst = *reinterpret_cast<const int4*>(
                        x_int8 + (size_t)gr * K + k_dense + col4 * 16);
                else
                    *dst = make_int4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        // ── Per-warp MMA computation: two mma.sp per M-tile per N-tile ──
        // Metadata indexing: k_meta0 for first 32 dense K (spsel=0),
        //                    k_meta1 for second 32 dense K (spsel=1)
        const int k_meta0 = k_dense / 32;
        const int k_meta1 = k_meta0 + 1;

        #pragma unroll
        for (int mi = 0; mi < SP_TILES_M; mi++) {
            const int m_local  = warp_wr + mi * 16;
            const int m_global = block_wr + m_local;

            // Load metadata E register for BOTH spsel=0 and spsel=1:
            //   tid=0: row groupID, K-chunk 0 (spsel=0)
            //   tid=1: row groupID+8, K-chunk 0 (spsel=0)
            //   tid=2: row groupID, K-chunk 1 (spsel=1)
            //   tid=3: row groupID+8, K-chunk 1 (spsel=1)
            uint32_t e_meta;
            if (tid == 0) {
                e_meta = (m_global + groupID < N_out)
                    ? w_meta[(size_t)(m_global + groupID) * meta_cols + k_meta0]
                    : 0x44444444u;
            } else if (tid == 1) {
                e_meta = (m_global + groupID + 8 < N_out)
                    ? w_meta[(size_t)(m_global + groupID + 8) * meta_cols + k_meta0]
                    : 0x44444444u;
            } else if (tid == 2) {
                e_meta = (m_global + groupID < N_out)
                    ? w_meta[(size_t)(m_global + groupID) * meta_cols + k_meta1]
                    : 0x44444444u;
            } else {  // tid == 3
                e_meta = (m_global + groupID + 8 < N_out)
                    ? w_meta[(size_t)(m_global + groupID + 8) * meta_cols + k_meta1]
                    : 0x44444444u;
            }

            // ── First half: K[0:31] in compressed space [0:16], spsel=0 ──
            uint32_t a_frag0[2];
            a_frag0[0] = *reinterpret_cast<const uint32_t*>(
                &smem_A[m_local + groupID][tid * 4]);
            a_frag0[1] = *reinterpret_cast<const uint32_t*>(
                &smem_A[m_local + groupID + 8][tid * 4]);

            // ── Second half: K[32:63] in compressed space [16:32], spsel=1 ──
            uint32_t a_frag1[2];
            a_frag1[0] = *reinterpret_cast<const uint32_t*>(
                &smem_A[m_local + groupID][16 + tid * 4]);
            a_frag1[1] = *reinterpret_cast<const uint32_t*>(
                &smem_A[m_local + groupID + 8][16 + tid * 4]);

            #pragma unroll
            for (int nj = 0; nj < SP_TILES_N; nj++) {
                const int n_local = warp_bc + nj * 8 + groupID;

                // B fragments: first 32 dense K
                uint32_t b_frag0[2];
                b_frag0[0] = *reinterpret_cast<const uint32_t*>(
                    &smem_B[n_local][tid * 4]);
                b_frag0[1] = *reinterpret_cast<const uint32_t*>(
                    &smem_B[n_local][16 + tid * 4]);

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

                // B fragments: second 32 dense K
                uint32_t b_frag1[2];
                b_frag1[0] = *reinterpret_cast<const uint32_t*>(
                    &smem_B[n_local][32 + tid * 4]);
                b_frag1[1] = *reinterpret_cast<const uint32_t*>(
                    &smem_B[n_local][48 + tid * 4]);

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

        __syncthreads();
    }

    // ════════════════════════════════════════════════════════════════
    //  Epilogue: dequant + bias + transposed write via smem staging
    //
    //  MMA output is C(N_out, M). We write output(M, N_out) transposed.
    //  Stage each 16×8 tile through smem with padded stride (9) to
    //  avoid bank conflicts during the column-wise readback.
    // ════════════════════════════════════════════════════════════════
    int32_t (*smem_stage)[16][SP_STAGE_PAD] =
        reinterpret_cast<int32_t (*)[16][SP_STAGE_PAD]>(smem_buf);

    #pragma unroll
    for (int mi = 0; mi < SP_TILES_M; mi++) {
        #pragma unroll
        for (int nj = 0; nj < SP_TILES_N; nj++) {
            // Store MMA fragment to per-warp staging buffer
            smem_stage[warp][groupID    ][tid * 2]     = acc[mi][nj][0];
            smem_stage[warp][groupID    ][tid * 2 + 1] = acc[mi][nj][1];
            smem_stage[warp][groupID + 8][tid * 2]     = acc[mi][nj][2];
            smem_stage[warp][groupID + 8][tid * 2 + 1] = acc[mi][nj][3];
            __syncwarp();

            // Tile position in global output
            const int tile_wr = block_wr + warp_wr + mi * 16;  // N_out base
            const int tile_bc = block_bc + warp_bc + nj * 8;    // M base

            // 16×8 = 128 elements, 32 threads → 4 per thread
            // Map: bc_local = i/16 (outer), wr_local = i%16 (inner, consecutive N)
            #pragma unroll
            for (int i = lane; i < 128; i += 32) {
                const int bc_local = i / 16;
                const int wr_local = i % 16;
                const int wr = tile_wr + wr_local;
                const int bc = tile_bc + bc_local;

                if (wr < N_out && bc < M) {
                    float val = static_cast<float>(smem_stage[warp][wr_local][bc_local]);
                    val *= act_scale[bc] * __half2float(w_scale[wr]);
                    if (bias != nullptr)
                        val += __half2float(bias[wr]);
                    output[(size_t)bc * N_out + wr] = __float2half(val);
                }
            }
            __syncwarp();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  C++ wrappers
// ═══════════════════════════════════════════════════════════════════════════════

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

    // Step 1: Prune + compress → compressed (N, K/2) + meta_raw (N*K/4) uint8
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

    // Step 2: Pack metadata → meta_packed (N, K/32) int32
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

torch::Tensor sparse_int8_linear(
    torch::Tensor x,           // (M, K) fp16 activation
    torch::Tensor w_comp,      // (N_out, K/2) int8 compressed weight
    torch::Tensor w_meta,      // (N_out, K/32) int32 packed metadata
    torch::Tensor w_scale,     // (N_out,) fp16 weight scale
    torch::Tensor bias         // (N_out,) fp16 or empty tensor
) {
    /*
     * Full sparse INT8 linear: quantize activation → sparse GEMM → dequant + bias.
     * Returns output (M, N_out) fp16.
     */
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16, "x must be fp16 CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (M, K)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w_comp.is_cuda() && w_comp.dtype() == torch::kInt8, "w_comp must be int8 CUDA");
    TORCH_CHECK(w_comp.is_contiguous(), "w_comp must be contiguous");
    TORCH_CHECK(w_meta.is_cuda() && w_meta.dtype() == torch::kInt32, "w_meta must be int32 CUDA");
    TORCH_CHECK(w_meta.is_contiguous(), "w_meta must be contiguous");

    const int M     = x.size(0);
    const int K     = x.size(1);
    const int N_out = w_comp.size(0);
    TORCH_CHECK(w_comp.size(1) == K / 2, "w_comp cols must be K/2");
    TORCH_CHECK(w_meta.size(0) == N_out && w_meta.size(1) == K / 32,
                "w_meta shape must be (N_out, K/32)");
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64 (BLOCK_K=64), got ", K);
    TORCH_CHECK(N_out % 16 == 0, "N_out must be multiple of 16, got ", N_out);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Step 1: Quantize activation (M, K) fp16 → int8 + scales
    auto x_int8    = torch::empty({M, K}, torch::dtype(torch::kInt8).device(x.device()));
    auto act_scale = torch::empty({M},    torch::dtype(torch::kFloat32).device(x.device()));
    {
        const int threads = min(256, ((K + 31) / 32) * 32);
        _sp_fused_quantize_kernel<<<M, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            x_int8.data_ptr<int8_t>(),
            act_scale.data_ptr<float>(),
            K
        );
    }

    // Step 2: Sparse GEMM + dequant + bias → output (M, N_out) fp16
    auto output = torch::empty({M, N_out},
        torch::dtype(torch::kFloat16).device(x.device()));

    const __half* bias_ptr = (bias.numel() > 0)
        ? reinterpret_cast<const __half*>(bias.data_ptr<at::Half>())
        : nullptr;

    dim3 grid(
        (N_out + SP_BLOCK_WR - 1) / SP_BLOCK_WR,
        (M     + SP_BLOCK_BC - 1) / SP_BLOCK_BC
    );

    sparse_gemm_kernel<<<grid, SP_THREADS, 0, stream>>>(
        w_comp.data_ptr<int8_t>(),
        x_int8.data_ptr<int8_t>(),
        reinterpret_cast<const uint32_t*>(w_meta.data_ptr<int32_t>()),
        act_scale.data_ptr<float>(),
        reinterpret_cast<const __half*>(w_scale.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        M, K, N_out
    );

    return output;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Python bindings
// ═══════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_prune_compress", &sparse_prune_compress,
          "Prune to 2:4 + compress: (N,K) int8 -> compressed (N,K/2) + raw metadata");
    m.def("sparse_prune_and_pack", &sparse_prune_and_pack,
          "Prune + compress + pack metadata: (N,K) int8 -> compressed + packed meta");
    m.def("sparse_int8_linear", &sparse_int8_linear,
          "Full sparse INT8 linear: quant -> sparse GEMM -> dequant+bias");
    m.def("test_mma_sp", &test_mma_sp,
          "Minimal mma.sp test: one 16x8 tile via m16n8k32 sparse INT8");
}
