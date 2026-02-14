/*
 * Custom INT8 GEMM kernel using Tensor Core wmma intrinsics.
 *
 * Directly drives the INT8 Tensor Cores via nvcuda::wmma, bypassing cuBLAS.
 * Fuses GEMM + dequantization + bias into a single kernel — no intermediate
 * int32 buffer hitting global memory.
 *
 * Y_fp16[m][n] = sum_k(A_int8[m][k] * B_int8[k][n]) * act_scale[m] * w_scale[n] + bias[n]
 *
 * Targets SM89 (RTX 4090 Ada Lovelace): 660 TOPS INT8 Tensor Core throughput.
 *
 * Two tile configurations, selected at runtime based on N:
 *   - 128x64:  Better for narrow N (≤2048). 12KB smem, 8 blocks/SM max occupancy.
 *   - 128x128: Better for wide N (>2048). 16KB smem, higher arithmetic intensity.
 *
 * Both use 8 warps (4x2), 256 threads, BLOCK_K=64 (4 wmma steps per load).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// ═══════════════════════════════════════════════════════════════════════════════
//  Templated GEMM kernel — parameterized on BLOCK_N_T (64 or 128)
//
//  BLOCK_M=128 always. 8 warps (4 in M × 2 in N).
//  WARP_N = BLOCK_N_T / 2:  32 for 128x64, 64 for 128x128.
//  Tiles per warp:           2×2 for 128x64, 2×4 for 128x128.
//  Accumulators per warp:    4 for 128x64, 8 for 128x128.
// ═══════════════════════════════════════════════════════════════════════════════

template <int BLOCK_N_T>
__global__ void int8_gemm_dequant_kernel(
    const int8_t* __restrict__ A,         // (M, K) row-major int8
    const int8_t* __restrict__ B,         // (N, K) row-major int8
    const float*  __restrict__ act_scale, // (M,) fp32
    const __half* __restrict__ w_scale,   // (N,) fp16
    const __half* __restrict__ bias,      // (N,) fp16, may be nullptr
    __half*       __restrict__ C,         // (M, N) fp16 output
    const int M, const int K, const int N
) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_K = 64;
    constexpr int WARPS_M = 4;
    constexpr int WARPS_N = 2;
    constexpr int WARP_M = BLOCK_M / WARPS_M;      // 32
    constexpr int WARP_N = BLOCK_N_T / WARPS_N;    // 32 or 64
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int TILES_M = WARP_M / WMMA_M;       // 2
    constexpr int TILES_N = WARP_N / WMMA_N;       // 2 or 4
    constexpr int K_STEPS = BLOCK_K / WMMA_K;      // 4
    constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;  // 256

    const int block_m = blockIdx.x * BLOCK_M;
    const int block_n = blockIdx.y * BLOCK_N_T;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_row = (warp_id / WARPS_N) * WARP_M;
    const int warp_col = (warp_id % WARPS_N) * WARP_N;

    // ── Shared memory for tiles ──
    __shared__ int8_t smem_A[BLOCK_M][BLOCK_K];
    __shared__ int8_t smem_B[BLOCK_N_T][BLOCK_K];

    // ── Accumulator fragments in registers ──
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc[TILES_M][TILES_N];
    #pragma unroll
    for (int i = 0; i < TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0);

    const int tid = threadIdx.x;

    // ── Outer K-loop ──
    for (int k_outer = 0; k_outer < K; k_outer += BLOCK_K) {

        // Load A tile: BLOCK_M × BLOCK_K via 128-bit (int4) loads
        {
            const int4* src = reinterpret_cast<const int4*>(A + (size_t)block_m * K + k_outer);
            int4* dst = reinterpret_cast<int4*>(&smem_A[0][0]);
            constexpr int total = (BLOCK_M * BLOCK_K) / 16;
            const int stride = K / 16;

            #pragma unroll
            for (int i = tid; i < total; i += NUM_THREADS) {
                const int row = i / (BLOCK_K / 16);
                const int col4 = i % (BLOCK_K / 16);
                if (block_m + row < M && k_outer + col4 * 16 < K)
                    dst[i] = src[row * stride + col4];
                else
                    dst[i] = make_int4(0, 0, 0, 0);
            }
        }

        // Load B tile: BLOCK_N_T × BLOCK_K via 128-bit loads
        {
            const int4* src = reinterpret_cast<const int4*>(B + (size_t)block_n * K + k_outer);
            int4* dst = reinterpret_cast<int4*>(&smem_B[0][0]);
            constexpr int total = (BLOCK_N_T * BLOCK_K) / 16;
            const int stride = K / 16;

            #pragma unroll
            for (int i = tid; i < total; i += NUM_THREADS) {
                const int row = i / (BLOCK_K / 16);
                const int col4 = i % (BLOCK_K / 16);
                if (block_n + row < N && k_outer + col4 * 16 < K)
                    dst[i] = src[row * stride + col4];
                else
                    dst[i] = make_int4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        // Inner K-loop: 4 wmma steps per shared memory load
        #pragma unroll
        for (int k_inner = 0; k_inner < K_STEPS; k_inner++) {
            const int k_offset = k_inner * WMMA_K;

            #pragma unroll
            for (int wi = 0; wi < TILES_M; wi++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    &smem_A[warp_row + wi * WMMA_M][k_offset], BLOCK_K);

                #pragma unroll
                for (int wj = 0; wj < TILES_N; wj++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(b_frag,
                        &smem_B[warp_col + wj * WMMA_N][k_offset], BLOCK_K);

                    wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]);
                }
            }
        }

        __syncthreads();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Epilogue: dequant + bias via per-warp staging buffer
    //
    //  Reuse smem_A for staging (tiles no longer needed).
    //  8 warps × 16 × 16 × 4 bytes = 8192 bytes ≤ smem_A size.
    // ═══════════════════════════════════════════════════════════════════════════

    int32_t (*smem_stage)[16][16] = reinterpret_cast<int32_t (*)[16][16]>(&smem_A[0][0]);

    const int half_warp = lane_id / 16;
    const int col_in_half = lane_id % 16;

    #pragma unroll
    for (int wi = 0; wi < TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < TILES_N; wj++) {
            wmma::store_matrix_sync(
                &smem_stage[warp_id][0][0],
                acc[wi][wj], 16, wmma::mem_row_major
            );

            #pragma unroll
            for (int r = 0; r < 16; r += 2) {
                const int local_row = r + half_warp;
                const int gr = block_m + warp_row + wi * WMMA_M + local_row;
                const int gc = block_n + warp_col + wj * WMMA_N + col_in_half;

                if (gr < M && gc < N) {
                    float val = static_cast<float>(smem_stage[warp_id][local_row][col_in_half]);
                    val *= act_scale[gr] * __half2float(w_scale[gc]);
                    if (bias != nullptr) {
                        val += __half2float(bias[gc]);
                    }
                    C[(size_t)gr * N + gc] = __float2half(val);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Activation quantization kernel (standalone copy for this module)
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void _tc_fused_quantize_kernel(
    const __half* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    const int K
) {
    const int row = blockIdx.x;
    const __half* in_row = input + (size_t)row * K;
    int8_t* out_row = output + (size_t)row * K;

    float thread_max = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        thread_max = fmaxf(thread_max, fabsf(__half2float(in_row[k])));
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));

    __shared__ float warp_maxes[32];
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) warp_maxes[warp_id] = thread_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_maxes[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        if (lane_id == 0) warp_maxes[0] = val;
    }
    __syncthreads();

    const float scale = fmaxf(warp_maxes[0], 1e-8f) / 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    const float inv_scale = 1.0f / scale;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int q = __float2int_rn(__half2float(in_row[k]) * inv_scale);
        out_row[k] = static_cast<int8_t>(max(-128, min(127, q)));
    }
}

static std::vector<torch::Tensor> _tc_fused_quantize(torch::Tensor input) {
    const int M = input.size(0);
    const int K = input.size(1);
    auto output = torch::empty({M, K}, torch::dtype(torch::kInt8).device(input.device()));
    auto scales = torch::empty({M}, torch::dtype(torch::kFloat32).device(input.device()));
    const int threads = min(256, ((K + 31) / 32) * 32);
    _tc_fused_quantize_kernel<<<M, threads>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        output.data_ptr<int8_t>(), scales.data_ptr<float>(), K
    );
    return {output, scales};
}


// ═══════════════════════════════════════════════════════════════════════════════
//  C++ wrappers — dispatch to 128x64 or 128x128 kernel based on N
// ═══════════════════════════════════════════════════════════════════════════════

// N threshold: use wide (128x128) tile when N > this value.
// 128x128 has higher arithmetic intensity but fewer N-blocks per SM.
// 128x64 is better when N is small (more blocks → better SM utilization).
static constexpr int WIDE_N_THRESHOLD = 2048;

torch::Tensor int8_gemm_tc(
    torch::Tensor A,           // (M, K) int8
    torch::Tensor B,           // (N, K) int8
    torch::Tensor act_scale,   // (M,) fp32
    torch::Tensor w_scale,     // (N,) fp16
    torch::Tensor bias         // (N,) fp16 or empty
) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kInt8, "A must be int8 CUDA");
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kInt8, "B must be int8 CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);
    TORCH_CHECK(B.size(1) == K, "A cols must match B cols (K dimension)");
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64, got ", K);
    TORCH_CHECK(N % 16 == 0, "N must be multiple of 16, got ", N);

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(A.device()));

    const __half* bias_ptr = (bias.numel() > 0)
        ? reinterpret_cast<const __half*>(bias.data_ptr<at::Half>())
        : nullptr;

    constexpr int BLOCK_M = 128;
    constexpr int NUM_THREADS = 256;

    if (N > WIDE_N_THRESHOLD) {
        // Wide tile: 128x128
        constexpr int BLOCK_N = 128;
        dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
        int8_gemm_dequant_kernel<128><<<grid, NUM_THREADS>>>(
            A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
            act_scale.data_ptr<float>(),
            reinterpret_cast<const __half*>(w_scale.data_ptr<at::Half>()),
            bias_ptr,
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            M, K, N
        );
    } else {
        // Narrow tile: 128x64
        constexpr int BLOCK_N = 64;
        dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
        int8_gemm_dequant_kernel<64><<<grid, NUM_THREADS>>>(
            A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
            act_scale.data_ptr<float>(),
            reinterpret_cast<const __half*>(w_scale.data_ptr<at::Half>()),
            bias_ptr,
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            M, K, N
        );
    }

    return output;
}


torch::Tensor int8_linear_tc(
    torch::Tensor x,           // (M, K) fp16
    torch::Tensor w_int8,      // (N, K) int8
    torch::Tensor w_scale,     // (N,) fp16
    torch::Tensor bias         // (N,) fp16 or empty
) {
    auto quant_result = _tc_fused_quantize(x);
    return int8_gemm_tc(quant_result[0], w_int8, quant_result[1], w_scale, bias);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemm_tc", &int8_gemm_tc,
          "Fused INT8 Tensor Core GEMM + dequant + bias (adaptive tile)");
    m.def("int8_linear_tc", &int8_linear_tc,
          "Full fused int8 linear: quant -> TC GEMM -> dequant+bias");
}
