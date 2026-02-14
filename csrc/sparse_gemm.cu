/*
 * Sparse INT8 GEMM via cuSPARSELt for 2:4 structured sparsity.
 *
 * The RTX 4090's Sparse Tensor Cores deliver 2x INT8 throughput (1321 TOPS)
 * when weights have 2:4 sparsity (2 zeros per group of 4 contiguous elements).
 *
 * cuSPARSELt handles the compressed format, metadata, and kernel dispatch.
 * We wrap it for PyTorch integration:
 *   - sparse_compress(): prune + compress weight (offline, once)
 *   - sparse_matmul():   compressed_W @ x -> int32 (runtime, per forward)
 *   - sparse_linear():   fused quant + sparse GEMM + dequant + bias
 *
 * Layout: cuSPARSELt 0.6.x requires column-major. PyTorch row-major tensors
 * are reinterpreted as column-major by transposing dimensions:
 *   PyTorch (N, K) row-major <=> cuSPARSELt (K, N) col-major with ld=K
 *
 * For the matmul C = A @ B:
 *   A (sparse) = weight: PyTorch (N, K) -> transposed to (K, N) contiguous
 *                        -> cuSPARSELt col-major (N, K) with ld=N
 *   B (dense)  = activation: PyTorch (M, K) row-major
 *                        -> cuSPARSELt col-major (K, M) with ld=K
 *   C (output) = cuSPARSELt col-major (N, M) with ld=N
 *                        -> PyTorch (M, N) row-major (same memory layout)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cusparseLt.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <mutex>

// ═══════════════════════════════════════════════════════════════════════════════
//  Error checking
// ═══════════════════════════════════════════════════════════════════════════════

#define CHECK_CUSPARSE(call)                                                    \
    do {                                                                        \
        cusparseStatus_t status = (call);                                       \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                \
            throw std::runtime_error(                                           \
                std::string("cuSPARSELt error: status=") +                      \
                std::to_string(static_cast<int>(status)) +                      \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));            \
        }                                                                       \
    } while (0)

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
//  Global cuSPARSELt handle (one per process)
// ═══════════════════════════════════════════════════════════════════════════════

static cusparseLtHandle_t g_handle;
static bool g_handle_init = false;
static std::mutex g_mutex;

static cusparseLtHandle_t* get_handle() {
    if (!g_handle_init) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_handle_init) {
            CHECK_CUSPARSE(cusparseLtInit(&g_handle));
            g_handle_init = true;
        }
    }
    return &g_handle;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Cached matmul plans (keyed by M, N, K dimensions)
//  cuSPARSELt plan creation is expensive; cache per unique shape.
// ═══════════════════════════════════════════════════════════════════════════════

struct PlanKey {
    int64_t M, N, K;
    bool operator==(const PlanKey& o) const { return M == o.M && N == o.N && K == o.K; }
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const {
        return std::hash<int64_t>()(k.M) ^ (std::hash<int64_t>()(k.N) << 16) ^
               (std::hash<int64_t>()(k.K) << 32);
    }
};

struct CachedPlan {
    cusparseLtMatmulPlan_t plan;
    size_t workspace_size;
    torch::Tensor workspace;  // cached workspace allocation
};

static std::unordered_map<PlanKey, CachedPlan, PlanKeyHash> g_plan_cache;
static std::mutex g_plan_mutex;

// ═══════════════════════════════════════════════════════════════════════════════
//  sparse_compress: prune + compress weight (offline)
//
//  Input:  weight_int8 (N, K) int8 — quantized dense weight (PyTorch row-major)
//  Output: compressed tensor (opaque bytes) + also returns the pruned weight
//          for correctness verification
//
//  The weight is transposed to (K, N) contiguous so cuSPARSELt can read it
//  as column-major (N, K) with ld=N.
// ═══════════════════════════════════════════════════════════════════════════════

torch::Tensor sparse_compress(torch::Tensor weight_int8) {
    TORCH_CHECK(weight_int8.is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(weight_int8.dtype() == torch::kInt8, "Weight must be int8");
    TORCH_CHECK(weight_int8.dim() == 2, "Weight must be 2D (N, K)");

    const int64_t N = weight_int8.size(0);
    const int64_t K = weight_int8.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for 2:4 sparsity");

    auto* handle = get_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Transpose to (K, N) contiguous — cuSPARSELt reads this as col-major (N, K)
    auto weight_t = weight_int8.t().contiguous();  // (K, N) C-contiguous

    // --- 1. Create descriptors (all column-major) ---
    cusparseLtMatDescriptor_t matA, matB, matC;

    // Sparse A: col-major (N, K) with ld=N. Data is (K, N) C-contiguous.
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
        handle, &matA, N, K, /*ld=*/N, /*alignment=*/16,
        CUDA_R_8I, CUSPARSE_ORDER_COL, CUSPARSELT_SPARSITY_50_PERCENT));

    // Dense B: col-major (K, 32) with ld=K — placeholder for compression
    // (cuSPARSELt requires minimum dimensions; actual M doesn't matter for prune/compress)
    const int64_t M_placeholder = 32;
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        handle, &matB, K, M_placeholder, /*ld=*/K, /*alignment=*/16,
        CUDA_R_8I, CUSPARSE_ORDER_COL));

    // Output C: col-major (N, 32) with ld=N
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        handle, &matC, N, M_placeholder, /*ld=*/N, /*alignment=*/16,
        CUDA_R_32I, CUSPARSE_ORDER_COL));

    // --- 2. Create matmul descriptor ---
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
        handle, &matmul,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matA, &matB, &matC, &matC,
        CUSPARSE_COMPUTE_32I));

    // --- 3. Prune weight to 2:4 pattern ---
    auto pruned_t = torch::empty_like(weight_t);  // (K, N) contiguous
    CHECK_CUSPARSE(cusparseLtSpMMAPrune(
        handle, &matmul,
        weight_t.data_ptr<int8_t>(),
        pruned_t.data_ptr<int8_t>(),
        CUSPARSELT_PRUNE_SPMMA_STRIP,
        stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // --- 4. Create plan for compression ---
    cusparseLtMatmulAlgSelection_t alg;
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
        handle, &alg, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

    cusparseLtMatmulPlan_t plan;
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(handle, &plan, &matmul, &alg));

    // --- 5. Get compressed size and compress ---
    size_t compressed_size, compressed_buf_size;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
        handle, &plan, &compressed_size, &compressed_buf_size));

    // Allocate compressed tensor as raw bytes
    auto compressed = torch::empty(
        {static_cast<int64_t>(compressed_size)},
        torch::dtype(torch::kUInt8).device(weight_int8.device()));

    auto comp_buf = torch::empty(
        {static_cast<int64_t>(compressed_buf_size)},
        torch::dtype(torch::kUInt8).device(weight_int8.device()));

    CHECK_CUSPARSE(cusparseLtSpMMACompress(
        handle, &plan,
        pruned_t.data_ptr<int8_t>(),
        compressed.data_ptr<uint8_t>(),
        comp_buf.data_ptr<uint8_t>(),
        stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // --- Cleanup ---
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan));
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA));
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB));
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC));

    return compressed;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  sparse_matmul: compressed_W (N, K) @ x^T (K, M) -> (M, N) int32
//
//  Inputs:
//    compressed_w: opaque compressed weight from sparse_compress()
//    x_int8:       (M, K) int8 activation
//    N, K:         original weight dimensions
//  Output:
//    (M, N) int32
//
//  Layout mapping:
//    x_int8 (M, K) C-contiguous -> cuSPARSELt col-major B (K, M) ld=K
//    output (M, N) C-contiguous <- cuSPARSELt col-major C (N, M) ld=N
// ═══════════════════════════════════════════════════════════════════════════════

torch::Tensor sparse_matmul(
    torch::Tensor compressed_w,
    torch::Tensor x_int8,
    int64_t N,
    int64_t K
) {
    TORCH_CHECK(compressed_w.is_cuda(), "compressed_w must be on CUDA");
    TORCH_CHECK(x_int8.is_cuda(), "x_int8 must be on CUDA");
    TORCH_CHECK(x_int8.dtype() == torch::kInt8, "x_int8 must be int8");
    TORCH_CHECK(x_int8.dim() == 2, "x_int8 must be 2D (M, K)");
    TORCH_CHECK(x_int8.size(1) == K, "x_int8 K dimension mismatch");

    const int64_t M = x_int8.size(0);
    auto* handle = get_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // x_int8 (M, K) C-contiguous: cuSPARSELt reads as col-major (K, M) ld=K
    auto x_cont = x_int8.contiguous();

    // --- Get or create plan ---
    PlanKey key{M, N, K};
    CachedPlan* cached = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_plan_mutex);
        auto it = g_plan_cache.find(key);
        if (it != g_plan_cache.end()) {
            cached = &it->second;
        }
    }

    if (!cached) {
        cusparseLtMatDescriptor_t matA, matB, matC;

        // A (sparse): col-major (N, K) ld=N — matches the compressed weight
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
            handle, &matA, N, K, /*ld=*/N, /*alignment=*/16,
            CUDA_R_8I, CUSPARSE_ORDER_COL, CUSPARSELT_SPARSITY_50_PERCENT));

        // B (dense): col-major (K, M) ld=K — x_int8 (M, K) row-major reinterpreted
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            handle, &matB, K, M, /*ld=*/K, /*alignment=*/16,
            CUDA_R_8I, CUSPARSE_ORDER_COL));

        // C (output): col-major (N, M) ld=N — becomes (M, N) row-major in PyTorch
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            handle, &matC, N, M, /*ld=*/N, /*alignment=*/16,
            CUDA_R_32I, CUSPARSE_ORDER_COL));

        cusparseLtMatmulDescriptor_t matmul;
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
            handle, &matmul,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &matA, &matB, &matC, &matC,
            CUSPARSE_COMPUTE_32I));

        cusparseLtMatmulAlgSelection_t alg;
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
            handle, &alg, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

        CachedPlan new_plan;
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(
            handle, &new_plan.plan, &matmul, &alg));

        CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(
            handle, &new_plan.plan, &new_plan.workspace_size));

        if (new_plan.workspace_size > 0) {
            new_plan.workspace = torch::empty(
                {static_cast<int64_t>(new_plan.workspace_size)},
                torch::dtype(torch::kUInt8).device(compressed_w.device()));
        } else {
            new_plan.workspace = torch::empty(
                {0}, torch::dtype(torch::kUInt8).device(compressed_w.device()));
        }

        CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA));
        CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB));
        CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC));

        std::lock_guard<std::mutex> lock(g_plan_mutex);
        auto [it, _] = g_plan_cache.emplace(key, std::move(new_plan));
        cached = &it->second;
    }

    // --- Execute sparse matmul ---
    // Output buffer: (M, N) C-contiguous = col-major (N, M) ld=N
    auto output = torch::empty({M, N}, torch::dtype(torch::kInt32).device(compressed_w.device()));

    int32_t alpha = 1, beta = 0;
    CHECK_CUSPARSE(cusparseLtMatmul(
        handle, &cached->plan,
        &alpha,
        compressed_w.data_ptr<uint8_t>(),
        x_cont.data_ptr<int8_t>(),
        &beta,
        output.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        cached->workspace_size > 0 ? cached->workspace.data_ptr<uint8_t>() : nullptr,
        &stream, 1));

    return output;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Activation quantization kernel (same as in int8_gemm_tc.cu)
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void _sparse_fused_quantize_kernel(
    const __half* __restrict__ input,   // (M, K) fp16
    int8_t* __restrict__ output,        // (M, K) int8
    float* __restrict__ scales,         // (M,) fp32
    const int K
) {
    const int row = blockIdx.x;
    const __half* in_row = input + (size_t)row * K;
    int8_t* out_row = output + (size_t)row * K;

    // Pass 1: find max|x| via parallel reduction
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

    // Pass 2: quantize
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = __half2float(in_row[k]);
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        out_row[k] = static_cast<int8_t>(q);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  sparse_linear: fused quant + sparse GEMM + dequant + bias
//
//  x (M, K) fp16 -> quantize -> sparse matmul -> dequant -> (M, N) fp16
// ═══════════════════════════════════════════════════════════════════════════════

torch::Tensor sparse_linear(
    torch::Tensor x,              // (M, K) fp16
    torch::Tensor compressed_w,   // opaque compressed weight
    torch::Tensor w_scale,        // (N,) fp16 per-channel weight scale
    torch::Tensor bias,           // (N,) fp16 or empty
    int64_t N,                    // output features
    int64_t K                     // input features
) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (M, K)");
    TORCH_CHECK(x.size(1) == K, "x K dimension mismatch");

    const int64_t M = x.size(0);
    auto x_cont = x.contiguous();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // --- 1. Quantize activations to int8 ---
    auto x_int8 = torch::empty({M, K}, torch::dtype(torch::kInt8).device(x.device()));
    auto act_scale = torch::empty({M}, torch::dtype(torch::kFloat32).device(x.device()));

    const int threads = min(256, ((int)K + 31) / 32 * 32);
    _sparse_fused_quantize_kernel<<<M, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_cont.data_ptr<at::Half>()),
        x_int8.data_ptr<int8_t>(),
        act_scale.data_ptr<float>(),
        (int)K);

    // --- 2. Sparse matmul: compressed_W (N,K) @ x_int8^T (K,M) -> (M, N) int32 ---
    auto y_int32 = sparse_matmul(compressed_w, x_int8, N, K);

    // --- 3. Dequant + bias ---
    auto y_f32 = y_int32.to(torch::kFloat32);
    y_f32 *= act_scale.unsqueeze(1);
    y_f32 *= w_scale.to(torch::kFloat32).unsqueeze(0);
    if (bias.numel() > 0) {
        y_f32 += bias.to(torch::kFloat32).unsqueeze(0);
    }

    return y_f32.to(torch::kFloat16);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Python bindings
// ═══════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_compress", &sparse_compress,
          "Prune weight to 2:4 + compress via cuSPARSELt (offline)");
    m.def("sparse_matmul", &sparse_matmul,
          "Sparse INT8 GEMM: compressed_W @ x_int8 -> int32");
    m.def("sparse_linear", &sparse_linear,
          "Fused: quant + sparse GEMM + dequant + bias");
}
