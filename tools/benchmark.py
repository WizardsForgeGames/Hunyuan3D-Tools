"""
Benchmark: FP16 vs Sparse INT8 vs Dense TC INT8 vs bitsandbytes.

Measures throughput, latency, VRAM, and correctness at UNet-relevant matrix sizes.

Usage:
    python tools/benchmark.py
    python tools/benchmark.py --no-bnb       # skip bitsandbytes (if not installed)
    python tools/benchmark.py --no-sparse    # skip sparse INT8
    python tools/benchmark.py --shapes 1280  # only test layers with dim 1280
    python tools/benchmark.py --hunyuan      # sparse vs FP16 at the pipeline's real shapes
    python tools/benchmark.py --hunyuan --compare-sparse-src old/int8_sparse_tc.cu   # A/B
"""

import argparse
import gc
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from quantize_utils import Int8Linear, int_mm_available, sp_available


def timed_forward(layer, x, warmup=5, repeats=50):
    """Time a layer's forward pass with CUDA sync."""
    # Warmup
    for _ in range(warmup):
        _ = layer(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        _ = layer(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    return elapsed


def compute_tops(M, K, N, elapsed_s):
    """Compute TOPS (tera operations per second) for a matmul."""
    ops = 2 * M * K * N  # multiply-add = 2 ops
    return ops / elapsed_s / 1e12


def measure_vram(factory_fn):
    """Measure VRAM used by a layer created via factory_fn."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    layer = factory_fn()
    after = torch.cuda.memory_allocated()
    return layer, after - before


def benchmark_shape(in_f, out_f, batch_sizes, use_bnb=True, use_sparse=True):
    """Benchmark a single layer shape across batch sizes."""
    print(f"\n{'='*80}")
    print(f"  Linear({in_f}, {out_f}) - {in_f*out_f/1e6:.1f}M params")
    print(f"{'='*80}")

    # Create layers
    ref_linear = nn.Linear(in_f, out_f, bias=True).cuda().half()
    int8_tc = Int8Linear.from_linear(ref_linear)

    int8_sparse = None
    if use_sparse and sp_available():
        try:
            int8_sparse = Int8Linear.from_linear_sparse(ref_linear)
        except Exception as e:
            print(f"  (sparse layer creation failed: {e})")

    bnb_layer = None
    if use_bnb:
        try:
            import bitsandbytes as bnb
            bnb_linear = nn.Linear(in_f, out_f, bias=True).cuda().half()
            bnb_layer = bnb.nn.Linear8bitLt(
                in_f, out_f, bias=True, has_fp16_weights=False, threshold=6.0,
            )
            bnb_layer.weight = bnb.nn.Int8Params(
                bnb_linear.weight.data.contiguous(),
                requires_grad=False, has_fp16_weights=False,
            )
            bnb_layer.bias = nn.Parameter(bnb_linear.bias.data.contiguous(), requires_grad=False)
            bnb_layer = bnb_layer.cuda()
            del bnb_linear
        except ImportError:
            bnb_layer = None
            print("  (bitsandbytes not available, skipping)")

    # VRAM comparison
    torch.cuda.empty_cache()
    gc.collect()

    _, vram_fp16 = measure_vram(lambda: nn.Linear(in_f, out_f, bias=True).cuda().half())
    _, vram_int8 = measure_vram(lambda: Int8Linear.from_linear(nn.Linear(in_f, out_f, bias=True).cuda().half()))

    vram_line = (f"\n  VRAM: fp16={vram_fp16/1e6:.1f}MB  int8={vram_int8/1e6:.1f}MB  "
                 f"savings={100*(vram_fp16-vram_int8)/max(vram_fp16,1):.0f}%")
    if int8_sparse is not None:
        _, vram_sparse = measure_vram(
            lambda: Int8Linear.from_linear_sparse(nn.Linear(in_f, out_f, bias=True).cuda().half()))
        vram_line += f"  sparse={vram_sparse/1e6:.1f}MB savings={100*(vram_fp16-vram_sparse)/max(vram_fp16,1):.0f}%"
    print(vram_line)

    # Header
    header = f"  {'Batch':>6} | {'FP16 ms':>9} {'TOPS':>7}"
    if int8_sparse is not None:
        header += f" | {'Sparse ms':>9} {'TOPS':>7} {'vs FP16':>8}"
    header += f" | {'Int8TC ms':>9} {'TOPS':>7} {'vs FP16':>8}"
    if bnb_layer:
        header += f" | {'BNB ms':>9} {'TOPS':>7}"
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    for M in batch_sizes:
        x = torch.randn(M, in_f, dtype=torch.float16, device="cuda")

        # FP16 baseline
        t_fp16 = timed_forward(ref_linear, x)
        tops_fp16 = compute_tops(M, in_f, out_f, t_fp16)

        line = f"  {M:>6} | {t_fp16*1000:>8.3f}ms {tops_fp16:>6.2f}T"

        # Sparse INT8
        if int8_sparse is not None:
            t_sp = timed_forward(int8_sparse, x)
            tops_sp = compute_tops(M, in_f, out_f, t_sp)
            sp_vs_fp16 = t_fp16 / t_sp
            line += f" | {t_sp*1000:>8.3f}ms {tops_sp:>6.2f}T {sp_vs_fp16:>7.2f}x"

        # Dense Int8 TC
        t_int8 = timed_forward(int8_tc, x)
        tops_int8 = compute_tops(M, in_f, out_f, t_int8)
        tc_vs_fp16 = t_fp16 / t_int8
        line += f" | {t_int8*1000:>8.3f}ms {tops_int8:>6.2f}T {tc_vs_fp16:>7.2f}x"

        # BNB
        if bnb_layer:
            t_bnb = timed_forward(bnb_layer, x)
            tops_bnb = compute_tops(M, in_f, out_f, t_bnb)
            line += f" | {t_bnb*1000:>8.3f}ms {tops_bnb:>6.2f}T"

        print(line)

    # Cleanup
    del ref_linear, int8_tc
    if int8_sparse is not None:
        del int8_sparse
    if bnb_layer:
        del bnb_layer
    torch.cuda.empty_cache()
    gc.collect()


# Sparse-mode layers of the paint pipeline at the token counts they see in a 6-view, 512px
# texture job (captured with forward hooks on every UNet and DINOv2 Linear during a real
# job; M = rows per call). Rows run every denoising step except DINOv2 (once per job).
# Omitted: K < 640 layers (FP16 in sparse mode) and the small-M rows that run only in the
# first step's reference pass (< 2% of the step's sparse FLOPs).
# (K, N, M, calls, label)
HUNYUAN_SHAPES = [
    (640, 640, 36864, 50, "UNet 640-wide attn proj, 64x64 latent"),
    (1280, 1280, 9216, 50, "UNet 1280-wide attn proj, 32x32 latent"),
    (640, 5120, 36864, 5, "UNet GEGLU up 640"),
    (1280, 10240, 9216, 5, "UNet GEGLU up 1280"),
    (1024, 1280, 37008, 12, "UNet K/V from 1024-d context"),
    (640, 640, 18432, 55, "UNet 640-wide attn proj"),
    (1280, 1280, 4608, 55, "UNet 1280-wide attn proj"),
    (1280, 320, 147456, 5, "UNet 1280->320 proj, 64x64 latent"),
    (2560, 640, 36864, 5, "UNet FFN down 640"),
    (5120, 1280, 9216, 5, "UNet FFN down 1280"),
    (1024, 640, 37008, 10, "UNet K/V from 1024-d context"),
    (1024, 320, 37008, 10, "UNet K/V from 1024-d context"),
    (640, 640, 3072, 35, "UNet 640-wide attn proj, small"),
    (1280, 1280, 768, 35, "UNet 1280-wide attn proj, small"),
    (1024, 1280, 2772, 12, "UNet K/V from 1024-d context, small"),
    (1280, 1280, 2304, 10, "UNet 1280-wide attn proj"),
    (1280, 1280, 1152, 11, "UNet 1280-wide attn proj"),
    (1536, 1536, 257, 160, "DINOv2-giant q/k/v/o (once per job)"),
    (1536, 8192, 257, 40, "DINOv2-giant SwiGLU w12 (once per job)"),
    (4096, 1536, 257, 40, "DINOv2-giant SwiGLU w3 (once per job)"),
]


def time_gpu(fn, warmup=100, reps=300, min_sample_ms=0.2):
    """GPU time per call: n calls captured in a CUDA graph, median over replays / n.

    Excludes host launch overhead (the pipeline runs the UNet under torch.compile).
    """
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        fn()
    e.record()
    torch.cuda.synchronize()
    est = s.elapsed_time(e) / 20
    n_inner = max(1, int(min_sample_ms / max(est, 1e-3) + 0.999))
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(st):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=st):
            for _ in range(n_inner):
                fn()
    torch.cuda.current_stream().wait_stream(st)
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(reps)]
    for s, e in ev:
        s.record(); g.replay(); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in ev) / n_inner


def _load_alt_sparse(src):
    """Build an alternate sparse kernel source (e.g. an older int8_sparse_tc.cu) for A/B."""
    from torch.utils.cpp_extension import load
    from csrc.build import _get_cuda_flags
    cflags, cuda_flags = _get_cuda_flags()
    return load(name="int8_sparse_tc_alt", sources=[src], extra_cflags=cflags,
                extra_cuda_cflags=cuda_flags, verbose=False)


def benchmark_hunyuan(rounds, alt_src=None):
    """Sparse layer vs FP16 on the pipeline's shapes; optional A/B against another kernel source.

    Rounds are interleaved (every variant of every shape is timed once per round, order
    alternating between rounds); the table shows the median across rounds [min-max].
    """
    from quantize_utils import _load_sp
    variants = [("fp16", None), ("sparse", _load_sp())]
    if alt_src:
        variants.insert(1, ("sparse_alt", _load_alt_sparse(alt_src)))
    print(f"Hunyuan shapes, GPU time per call (CUDA graph), {rounds} interleaved rounds, "
          f"median [min-max] ms" + (f"\n  sparse_alt = {alt_src}" if alt_src else ""))

    torch.manual_seed(0)
    cases = []
    for K, N, M, calls, label in HUNYUAN_SHAPES:
        lin = nn.Linear(K, N, bias=True).cuda().half()
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        layers = {}
        for name, mod in variants:
            if mod is None:
                layers[name] = lin
            else:
                layer = Int8Linear.from_linear_sparse(lin)
                layer._sp_ref = mod  # instance-level: this layer runs on `mod`
                layers[name] = layer
        cases.append((K, N, M, calls, label, x, layers))

    times = {}  # (case_idx, variant) -> [ms per round]
    for r in range(rounds):
        for ci, (_, _, _, _, _, x, layers) in enumerate(cases):
            names = [v for v, _ in variants]
            if r % 2:
                names = names[::-1]
            for name in names:
                layer = layers[name]
                with torch.no_grad():
                    t = time_gpu(lambda: layer(x))
                times.setdefault((ci, name), []).append(t)

    def fmt(ts):
        return f"{statistics.median(ts):.4f} [{min(ts):.4f}-{max(ts):.4f}]"

    names = [v for v, _ in variants]
    hdr = f"  {'K x N':>11} {'M':>6} | " + " | ".join(f"{n:>24}" for n in names)
    hdr += " | sparse vs fp16" + (" | alt -> new" if alt_src else "")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    step = {n: 0.0 for n in names}
    dino = {n: 0.0 for n in names}
    for ci, (K, N, M, calls, label, _, _) in enumerate(cases):
        med = {n: statistics.median(times[(ci, n)]) for n in names}
        for n in names:
            (dino if label.startswith("DINOv2") else step)[n] += med[n] * calls
        line = f"  {K:>5}x{N:<5} {M:>6} | " + " | ".join(f"{fmt(times[(ci, n)]):>24}" for n in names)
        line += f" | {med['fp16'] / med['sparse']:>13.2f}x"
        if alt_src:
            line += f" | {med['sparse_alt'] / med['sparse']:>9.2f}x"
        print(line + f"   {label} (x{calls}/step)")
    print("  UNet rows, one denoising step (sum of median x calls): " +
          "  ".join(f"{n}={step[n]:.2f} ms" for n in names))
    print("  DINOv2 rows, once per job (sum of median x calls):     " +
          "  ".join(f"{n}={dino[n]:.2f} ms" for n in names))


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantization methods")
    parser.add_argument("--no-bnb", action="store_true", help="Skip bitsandbytes benchmark")
    parser.add_argument("--no-sparse", action="store_true", help="Skip sparse INT8 benchmark")
    parser.add_argument("--shapes", type=int, nargs="*", help="Only test layers with these dimensions")
    parser.add_argument("--hunyuan", action="store_true",
                        help="Sparse vs FP16 on the paint pipeline's sparse-mode shapes and token counts")
    parser.add_argument("--rounds", type=int, default=3, help="Interleaved rounds for --hunyuan")
    parser.add_argument("--compare-sparse-src", default=None,
                        help="With --hunyuan: also time another int8_sparse_tc.cu (A/B)")
    args = parser.parse_args()

    if args.hunyuan:
        print(f"PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}  |  {torch.cuda.get_device_name(0)}")
        benchmark_hunyuan(args.rounds, args.compare_sparse_src)
        return

    print(f"PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}  |  {torch.cuda.get_device_name(0)}")
    print(f"torch._int_mm available: {int_mm_available()}")
    print(f"Sparse INT8 TC available: {sp_available()}")

    # UNet-relevant layer shapes (in_features, out_features)
    all_shapes = [
        (320, 320),
        (640, 640),
        (1280, 1280),
        (1280, 5120),    # representative 4x FFN up-projection (the paint UNet's GEGLU uses 1280->10240)
        (5120, 1280),    # representative 4x FFN down-projection
        (1024, 1024),    # cross-attention
        (1536, 6144),    # representative 4x FFN at width 1536
        (6144, 1536),    # representative 4x FFN down at width 1536
        (1536, 8192),    # dinov2-giant SwiGLU w12 (gate + up)
        (4096, 1536),    # dinov2-giant SwiGLU w3 (down)
    ]

    if args.shapes:
        all_shapes = [(i, o) for i, o in all_shapes if i in args.shapes or o in args.shapes]

    batch_sizes = [32, 128, 512, 1024, 4096]

    for in_f, out_f in all_shapes:
        benchmark_shape(in_f, out_f, batch_sizes,
                        use_bnb=not args.no_bnb,
                        use_sparse=not args.no_sparse)

    print(f"\n{'='*80}")
    print("  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
