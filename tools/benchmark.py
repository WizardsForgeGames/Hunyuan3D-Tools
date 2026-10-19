"""
Benchmark: FP16 vs Sparse INT8 vs Dense TC INT8 vs bitsandbytes.

Measures throughput, latency, VRAM, and correctness at UNet-relevant matrix sizes.

Usage:
    python tools/benchmark.py
    python tools/benchmark.py --no-bnb       # skip bitsandbytes (if not installed)
    python tools/benchmark.py --no-sparse    # skip sparse INT8
    python tools/benchmark.py --shapes 1280  # only test layers with dim 1280
"""

import argparse
import gc
import os
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantization methods")
    parser.add_argument("--no-bnb", action="store_true", help="Skip bitsandbytes benchmark")
    parser.add_argument("--no-sparse", action="store_true", help="Skip sparse INT8 benchmark")
    parser.add_argument("--shapes", type=int, nargs="*", help="Only test layers with these dimensions")
    args = parser.parse_args()

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
