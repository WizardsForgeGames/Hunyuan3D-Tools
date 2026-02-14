"""
Test sparse INT8 quantization on the Hunyuan3D multiview UNet architecture.

Creates a UNet2DConditionModel with the same config, applies quantization,
measures VRAM, and runs a forward pass to verify correctness.

Usage:
    python test_sparse_unet.py
    python test_sparse_unet.py --mode sparse       # 2:4 Sparse INT8 (default)
    python test_sparse_unet.py --mode tensorcore    # Dense INT8 TC
    python test_sparse_unet.py --mode fp16          # No quantization (baseline)
    python test_sparse_unet.py --mode compare       # Run all three, compare VRAM + timing
"""

import argparse
import gc
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn as nn


def get_vram_mb():
    return torch.cuda.memory_allocated() / 1e6


def count_linear_layers(model):
    """Count nn.Linear layers and total params."""
    count = 0
    params = 0
    shapes = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            count += 1
            n = m.in_features * m.out_features
            params += n
            key = (m.in_features, m.out_features)
            shapes[key] = shapes.get(key, 0) + 1
    return count, params, shapes


def count_int8_layers(model):
    """Count Int8Linear layers by mode."""
    from quantize_utils import Int8Linear
    counts = {"sparse": 0, "tc": 0, "other": 0}
    for m in model.modules():
        if isinstance(m, Int8Linear):
            if m.mode == "sparse":
                counts["sparse"] += 1
            elif m.mode == "tc":
                counts["tc"] += 1
            else:
                counts["other"] += 1
    return counts


def create_unet():
    """Create UNet2DConditionModel with Hunyuan3D-2.1 paint config."""
    from diffusers import UNet2DConditionModel

    # Exact config from tencent/Hunyuan3D-2.1 hunyuan3d-paintpbr-v2-1/unet/config.json
    # Note: actual model has in_channels=12 (4+4+4 for latent+normal+position)
    # but the base config says 4. The custom pipeline patches conv_in at load time.
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=[320, 640, 1280, 1280],
        down_block_types=[
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ],
        up_block_types=[
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ],
        cross_attention_dim=1024,
        attention_head_dim=[5, 10, 20, 20],
        use_linear_projection=True,
    )

    return unet


def test_mode(mode):
    """Create UNet, apply quantization, measure VRAM, run forward pass."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n  Creating UNet2DConditionModel...")
    t0 = time.time()
    unet = create_unet().half().cuda()
    print(f"  Created in {time.time() - t0:.1f}s")

    # Count layers
    n_layers, n_params, shapes = count_linear_layers(unet)
    print(f"\n  Linear layers: {n_layers}")
    print(f"  Total Linear params: {n_params / 1e6:.1f}M ({n_params * 2 / 1e6:.0f} MB as FP16)")
    print(f"  Layer shapes:")
    for (inf, outf), cnt in sorted(shapes.items(), key=lambda x: -(x[0][0] * x[0][1] * x[1])):
        print(f"    ({inf:>5}, {outf:>5}) x {cnt:>3}  [{inf*outf/1e6:.1f}M params each]")

    vram_model = get_vram_mb()
    print(f"\n  VRAM (FP16 model loaded): {vram_model:.0f} MB")

    if mode == "sparse":
        from quantize_utils import quantize_model_sparse
        print(f"\n  Applying sparse INT8 quantization (2:4 pruning)...")
        t0 = time.time()
        stats = quantize_model_sparse(unet)
        elapsed = time.time() - t0
        print(f"  Quantized in {elapsed:.1f}s")
    elif mode == "tensorcore":
        from quantize_utils import quantize_model_tensorcore
        print(f"\n  Applying dense TC INT8 quantization...")
        t0 = time.time()
        stats = quantize_model_tensorcore(unet)
        elapsed = time.time() - t0
        print(f"  Quantized in {elapsed:.1f}s")
    else:  # fp16
        print(f"\n  No quantization (FP16 baseline)")
        elapsed = 0

    torch.cuda.empty_cache()
    gc.collect()
    vram_after = get_vram_mb()

    print(f"\n  VRAM after quantization: {vram_after:.0f} MB")
    print(f"  VRAM savings vs FP16 model: {vram_model - vram_after:+.0f} MB")

    if mode != "fp16":
        int8_counts = count_int8_layers(unet)
        print(f"  Int8Linear layers: {int8_counts}")

    # Forward pass test
    print(f"\n  Running forward pass (M=1536 equivalent, 6 views x 64x64 latent)...")
    B = 1
    H, W = 64, 64
    latent = torch.randn(B * 6, 4, H, W, dtype=torch.float16, device="cuda")
    timestep = torch.tensor([500], device="cuda")
    encoder_hidden = torch.randn(B * 6, 77, 1024, dtype=torch.float16, device="cuda")

    # Warmup
    with torch.no_grad():
        try:
            _ = unet(latent, timestep, encoder_hidden_states=encoder_hidden)
            print("  Warmup forward: OK")
        except Exception as e:
            print(f"  Warmup forward FAILED: {e}")
            return vram_after, elapsed, False

    # Timed forward
    torch.cuda.synchronize()
    with torch.no_grad():
        t0 = time.time()
        n_iters = 3
        for _ in range(n_iters):
            out = unet(latent, timestep, encoder_hidden_states=encoder_hidden)
        torch.cuda.synchronize()
        fwd_ms = (time.time() - t0) / n_iters * 1000

    sample = out.sample
    print(f"  Output shape: {sample.shape}")
    print(f"  Output range: [{sample.min().item():.3f}, {sample.max().item():.3f}]")
    nan_count = sample.isnan().sum().item()
    print(f"  NaN count: {nan_count}")
    print(f"  Forward time: {fwd_ms:.1f} ms (avg of {n_iters})")

    peak_vram = torch.cuda.max_memory_allocated() / 1e6
    print(f"  Peak VRAM (incl. activations): {peak_vram:.0f} MB")

    ok = nan_count == 0
    if ok:
        print(f"  PASS: Forward pass OK")
    else:
        print(f"  FAIL: NaN detected in output!")

    del unet, latent, encoder_hidden, out
    torch.cuda.empty_cache()
    gc.collect()

    return vram_after, elapsed, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sparse",
                        choices=["sparse", "tensorcore", "fp16", "compare"])
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}  |  {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.mode == "compare":
        results = {}
        for mode in ["fp16", "tensorcore", "sparse"]:
            print(f"\n{'='*60}")
            print(f"  Mode: {mode.upper()}")
            print(f"{'='*60}")
            vram, elapsed, ok = test_mode(mode)
            results[mode] = (vram, elapsed, ok)

        print(f"\n{'='*60}")
        print(f"  VRAM Comparison (UNet model weights only)")
        print(f"{'='*60}")
        fp16_vram = results["fp16"][0]
        for mode, (vram, elapsed, ok) in results.items():
            savings = fp16_vram - vram
            pct = 100 * savings / fp16_vram if fp16_vram > 0 else 0
            status = "OK" if ok else "FAIL"
            quant_time = f"({elapsed:.1f}s)" if elapsed > 0 else ""
            print(f"  {mode:>12}: {vram:>7.0f} MB  "
                  f"(savings: {savings:>+6.0f} MB / {pct:>5.1f}%)  "
                  f"[{status}] {quant_time}")
    else:
        print(f"\n{'='*60}")
        print(f"  Mode: {args.mode.upper()}")
        print(f"{'='*60}")
        test_mode(args.mode)


if __name__ == "__main__":
    main()
