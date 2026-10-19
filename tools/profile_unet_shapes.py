"""
Profile sparse vs FP16 at exact UNet layer shapes and batch sizes.
Identifies which layers contribute most to the overall speed gap.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import torch
import torch.nn as nn
from quantize_utils import Int8Linear, sp_available

def timed(layer, x, warmup=10, repeats=50):
    for _ in range(warmup):
        _ = layer(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = layer(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000  # ms

# UNet shapes: (in_features, out_features, count_in_unet)
# M values at each channel level:
#   320-channel: spatial=64x64=4096, M=6*4096=24576
#   640-channel: spatial=32x32=1024, M=6*1024=6144
#   1280-channel: spatial=16x16=256, M=6*256=1536
# Cross-attention K-proj: M same as spatial at that level
# Cross-attention V/out: M = 6 * num_tokens (77 for text)

SHAPES = [
    # (in_f, out_f, count, M, description)
    (320,  320,   40, 24576, "320ch self-attn/conv proj"),
    (320,  2560,   5, 24576, "320ch FFN up"),
    (1280, 320,   10, 24576, "320ch cross-attn K/V from 1280"),
    (1024, 320,   10,   462, "320ch cross-attn K/V (77 tokens)"),

    (640,  640,   40,  6144, "640ch self-attn/conv proj"),
    (640,  5120,   5,  6144, "640ch FFN up"),
    (2560, 640,    5,  6144, "640ch FFN down"),
    (1024, 640,   10,   462, "640ch cross-attn K/V"),

    (1280, 1280,  61,  1536, "1280ch self-attn/conv proj"),
    (1280, 10240,  6,  1536, "1280ch FFN up"),
    (5120, 1280,   6,  1536, "1280ch FFN down"),
    (1024, 1280,  12,   462, "1280ch cross-attn K/V"),

    (1280, 640,    5,  6144, "up-block channel proj"),
    (320, 1280,    1, 24576, "skip-connection proj"),
]

print(f"{'Shape':>20} {'Count':>5} {'M':>6} | {'FP16 ms':>8} {'Sparse ms':>10} {'Ratio':>6} | {'FP16 tot':>9} {'Sparse tot':>10} {'Delta':>7} | Description")
print("-" * 140)

total_fp16 = 0
total_sparse = 0

for in_f, out_f, count, M, desc in SHAPES:
    ref = nn.Linear(in_f, out_f, bias=True).cuda().half()
    sp = Int8Linear.from_linear_sparse(ref)

    x = torch.randn(M, in_f, dtype=torch.float16, device="cuda")

    t_fp16 = timed(ref, x)
    t_sp = timed(sp, x)

    ratio = t_sp / t_fp16
    tot_fp16 = t_fp16 * count
    tot_sp = t_sp * count
    delta = tot_sp - tot_fp16

    total_fp16 += tot_fp16
    total_sparse += tot_sp

    print(f"({in_f:>4},{out_f:>5}) x{count:>3}  M={M:>5} | {t_fp16:>7.3f}ms {t_sp:>9.3f}ms {ratio:>5.2f}x | {tot_fp16:>8.2f}ms {tot_sp:>9.2f}ms {delta:>+6.1f}ms | {desc}")

    del ref, sp, x
    torch.cuda.empty_cache()

print("-" * 140)
print(f"{'TOTAL':>49} | {total_fp16:>8.2f}ms {total_sparse:>9.2f}ms {total_sparse - total_fp16:>+6.1f}ms")
print(f"\nOverall sparse/FP16 ratio: {total_sparse/total_fp16:.3f}x")
