"""Check the sparse Int8Linear against FP16 math on the 2:4-pruned weights.

The reference is the layer's own pruned weight: the kernel's compressed int8 values and
metadata are expanded back to a dense (N, K) matrix, dequantized with the per-channel
scales, and multiplied in FP32. What remains is per-token INT8 activation quantization
error, so the relative error should be ~1% (the same metric sparsemma's tests use).
The error against the unpruned layer (~45% on random weights) is printed for context;
that is the cost of 2:4 pruning without fine-tuning, not a kernel error.

Shapes are the Linear layers that run in sparse mode in the paint pipeline, at the token
counts measured in a 6-view 512px job (HUNYUAN_SHAPES in tools/benchmark.py).

Run from the repo root: python tools/test_sparse_numerics.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from quantize_utils import Int8Linear, _load_sp

# (K, N, M, label)
SHAPES = [
    (640, 640, 36864, "UNet 640-wide attention projection"),
    (1280, 1280, 9216, "UNet 1280-wide attention projection"),
    (640, 5120, 36864, "UNet GEGLU up 640"),
    (1280, 10240, 9216, "UNet GEGLU up 1280"),
    (2560, 640, 36864, "UNet FFN down 640"),
    (5120, 1280, 9216, "UNet FFN down 1280"),
    (1024, 1280, 37008, "UNet K/V from 1024-d context"),
    (1024, 320, 2772, "UNet K/V from 1024-d context"),
    (1280, 320, 147456, "UNet 1280->320 projection"),
    (1280, 1280, 3, "UNet 1280-wide projection, tiny M"),
    (1536, 1536, 257, "DINOv2-giant q/k/v/o"),
    (1536, 8192, 257, "DINOv2-giant SwiGLU w12"),
    (4096, 1536, 257, "DINOv2-giant SwiGLU w3"),
    (1536, 4608, 257, "fused-qkv style 1536->4608"),
]


def pruned_dense_weight(layer, sp_mod, w_int8):
    """Expand the kernel's 2:4 compressed weight back to a dense dequantized (N, K) matrix."""
    comp, meta_raw = sp_mod.sparse_prune_compress(w_int8)
    N, K = w_int8.shape
    G = K // 4
    vals = comp.view(N, G, 2).float()
    nib = meta_raw.view(N, G).to(torch.int64)
    idx = torch.stack([nib & 3, (nib >> 2) & 3], dim=-1)
    dense = torch.zeros(N, G, 4, device=w_int8.device)
    dense.scatter_(2, idx, vals)
    assert torch.equal(comp, layer.weight_sparse_comp), "compressed weight mismatch"
    return dense.view(N, K) * layer.weight_scale.float()[:, None]


def main(tol=0.02):
    torch.manual_seed(0)
    sp_mod = _load_sp()
    assert sp_mod is not None, "sparse kernel unavailable"
    worst = 0.0
    for K, N, M, label in SHAPES:
        lin = nn.Linear(K, N).cuda().half()
        sp = Int8Linear.from_linear_sparse(lin)
        w32 = lin.weight.float()
        scale = w32.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        w_int8 = (w32 / scale[:, None]).round().clamp(-128, 127).to(torch.int8).contiguous()
        w_pruned = pruned_dense_weight(sp, sp_mod, w_int8)
        for m in sorted({M, 64}):
            x = torch.randn(m, K, device="cuda", dtype=torch.half)
            y = sp(x).float()
            ref = x.float() @ w_pruned.t() + lin.bias.float()
            full = lin(x).float()
            err = ((y - ref).norm() / ref.norm()).item()
            err_full = ((y - full).norm() / full.norm()).item()
            worst = max(worst, err)
            print(f"  {K:5d}x{N:<5d} M={m:5d}  rel err vs pruned FP32 {err:.4f}  "
                  f"(vs unpruned {err_full:.3f})  {label}")
            assert err < tol, f"{K}x{N} M={m}: rel err {err:.4f} >= {tol}"
    print(f"OK: worst rel err vs pruned weights {worst:.4f} (< {tol})")


if __name__ == "__main__":
    main()
