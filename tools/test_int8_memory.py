"""Check Int8Linear weight memory and output accuracy for each compute path.

Run from the repo root: python tools/test_int8_memory.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from quantize_utils import Int8Linear


def nbytes(module):
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()) if t is not None)


def main():
    torch.manual_seed(0)
    for K, N in [(1536, 6144), (1280, 5120), (5120, 1280)]:
        lin = nn.Linear(K, N).cuda().half()
        q = Int8Linear.from_linear(lin)
        sp = Int8Linear.from_linear_sparse(lin)
        print(f"Linear({K},{N}): fp16 {nbytes(lin)/2**20:.2f} MiB | int8 tc {nbytes(q)/2**20:.2f} MiB "
              f"({100*(1-nbytes(q)/nbytes(lin)):.0f}% less) | sparse {nbytes(sp)/2**20:.2f} MiB ({100*(1-nbytes(sp)/nbytes(lin)):.0f}% less)")
        for M in (64, 4096):
            x = torch.randn(M, K, device="cuda", dtype=torch.half)
            ref = lin(x).float()
            for mode in ("tc", "dequant", "int_mm"):
                q.mode = mode
                y = q(x).float()
                err = ((y - ref).norm() / ref.norm()).item()
                assert err < 0.02, f"{mode} M={M} rel err {err}"
                print(f"   M={M:5d} {mode:8s} rel err {err:.4f}")


if __name__ == "__main__":
    main()
