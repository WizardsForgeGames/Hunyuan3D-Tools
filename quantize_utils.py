"""
Int8 quantization utilities for Hunyuan3D texture pipeline.

Four compute backends (auto-selected per layer):
  1. "sparse" — 2:4 Sparse INT8 Tensor Core GEMM via PTX mma.sp.
     Prunes 50% of weights (2:4 pattern), compresses, and uses Sparse
     Tensor Cores. Requires K % 64 == 0, N % 16 == 0. Measured: 69% weight
     memory reduction per layer vs FP16 (incl. metadata/scales). Kernel is
     sparsemma's optimized one; on the paint UNet's sparse-mode shapes it
     runs 1.2-2.8x faster than cuBLAS FP16 per layer (GPU time, RTX 4090,
     tools/benchmark.py --hunyuan). Narrow layers at large M (1280->320 at
     147k tokens) fuse the activation quantize into the GEMM, so the int8
     copy of the activation never round-trips through DRAM.

  2. "tc" — Custom INT8 Tensor Core GEMM via wmma intrinsics. Fuses
     GEMM + dequant + bias in one kernel. Beats FP16 on wide layers
     (FFN up-projections) by 1.1-1.3x. Requires K % 64 == 0, N % 16 == 0.

  3. "dequant" — Dequantize int8 weights to fp16 on-the-fly, then fp16
     matmul. ~Same speed as nn.Linear; 50% less weight memory than FP16.
     Used as fallback when TC requirements aren't met.

  4. "bnb" — bitsandbytes Linear8bitLt with mixed-precision decomposition.
     Memory savings only (outlier overhead prevents speed gains).

Usage:
    from quantize_utils import quantize_model_sparse, quantize_model_tensorcore

    # Best: sparse INT8 (2:4 pruning + Sparse Tensor Cores)
    quantize_model_sparse(model.unet)

    # Dense INT8: custom TC GEMM (auto-fallback to dequant)
    quantize_model_tensorcore(model.unet)
"""

import torch
import torch.nn as nn

import os
import sys
import warnings

# Set INT8_KERNELS_STRICT=1 to raise instead of falling back when a kernel fails to build.
_STRICT = os.environ.get("INT8_KERNELS_STRICT", "") not in ("", "0")
_FAILED = object()  # cached load failure, so a failed build is attempted only once


def _kernel_unavailable(name, fallback, err):
    msg = (f"{name} kernel failed to build or load; falling back to {fallback}. "
           f"Error: {type(err).__name__}: {err}. "
           "On Windows with a non-UTF-8 code page, setting PYTHONUTF8=1 may help. "
           "Set INT8_KERNELS_STRICT=1 to make this an error.")
    if _STRICT:
        raise RuntimeError(msg) from err
    banner = "!" * 78
    print(banner, "  WARNING: " + msg, banner, sep="\n", file=sys.stderr, flush=True)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)



# ═══════════════════════════════════════════════════════════════════════════════
#  Custom TC GEMM module (lazy-loaded)
# ═══════════════════════════════════════════════════════════════════════════════

_TC_MODULE = None


def _load_tc():
    """Lazy-load the custom Tensor Core INT8 GEMM extension."""
    global _TC_MODULE
    if _TC_MODULE is _FAILED:
        return None
    if _TC_MODULE is not None:
        return _TC_MODULE
    try:
        from csrc.build import load_tc_gemm
        _TC_MODULE = load_tc_gemm()
        return _TC_MODULE
    except Exception as e:
        _TC_MODULE = _FAILED
        _kernel_unavailable("Dense INT8 Tensor Core GEMM (int8_gemm_tc)", "the dequantize + FP16 matmul path", e)
        return None


def tc_available() -> bool:
    """Check if the custom TC GEMM kernel is available."""
    return _load_tc() is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  Sparse TC GEMM module (lazy-loaded)
# ═══════════════════════════════════════════════════════════════════════════════

_SP_MODULE = None


def _load_sp():
    """Lazy-load the sparse INT8 Tensor Core GEMM extension (PTX mma.sp)."""
    global _SP_MODULE
    if _SP_MODULE is _FAILED:
        return None
    if _SP_MODULE is not None:
        return _SP_MODULE
    try:
        from csrc.build import load_sparse_tc
        _SP_MODULE = load_sparse_tc()
        return _SP_MODULE
    except Exception as e:
        _SP_MODULE = _FAILED
        _kernel_unavailable("Sparse INT8 Tensor Core GEMM (int8_sparse_tc)", "the dense path", e)
        return None


def sp_available() -> bool:
    """Check if the sparse INT8 TC GEMM kernel is available."""
    return _load_sp() is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  Detect torch._int_mm availability
# ═══════════════════════════════════════════════════════════════════════════════

def _check_int_mm() -> bool:
    """Test whether torch._int_mm actually works on this platform."""
    if not torch.cuda.is_available():
        return False
    try:
        a = torch.zeros(32, 32, dtype=torch.int8, device="cuda")
        b = torch.zeros(32, 32, dtype=torch.int8, device="cuda")
        torch._int_mm(a, b)
        return True
    except (RuntimeError, AttributeError):
        return False


_INT_MM_AVAILABLE: bool | None = None  # lazy init


def int_mm_available() -> bool:
    """Check if torch._int_mm works (cached after first call)."""
    global _INT_MM_AVAILABLE
    if _INT_MM_AVAILABLE is None:
        _INT_MM_AVAILABLE = _check_int_mm()
    return _INT_MM_AVAILABLE


# ═══════════════════════════════════════════════════════════════════════════════
#  Int8Linear — stores weights as int8, computes via Tensor Cores or fallback
# ═══════════════════════════════════════════════════════════════════════════════

class Int8Linear(nn.Module):
    """Drop-in nn.Linear replacement with int8 weight storage.

    Four compute paths (auto-selected):
      1. sparse: 2:4 Sparse INT8 TC GEMM via PTX mma.sp (fused quant+GEMM+dequant+bias).
         Requires K % 64 == 0, N % 16 == 0. Uses from_linear_sparse().
      2. TC: Dense INT8 Tensor Core GEMM (fused quant+GEMM+dequant+bias).
         Requires K % 64 == 0, N % 16 == 0.
      3. dequant: Dequantize int8->fp16 on-the-fly + fp16 matmul. Fallback.
      4. int_mm: torch._int_mm (legacy, slower than both due to unfused dequant).

    Weight quantization (offline, per output-channel):
        scale_j = max(|W[j,:]|) / 127
        W_int8[j,:] = round(W[j,:] / scale_j).clamp(-128, 127)
    """

    __constants__ = ["in_features", "out_features", "_tc_ok", "_sparse_ok"]
    _MIN_ROWS = 32  # torch._int_mm cuBLAS constraint

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # TC kernel requires K % 64 == 0 and N % 16 == 0
        self._tc_ok = (in_features % 64 == 0) and (out_features % 16 == 0)
        # Sparse kernel requires K % 64 == 0 and N % 16 == 0 (BLOCK_K=64)
        self._sparse_ok = (in_features % 64 == 0) and (out_features % 16 == 0)

        # Dense weight buffers (populated by from_linear, empty in sparse mode)
        self.register_buffer(
            "weight_int8_nk",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        # Per-channel fp16 scale: (out_features,)
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.float16),
        )
        # Sparse weight buffers (None until from_linear_sparse populates them)
        self.register_buffer("weight_sparse_comp", None)
        self.register_buffer("weight_sparse_meta", None)

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float16),
            )
        else:
            self.bias = None

    @staticmethod
    def from_linear(linear: nn.Linear) -> "Int8Linear":
        """Convert an nn.Linear to Int8Linear with per-channel weight quantization."""
        has_bias = linear.bias is not None
        layer = Int8Linear(linear.in_features, linear.out_features, bias=has_bias)

        # Per-channel quantization: scale_j = max(|W[j,:]|) / 127
        weight_f32 = linear.weight.data.float()
        scale = weight_f32.abs().amax(dim=1) / 127.0
        scale = scale.clamp(min=1e-8)

        weight_int8 = (weight_f32 / scale[:, None]).round().clamp(-128, 127).to(torch.int8)

        # One (N, K) copy. The int_mm and dequant paths read it through a
        # transposed view, so no second (K, N) copy is stored.
        layer.weight_int8_nk = weight_int8.contiguous()          # (N, K)
        layer.weight_scale = scale.to(torch.float16)

        if has_bias:
            layer.bias = linear.bias.data.to(torch.float16)

        return layer.to(linear.weight.device)

    @staticmethod
    def from_linear_sparse(linear: nn.Linear) -> "Int8Linear":
        """Convert nn.Linear to sparse Int8Linear (2:4 pruning + int8 quantization).

        Prunes 50% of weights (2 per group of 4 along K), quantizes to int8,
        compresses, and packs metadata for Sparse Tensor Core GEMM.

        Weight memory: ~62.5% of dense int8 (compressed + metadata).
        """
        sp_mod = _load_sp()
        if sp_mod is None:
            raise RuntimeError("Sparse TC module not available. Cannot create sparse layer.")

        has_bias = linear.bias is not None
        layer = Int8Linear(linear.in_features, linear.out_features, bias=has_bias)

        # Per-channel quantization: scale_j = max(|W[j,:]|) / 127
        weight_f32 = linear.weight.data.float()
        scale = weight_f32.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        weight_int8 = (weight_f32 / scale[:, None]).round().clamp(-128, 127).to(torch.int8)

        # Prune + compress + pack metadata on GPU
        device = linear.weight.device
        results = sp_mod.sparse_prune_and_pack(weight_int8.to(device))
        layer.weight_sparse_comp = results[0]   # (N, K/2) int8
        layer.weight_sparse_meta = results[1]   # (N, K/32) int32
        layer.weight_scale = scale.to(torch.float16)

        # Free dense weight buffers (not needed in sparse mode)
        layer.register_buffer("weight_int8_nk", torch.empty(0, dtype=torch.int8))

        if has_bias:
            layer.bias = linear.bias.data.to(torch.float16)

        layer.mode = "sparse"
        return layer.to(device)

    _tc_ref = None  # cached module reference
    _sp_ref = None  # cached sparse module reference
    _empty_bias = None  # cached empty tensor for no-bias layers

    def _forward_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse TC GEMM: fused quant -> sparse INT8 GEMM -> dequant + bias."""
        if self._sp_ref is None:
            self._sp_ref = _load_sp()
        if self.bias is not None:
            bias = self.bias
        else:
            if self._empty_bias is None or self._empty_bias.device != x.device:
                self._empty_bias = torch.empty(0, dtype=torch.float16, device=x.device)
            bias = self._empty_bias
        return self._sp_ref.sparse_int8_linear(
            x, self.weight_sparse_comp, self.weight_sparse_meta,
            self.weight_scale, bias
        )

    def _forward_tc(self, x: torch.Tensor) -> torch.Tensor:
        """Custom TC GEMM: fused quant → INT8 Tensor Core GEMM → dequant + bias."""
        if self._tc_ref is None:
            self._tc_ref = _load_tc()
        if self.bias is not None:
            bias = self.bias
        else:
            if self._empty_bias is None or self._empty_bias.device != x.device:
                self._empty_bias = torch.empty(0, dtype=torch.float16, device=x.device)
            bias = self._empty_bias
        return self._tc_ref.int8_linear_tc(x, self.weight_int8_nk, self.weight_scale, bias)

    def _forward_int_mm(self, x: torch.Tensor, M: int) -> torch.Tensor:
        """Legacy path: int8 activation quant → torch._int_mm → dequant."""
        x_f32 = x.float()
        act_scale = (x_f32.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        x_int8 = (x_f32 / act_scale[:, None]).round().clamp(-128, 127).to(torch.int8)

        if M < self._MIN_ROWS:
            pad = self._MIN_ROWS - M
            x_int8 = torch.nn.functional.pad(x_int8, (0, 0, 0, pad))
            act_scale = torch.nn.functional.pad(act_scale, (0, pad), value=1.0)

        y_int32 = torch._int_mm(x_int8, self.weight_int8_nk.t())
        y = (y_int32.float() * (act_scale[:, None] * self.weight_scale[None, :].float())).half()

        if M < self._MIN_ROWS:
            y = y[:M]
        return y

    def _forward_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback: dequantize int8 weights to fp16, then fp16 matmul."""
        w_fp16 = self.weight_int8_nk.t().to(torch.float16) * self.weight_scale[None, :]
        return torch.mm(x.to(torch.float16), w_fp16)

    # "tc" (default, auto-falls back to dequant at small M), "dequant", or "int_mm"
    mode: str = "tc"

    # Below this M the TC kernel's fixed overhead dominates. Re-measured on an
    # RTX 4090 (2026-10-19) at DINOv2-giant's 257 tokens: TC beats the dequant
    # fallback on 1536x1536 (0.044 vs 0.077 ms) and 1536x8192 (0.077 vs 0.129 ms),
    # ties on 4096x1536 and loses on 1536x4608 (0.172 vs 0.087 ms); net neutral
    # per block. 256 lets DINOv2's layers use the TC kernel.
    _TC_MIN_M: int = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        M = x.shape[0]

        if self.mode == "sparse" and self._sparse_ok and sp_available():
            # Sparse path: fused quant + sparse INT8 GEMM + dequant + bias
            y = self._forward_sparse(x)
        elif self.mode == "tc" and self._tc_ok and tc_available() and M >= self._TC_MIN_M:
            # TC path: fused quant + GEMM + dequant + bias in one kernel
            y = self._forward_tc(x)
        elif self.mode == "int_mm" and int_mm_available():
            y = self._forward_int_mm(x, M)
            if self.bias is not None:
                y = y + self.bias
            if len(orig_shape) > 2:
                y = y.reshape(*orig_shape[:-1], self.out_features)
            return y
        else:
            y = self._forward_dequant(x)
            if self.bias is not None:
                y = y + self.bias
            if len(orig_shape) > 2:
                y = y.reshape(*orig_shape[:-1], self.out_features)
            return y

        # Sparse and TC paths already include dequant + bias in the kernel
        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], self.out_features)
        return y

    def extra_repr(self) -> str:
        if self.mode == "sparse" and self._sparse_ok and sp_available():
            mode_str = "sparse_tc (2:4 Sparse INT8 Tensor Core)"
        elif self.mode == "tc" and self._tc_ok and tc_available():
            mode_str = "tc_gemm (custom INT8 Tensor Core)"
        elif self.mode == "int_mm" and int_mm_available():
            mode_str = "int_mm (cuBLAS)"
        else:
            mode_str = "dequant (int8->fp16 + fp16 matmul)"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={mode_str}"
        )


def quantize_model_tensorcore(model, min_features=128, verbose=True):
    """Replace nn.Linear layers with Int8Linear (Tensor Core int8 or dequant fallback).

    Args:
        model: The nn.Module to quantize in-place.
        min_features: Only quantize layers where both in/out >= this value.
        verbose: Print summary of what was quantized.

    Returns:
        dict with stats.
    """
    stats = {"total_params": 0, "quantized_params": 0, "skipped_params": 0, "num_replaced": 0}
    _replace_linear_tensorcore(model, min_features, stats)

    if verbose:
        saved_mb = stats["quantized_params"] * 1 / 1e6
        tc_ok = tc_available()
        mode = "TC GEMM (custom INT8 Tensor Core)" if tc_ok else "dequant fallback"
        print(f"  Quantized {stats['num_replaced']} Linear layers -> Int8Linear [{mode}]")
        print(f"  TC-eligible: {stats.get('tc_eligible', 0)} layers")
        print(f"  Linear params: {stats['total_params'] / 1e6:.1f}M total, "
              f"{stats['quantized_params'] / 1e6:.1f}M quantized, "
              f"{stats['skipped_params'] / 1e6:.1f}M skipped (below {min_features} features)")
        print(f"  Estimated VRAM saved: ~{saved_mb:.0f} MB")

    return stats


def _replace_linear_tensorcore(module, min_features, stats):
    """Walk the module tree and swap Linear -> Int8Linear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            num_params = child.in_features * child.out_features
            stats["total_params"] += num_params

            if child.in_features >= min_features and child.out_features >= min_features:
                int8_layer = Int8Linear.from_linear(child)
                setattr(module, name, int8_layer)
                stats["quantized_params"] += num_params
                stats["num_replaced"] += 1
                if int8_layer._tc_ok:
                    stats["tc_eligible"] = stats.get("tc_eligible", 0) + 1
            else:
                stats["skipped_params"] += num_params
        else:
            _replace_linear_tensorcore(child, min_features, stats)


# ═══════════════════════════════════════════════════════════════════════════════
#  Sparse INT8 model quantization (2:4 pruning + Sparse Tensor Cores)
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_model_sparse(model, min_features=128, sparse_min_k=640, verbose=True):
    """Replace nn.Linear layers with sparse Int8Linear (2:4 pruning + Sparse TC GEMM).

    Uses mixed quantization: only applies sparse INT8 where the kernel is faster
    than FP16 cuBLAS (large K dimension). Layers with small K stay as nn.Linear.

    Sparse wins when K >= sparse_min_k because the K-loop has enough iterations
    to amortize kernel overhead. Small layers (320-dim, 640-dim) are faster in FP16.

    Args:
        model: The nn.Module to quantize in-place.
        min_features: Only quantize layers where both in/out >= this value.
        sparse_min_k: Minimum in_features (K dimension) for sparse quantization.
            Layers with in_features < sparse_min_k stay as FP16. Default 640.
            Chosen with the previous sparse kernel (k>=640 gave ~1.01x FP16 speed on an
            RTX 4090); not re-tuned for the current kernel.
        verbose: Print summary of what was quantized.

    Returns:
        dict with stats.
    """
    if not sp_available():
        raise RuntimeError(
            "Sparse TC module not available. Compile with: python csrc/build.py sparse_tc"
        )

    stats = {"total_params": 0, "quantized_params": 0, "skipped_params": 0,
             "num_replaced": 0, "sparse_eligible": 0, "kept_fp16": 0}
    _replace_linear_sparse(model, min_features, sparse_min_k, stats)

    if verbose:
        fp16_mb = stats["quantized_params"] * 2 / 1e6
        # 0.5 B/weight int8 values (K/2 per row) + 0.125 B/weight metadata (K/32 int32 per row)
        sparse_mb = stats["quantized_params"] * 0.625 / 1e6
        saved_mb = fp16_mb - sparse_mb
        print(f"  Quantized {stats['num_replaced']} Linear layers -> sparse Int8Linear [2:4 Sparse TC]")
        print(f"  Sparse-eligible: {stats['sparse_eligible']} layers")
        print(f"  Kept as FP16: {stats['kept_fp16']} layers "
              f"(K < {sparse_min_k}, K % 64 != 0, or N % 16 != 0)")
        print(f"  Linear params: {stats['total_params'] / 1e6:.1f}M total, "
              f"{stats['quantized_params'] / 1e6:.1f}M quantized, "
              f"{stats['skipped_params'] / 1e6:.1f}M skipped (below {min_features} features)")
        print(f"  Estimated VRAM saved: ~{saved_mb:.0f} MB (vs FP16)")

    return stats


def _replace_linear_sparse(module, min_features, sparse_min_k, stats):
    """Walk the module tree and swap Linear -> sparse Int8Linear where beneficial."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            num_params = child.in_features * child.out_features
            stats["total_params"] += num_params

            if child.in_features < min_features or child.out_features < min_features:
                stats["skipped_params"] += num_params
            elif (child.in_features < sparse_min_k
                  or child.in_features % 64 != 0
                  or child.out_features % 16 != 0):
                # K too small for the sparse kernel to beat FP16, or a shape the kernel
                # cannot run (needs K % 64 == 0, N % 16 == 0). A sparse Int8Linear with an
                # unsupported shape would fall through to the dequant path, which has no
                # dense weights in sparse mode, so keep these layers as nn.Linear.
                stats["kept_fp16"] = stats.get("kept_fp16", 0) + 1
            else:
                int8_layer = Int8Linear.from_linear_sparse(child)
                setattr(module, name, int8_layer)
                stats["quantized_params"] += num_params
                stats["num_replaced"] += 1
                if int8_layer._sparse_ok:
                    stats["sparse_eligible"] += 1
        else:
            _replace_linear_sparse(child, min_features, sparse_min_k, stats)


# ═══════════════════════════════════════════════════════════════════════════════
#  bitsandbytes Int8 — mixed-precision decomposition (memory-only savings)
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_model_int8(model, min_features=128, verbose=True):
    """Replace nn.Linear layers with bitsandbytes Linear8bitLt (int8).

    Args:
        model: The nn.Module to quantize in-place.
        min_features: Only quantize layers where both in/out >= this value.
        verbose: Print summary of what was quantized.

    Returns:
        dict with stats.
    """
    import bitsandbytes as bnb  # lazy import

    stats = {"total_params": 0, "quantized_params": 0, "skipped_params": 0, "num_replaced": 0}
    _replace_linear_bnb(model, min_features, stats, bnb)

    if verbose:
        saved_mb = stats["quantized_params"] * 1 / 1e6
        print(f"  Quantized {stats['num_replaced']} Linear layers to int8 (bitsandbytes)")
        print(f"  Linear params: {stats['total_params'] / 1e6:.1f}M total, "
              f"{stats['quantized_params'] / 1e6:.1f}M quantized, "
              f"{stats['skipped_params'] / 1e6:.1f}M skipped (below {min_features} features)")
        print(f"  Estimated VRAM saved: ~{saved_mb:.0f} MB")

    return stats


def _replace_linear_bnb(module, min_features, stats, bnb):
    """Walk the module tree and swap Linear -> Linear8bitLt."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            num_params = child.in_features * child.out_features
            stats["total_params"] += num_params

            if child.in_features >= min_features and child.out_features >= min_features:
                int8_layer = _make_bnb_int8_linear(child, bnb)
                setattr(module, name, int8_layer)
                stats["quantized_params"] += num_params
                stats["num_replaced"] += 1
            else:
                stats["skipped_params"] += num_params
        else:
            _replace_linear_bnb(child, min_features, stats, bnb)


def _make_bnb_int8_linear(linear, bnb):
    """Convert a single nn.Linear to bnb.nn.Linear8bitLt, preserving weights."""
    has_bias = linear.bias is not None
    device = linear.weight.device

    int8_linear = bnb.nn.Linear8bitLt(
        linear.in_features,
        linear.out_features,
        bias=has_bias,
        has_fp16_weights=False,
        threshold=6.0,
    )

    int8_linear.weight = bnb.nn.Int8Params(
        linear.weight.data.contiguous(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    if has_bias:
        int8_linear.bias = nn.Parameter(linear.bias.data.contiguous(), requires_grad=False)

    return int8_linear.to(device)
