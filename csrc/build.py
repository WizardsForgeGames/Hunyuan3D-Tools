"""
JIT-compile the fused int8 CUDA kernels.

Four modules:
  1. int8_kernels   — fused quant/dequant + cuBLAS _int_mm path
  2. int8_gemm_tc   — custom Tensor Core GEMM via wmma (replaces _int_mm)
  3. sparse_gemm    — 2:4 sparse INT8 GEMM via cuSPARSELt (BLOCKED on Windows)
  4. int8_sparse_tc — hand-rolled sparse INT8 GEMM via PTX mma.sp (bypasses cuSPARSELt)

Usage:
    from csrc.build import load_int8_kernels, load_tc_gemm, load_sparse_tc
    kernels = load_int8_kernels()
    tc = load_tc_gemm()
    tc.int8_linear_tc(x, w_int8_nk, w_scale, bias)
    sp = load_sparse_tc()
    sp.test_mma_sp(A_comp, B_dense, meta)
"""

import os
import torch
from torch.utils.cpp_extension import load

_CACHED_MODULE = None
_CACHED_TC_MODULE = None
_CACHED_SPARSE_MODULE = None
_CACHED_SPARSE_TC_MODULE = None

# MSVC path for Windows builds
_MSVC_PATH = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822"

# cuSPARSELt paths (pip install nvidia-cusparselt-cu12)
_CUSPARSELT_BASE = None


def _get_cuda_flags():
    """Common CUDA compilation flags."""
    extra_cflags = []
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_89,code=sm_89",  # RTX 4090 (Ada Lovelace)
        "-gencode=arch=compute_86,code=sm_86",   # RTX 3090 fallback
        "--allow-unsupported-compiler",
    ]

    if os.name == "nt":
        cl_dir = os.path.join(_MSVC_PATH, "bin", "Hostx64", "x64")
        if os.path.isdir(cl_dir) and cl_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cl_dir + os.pathsep + os.environ.get("PATH", "")

        cl_exe = os.path.join(_MSVC_PATH, "bin", "Hostx64", "x64", "cl.exe")
        if os.path.isfile(cl_exe):
            extra_cuda_cflags.append(f"-ccbin={cl_exe}")

    return extra_cflags, extra_cuda_cflags


def load_int8_kernels():
    """Load the fused quant/dequant kernels (cuBLAS _int_mm path)."""
    global _CACHED_MODULE
    if _CACHED_MODULE is not None:
        return _CACHED_MODULE

    csrc_dir = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(csrc_dir, "int8_kernels.cu")
    extra_cflags, extra_cuda_cflags = _get_cuda_flags()

    _CACHED_MODULE = load(
        name="int8_kernels",
        sources=[source],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    return _CACHED_MODULE


def load_tc_gemm():
    """Load the custom Tensor Core INT8 GEMM kernel (replaces cuBLAS _int_mm)."""
    global _CACHED_TC_MODULE
    if _CACHED_TC_MODULE is not None:
        return _CACHED_TC_MODULE

    csrc_dir = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(csrc_dir, "int8_gemm_tc.cu")
    extra_cflags, extra_cuda_cflags = _get_cuda_flags()

    _CACHED_TC_MODULE = load(
        name="int8_gemm_tc",
        sources=[source],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    return _CACHED_TC_MODULE


def _find_cusparselt():
    """Find cuSPARSELt install (pip package or manual)."""
    global _CUSPARSELT_BASE
    if _CUSPARSELT_BASE is not None:
        return _CUSPARSELT_BASE

    # Try pip package: nvidia-cusparselt-cu12
    # v0.8+: nvidia.cusparselt namespace; v0.6.x: top-level cusparselt/ dir
    try:
        import nvidia.cusparselt
        base = nvidia.cusparselt.__path__[0]
        if os.path.isfile(os.path.join(base, "include", "cusparseLt.h")):
            _CUSPARSELT_BASE = base
            return base
    except ImportError:
        pass

    # v0.6.x layout: cusparselt/ in site-packages (no nvidia namespace)
    import importlib.metadata
    try:
        dist = importlib.metadata.distribution("nvidia-cusparselt-cu12")
        base = os.path.join(os.path.dirname(dist._path), "cusparselt")
        if os.path.isfile(os.path.join(base, "include", "cusparseLt.h")):
            _CUSPARSELT_BASE = base
            return base
    except importlib.metadata.PackageNotFoundError:
        pass

    # Try common manual install locations
    for path in [r"C:\libs\cusparselt", r"G:\libs\cusparselt"]:
        if os.path.isfile(os.path.join(path, "include", "cusparseLt.h")):
            _CUSPARSELT_BASE = path
            return path

    return None


def load_sparse_gemm():
    """Load the 2:4 sparse INT8 GEMM via cuSPARSELt."""
    global _CACHED_SPARSE_MODULE
    if _CACHED_SPARSE_MODULE is not None:
        return _CACHED_SPARSE_MODULE

    cusparselt_base = _find_cusparselt()
    if cusparselt_base is None:
        raise RuntimeError(
            "cuSPARSELt not found. Install with: pip install nvidia-cusparselt-cu12"
        )

    # Add DLL directory to search path so cusparseLt.dll can be found at runtime
    dll_dir = os.path.join(cusparselt_base, "bin")
    if os.path.isdir(dll_dir):
        if dll_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        # Python 3.8+ on Windows: must use add_dll_directory for DLL search
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)

    csrc_dir = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(csrc_dir, "sparse_gemm.cu")
    extra_cflags, extra_cuda_cflags = _get_cuda_flags()

    include_dir = os.path.join(cusparselt_base, "include")
    lib_dir = os.path.join(cusparselt_base, "lib")

    extra_include_paths = [include_dir]

    # Windows linking
    if os.name == "nt":
        extra_ldflags = [f"/LIBPATH:{lib_dir}", "cusparseLt.lib"]
    else:
        extra_ldflags = [f"-L{lib_dir}", "-lcusparseLt"]

    _CACHED_SPARSE_MODULE = load(
        name="sparse_gemm",
        sources=[source],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        extra_ldflags=extra_ldflags,
        verbose=True,
    )
    return _CACHED_SPARSE_MODULE


def load_sparse_tc():
    """Load the hand-rolled sparse INT8 GEMM via PTX mma.sp (no cuSPARSELt)."""
    global _CACHED_SPARSE_TC_MODULE
    if _CACHED_SPARSE_TC_MODULE is not None:
        return _CACHED_SPARSE_TC_MODULE

    csrc_dir = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(csrc_dir, "int8_sparse_tc.cu")
    extra_cflags, extra_cuda_cflags = _get_cuda_flags()

    _CACHED_SPARSE_TC_MODULE = load(
        name="int8_sparse_tc",
        sources=[source],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    return _CACHED_SPARSE_TC_MODULE


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("all", "kernels"):
        print("Compiling int8_kernels...")
        m = load_int8_kernels()
        print(f"  {[x for x in dir(m) if not x.startswith('_')]}")

    if target in ("all", "tc"):
        print("Compiling int8_gemm_tc (Tensor Core GEMM)...")
        m = load_tc_gemm()
        print(f"  {[x for x in dir(m) if not x.startswith('_')]}")

    if target in ("all", "sparse"):
        print("Compiling sparse_gemm (cuSPARSELt)...")
        m = load_sparse_gemm()
        print(f"  {[x for x in dir(m) if not x.startswith('_')]}")

    if target in ("all", "sparse_tc"):
        print("Compiling int8_sparse_tc (PTX mma.sp)...")
        m = load_sparse_tc()
        print(f"  {[x for x in dir(m) if not x.startswith('_')]}")

    print("Done.")
