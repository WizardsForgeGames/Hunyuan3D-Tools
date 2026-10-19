# Hunyuan3D-2.1 Optimized Pipeline Tools

Performance-tuned tooling for [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) on a single 24 GB consumer GPU. On an RTX 4090 24 GB:

- **Texture stage: ~35 s per model** with 10 denoising steps (upstream default is 15), INT8 tensorcore mode, 6 views at 512 px, 4096 px texture maps.
- **End-to-end: ~80 s per model** (shape ~45 s at 50 steps + texture ~35 s).
- Model load is excluded from these numbers (amortized across the batch).

**Context:** upstream Hunyuan3D-2.1 [documents 29 GB VRAM](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) for shape + texture generation; this pipeline runs both on a single 24 GB RTX 4090. For reference, a third-party RTX 4090 benchmark ([SaladCloud, Oct 2025](https://blog.salad.com/hunyuan3d-2-1/), ComfyUI custom node, settings not published) reported a 139 s median per generation. That is not a controlled comparison (different settings).

**Status:** a personal tool built for my own projects, shared for anyone who finds it useful. It is not regularly maintained; I update it when I return to it.

![10 sample models generated with the optimized pipeline](screenshots/grid.png)

*10 models generated end-to-end (image to textured PBR mesh) on an RTX 4090 24 GB. Texture stage averaged ~35 s per model at 10 denoising steps, INT8 tensorcore mode, 6 views at 512 px; shape stage ~45 s per model at 50 steps. Model load excluded.*

## What's Here

Wrapper scripts and custom CUDA kernels that optimize the Hunyuan3D-2.1 texture painting pipeline:

| Component | What it does |
|-----------|-------------|
| `texture.py` | Single-model texture generation with all optimizations |
| `texture_batch.py` | Batch texture generation (load pipeline once, texture many) |
| `generate_shape.py` | Image-to-mesh shape generation |
| `batch_demo.py` | Full end-to-end: shape gen + batch texture for N models |
| `quantize_utils.py` | INT8 quantization with custom Tensor Core GEMM kernels |
| `csrc/` | CUDA kernels: INT8 TC GEMM (wmma), 2:4 Sparse TC GEMM (mma.sp) |
| `generate.py` | One-shot CLI: image to mesh, optional `--texture` (uses the INT8 texture path) |
| `start_server.py` / `client.py` | Launch the upstream Hunyuan3D API server (keeps models resident, default `127.0.0.1:8081`, `--low-vram` option) and send image requests to it |
| `hy3d.bat` / `activate.bat` | Windows helpers: run `generate.py` with the local `venv`, or activate the `venv` |
| `render_screenshots.py` / `repack_glb.py` | Render preview thumbnails/grid; re-export OBJ outputs as GLB with embedded metallic-roughness maps |
| `tools/` | Development benchmarks, profilers and kernel tests (not needed to run the pipeline) |

## Quick Start

### Prerequisites

- **GPU**: NVIDIA RTX 4090 (24 GB VRAM) or similar Ada/Ampere GPU
- **CUDA Toolkit**: 12.6+
- **PyTorch**: 2.8+ with cu126
- **Python**: 3.11+
- [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) cloned as `./Hunyuan3D-2.1/`

### Installation

```bash
# Clone Hunyuan3D-2.1 into this directory
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git

# Install PyTorch 2.8+ with CUDA 12.6
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install Hunyuan3D dependencies
pip install -r Hunyuan3D-2.1/requirements.txt

# Additional dependencies
pip install pymeshlab fast_simplification timm
```

### Single Model (Texture Only)

```bash
# Texture an existing mesh with INT8 quantization (default)
python texture.py --mesh untextured.glb --image reference.png

# Fewer denoising steps for speed (10 vs default 15)
python texture.py --mesh mesh.glb --image photo.png --steps 10

# With torch.compile (slower first run, faster subsequent)
python texture.py --mesh mesh.glb --image photo.png --compile
```

### Full Pipeline (Image to Textured 3D)

```bash
# Generate shape + texture for one image
python generate_shape.py --image photo.png --output mesh.glb
python texture.py --mesh mesh.glb --image photo.png --steps 10
```

### Batch Mode (10 Models)

```bash
# Full pipeline: shape gen + texture for 10 built-in examples
python batch_demo.py --num 10 --texture-steps 10

# Texture only (reuse existing meshes)
python batch_demo.py --num 10 --skip-shape --texture-steps 10

# Custom images
python batch_demo.py --images img1.png img2.png img3.png
```

## Performance

Benchmarked on RTX 4090 24GB, PyTorch 2.8, CUDA 12.6, Windows 11.

Settings: texture 10 denoising steps (`--texture-steps 10`; upstream default 15), `--quantize-mode tensorcore`, 6 views at 512 px multiview resolution, 4096 px texture maps (views, resolution and texture size are the upstream defaults); shape 50 steps. Per-model times exclude model load, which is paid once per batch.

### Texture Generation (per model)

| Step | Time | Notes |
|------|------|-------|
| Remesh (pymeshlab) | 4-9s | Varies with mesh complexity |
| UV unwrap | 3-6s | CPU-bound |
| Multiview denoising | 9-17s | 10 steps, INT8 quantized UNet |
| RealESRGAN enhance | 4s | GPU, requires empty_cache() before |
| Texture baking | 2-3s | GPU |
| Inpainting | 7-8s | CPU (TELEA, parallelized albedo+MR) |
| Save + GLB convert | 1s | |
| **Total per model** | **~35s** | |
| Pipeline load (once) | ~14s | Amortized across batch |

### Shape Generation

| Metric | Value |
|--------|-------|
| Per model (50 steps) | ~45s |
| Model load (once) | ~30s |

### End-to-End (10 Models, Batch)

| Phase | Time |
|-------|------|
| Shape gen (10 models) | ~7.5 min |
| Texture pipeline load | ~14s |
| Texture gen (10 models) | ~5.5 min |
| **Total** | **~13 min** |

## Optimizations

### INT8 Tensor Core Quantization (`quantize_utils.py`)

Custom CUDA kernels replace all Linear layers in the multiview UNet and DINOv2:

- **Tensorcore mode**: Custom wmma INT8 GEMM kernel. Fuses GEMM + dequant + bias. ~1.0-1.3x FP16 speed. 50% less weight memory than FP16 (one INT8 N×K copy plus per-channel scales; measured with `tools/test_int8_memory.py`). Sparse mode cuts it by 69%.
- **Sparse mode**: 2:4 structured sparsity + Sparse Tensor Cores via PTX `mma.sp`, using [sparsemma](https://github.com/WizardsForgeGames/sparsemma)'s optimized kernel. 69% weight-memory reduction per layer vs FP16. Measured on an RTX 4090 (2026-10-20):
  - Per layer, GPU time at the pipeline's real shapes and token counts (`python tools/benchmark.py --hunyuan`): 1.1-2.9x faster than cuBLAS FP16 on 19 of the 20 sparse-mode shapes; 0.55x on the 1280->320 projection at M = 147,456. The sparse layers of one UNet step add up to 44 ms vs 73 ms in FP16 (83 ms with the previous kernel).
  - End to end (6 views, 512px, 10 steps, sample `004`, median of 3 runs): UNet step 0.682 -> 0.643 s and multiview stage 8.69 -> 8.29 s vs the previous kernel. Linear layers are a small part of each step (attention dominates), so the ~2x layer speedup shows up as ~5% of denoising time.
  - Numerics: `python tools/test_sparse_numerics.py` checks each shape against FP32 math on the pruned weights (rel. error <= 0.9%). Pruning itself (no fine-tuning) changes outputs; its quality cost is not measured.

```bash
# Tensorcore mode (default, recommended)
python texture.py --mesh mesh.glb --image photo.png --quantize-mode tensorcore

# Sparse mode (2:4 pruning without fine-tuning; quality cost not measured, check outputs)
python texture.py --mesh mesh.glb --image photo.png --quantize-mode sparse
```

### CUDA Allocator Fragmentation Fix

The pipeline sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and calls `torch.cuda.empty_cache()` before RealESRGAN to prevent allocator fragmentation that would otherwise cause 10-200x slowdowns in downstream operations.

### Parallel Inpainting

Albedo and metallic-roughness texture inpainting run concurrently via ThreadPoolExecutor (OpenCV releases the GIL). Uses TELEA (Fast Marching) instead of the slower Navier-Stokes method.

### Batch Pipeline Loading

`texture_batch.py` loads the ~2GB texture pipeline once and processes all meshes sequentially, saving ~14s per additional model vs individual `texture.py` calls.

### PyTorch 2.8 Workarounds

- Monkey-patches `torch.load` with `mmap=False` to work around intermittent `PytorchStreamReader` corruption
- Includes torchvision compatibility fix for removed `functional_tensor` module

## CUDA Kernels (`csrc/`)

| File | Purpose |
|------|---------|
| `int8_gemm_tc.cu` | Dense INT8 GEMM via wmma intrinsics (Tensor Cores) |
| `int8_sparse_tc.cu` | 2:4 Sparse INT8 GEMM via PTX mma.sp (Sparse Tensor Cores) |
| `int8_kernels.cu` | Activation quantization (fused per-row absmax + cast) |
| `build.py` | JIT compilation with ninja |

`int8_sparse_tc.cu` is [sparsemma](https://github.com/WizardsForgeGames/sparsemma)'s current kernel (synced at sparsemma `6ed9a75`); `int8_gemm_tc.cu` matches sparsemma's and `int8_kernels.cu` differs only in comments. The copies here are ASCII-only so torch's JIT build can read them under non-UTF-8 Windows code pages (e.g. cp932).

Kernels are JIT-compiled on first use via `torch.utils.cpp_extension`. Requires CUDA Toolkit 12.6+ and a C++ compiler (MSVC on Windows, gcc on Linux).

## Output Format

Each model produces:
- `{name}_mesh.glb` - Untextured mesh from shape generation
- `{name}_textured.glb` - Final textured mesh (GLB with embedded PBR textures)
- `{name}_textured.obj` - OBJ format with separate texture files
- `{name}_textured.jpg` - Albedo texture map
- `{name}_textured_metallic.jpg` - Metallic texture map
- `{name}_textured_roughness.jpg` - Roughness texture map

## Requirements

```
torch >= 2.8.0 (cu126)
torchvision >= 0.23.0
diffusers >= 0.30.0
trimesh
pymeshlab
fast_simplification
timm
Pillow
opencv-python
numpy
omegaconf
huggingface_hub
```

## Known Issues

**If a kernel fails to build**, `Int8Linear` falls back to a slower path and prints a boxed warning naming the kernel and the error (once per process). Set `INT8_KERNELS_STRICT=1` to make that an error instead. On Windows with a non-UTF-8 code page, older PyTorch versions can fail to read kernel sources; the sources are ASCII-only for that reason, and `PYTHONUTF8=1` also avoids it.

- **Intermittent CUDA slowdowns**: 1-2 out of 10 models may experience 2-5x slower denoising in batch mode. Re-running typically resolves it. Caused by PyTorch CUDA allocator behavior under sustained load.
- **PyTorch 2.8 torch.load corruption**: Intermittent `PytorchStreamReader` errors. Mitigated by `mmap=False` monkey-patch and retry logic.
- **Windows path separator**: Batch mode uses `|` as pair separator (not `:`) to avoid conflicts with Windows drive letters.

## License

The code in this repository (scripts and `csrc/` kernels) is released under the [MIT License](LICENSE), Copyright (c) 2026 Darren Sander.

The Hunyuan3D-2.1 model, weights and upstream code are not covered by that license. They remain under Tencent's [Hunyuan 3D 2.1 Community License](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE), which governs what you can do with the model and its outputs. Using these tools with the model is intended for research and non-commercial use, consistent with that license.

---

If you found this useful, you can [buy me a coffee](https://ko-fi.com/thewizardsforge).
