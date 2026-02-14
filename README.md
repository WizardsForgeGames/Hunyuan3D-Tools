# Hunyuan3D-2.1 Optimized Pipeline Tools

Performance-tuned tooling for [Hunyuan3D-2.1](https://github.com/Tencent/Hunyuan3D-2.1) on consumer GPUs. Full image-to-PBR-textured-3D pipeline in **~35s per model** on an RTX 4090 (texture only) or **~80s** end-to-end including shape generation.

![10 sample models generated with the optimized pipeline](screenshots/grid.png)

*10 models generated end-to-end (image to textured PBR mesh) averaging ~35s texture time per model on RTX 4090.*

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

## Quick Start

### Prerequisites

- **GPU**: NVIDIA RTX 4090 (24 GB VRAM) or similar Ada/Ampere GPU
- **CUDA Toolkit**: 12.6+
- **PyTorch**: 2.8+ with cu126
- **Python**: 3.11+
- [Hunyuan3D-2.1](https://github.com/Tencent/Hunyuan3D-2.1) cloned as `./Hunyuan3D-2.1/`

### Installation

```bash
# Clone Hunyuan3D-2.1 into this directory
git clone https://github.com/Tencent/Hunyuan3D-2.1.git

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

- **Tensorcore mode**: Custom wmma INT8 GEMM kernel. Fuses GEMM + dequant + bias. 50% weight VRAM savings, ~1.0-1.3x FP16 speed.
- **Sparse mode**: 2:4 structured sparsity + Sparse Tensor Cores via PTX `mma.sp`. 75% weight VRAM savings, 2x INT8 throughput on eligible layers.

```bash
# Tensorcore mode (default, recommended)
python texture.py --mesh mesh.glb --image photo.png --quantize-mode tensorcore

# Sparse mode (more aggressive, slight quality tradeoff)
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
| `sparse_gemm.cu` | Sparse weight preprocessing and compression |
| `build.py` | JIT compilation with ninja |

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

- **Intermittent CUDA slowdowns**: 1-2 out of 10 models may experience 2-5x slower denoising in batch mode. Re-running typically resolves it. Caused by PyTorch CUDA allocator behavior under sustained load.
- **PyTorch 2.8 torch.load corruption**: Intermittent `PytorchStreamReader` errors. Mitigated by `mmap=False` monkey-patch and retry logic.
- **Windows path separator**: Batch mode uses `|` as pair separator (not `:`) to avoid conflicts with Windows drive letters.

## License

These tools are provided for research and non-commercial use, consistent with the [Tencent Hunyuan Non-Commercial License](https://github.com/Tencent/Hunyuan3D-2.1/blob/main/LICENSE) of the underlying Hunyuan3D-2.1 model.
