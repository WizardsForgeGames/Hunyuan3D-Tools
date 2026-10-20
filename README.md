# Hunyuan3D-2.1 Optimized Pipeline Tools

Performance-tuned tooling for [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) on a single 24 GB consumer GPU. On an RTX 4090 24 GB (measured 2026-10-20, PyTorch 2.8.0+cu126, 5 sample images):

- **Texture stage: ~23 s per model** (22-24 s) with 10 denoising steps (upstream default is 15), INT8 tensorcore mode, 6 views at 512 px, 4096 px texture maps. Remesh and UV unwrap run on a CPU worker while the GPU textures the previous mesh.
- **End-to-end: ~63 s per model** (shape ~40 s at 50 steps + texture ~23 s), vs ~80 s in February 2026.
- Model load is excluded from these numbers; `batch_demo.py` loads the shape and texture models once per batch each. A 5-model batch takes {TOTAL} wall time (see [Performance](#performance)).
- Earlier the same day the February figures did not reproduce (~48 s texture, ~92 s end-to-end): Windows was running the CPU stages on the E-cores. The scripts now opt out of that; see [Performance](#performance).

**Context:** upstream Hunyuan3D-2.1 [documents 29 GB VRAM](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) for shape + texture generation; this pipeline runs both on a single 24 GB RTX 4090 (measured peaks: 8.5 GiB reserved for shape, 19.0 GiB reserved / 11.0 GiB allocated for texture, in separate processes; see [GPU memory](#gpu-memory)). For reference, a third-party RTX 4090 benchmark ([SaladCloud, Oct 2025](https://blog.salad.com/hunyuan3d-2-1/), ComfyUI custom node, settings not published) reported a 139 s median per generation. That is not a controlled comparison (different settings).

**Status:** a personal tool built for my own projects, shared for anyone who finds it useful. It is not regularly maintained; I update it when I return to it.

![10 sample models generated with the optimized pipeline](screenshots/grid.png)

*10 models generated end-to-end (image to textured PBR mesh) on an RTX 4090 24 GB in February 2026. That run's texture stage averaged ~35 s per model at 10 denoising steps, INT8 tensorcore mode, 6 views at 512 px; shape stage ~45 s per model at 50 steps; model load excluded. The 2026-10-20 measurement is in [Performance](#performance).*

## What's Here

Wrapper scripts and custom CUDA kernels that optimize the Hunyuan3D-2.1 texture painting pipeline:

| Component | What it does |
|-----------|-------------|
| `texture.py` | Single-model texture generation with all optimizations |
| `texture_batch.py` | Batch texture generation (load pipeline once, texture many) |
| `mesh_prep.py` | Remesh + UV unwrap in a background worker process, ahead of the texture pipeline |
| `win_qos.py` | Opts the process out of Windows power throttling so CPU stages run on P-cores |
| `generate_shape.py` | Image-to-mesh shape generation |
| `batch_demo.py` | Full end-to-end: shape gen + batch texture for N models |
| `quantize_utils.py` | INT8 quantization with custom Tensor Core GEMM kernels |
| `csrc/` | CUDA kernels: INT8 TC GEMM (wmma), 2:4 Sparse TC GEMM (mma.sp) |
| `generate.py` | One-shot CLI: image to mesh, optional `--texture` (uses the INT8 texture path) |
| `start_server.py` / `client.py` | Launch the upstream Hunyuan3D API server (keeps models resident, default `127.0.0.1:8081`, `--low-vram` option) and send image requests to it |
| `hy3d.bat` / `activate.bat` | Windows helpers: run `generate.py` with the local `venv`, or activate the `venv` (they use `venv`, not `venv-cu126`) |
| `requirements-cu126.txt` | Pinned dependencies for the tested PyTorch 2.8.0 + cu126 environment |
| `patches/hunyuan3d-2.1-windows.patch` | Changes to the Hunyuan3D-2.1 checkout that these scripts depend on (see Installation) |
| `render_screenshots.py` / `repack_glb.py` | Render preview thumbnails/grid; re-export OBJ outputs as GLB with embedded metallic-roughness maps |
| `tools/` | Development benchmarks, profilers and kernel tests (not needed to run the pipeline) |

## Quick Start

### Prerequisites

- **GPU**: NVIDIA RTX 4090 (24 GB VRAM) or similar Ada/Ampere GPU
- **PyTorch**: 2.8.0 + cu126 (tested). The upstream pins with 2.5.1 + cu124 also work and measured the same speed.
- **CUDA Toolkit + C++ compiler**: needed to build `custom_rasterizer` and the INT8 kernels (MSVC on Windows, gcc on Linux). Tested with the CUDA 12.4 toolkit against the cu126 wheels: PyTorch warns about the minor-version mismatch (12.4 vs 12.6), and the builds and kernel tests pass.
- **Python**: 3.11 (tested 3.11.3)
- [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) cloned as `./Hunyuan3D-2.1/` at commit `82920d6`, with `patches/hunyuan3d-2.1-windows.patch` applied

Tested 2026-10-20 on Windows 11, RTX 4090, driver 617.14.

### Installation

```bash
# 1. Clone Hunyuan3D-2.1 into this directory and apply this repo's patch
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1
git checkout 82920d6
git apply ../patches/hunyuan3d-2.1-windows.patch
mkdir -p hy3dpaint/ckpt
curl -L -o hy3dpaint/ckpt/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
cd ..

# 2. A new virtual environment (Windows; on Linux use venv-cu126/bin/activate)
python -m venv venv-cu126
source venv-cu126/Scripts/activate      # cmd.exe: venv-cu126\Scripts\activate
python -m pip install --upgrade pip wheel

# 3. PyTorch 2.8.0 + CUDA 12.6
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

# 4. Pipeline dependencies, pinned
pip install -r requirements-cu126.txt

# 5. Native extensions, compiled against the torch installed above
#    (--no-build-isolation so the build sees that torch)
pip install --no-build-isolation ./Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
pip install --no-build-isolation ./Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer

# 6. Optional: build and check the INT8 kernels now instead of on first use
INT8_KERNELS_STRICT=1 python tools/test_int8_memory.py
INT8_KERNELS_STRICT=1 python tools/test_sparse_numerics.py
```

The Hunyuan3D-2.1 and DINOv2 weights download from Hugging Face on first run.

What the patch changes in Hunyuan3D-2.1 (the scripts here depend on it; without it, for example, `--steps` is silently ignored):

- `custom_rasterizer`: `long` -> `int64_t` (Windows `long` is 32-bit) and `--allow-unsupported-compiler`, so it builds with MSVC.
- `DifferentiableRenderer`: a `setup.py` for `mesh_inpaint_processor` (upstream ships only a Linux shell script), a shim module for it, ASCII-only comments in its C++ source, and TELEA inpainting.
- Texture pipeline: configurable denoising steps, per-stage timing output, DINOv2 moved to CPU after use, `empty_cache()` before RealESRGAN, albedo and metallic-roughness inpainting in parallel, pymeshlab-only remesh, trimesh fallback for OBJ -> GLB when Blender's `bpy` is not installed.

`texture.py` and `texture_batch.py` import the `custom_rasterizer` installed in the active environment first. A `custom_rasterizer_kernel` `.pyd` built in place under `Hunyuan3D-2.1/hy3dpaint/custom_rasterizer/` for a different torch version fails to load ("The specified procedure could not be found"), so it is used only if nothing is installed.

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

**Measured 2026-10-20.** RTX 4090 24 GB (driver 617.14), i9-13900K (8 P + 16 E cores), Windows 11, Python 3.11.3, `venv-cu126`: PyTorch 2.8.0+cu126 / torchvision 0.23.0+cu126, packages from `requirements-cu126.txt`, kernels built with the CUDA 12.4 toolkit. Desktop applications were open (2.2-4.6 GB of VRAM in use before the runs). On this hybrid CPU, Windows 11 schedules a process started without a foreground window (background shell, IDE task) on the E-cores, which made the CPU stages 1.3-1.6x slower; every entry script now opts out of that power throttling for its own process (`win_qos.py`), and that was the gap to the February numbers.

Settings: texture 10 denoising steps (`--texture-steps 10`; upstream default 15), `--quantize-mode tensorcore`, 6 views at 512 px multiview resolution, 4096 px texture maps (views, resolution and texture size are the upstream defaults); shape 50 steps. Inputs: the first 5 `batch_demo.py` sample images (`ainimal_9`, `ainimal_14`, `004`, `073`, `202`). Per-stage texture times pool two `texture_batch.py` runs per mode (10 textures each) on the meshes of one batch; per-model times exclude model load.

### Texture Generation (per model)

| Step | Feb 2026 | 2026-10-20, median (range) |
|------|----------|----------------------------|
| Remesh (pymeshlab, CPU) | 4-9 s | 5.0 s (3.2-7.0), on a background worker: overlapped with the GPU stages |
| UV unwrap (xatlas, CPU) | 3-6 s | 4.0 s (2.9-4.5), on the same worker: overlapped |
| Load mesh, select views, render normal/position maps | not listed | 0.9 s |
| Multiview denoising (10 steps, INT8 UNet) | 9-17 s | 8.5 s (8.5-8.6) |
| RealESRGAN enhance (GPU) | 4 s | 3.7 s |
| Texture baking (GPU) | 2-3 s | 1.7 s |
| Inpainting (CPU, TELEA, albedo and MR in parallel) | 7-8 s | 7.1 s (6.8-8.9) |
| Save + GLB convert | 1 s | 0.5 s |
| **Total per model** | **~35 s** | **23.0 s mean (22.2-24.2)**; 32.1 s (30.2-34.6) with `--no-prefetch` |
| Pipeline load (once per batch) | ~14 s | 12.7-14.1 s |

With the prefetch worker no model waits for its mesh, the first one included (its remesh runs during the pipeline load). Outputs are identical with and without prefetch (geometry, UVs and all texture images). The first texture run in a fresh environment also compiles the INT8 kernels (about 40 s, once; cached under `torch_extensions`).

### Shape Generation

| Metric | Feb 2026 | 2026-10-20 |
|--------|----------|------------|
| Per model (50 steps) | ~45 s | 39.7 s (39.6-39.9) |
| Model load | ~30 s | 23.6-34.8 s, once per batch |

### End-to-End (5 Models, Batch)

`python batch_demo.py --num 5 --texture-steps 10`:

| Phase | 2026-10-20 |
|-------|------------|
| Shape phase (one process: start-up, one model load, 5 x ~39.7 s) | {SHAPE} |
| Texture phase (process start, pipeline load ~13 s, quantization, 5 x ~23 s) | {TEX} |
| **Total wall time** | **{TOTAL}** |
| **Per model, generation only (shape + texture, model load excluded)** | **~63 s** (Feb 2026: ~80 s) |

The February table gave ~7.5 min for 10 shapes, which is 10 x 45 s of generation only; at the time `batch_demo.py` also reloaded the shape model in a new process for every image, which the table did not count.

### GPU Memory

Peaks per stage from `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` (reset at each stage boundary; GiB), from 5-model batch runs (re-checked with mesh prefetch and the single shape process: unchanged):

| Stage | Peak allocated | Peak reserved |
|-------|----------------|---------------|
| Shape generation (its own process) | 7.7 GiB | 8.5 GiB |
| Texture: models resident between stages (after INT8 quantization, DINOv2 on CPU) | 4.9 GiB | - |
| Texture: multiview denoising, tensorcore | 11.0 GiB | 19.0 GiB |
| Texture: multiview denoising, sparse | 10.7 GiB | 17.9 GiB |
| Texture: baking | 8.5 GiB | 9.9 GiB |
| Whole GPU (`nvidia-smi`, including ~2.2-4.6 GB used by the desktop) | 21.4 GiB max | |

How it fits in 24 GB: shape and texture never share a process, so their peaks do not add; the UNet and DINOv2 Linear weights are INT8; DINOv2 moves to the CPU after it runs. During denoising the allocator caches up to 19 GiB while about 11 GiB is in use; `empty_cache()` before RealESRGAN returns it.

### History: the earlier 2026-10-20 measurement

The same day, before the power-throttling opt-out, the single shape-model load and mesh prefetch, the same 5 images measured: texture 48.2 s mean (44.6-53.3; remesh 8.2 s, UV 5.2 s, inpainting 12.6 s, denoising 9.9 s), pipeline load 20-25 s, shape 44.0 s, ~92 s per model generation only, and a 973 s batch (626 s of it the shape phase, with one process and model load per image). Sparse mode measured 46.1 s per texture in that setup. A second environment with torch 2.5.1+cu124 and the upstream Hunyuan3D-2.1 pins (diffusers 0.30.0, transformers 4.46.0) measured the same (shape 42.3 s, texture 48.9 s), so the PyTorch version was not the cause, and the library versions in the February environment (pymeshlab 2025.7, opencv 4.13) were checked directly and did not explain it either. The cause was the E-core scheduling described above: two concurrent 4096 px `cv2.inpaint` calls took 7.1-7.4 s pinned to the P-cores, 11.8-12.6 s pinned to the E-cores, and 7.4-7.8 s with the opt-out.

## Optimizations

### INT8 Tensor Core Quantization (`quantize_utils.py`)

Custom CUDA kernels replace all Linear layers in the multiview UNet and DINOv2:

- **Tensorcore mode**: Custom wmma INT8 GEMM kernel. Fuses GEMM + dequant + bias. ~1.0-1.3x FP16 speed. 50% less weight memory than FP16 (one INT8 N×K copy plus per-channel scales; measured with `tools/test_int8_memory.py`). Sparse mode cuts it by 69%.
- **Sparse mode**: 2:4 structured sparsity + Sparse Tensor Cores via PTX `mma.sp`, using [sparsemma](https://github.com/WizardsForgeGames/sparsemma)'s optimized kernel. 69% weight-memory reduction per layer vs FP16. Measured on an RTX 4090 (2026-10-20):
  - Per layer, GPU time at the pipeline's real shapes and token counts (`python tools/benchmark.py --hunyuan`): 1.2-2.8x faster than cuBLAS FP16 on all 17 UNet sparse-mode shapes, 0.99-1.10x on the three DINOv2 shapes (M = 257). The 1280->320 projection at M = 147,456 (0.55x FP16 before) now fuses the activation quantize into the GEMM: 1.53 -> 0.69 ms, 1.22x faster than FP16. The sparse layers of one UNet step add up to 40 ms vs 75 ms in FP16 (45 ms before the fused path, 83 ms with the original kernel).
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

`texture_batch.py` loads the ~2GB texture pipeline once and processes all meshes sequentially, saving the pipeline load (12.7-14.1 s measured 2026-10-20; ~14 s in February) plus quantization for each additional model vs individual `texture.py` calls. `batch_demo.py` likewise generates all shapes in one process after a single shape-model load (`generate_shape.py --pairs "image|output" ...`).

### Mesh Prefetch (`mesh_prep.py`)

Remeshing and UV unwrapping (~9 s of CPU work per model) depend only on the input mesh. A worker process runs the same upstream functions ahead of time: `texture.py` overlaps them with the pipeline load, and `texture_batch.py` prepares mesh N+1 while mesh N is textured on the GPU. The pipeline takes the prepared mesh via `mesh=` and skips both steps. 32.1 -> 23.0 s per model, identical output; `--no-prefetch` restores the in-pipeline path.

### Windows Power Throttling Opt-out (`win_qos.py`)

Each entry script calls `SetProcessInformation(ProcessPowerThrottling)` for its own process (no system or power-plan change; no-op off Windows), so Windows 11 does not move it to the E-cores when it was started without a foreground window. Texture time per model: 47.0 -> 31.7 s mean before prefetch, with byte-identical outputs.

### Load Workarounds

- Monkey-patches `torch.load` with `mmap=False`, and `batch_demo.py` retries the texture process, to work around intermittent `PytorchStreamReader` errors (see Known Issues)
- Includes torchvision compatibility fix for removed `functional_tensor` module (torchvision 0.17+)

## CUDA Kernels (`csrc/`)

| File | Purpose |
|------|---------|
| `int8_gemm_tc.cu` | Dense INT8 GEMM via wmma intrinsics (Tensor Cores) |
| `int8_sparse_tc.cu` | 2:4 Sparse INT8 GEMM via PTX mma.sp (Sparse Tensor Cores) |
| `int8_kernels.cu` | Activation quantization (fused per-row absmax + cast) |
| `build.py` | JIT compilation with ninja |

`int8_sparse_tc.cu` is [sparsemma](https://github.com/WizardsForgeGames/sparsemma)'s current kernel (synced at sparsemma `6ed9a75`); `int8_gemm_tc.cu` matches sparsemma's and `int8_kernels.cu` differs only in comments. The copies here are ASCII-only so torch's JIT build can read them under non-UTF-8 Windows code pages (e.g. cp932).

Kernels are JIT-compiled on first use via `torch.utils.cpp_extension`. Requires a CUDA 12.x toolkit and a C++ compiler (MSVC on Windows, gcc on Linux); tested with the 12.4 toolkit and PyTorch 2.8.0+cu126.

## Output Format

Each model produces:
- `{name}_mesh.glb` - Untextured mesh from shape generation
- `{name}_textured.glb` - Final textured mesh (GLB with embedded PBR textures)
- `{name}_textured.obj` - OBJ format with separate texture files
- `{name}_textured.jpg` - Albedo texture map
- `{name}_textured_metallic.jpg` - Metallic texture map
- `{name}_textured_roughness.jpg` - Roughness texture map

## Requirements

The tested set is pinned in `requirements-cu126.txt` (install torch 2.8.0 / torchvision 0.23.0 from the cu126 index first; see Installation). The main pins: diffusers 0.30.0, transformers 4.46.0, accelerate 1.1.1, huggingface-hub 0.30.2, numpy 2.2.6, opencv-python 4.10.0.84, pymeshlab 2022.2.post3, trimesh 4.4.7, xatlas 0.0.9, realesrgan 0.3.0, timm 1.0.24. `custom_rasterizer` and `mesh_inpaint_processor` are built from the patched Hunyuan3D-2.1 checkout.

## Known Issues

**If a kernel fails to build**, `Int8Linear` falls back to a slower path and prints a boxed warning naming the kernel and the error (once per process). Set `INT8_KERNELS_STRICT=1` to make that an error instead. On Windows with a non-UTF-8 code page, older PyTorch versions can fail to read kernel sources; the sources are ASCII-only for that reason, and `PYTHONUTF8=1` also avoids it.

- **Intermittent CUDA slowdowns**: 1-2 out of 10 models may experience 2-5x slower denoising in batch mode. Re-running typically resolves it. Caused by PyTorch CUDA allocator behavior under sustained load.
- **torch.load `PytorchStreamReader` errors**: intermittent, while the texture pipeline loads. Not specific to PyTorch 2.8: on 2026-10-20 it hit 2 of 4 batch runs, one on 2.5.1 and one on 2.8.0, and the retry recovered both. Loading the RealESRGAN checkpoint alone in 20 fresh processes did not reproduce it. Mitigated by the `mmap=False` monkey-patch and retry logic.
- **Windows path separator**: Batch mode uses `|` as pair separator (not `:`) to avoid conflicts with Windows drive letters.

## License

The code in this repository (scripts and `csrc/` kernels) is released under the [MIT License](LICENSE), Copyright (c) 2026 Darren Sander.

The Hunyuan3D-2.1 model, weights and upstream code are not covered by that license. They remain under Tencent's [Hunyuan 3D 2.1 Community License](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE), which governs what you can do with the model and its outputs. Using these tools with the model is intended for research and non-commercial use, consistent with that license.

---

If you found this useful, you can [buy me a coffee](https://ko-fi.com/thewizardsforge).
