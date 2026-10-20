"""
Batch texture generation — load the pipeline once, texture multiple meshes.
Avoids the ~20s model load overhead per model.

Usage:
    python texture_batch.py --pairs "mesh1.glb|img1.png" "mesh2.glb|img2.png"
    python texture_batch.py --mesh-dir output/samples/ --image-dir Hunyuan3D-2.1/assets/example_images/
"""

import argparse
import os
import sys
import time

# Prevent CUDA allocator fragmentation that causes random 10x slowdowns
# during batch processing. expandable_segments uses virtual memory instead
# of fixed-size blocks, avoiding the splitting/coalescing pathology.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "Hunyuan3D-2.1")

# Run at High QoS so Windows keeps the CPU stages on the P-cores (see win_qos.py)
from win_qos import request_high_qos
request_high_qos()

# CUDA DLL directory (Windows)
_cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME", "")
if _cuda_path:
    _cuda_bin = os.path.join(_cuda_path, "bin")
    if os.path.isdir(_cuda_bin) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_cuda_bin)

sys.path.insert(0, os.path.join(REPO_DIR, "hy3dpaint"))
sys.path.insert(0, REPO_DIR)
# custom_rasterizer: prefer the copy pip-installed into this environment (built against its
# torch); an in-place build in custom_rasterizer/ is only a fallback, since a .pyd built
# for another torch version fails to load.
sys.path.append(os.path.join(REPO_DIR, "hy3dpaint", "custom_rasterizer"))


def main():
    parser = argparse.ArgumentParser(description="Batch texture generation")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="mesh|image pairs (e.g. \"mesh.glb|photo.png\" - quote it; from Git Bash set MSYS_NO_PATHCONV=1)")
    parser.add_argument("--mesh-dir", default=None,
                        help="Directory of *_mesh.glb files to texture")
    parser.add_argument("--image-dir", default=None,
                        help="Directory of matching images (same stem as mesh)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as mesh)")
    parser.add_argument("--steps", type=int, default=15,
                        help="Denoising steps (default: 15)")
    parser.add_argument("--quantize-mode", default="tensorcore",
                        choices=["sparse", "tensorcore", "bnb", "none"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Remesh + UV-unwrap inside the pipeline instead of in a background worker")
    args = parser.parse_args()

    # Apply torchvision compatibility fix
    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except (ImportError, Exception):
        pass

    # Workaround: PyTorch 2.8 intermittent PytorchStreamReader corruption.
    # Disabling mmap in torch.load avoids the corrupted memory-mapped reads.
    import torch
    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        kwargs.setdefault("mmap", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load

    # Resolve pairs
    pairs = []  # [(mesh_path, image_path, output_path), ...]

    if args.pairs:
        for pair in args.pairs:
            mesh, image = pair.split("|")
            name = os.path.splitext(os.path.basename(mesh))[0].replace("_mesh", "")
            out_dir = args.output_dir or os.path.dirname(mesh)
            output = os.path.join(out_dir, f"{name}_textured.obj")
            pairs.append((mesh, image, output))
    elif args.mesh_dir:
        import glob
        meshes = sorted(glob.glob(os.path.join(args.mesh_dir, "*_mesh.glb")))
        for mesh in meshes:
            name = os.path.basename(mesh).replace("_mesh.glb", "")
            # Find matching image
            image = None
            if args.image_dir:
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = os.path.join(args.image_dir, name + ext)
                    if os.path.isfile(candidate):
                        image = candidate
                        break
            if image is None:
                print(f"Warning: no image found for {name}, skipping")
                continue
            out_dir = args.output_dir or args.mesh_dir
            output = os.path.join(out_dir, f"{name}_textured.obj")
            pairs.append((mesh, image, output))
    else:
        parser.error("Provide --pairs or --mesh-dir")

    if not pairs:
        print("No valid mesh/image pairs found")
        sys.exit(1)

    import torch
    n = len(pairs)
    print(f"Batch texture: {n} models, {args.steps} steps, quantize={args.quantize_mode}")
    print()

    # Remesh + UV-unwrap every mesh in a background process (CPU), in order, while the
    # pipeline loads and earlier meshes are textured on the GPU. Same functions as the
    # pipeline runs itself, so the result is identical (see mesh_prep.py).
    prefetcher = None
    prepared = [None] * n
    if not args.no_prefetch:
        from mesh_prep import MeshPrefetcher
        prefetcher = MeshPrefetcher()
        prepared = [prefetcher.submit(mesh) for mesh, _, _ in pairs]

    # Load pipeline ONCE
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    conf = Hunyuan3DPaintConfig(args.views, args.resolution)
    conf.num_inference_steps = args.steps
    conf.realesrgan_ckpt_path = os.path.join(REPO_DIR, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = os.path.join(REPO_DIR, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(REPO_DIR, "hy3dpaint/hunyuanpaintpbr")

    print("Loading texture pipeline...")
    t_load = time.time()
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    t_load = time.time() - t_load
    print(f"  Pipeline loaded in {t_load:.1f}s")

    # Quantize
    if args.quantize_mode != "none":
        sys.path.insert(0, SCRIPT_DIR)
        mv_model = paint_pipeline.models["multiview_model"]
        if args.quantize_mode == "sparse":
            from quantize_utils import quantize_model_sparse
            quantize_model_sparse(mv_model.pipeline.unet)
            if hasattr(mv_model, "dino_v2"):
                quantize_model_sparse(mv_model.dino_v2)
        elif args.quantize_mode == "tensorcore":
            from quantize_utils import quantize_model_tensorcore
            quantize_model_tensorcore(mv_model.pipeline.unet)
            if hasattr(mv_model, "dino_v2"):
                quantize_model_tensorcore(mv_model.dino_v2)

    # torch.compile
    if args.compile:
        mv_model = paint_pipeline.models["multiview_model"]
        print("Compiling UNet...")
        mv_model.pipeline.unet = torch.compile(mv_model.pipeline.unet, mode="reduce-overhead")

    print()

    # Process all meshes
    times = []
    for idx, (mesh, image, output) in enumerate(pairs):
        name = os.path.splitext(os.path.basename(mesh))[0].replace("_mesh", "")
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

        # Clear allocator fragmentation between models
        torch.cuda.empty_cache()

        print(f"[{idx+1}/{n}] {name}")
        t0 = time.time()
        try:
            mesh_obj = None
            if prepared[idx] is not None:
                try:
                    mesh_obj = prepared[idx].result()  # waiting time counts toward this model
                except Exception as e:
                    print(f"  (background remesh/UV failed: {e}; redoing it in the pipeline)")
            paint_pipeline(
                mesh_path=mesh,
                image_path=image,
                output_mesh_path=output,
                mesh=mesh_obj,
            )
            t = time.time() - t0
            times.append(t)
            glb = output.replace(".obj", ".glb")
            print(f"  -> {t:.1f}s  {glb}")
        except Exception as e:
            t = time.time() - t0
            times.append(-1)
            print(f"  -> FAILED ({t:.1f}s): {e}")
        print()

    if prefetcher is not None:
        prefetcher.close()

    # Summary
    valid = [t for t in times if t > 0]
    print("=" * 50)
    print(f"  Completed: {len(valid)}/{n}")
    if valid:
        print(f"  Average: {sum(valid)/len(valid):.1f}s per model")
        print(f"  Total generation: {sum(valid):.1f}s")
        print(f"  + one-time load: {t_load:.1f}s")

    # Print machine-readable results
    for idx, (mesh, image, output) in enumerate(pairs):
        name = os.path.splitext(os.path.basename(mesh))[0].replace("_mesh", "")
        t = times[idx]
        tag = f"{t:.2f}" if t > 0 else "FAIL"
        print(f"TEXTURE_TIME:{name}={tag}")


if __name__ == "__main__":
    main()
