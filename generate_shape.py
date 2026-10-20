"""
Hunyuan3D-2.1 Shape Generator — Generate untextured 3D mesh from an image.

Usage:
    python generate_shape.py --image photo.png --output mesh.glb
    python generate_shape.py --pairs "a.png|a_mesh.glb" "b.png|b_mesh.glb"   # one model load
"""

import argparse
import os
import sys
import time

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

sys.path.insert(0, os.path.join(REPO_DIR, "hy3dshape"))
sys.path.insert(0, REPO_DIR)


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from image")
    parser.add_argument("--image", "-i", help="Input image path")
    parser.add_argument("--output", "-o", help="Output mesh path (.glb)")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Batch mode: 'image|output.glb' pairs, generated with one model load")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps (default: 50)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the initial noise of each shape (default: unseeded)")
    args = parser.parse_args()

    if args.pairs:
        if args.image or args.output:
            parser.error("use either --pairs or --image/--output")
        jobs = [tuple(p.split("|", 1)) for p in args.pairs]
        if any(len(j) != 2 for j in jobs):
            parser.error("--pairs entries must be 'image|output'")
    elif args.image and args.output:
        jobs = [(args.image, args.output)]
    else:
        parser.error("provide --image and --output, or --pairs")

    for image_path, _ in jobs:
        if not os.path.isfile(image_path):
            print(f"Error: image not found: {image_path}")
            sys.exit(1)

    import torch
    from PIL import Image

    # Workaround: PyTorch 2.8 intermittent PytorchStreamReader corruption
    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        kwargs.setdefault("mmap", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    print(f"Steps:  {args.steps}")

    # Try background removal
    try:
        from hy3dshape.rembg import BackgroundRemover
        rembg = BackgroundRemover()
    except Exception:
        rembg = None

    # Load the model once for all jobs
    print("Loading shape model...")
    t_load = time.time()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    t_load = time.time() - t_load
    print(f"  Model loaded in {t_load:.1f}s")
    print(f"SHAPE_LOAD={t_load:.2f}")

    batch = args.pairs is not None
    failed = 0
    for idx, (image_path, output) in enumerate(jobs):
        print(f"[{idx+1}/{len(jobs)}] Image:  {image_path}")
        print(f"  Output: {output}")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            image = Image.open(image_path).convert("RGBA")
            if rembg and image.mode == "RGB":
                image = rembg(image)

            generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None

            print("  Generating shape...")
            t0 = time.time()
            mesh = pipeline(image=image, num_inference_steps=args.steps, generator=generator)[0]
            t_shape = time.time() - t0

            mesh.export(output)
            print(f"  Shape generated in {t_shape:.1f}s")
            print(f"  Saved to: {output}")
        except Exception as e:
            if not batch:
                raise
            failed += 1
            print(f"  FAILED: {e}")
            print(f"SHAPE_TIME:{idx}=FAIL", flush=True)
            continue

        # Machine-readable timing for batch_demo to parse
        if batch:
            print(f"SHAPE_TIME:{idx}={t_shape:.2f}", flush=True)
        else:
            print(f"SHAPE_TIME={t_shape:.2f}", flush=True)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
