"""
Hunyuan3D-2.1 Shape Generator — Generate untextured 3D mesh from an image.

Usage:
    python generate_shape.py --image photo.png --output mesh.glb
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
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output mesh path (.glb)")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps (default: 50)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    import torch
    from PIL import Image

    # Workaround: PyTorch 2.8 intermittent PytorchStreamReader corruption
    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        kwargs.setdefault("mmap", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    print(f"Image:  {args.image}")
    print(f"Output: {args.output}")
    print(f"Steps:  {args.steps}")

    # Try background removal
    try:
        from hy3dshape.rembg import BackgroundRemover
        rembg = BackgroundRemover()
    except Exception:
        rembg = None

    print("Loading shape model...")
    t_load = time.time()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    print(f"  Model loaded in {time.time() - t_load:.1f}s")

    image = Image.open(args.image).convert("RGBA")
    if rembg and image.mode == "RGB":
        image = rembg(image)

    print("Generating shape...")
    t0 = time.time()
    mesh = pipeline(image=image, num_inference_steps=args.steps)[0]
    t_shape = time.time() - t0

    mesh.export(args.output)
    print(f"  Shape generated in {t_shape:.1f}s")
    print(f"  Saved to: {args.output}")

    # Print machine-readable timing for batch_demo to parse
    print(f"SHAPE_TIME={t_shape:.2f}")


if __name__ == "__main__":
    main()
