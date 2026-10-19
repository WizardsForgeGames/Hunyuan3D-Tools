"""
Hunyuan3D-2.1 Texture Generator — Apply PBR textures to an untextured mesh.

Usage:
    python texture.py --mesh untextured.glb --image photo.png
    python texture.py --mesh mesh.glb --image photo.png --output textured.obj
"""

import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "Hunyuan3D-2.1")

# Add CUDA DLL directory for Windows (must happen before torch import)
_cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME", "")
if _cuda_path:
    _cuda_bin = os.path.join(_cuda_path, "bin")
    if os.path.isdir(_cuda_bin) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_cuda_bin)

# Add Hunyuan3D submodule paths
sys.path.insert(0, os.path.join(REPO_DIR, "hy3dpaint"))
sys.path.insert(0, REPO_DIR)
# custom_rasterizer_kernel.pyd lives in custom_rasterizer/ (needed by render.py)
sys.path.insert(0, os.path.join(REPO_DIR, "hy3dpaint", "custom_rasterizer"))


def main():
    parser = argparse.ArgumentParser(description="Apply PBR textures to a mesh")
    parser.add_argument("--mesh", "-m", required=True, help="Path to untextured mesh (.glb or .obj)")
    parser.add_argument("--image", "-i", required=True, help="Path to reference image")
    parser.add_argument("--output", "-o", default=None, help="Output path (.obj). Defaults to <mesh>_textured.obj")
    parser.add_argument("--views", type=int, default=6, choices=range(6, 10), help="Number of texture views (default: 6)")
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 768], help="Texture resolution (default: 512)")
    parser.add_argument("--quantize-mode", default="tensorcore",
                        choices=["sparse", "tensorcore", "bnb", "none"],
                        help="Int8 quantization backend (default: tensorcore). "
                             "'sparse' uses 2:4 Sparse INT8 Tensor Cores (~69%% weight memory reduction), "
                             "'tensorcore' uses custom Int8Linear (50%% weight VRAM savings), "
                             "'bnb' uses bitsandbytes, 'none' disables quantization.")
    parser.add_argument("--steps", type=int, default=15,
                        help="Denoising steps (default: 15, try 10 for speed)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the UNet for faster denoising (adds ~60s first-run warmup)")
    # Legacy alias
    parser.add_argument("--no-quantize", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.no_quantize:
        args.quantize_mode = "none"

    if not os.path.isfile(args.mesh):
        print(f"Error: mesh not found: {args.mesh}")
        sys.exit(1)
    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(args.mesh)[0]
        # Remove _untextured suffix if present
        if base.endswith("_untextured"):
            base = base[:-len("_untextured")]
        args.output = base + "_textured.obj"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Apply torchvision compatibility fix
    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except (ImportError, Exception):
        pass

    print(f"Mesh:       {args.mesh}")
    print(f"Image:      {args.image}")
    print(f"Output:     {args.output}")
    print(f"Views:      {args.views}  |  Resolution: {args.resolution}")
    print()

    t0 = time.time()

    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    conf = Hunyuan3DPaintConfig(args.views, args.resolution)
    conf.num_inference_steps = args.steps
    conf.realesrgan_ckpt_path = os.path.join(REPO_DIR, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = os.path.join(REPO_DIR, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(REPO_DIR, "hy3dpaint/hunyuanpaintpbr")

    print("Loading paint pipeline...")
    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    # Quantize the multiview UNet and DINOv2 to int8 (saves ~2-4 GB peak VRAM)
    if args.quantize_mode != "none":
        import torch
        sys.path.insert(0, SCRIPT_DIR)

        mv_model = paint_pipeline.models["multiview_model"]

        if args.quantize_mode == "sparse":
            from quantize_utils import quantize_model_sparse
            print("Quantizing multiview UNet (Sparse INT8 - 2:4 pruning)...")
            quantize_model_sparse(mv_model.pipeline.unet)
            if hasattr(mv_model, "dino_v2"):
                print("Quantizing DINOv2 (Sparse INT8 - 2:4 pruning)...")
                quantize_model_sparse(mv_model.dino_v2)
        elif args.quantize_mode == "tensorcore":
            from quantize_utils import quantize_model_tensorcore
            print("Quantizing multiview UNet (Int8Linear)...")
            quantize_model_tensorcore(mv_model.pipeline.unet)
            if hasattr(mv_model, "dino_v2"):
                print("Quantizing DINOv2 (Int8Linear)...")
                quantize_model_tensorcore(mv_model.dino_v2)
        else:  # bnb
            from quantize_utils import quantize_model_int8
            print("Quantizing multiview UNet (bitsandbytes)...")
            quantize_model_int8(mv_model.pipeline.unet)
            if hasattr(mv_model, "dino_v2"):
                print("Quantizing DINOv2 (bitsandbytes)...")
                quantize_model_int8(mv_model.dino_v2)

    # torch.compile the UNet for faster denoising via CUDA graphs
    if args.compile:
        import torch
        mv_model = paint_pipeline.models["multiview_model"]
        print("Compiling UNet (first run will be slower)...")
        mv_model.pipeline.unet = torch.compile(
            mv_model.pipeline.unet, mode="reduce-overhead"
        )

    print("Generating textures...")
    paint_pipeline(
        mesh_path=args.mesh,
        image_path=args.image,
        output_mesh_path=args.output,
    )

    tex_time = time.time() - t0
    print(f"  Textures generated in {tex_time:.1f}s")

    glb_path = args.output.replace(".obj", ".glb")
    if os.path.isfile(glb_path):
        print(f"\nDone! Textured mesh saved to: {glb_path}")
    else:
        print(f"\nDone! Textured mesh saved to: {args.output}")


if __name__ == "__main__":
    main()
