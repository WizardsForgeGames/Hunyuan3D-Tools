"""
Hunyuan3D-2.1 CLI — Generate 3D meshes from images.

Usage:
    python generate.py --image path/to/image.png
    python generate.py --image image.png --output model.glb --steps 50 --seed 42
    python generate.py --image image.png --texture  (requires CUDA toolkit)

The first run will download model weights from HuggingFace (~6GB).
"""

import argparse
import os
import sys
import time

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
sys.path.insert(0, os.path.join(REPO_DIR, "hy3dshape"))
sys.path.insert(0, os.path.join(REPO_DIR, "hy3dpaint"))
sys.path.insert(0, REPO_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D meshes from images using Hunyuan3D-2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Shape only (fast, ~10GB VRAM):
    python generate.py --image photo.png

  Shape with more detail:
    python generate.py --image photo.png --steps 50 --octree-resolution 384

  With PBR textures (needs CUDA toolkit compiled):
    python generate.py --image photo.png --texture

  Custom output path and seed:
    python generate.py --image photo.png -o my_model.glb --seed 42
""",
    )

    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to input image (PNG/JPG/WEBP). Background will be auto-removed.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (.glb or .obj). Defaults to <input_name>.glb in ./output/",
    )
    parser.add_argument(
        "--texture", "-t",
        action="store_true",
        help="Generate PBR textures (albedo, roughness, metallic). Requires compiled CUDA rasterizer.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion inference steps (default: 50). Fewer = faster, more = higher quality.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible generation (default: 1234).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale (default: 5.0). Higher = stronger adherence to image.",
    )
    parser.add_argument(
        "--octree-resolution",
        type=int,
        default=384,
        help="Mesh octree resolution (default: 384). Higher = more geometric detail, more VRAM.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=8000,
        help="Number of chunks for mesh decoding (default: 8000).",
    )
    parser.add_argument(
        "--face-count",
        type=int,
        default=40000,
        help="Target face count for mesh simplification during texturing (default: 40000).",
    )
    parser.add_argument(
        "--texture-views",
        type=int,
        default=6,
        choices=range(6, 10),
        help="Number of views for texture generation (default: 6, max: 9).",
    )
    parser.add_argument(
        "--texture-resolution",
        type=int,
        default=512,
        choices=[512, 768],
        help="Texture resolution (default: 512). 768 uses more VRAM.",
    )
    parser.add_argument(
        "--model-path",
        default="tencent/Hunyuan3D-2.1",
        help="HuggingFace model path or local directory (default: tencent/Hunyuan3D-2.1).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda).",
    )
    parser.add_argument(
        "--remove-floaters",
        action="store_true",
        default=True,
        help="Remove small disconnected mesh components (default: on).",
    )
    parser.add_argument(
        "--no-remove-floaters",
        action="store_false",
        dest="remove_floaters",
    )
    parser.add_argument(
        "--quantize-mode",
        default="tensorcore",
        choices=["tensorcore", "bnb", "none"],
        help="Int8 quantization for texture pipeline (default: tensorcore). "
             "'tensorcore' uses custom Int8Linear, 'bnb' uses bitsandbytes, 'none' disables.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy alias
    )

    args = parser.parse_args()
    if args.no_quantize:
        args.quantize_mode = "none"
    return args


def resolve_output_path(image_path, output_path, textured):
    """Determine the output file path."""
    if output_path:
        return output_path

    os.makedirs(os.path.join(SCRIPT_DIR, "output"), exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "_textured" if textured else ""
    return os.path.join(SCRIPT_DIR, "output", f"{base}{suffix}.glb")


def main():
    args = parse_args()

    # Validate input image exists
    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    output_path = resolve_output_path(args.image, args.output, args.texture)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Input:  {args.image}")
    print(f"Output: {output_path}")
    print(f"Mode:   {'Shape + PBR Texture' if args.texture else 'Shape only'}")
    print(f"Steps:  {args.steps}  |  Seed: {args.seed}  |  Guidance: {args.guidance_scale}")
    print(f"Octree: {args.octree_resolution}  |  Device: {args.device}")
    print()

    # Apply torchvision compatibility fix
    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except (ImportError, Exception):
        pass

    import torch
    from PIL import Image
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover

    # ── Load image and remove background ──
    print("Loading image and removing background...")
    t0 = time.time()
    image = Image.open(args.image).convert("RGBA")
    if image.mode == "RGB":
        rembg = BackgroundRemover()
        image = rembg(image)
    print(f"  Image prepared in {time.time() - t0:.1f}s")

    # ── Load shape pipeline ──
    print(f"Loading shape model from {args.model_path}...")
    t0 = time.time()

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)
    pipeline.enable_model_cpu_offload()
    pipeline.enable_flashvdm()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # ── Generate mesh ──
    print(f"Generating mesh ({args.steps} steps)...")
    t0 = time.time()
    mesh = pipeline(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        generator=generator,
    )[0]
    shape_time = time.time() - t0
    print(f"  Shape generated in {shape_time:.1f}s")
    print(f"  Faces: {len(mesh.faces)}  |  Vertices: {len(mesh.vertices)}")

    # ── Post-processing ──
    if args.remove_floaters:
        from hy3dshape import FloaterRemover
        print("Removing floaters...")
        remover = FloaterRemover()
        mesh = remover(mesh)

    if not args.texture:
        # Shape-only: export directly
        mesh.export(output_path)
        print(f"\nDone! Mesh saved to: {output_path}")
        print(f"Total time: {shape_time:.1f}s")
    else:
        # Texture generation — runs in a subprocess to avoid VRAM/memory conflicts
        print("\nStarting PBR texture generation...")
        t0 = time.time()

        # Save intermediate untextured mesh
        intermediate_path = output_path.replace(".glb", "_untextured.glb")
        mesh.export(intermediate_path)
        print(f"  Untextured mesh saved to: {intermediate_path}")

        # Free shape pipeline VRAM before texture subprocess needs the GPU
        import gc
        del mesh, pipeline, generator
        torch.cuda.empty_cache()
        gc.collect()
        print("  Shape pipeline released from VRAM.")

        # The paint pipeline works with OBJ internally, then converts to GLB
        obj_output = output_path.replace(".glb", ".obj")

        # Run texture generation in a separate process (clean VRAM)
        import subprocess
        texture_script = os.path.join(SCRIPT_DIR, "texture.py")
        python_exe = sys.executable
        cmd = [
            python_exe, texture_script,
            "--mesh", intermediate_path,
            "--image", args.image,
            "--output", obj_output,
            "--views", str(args.texture_views),
            "--resolution", str(args.texture_resolution),
            "--quantize-mode", args.quantize_mode,
        ]
        print(f"  Launching texture subprocess...")
        result = subprocess.run(cmd)

        tex_time = time.time() - t0
        if result.returncode != 0:
            print(f"\nError: Texture generation failed (exit code {result.returncode}).")
            print(f"  Untextured mesh saved to: {intermediate_path}")
            sys.exit(1)

        glb_path = obj_output.replace(".obj", ".glb")
        if os.path.isfile(glb_path):
            print(f"\nDone! Total time: {shape_time:.1f}s (shape) + {tex_time - shape_time:.0f}s (texture)")
        else:
            print(f"\nDone! Total time: {shape_time:.1f}s (shape) + {tex_time - shape_time:.0f}s (texture)")


if __name__ == "__main__":
    main()
