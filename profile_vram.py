"""
VRAM profiler for Hunyuan3D-2.1 texture pipeline.
Logs GPU memory at each stage to find where the 14-18 GB of activations live.

Usage:
    python profile_vram.py
"""

import os
import sys
import time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "Hunyuan3D-2.1")

# CUDA DLL directory
_cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME", "")
if _cuda_path:
    _cuda_bin = os.path.join(_cuda_path, "bin")
    if os.path.isdir(_cuda_bin) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_cuda_bin)

sys.path.insert(0, os.path.join(REPO_DIR, "hy3dpaint"))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "hy3dpaint", "custom_rasterizer"))

# Torchvision fix
try:
    from torchvision_fix import apply_fix
    apply_fix()
except (ImportError, Exception):
    pass


def mb(bytes_val):
    return f"{bytes_val / 1024**2:.0f} MB"


def gpu_stats():
    """Return current / peak / reserved VRAM."""
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return alloc, peak, reserved


def log_vram(label):
    alloc, peak, reserved = gpu_stats()
    print(f"  [{label:>40s}]  alloc={mb(alloc):>8s}  peak={mb(peak):>8s}  reserved={mb(reserved):>8s}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize-mode", default="none",
                        choices=["sparse", "tensorcore", "none"])
    parser.add_argument("--compile", action="store_true", help="torch.compile the UNet")
    parser.add_argument("--expandable-segments", action="store_true",
                        help="Enable CUDA expandable segments allocator")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Abort after N UNet calls (0 = full run)")
    args = parser.parse_args()

    if args.expandable_segments:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("Enabled expandable_segments allocator")

    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {mb(torch.cuda.get_device_properties(0).total_memory)}")
    print()

    torch.cuda.reset_peak_memory_stats()
    log_vram("baseline (empty)")

    # ── Load pipeline ──
    print("\n=== Loading pipeline ===")
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    conf = Hunyuan3DPaintConfig(6, 512)
    conf.realesrgan_ckpt_path = os.path.join(REPO_DIR, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = os.path.join(REPO_DIR, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(REPO_DIR, "hy3dpaint/hunyuanpaintpbr")

    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    log_vram("pipeline loaded (all models)")

    # ── Inspect individual model sizes ──
    print("\n=== Model VRAM breakdown ===")
    mv_model = paint_pipeline.models["multiview_model"]

    # UNet
    unet = mv_model.pipeline.unet
    unet_params = sum(p.numel() * p.element_size() for p in unet.parameters())
    unet_buffers = sum(b.numel() * b.element_size() for b in unet.buffers())
    print(f"  UNet params:   {mb(unet_params)}")
    print(f"  UNet buffers:  {mb(unet_buffers)}")

    # DINOv2
    if hasattr(mv_model, "dino_v2"):
        dino = mv_model.dino_v2
        dino_params = sum(p.numel() * p.element_size() for p in dino.parameters())
        dino_buffers = sum(b.numel() * b.element_size() for b in dino.buffers())
        print(f"  DINOv2 params: {mb(dino_params)}")
        print(f"  DINOv2 buffers:{mb(dino_buffers)}")

    # VAE
    vae = mv_model.pipeline.vae
    vae_params = sum(p.numel() * p.element_size() for p in vae.parameters())
    print(f"  VAE params:    {mb(vae_params)}")

    # Text encoder
    if hasattr(mv_model.pipeline, "text_encoder") and mv_model.pipeline.text_encoder is not None:
        te = mv_model.pipeline.text_encoder
        te_params = sum(p.numel() * p.element_size() for p in te.parameters())
        print(f"  TextEnc params:{mb(te_params)}")

    # Scheduler (no params)
    print(f"  Scheduler:     {type(mv_model.pipeline.scheduler).__name__}")

    log_vram("after model inspection")

    # ── Quantize ──
    if args.quantize_mode != "none":
        print(f"\n=== Quantizing ({args.quantize_mode}) ===")
        sys.path.insert(0, SCRIPT_DIR)
        torch.cuda.reset_peak_memory_stats()

        if args.quantize_mode == "sparse":
            from quantize_utils import quantize_model_sparse
            print("Quantizing UNet...")
            quantize_model_sparse(unet)
            log_vram("after UNet sparse quantization")

            if hasattr(mv_model, "dino_v2"):
                print("Quantizing DINOv2...")
                quantize_model_sparse(mv_model.dino_v2)
                log_vram("after DINOv2 sparse quantization")

        elif args.quantize_mode == "tensorcore":
            from quantize_utils import quantize_model_tensorcore
            print("Quantizing UNet...")
            quantize_model_tensorcore(unet)
            log_vram("after UNet TC quantization")

            if hasattr(mv_model, "dino_v2"):
                print("Quantizing DINOv2...")
                quantize_model_tensorcore(mv_model.dino_v2)
                log_vram("after DINOv2 TC quantization")

        torch.cuda.empty_cache()
        log_vram("after quantization + cache clear")

    # ── torch.compile ──
    if args.compile:
        print("\n=== Compiling UNet ===")
        t0 = time.time()
        mv_model.pipeline.unet = torch.compile(
            mv_model.pipeline.unet,
            mode="reduce-overhead",
        )
        print(f"  torch.compile() call took {time.time()-t0:.1f}s")
        log_vram("after torch.compile")

    # ── Run texture generation with per-step VRAM hooks ──
    print("\n=== Running texture generation ===")
    mesh_path = os.path.join(REPO_DIR, "hy3dpaint/assets/case_1/mesh.glb")
    image_path = os.path.join(REPO_DIR, "hy3dpaint/assets/case_1/image.png")
    output_path = os.path.join(SCRIPT_DIR, "output/vram_profile.obj")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.isfile(mesh_path) or not os.path.isfile(image_path):
        print(f"  Test assets not found, skipping generation")
        return

    # Hook into the UNet forward to log VRAM per call
    real_unet = mv_model.pipeline.unet
    step_count = [0]
    step_times = []

    class EarlyAbort(Exception):
        pass

    def unet_hook(module, input, output):
        step_count[0] += 1
        alloc, peak, reserved = gpu_stats()
        t = time.time()
        step_times.append(t)
        elapsed = f"{t - step_times[0]:.1f}s" if len(step_times) > 1 else "0.0s"
        # Log input batch size to see CFG batching
        if isinstance(input, tuple) and len(input) > 0:
            inp_shape = input[0].shape if hasattr(input[0], 'shape') else '?'
        else:
            inp_shape = '?'
        print(f"    unet call {step_count[0]:>3d}: alloc={mb(alloc):>8s}  peak={mb(peak):>8s}  elapsed={elapsed}  input={inp_shape}")
        if args.max_steps > 0 and step_count[0] >= args.max_steps:
            raise EarlyAbort(f"Stopped after {step_count[0]} UNet calls")
        return output

    hook_handle = real_unet.register_forward_hook(unet_hook)

    torch.cuda.reset_peak_memory_stats()
    log_vram("before generation")

    t_start = time.time()
    try:
        paint_pipeline(
            mesh_path=mesh_path,
            image_path=image_path,
            output_mesh_path=output_path,
        )
    except EarlyAbort as e:
        print(f"\n  Early abort: {e}")
    except Exception as e:
        print(f"\n  Generation failed: {e}")
        import traceback
        traceback.print_exc()

    t_total = time.time() - t_start
    alloc, peak, reserved = gpu_stats()
    print(f"\n=== Results ===")
    print(f"  Total time:       {t_total:.1f}s")
    print(f"  Peak VRAM:        {mb(peak)}")
    print(f"  Final alloc:      {mb(alloc)}")
    print(f"  Denoising steps:  {step_count[0]}")

    hook_handle.remove()


if __name__ == "__main__":
    main()
