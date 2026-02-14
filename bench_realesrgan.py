"""
Isolated RealESRGAN benchmark — measure actual upscale time per image.
"""
import os
import sys
import time
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "Hunyuan3D-2.1")

# CUDA DLL
_cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME", "")
if _cuda_path:
    _cuda_bin = os.path.join(_cuda_path, "bin")
    if os.path.isdir(_cuda_bin) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_cuda_bin)

# Apply torchvision compatibility fix before importing realesrgan
sys.path.insert(0, REPO_DIR)
from torchvision_fix import apply_fix
apply_fix()

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


def mb(b):
    return f"{b / 1024**2:.0f} MB"


def main():
    ckpt = os.path.join(REPO_DIR, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
    if not os.path.isfile(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Build model exactly as the pipeline does
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=ckpt,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None,
    )

    print(f"Model device: {upsampler.device}")
    print(f"Model dtype:  {next(upsampler.model.parameters()).dtype}")
    param_bytes = sum(p.numel() * p.element_size() for p in upsampler.model.parameters())
    print(f"Model size:   {mb(param_bytes)}")
    print()

    # Test at different input sizes
    for h, w in [(512, 512), (256, 256), (1024, 1024)]:
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Warm up
        print(f"--- Input: {h}x{w} ---")
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated()

        t0 = time.time()
        output, _ = upsampler.enhance(img)
        torch.cuda.synchronize()
        t_warmup = time.time() - t0
        print(f"  Warmup:  {t_warmup:.2f}s  output={output.shape}  peak_vram={mb(torch.cuda.max_memory_allocated())}")

        # Benchmark 3 images
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(3):
            t0 = time.time()
            output, _ = upsampler.enhance(img)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        avg = sum(times) / len(times)
        print(f"  3 runs:  {[f'{t:.2f}s' for t in times]}  avg={avg:.2f}s")
        print(f"  Peak VRAM: {mb(torch.cuda.max_memory_allocated())}")
        print()


if __name__ == "__main__":
    main()
