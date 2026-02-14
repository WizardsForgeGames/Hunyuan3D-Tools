"""
A/B test: all-sparse vs mixed-sparse vs FP16 on UNet.
More iterations for stable timing.
"""
import gc
import sys
import time
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from quantize_utils import quantize_model_sparse, sp_available

def create_unet():
    from diffusers import UNet2DConditionModel
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=[320, 640, 1280, 1280],
        down_block_types=["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"],
        up_block_types=["UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"],
        cross_attention_dim=1024, attention_head_dim=[5, 10, 20, 20],
        use_linear_projection=True,
    )

def count_layers(model):
    from quantize_utils import Int8Linear
    sparse = sum(1 for m in model.modules() if isinstance(m, Int8Linear) and m.mode == "sparse")
    fp16 = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    return sparse, fp16

def bench(unet, label, n_warmup=5, n_iters=20):
    B = 6
    latent = torch.randn(B, 4, 64, 64, dtype=torch.float16, device="cuda")
    timestep = torch.tensor([500], device="cuda")
    enc = torch.randn(B, 77, 1024, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = unet(latent, timestep, encoder_hidden_states=enc)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = unet(latent, timestep, encoder_hidden_states=enc)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_iters * 1000

    vram = torch.cuda.memory_allocated() / 1e6
    nans = out.sample.isnan().sum().item()
    print(f"  {label:>20}: {ms:>7.1f} ms  |  VRAM: {vram:.0f} MB  |  NaN: {nans}")
    return ms

print(f"PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}  |  {torch.cuda.get_device_name(0)}")
print()

# FP16 baseline
torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
unet_fp16 = create_unet().half().cuda()
t_fp16 = bench(unet_fp16, "FP16")
del unet_fp16; torch.cuda.empty_cache(); gc.collect()

# All-sparse (sparse_min_k=0 forces all layers to sparse)
torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
unet_all = create_unet().half().cuda()
quantize_model_sparse(unet_all, sparse_min_k=0, verbose=False)
sp, fp = count_layers(unet_all)
print(f"  {'':>20}  [sparse={sp}, fp16={fp}]")
t_all = bench(unet_all, "All-sparse (k>=0)")
del unet_all; torch.cuda.empty_cache(); gc.collect()

# Mixed-sparse with sparse_min_k=1024
torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
unet_mix = create_unet().half().cuda()
quantize_model_sparse(unet_mix, sparse_min_k=1024, verbose=False)
sp, fp = count_layers(unet_mix)
print(f"  {'':>20}  [sparse={sp}, fp16={fp}]")
t_mix = bench(unet_mix, "Mixed (k>=1024)")
del unet_mix; torch.cuda.empty_cache(); gc.collect()

# Mixed-sparse with sparse_min_k=640
torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
unet_640 = create_unet().half().cuda()
quantize_model_sparse(unet_640, sparse_min_k=640, verbose=False)
sp, fp = count_layers(unet_640)
print(f"  {'':>20}  [sparse={sp}, fp16={fp}]")
t_640 = bench(unet_640, "Mixed (k>=640)")
del unet_640; torch.cuda.empty_cache(); gc.collect()

print(f"\n  Summary:")
print(f"    FP16:           {t_fp16:.1f} ms (baseline)")
print(f"    All-sparse:     {t_all:.1f} ms ({t_all/t_fp16:.2f}x FP16)")
print(f"    Mixed (k>=1024):{t_mix:.1f} ms ({t_mix/t_fp16:.2f}x FP16)")
print(f"    Mixed (k>=640): {t_640:.1f} ms ({t_640/t_fp16:.2f}x FP16)")
