"""
Timing breakdown: how much time each step takes in the streaming path.
Patches engine internals to measure encode/decode/unet separately.
"""
import sys, os
sys.path.insert(0, r"H:\RTDiffusion")
os.chdir(r"H:\RTDiffusion")

# Patch before import
import time as _time

# Load env
from backend.app.config import load_local_env
load_local_env()

import torch
from PIL import Image
import numpy as np

print("Loading pipeline...")
t0 = _time.perf_counter()
from diffusers import StableDiffusionXLInpaintPipeline, LCMScheduler, AutoencoderTiny
print(f"  imports: {(_time.perf_counter()-t0)*1000:.0f}ms")

model_path = r"D:\comfyui\comfy\models\checkpoints\cyberrealisticXL_v80-inpainting.safetensors"
device = "cuda"
dtype = torch.float16

t0 = _time.perf_counter()
pipe = StableDiffusionXLInpaintPipeline.from_single_file(
    model_path, torch_dtype=dtype, use_safetensors=True
).to(device)
print(f"  pipeline load: {(_time.perf_counter()-t0)*1000:.0f}ms")

t0 = _time.perf_counter()
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
print(f"  scheduler swap: {(_time.perf_counter()-t0)*1000:.0f}ms")

# Load LCM LoRA
t0 = _time.perf_counter()
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="_lcm")
pipe.set_adapters(["_lcm"], [1.0])
print(f"  LCM LoRA inject: {(_time.perf_counter()-t0)*1000:.0f}ms")

# Full VAE timings
full_vae = pipe.vae
t0 = _time.perf_counter()
tiny_vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=dtype).to(device)
print(f"  TinyVAE load: {(_time.perf_counter()-t0)*1000:.0f}ms")

print("\n--- Per-component timing (avg over 5 runs) ---\n")

W, H = 512, 512
latent_h, latent_w = H // 8, W // 8
img = Image.fromarray(np.random.randint(50, 200, (H, W, 3), dtype=np.uint8))
img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 127.5 - 1.0).permute(2,0,1).unsqueeze(0).to(device, dtype=dtype)

mask = Image.new("L", (W, H), 255)
mask_t = torch.ones(1, 1, latent_h, latent_w, device=device, dtype=dtype)

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents

N = 5

def bench(name, fn):
    torch.cuda.synchronize()
    t0 = _time.perf_counter()
    for _ in range(N):
        result = fn()
        torch.cuda.synchronize()
    ms = (_time.perf_counter() - t0) * 1000 / N
    print(f"  {name}: {ms:.1f}ms")
    return result

# VAE encode (full)
with torch.no_grad():
    latent_full = bench("Full VAE encode", lambda: retrieve_latents(full_vae.encode(img_t)))
    bench("Full VAE decode", lambda: full_vae.decode(latent_full * 0.13025 / 0.13025, return_dict=False)[0])

# TinyVAE encode/decode
with torch.no_grad():
    latent_tiny = bench("TinyVAE encode", lambda: tiny_vae.encode(img_t).latents)
    bench("TinyVAE decode", lambda: tiny_vae.decode(latent_tiny, return_dict=False)[0])

# UNet forward (9-channel, 2-step batch)
unet = pipe.unet
print()
with torch.inference_mode():
    # Prompt embed
    scheduler = pipe.scheduler
    scheduler.set_timesteps(50, device=device)
    ts = scheduler.timesteps
    sub_ts = torch.stack([ts[32], ts[45]]).to(device)

    pe, ne, po, pn = pipe.encode_prompt("a forest", device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
    pe = pe.to(dtype).repeat(2, 1, 1)  # batch=2 (2 steps)
    po_b = po.to(dtype).repeat(2, 1)

    time_ids = pipe._get_add_time_ids((H,W),(0,0),(H,W),6.0,2.5,(H,W),(0,0),(H,W), dtype, text_encoder_projection_dim=po.shape[-1])
    if isinstance(time_ids, tuple): time_ids = time_ids[0]
    time_ids = time_ids.to(device).repeat(2, 1)

    latent = latent_full.repeat(2, 1, 1, 1)
    alpha_sq = scheduler.alphas_cumprod[sub_ts].sqrt().view(2,1,1,1).to(dtype)
    beta_sq = (1 - scheduler.alphas_cumprod[sub_ts]).sqrt().view(2,1,1,1).to(dtype)
    noise = torch.randn_like(latent)
    x_in = alpha_sq * latent + beta_sq * noise

    mask_latents = mask_t.expand(2, 1, latent_h, latent_w)
    masked_image_latents = torch.zeros(2, 4, latent_h, latent_w, device=device, dtype=dtype)
    x_in_9ch = torch.cat([x_in, mask_latents, masked_image_latents], dim=1)

    unet_kwargs = {
        "encoder_hidden_states": pe,
        "return_dict": False,
        "added_cond_kwargs": {"text_embeds": po_b, "time_ids": time_ids}
    }
    bench("UNet 9ch forward (batch=2, 2-step)", lambda: unet(x_in_9ch, sub_ts, **unet_kwargs)[0])

print()
print("=== TOTAL expected per frame ===")
print("  Full VAE: encode×1 + masked_encode×1 + decode×1")
print("  UNet 9ch batch=2 (2 denoising steps)")
print()
print("Optimization potential:")
print("  TinyVAE: saves ~60-100ms on encode+decode")
print("  1 step (batch=1): saves ~50% UNet time")
