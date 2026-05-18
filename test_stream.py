"""
StreamDiffusion standalone benchmark & demo.

Implements the EXACT algorithm from https://github.com/cumulo-autumn/streamdiffusion:
  - Cascading latent buffer for temporal coherence
  - RCFG "self" mode: stock_noise * delta as negative, rotated each frame
  - LCMScheduler with precomputed c_skip/c_out/alpha/beta at selected timesteps
  - TinyVAE (taesdxl) for fast encode/decode
  - Stochastic similarity filter (optional)
  - A->B prompt interpolation via embedding lerp

Usage:
    python test_stream.py [--model PATH] [--lora PATH] [--steps 2] [--fps-only]

Output: outputs/stream_test/  (frame_XXXX.png + demo.gif + bench.txt)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults — override with env vars or CLI args
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIRS = [
    r"D:\comfyui\comfy\models\diffusers\sdxl-turbo",
    r"D:\models\diffusers\sdxl-turbo",
    "stabilityai/sdxl-turbo",           # HF fallback
]
DEFAULT_LCM_LORA = "latent-consistency/lcm-lora-sdxl"
TAESD_ID = "madebyollin/taesdxl"
OUTPUT_DIR = Path("outputs/stream_test")

# ---------------------------------------------------------------------------
# Pure-algorithm helpers (no class needed for the test)
# ---------------------------------------------------------------------------

def find_model(candidates: list[str]) -> str:
    for c in candidates:
        if Path(c).exists():
            return c
    return candidates[-1]  # HF id


def build_scheduler_tables(scheduler, t_index_list: list[int], batch_size: int, dtype, device):
    """Pre-compute scalings for the selected timestep indices."""
    scheduler.set_timesteps(50)
    ts = scheduler.timesteps                        # shape: (50,)
    sub_ts = ts[t_index_list]                       # selected timesteps

    # Broadcast to full batch (steps * frame_buffer ≡ batch_size)
    sub_ts_rep = sub_ts.repeat_interleave(batch_size // len(t_index_list)) if batch_size > len(t_index_list) else sub_ts

    alpha_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    alpha_sq = alpha_cumprod[sub_ts].sqrt()
    beta_sq = (1 - alpha_cumprod[sub_ts]).sqrt()

    c_skip_list, c_out_list = [], []
    for t in sub_ts:
        c_skip, c_out = scheduler.get_scalings_for_boundary_condition_discrete(t)
        c_skip_list.append(c_skip)
        c_out_list.append(c_out)

    def expand(vals):
        return torch.stack(vals).view(-1, 1, 1, 1).to(device=device, dtype=dtype)

    return {
        "sub_timesteps": sub_ts.to(device),
        "alpha_sq": expand(alpha_sq.unbind()),
        "beta_sq": expand(beta_sq.unbind()),
        "c_skip": expand(c_skip_list),
        "c_out": expand(c_out_list),
    }


def encode_image(vae, image: Image.Image, device, dtype, latent_scale: float) -> torch.Tensor:
    img = image.convert("RGB").resize((512, 512), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        enc = vae.encode(t)
        # AutoencoderKL → .latent_dist.mean | AutoencoderTiny → .latents
        latent = enc.latent_dist.mean if hasattr(enc, "latent_dist") else enc.latents
    return latent * latent_scale                    # (1, 4, h, w)


def decode_latent(vae, latent: torch.Tensor, latent_scale: float) -> Image.Image:
    with torch.no_grad():
        img_t = vae.decode(latent / latent_scale).sample
    img_t = img_t.squeeze(0).permute(1, 2, 0).float().clamp(-1, 1)
    arr = ((img_t.cpu().numpy() + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(arr)


def add_noise(latent: torch.Tensor, noise: torch.Tensor, alpha_sq: torch.Tensor, beta_sq: torch.Tensor) -> torch.Tensor:
    """Mix latent with noise according to the first timestep's schedule."""
    return alpha_sq * latent + beta_sq * noise


def lcm_step(model_pred: torch.Tensor, x_t: torch.Tensor, alpha_sq, beta_sq, c_skip, c_out) -> torch.Tensor:
    """Single LCM denoising step.  Returns predicted x_0."""
    F_theta = (x_t - beta_sq * model_pred) / alpha_sq.clamp(min=1e-8)
    return c_out * F_theta + c_skip * x_t


# ---------------------------------------------------------------------------
# The StreamDiffusion session
# ---------------------------------------------------------------------------

class StreamSession:
    """One persistent StreamDiffusion session.  Mimics pipeline.StreamDiffusion exactly."""

    def __init__(
        self,
        unet,
        vae,
        scheduler,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor | None,
        neg_embeds: torch.Tensor | None,
        pooled_neg: torch.Tensor | None,
        t_index_list: list[int],
        guidance_scale: float,
        cfg_type: str,             # "self" | "none" | "full"
        delta: float,
        do_add_noise: bool,
        device,
        dtype,
        latent_scale: float,
        is_sdxl: bool,
    ):
        self.unet = unet
        self.vae = vae
        self.guidance_scale = guidance_scale
        self.cfg_type = cfg_type
        self.delta = delta
        self.do_add_noise = do_add_noise
        self.device = device
        self.dtype = dtype
        self.latent_scale = latent_scale
        self.is_sdxl = is_sdxl
        self.steps = len(t_index_list)

        # Encoder hidden states
        self.prompt_embeds = prompt_embeds      # (1, seq, dim)
        self.pooled_embeds = pooled_embeds      # (1, pool_dim) or None
        self.neg_embeds = neg_embeds
        self.pooled_neg = pooled_neg

        # Scheduler tables
        tables = build_scheduler_tables(scheduler, t_index_list, self.steps, dtype, device)
        self.sub_timesteps = tables["sub_timesteps"]    # (steps,)
        self.alpha_sq = tables["alpha_sq"]              # (steps, 1, 1, 1)
        self.beta_sq = tables["beta_sq"]
        self.c_skip = tables["c_skip"]
        self.c_out = tables["c_out"]

        # Latent dimensions (inferred after first encode)
        self.latent_h = 64
        self.latent_w = 64
        self._buffer_initialized = False

    def _init_buffer(self, h: int, w: int):
        self.latent_h = h
        self.latent_w = w
        shape = (max(1, self.steps - 1), 4, h, w)
        self.latent_buffer = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.init_noise = torch.randn((self.steps, 4, h, w), device=self.device, dtype=self.dtype)
        self.stock_noise = torch.randn((self.steps, 4, h, w), device=self.device, dtype=self.dtype)
        self._buffer_initialized = True

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        prompt_embeds_b: torch.Tensor | None = None,
        pooled_b: torch.Tensor | None = None,
        lerp_t: float = 0.0,
    ) -> Image.Image:
        latent = encode_image(self.vae, image, self.device, self.dtype, self.latent_scale)
        _, _, h, w = latent.shape

        if not self._buffer_initialized:
            self._init_buffer(h, w)

        # Add noise at first timestep level
        x_t = add_noise(latent, self.init_noise[0:1], self.alpha_sq[0:1], self.beta_sq[0:1])

        # Build batch input: current frame + buffer from previous frames
        if self.steps > 1:
            x_in = torch.cat([x_t, self.latent_buffer], dim=0)    # (steps, 4, h, w)
            # Rotate stock noise (RCFG): shift previous stock noise by one frame
            self.stock_noise = torch.cat([self.init_noise[0:1], self.stock_noise[:-1]], dim=0)
        else:
            x_in = x_t

        # Prompt interpolation
        pe = self.prompt_embeds
        po = self.pooled_embeds
        if prompt_embeds_b is not None and lerp_t > 0:
            pe = (1 - lerp_t) * pe + lerp_t * prompt_embeds_b
            if po is not None and pooled_b is not None:
                po = (1 - lerp_t) * po + lerp_t * pooled_b

        # Expand prompt embeds to match batch size
        bsz = x_in.shape[0]
        pe_batch = pe.expand(bsz, -1, -1)

        # CFG preparation
        if self.cfg_type == "full" and self.guidance_scale > 1:
            x_cfg = torch.cat([x_in, x_in], dim=0)
            t_cfg = torch.cat([self.sub_timesteps, self.sub_timesteps])
            pe_cfg = torch.cat([self.neg_embeds.expand(bsz, -1, -1), pe_batch], dim=0)
            po_cfg = torch.cat([self.pooled_neg.expand(bsz, -1), po.expand(bsz, -1)], dim=0) if (po is not None and self.pooled_neg is not None) else None
        else:
            x_cfg = x_in
            t_cfg = self.sub_timesteps[:bsz] if bsz < len(self.sub_timesteps) else self.sub_timesteps
            pe_cfg = pe_batch
            po_cfg = po.expand(bsz, -1) if po is not None else None

        # UNet forward
        unet_kwargs: dict = {"encoder_hidden_states": pe_cfg, "return_dict": False}
        if self.is_sdxl and po_cfg is not None:
            # SDXL needs time_ids and text_embeds for added_cond_kwargs
            # Use canonical values: original_size=target_size=(512,512), crop=(0,0)
            orig_size = (512, 512)
            time_ids = torch.tensor(
                [orig_size[0], orig_size[1], 0, 0, orig_size[0], orig_size[1]],
                dtype=self.dtype, device=self.device
            ).unsqueeze(0).expand(x_cfg.shape[0], -1)
            unet_kwargs["added_cond_kwargs"] = {"text_embeds": po_cfg, "time_ids": time_ids}

        # Make timesteps match batch dim
        if t_cfg.shape[0] != x_cfg.shape[0]:
            t_cfg = t_cfg[:x_cfg.shape[0]]

        model_pred = self.unet(x_cfg, t_cfg, **unet_kwargs)[0]

        # Apply CFG
        if self.cfg_type == "full" and self.guidance_scale > 1:
            noise_uncond, noise_text = model_pred.chunk(2)
            model_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)
        elif self.cfg_type == "self" and self.guidance_scale > 1:
            noise_text = model_pred
            noise_uncond = self.stock_noise[:bsz] * self.delta
            model_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)
        # else cfg_type=="none": model_pred unchanged

        # LCM scheduler step for all items in batch
        x0_pred = lcm_step(
            model_pred, x_in,
            self.alpha_sq[:bsz], self.beta_sq[:bsz],
            self.c_skip[:bsz], self.c_out[:bsz],
        )

        # Current frame output = last item
        x0_out = x0_pred[-1:]

        # Update latent buffer for next frame
        if self.steps > 1:
            if self.do_add_noise:
                self.latent_buffer = (
                    self.alpha_sq[1:bsz] * x0_pred[:-1]
                    + self.beta_sq[1:bsz] * self.init_noise[1:bsz]
                )
            else:
                self.latent_buffer = self.alpha_sq[1:bsz] * x0_pred[:-1]

        return decode_latent(self.vae, x0_out, self.latent_scale)


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(model_path: str, lora_path: str | None, device: str, dtype: torch.dtype):
    from diffusers import StableDiffusionXLPipeline, AutoencoderTiny, LCMScheduler
    from diffusers import EulerDiscreteScheduler

    print(f"[load] model: {model_path}")
    is_file = Path(model_path).is_file()

    if is_file:
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path, torch_dtype=dtype, use_safetensors=True
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path, torch_dtype=dtype, use_safetensors=True, variant="fp16"
        )

    pipe = pipe.to(device)

    # Swap scheduler → LCM
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # LCM LoRA if model isn't already turbo/lcm
    model_name = str(model_path).lower()
    needs_lcm = not any(m in model_name for m in ("turbo", "lcm", "lightning"))
    if needs_lcm and lora_path:
        print(f"[load] injecting LCM LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name="_lcm")
        pipe.set_adapters(["_lcm"], [1.0])

    # TinyVAE for fast decode
    print(f"[load] swapping VAE -> {TAESD_ID}")
    pipe.vae = AutoencoderTiny.from_pretrained(TAESD_ID, torch_dtype=dtype).to(device)

    pipe.unet.eval()
    pipe.vae.eval()

    return pipe, needs_lcm


def encode_prompt(pipe, prompt: str, negative_prompt: str, device, dtype):
    """Returns (prompt_embeds, pooled_embeds, neg_embeds, pooled_neg)."""
    with torch.no_grad():
        result = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
    # encode_prompt returns (prompt_embeds, neg_embeds, pooled, neg_pooled)
    if len(result) == 4:
        pe, ne, po, pn = result
    else:
        pe, ne = result
        po = pn = None
    return (
        pe.to(dtype=dtype),
        po.to(dtype=dtype) if po is not None else None,
        ne.to(dtype=dtype),
        pn.to(dtype=dtype) if pn is not None else None,
    )


# ---------------------------------------------------------------------------
# Cosine similarity filter
# ---------------------------------------------------------------------------

class SimilarityFilter:
    def __init__(self, threshold: float = 0.98, max_skip: int = 10):
        self.threshold = threshold
        self.max_skip = max_skip
        self.prev = None
        self.skip_count = 0

    def __call__(self, img: Image.Image) -> bool:
        """Returns True if the frame should be processed (not skipped)."""
        import torch.nn.functional as F
        arr = torch.from_numpy(np.array(img, dtype=np.float32)).flatten()
        if self.prev is None:
            self.prev = arr
            return True
        sim = F.cosine_similarity(self.prev.unsqueeze(0), arr.unsqueeze(0)).item()
        import random
        skip_prob = max(0.0, 1.0 - (1.0 - sim) / max(1e-8, 1.0 - self.threshold)) if self.threshold < 1.0 else 0.0
        if skip_prob < random.random() or self.skip_count >= self.max_skip:
            self.prev = arr
            self.skip_count = 0
            return True
        self.skip_count += 1
        return False


# ---------------------------------------------------------------------------
# Tests / demo
# ---------------------------------------------------------------------------

def make_gradient_image(size=(512, 512), color_a=(30, 60, 120), color_b=(120, 60, 30)) -> Image.Image:
    """Simple gradient base image to show motion."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        for c in range(3):
            arr[y, :, c] = int(color_a[c] * (1 - t) + color_b[c] * t)
    return Image.fromarray(arr)


def save_gif(frames: list[Image.Image], path: Path, fps: float = 12.0):
    dur = int(1000 / fps)
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, duration=dur)
    print(f"[demo] saved GIF → {path}  ({len(frames)} frames @ {fps:.0f} fps)")


def run_tests(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[init] device={device}  dtype={dtype}")

    # ── Load model ─────────────────────────────────────────────────────────
    model_path = args.model or find_model(DEFAULT_MODEL_DIRS)
    lora_path = args.lora or DEFAULT_LCM_LORA
    pipe, _ = load_pipeline(model_path, lora_path, device, dtype)

    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    latent_scale = getattr(vae.config, "scaling_factor", 0.13025)
    is_sdxl = hasattr(pipe, "text_encoder_2")

    # ── Prompts ─────────────────────────────────────────────────────────────
    PROMPT_A = "a warrior standing in a forest, cinematic lighting, sharp focus, 4k"
    PROMPT_B = "a warrior rising from the ground, epic pose, volumetric light, 4k"
    NEG = "blurry, ugly, bad quality, disfigured"

    print("[encode] encoding prompts …")
    pe_a, po_a, ne_a, pn_a = encode_prompt(pipe, PROMPT_A, NEG, device, dtype)
    pe_b, po_b, ne_b, pn_b = encode_prompt(pipe, PROMPT_B, NEG, device, dtype)

    # ── Session ─────────────────────────────────────────────────────────────
    t_index_list = list(range(0, 50, 50 // args.steps))[:args.steps]
    # Standard StreamDiffusion values: 2 steps → [32, 45]; 3 → [22, 32, 45]
    if args.steps == 2:
        t_index_list = [32, 45]
    elif args.steps == 3:
        t_index_list = [22, 32, 45]
    elif args.steps == 4:
        t_index_list = [0, 16, 32, 45]
    else:
        t_index_list = [45]

    session = StreamSession(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        prompt_embeds=pe_a,
        pooled_embeds=po_a,
        neg_embeds=ne_a,
        pooled_neg=pn_a,
        t_index_list=t_index_list,
        guidance_scale=1.2,
        cfg_type="self",
        delta=0.5,
        do_add_noise=True,
        device=device,
        dtype=dtype,
        latent_scale=latent_scale,
        is_sdxl=is_sdxl,
    )

    base_image = make_gradient_image()

    # ── TEST 1: Warmup + steady breathing ──────────────────────────────────
    print("\n[test 1] warmup + breathing (static prompt A, 30 frames) …")
    frames_breathing: list[Image.Image] = []
    warmup = args.steps + 2

    t0 = time.perf_counter()
    for i in range(30):
        out = session(base_image)
        if i >= warmup:
            frames_breathing.append(out.resize((256, 256)))
        if i % 5 == 0:
            out.save(OUTPUT_DIR / f"frame_{i:04d}.png")
            print(f"  frame {i:02d}")
    t1 = time.perf_counter()

    n_frames = 30 - warmup
    fps_breathing = n_frames / (t1 - t0 - warmup * (t1 - t0) / 30)
    print(f"  steady-state FPS: {fps_breathing:.1f}")

    # ── TEST 2: A→B prompt transition ──────────────────────────────────────
    print("\n[test 2] A→B prompt transition (30 frames) …")
    frames_ab: list[Image.Image] = []
    for i in range(30):
        lerp_t = i / 29.0
        out = session(base_image, prompt_embeds_b=pe_b, pooled_b=po_b, lerp_t=lerp_t)
        frames_ab.append(out.resize((256, 256)))
        if i % 5 == 0:
            out.save(OUTPUT_DIR / f"ab_{i:04d}.png")
            print(f"  lerp={lerp_t:.2f}")

    # ── TEST 3: FPS benchmark ───────────────────────────────────────────────
    print("\n[test 3] FPS benchmark (100 frames, no save overhead) …")
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        session(base_image)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    fps_bench = 100 / (t1 - t0)
    print(f"  benchmark FPS: {fps_bench:.1f}  ({(t1-t0)*10:.0f} ms/frame avg)")

    # ── TEST 4: Temporal coherence check ───────────────────────────────────
    print("\n[test 4] temporal coherence (cosine similarity between consecutive frames) …")
    sims = []
    prev_arr = None
    for i, fr in enumerate(frames_breathing):
        arr = np.array(fr, dtype=np.float32).flatten() / 255.0
        if prev_arr is not None:
            cos = np.dot(arr, prev_arr) / (np.linalg.norm(arr) * np.linalg.norm(prev_arr) + 1e-8)
            sims.append(cos)
        prev_arr = arr
    if sims:
        avg_sim = float(np.mean(sims))
        print(f"  avg frame-to-frame cosine similarity: {avg_sim:.4f}  (higher = smoother motion)")
        assert avg_sim > 0.7, f"Temporal coherence too low: {avg_sim:.4f} — session may be broken"
        print("  [PASS] temporal coherence OK")

    # ── Save demo GIFs ──────────────────────────────────────────────────────
    if frames_breathing:
        save_gif(frames_breathing, OUTPUT_DIR / "demo_breathing.gif", fps=10)
    if frames_ab:
        save_gif(frames_ab, OUTPUT_DIR / "demo_ab_transition.gif", fps=10)

    # ── Write bench report ──────────────────────────────────────────────────
    report = (
        f"StreamDiffusion test results\n"
        f"Model: {model_path}\n"
        f"Steps: {args.steps}  t_index_list: {t_index_list}\n"
        f"Device: {device}  dtype: {dtype}\n"
        f"Breathing FPS (steady-state): {fps_breathing:.1f}\n"
        f"Benchmark FPS (100 frames):   {fps_bench:.1f}\n"
        f"Frame-to-frame coherence:     {avg_sim:.4f}\n"
    )
    (OUTPUT_DIR / "bench.txt").write_text(report)
    print(f"\n{report}")
    print(f"[done] outputs in {OUTPUT_DIR.resolve()}")

    return fps_bench


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion standalone test")
    parser.add_argument("--model", default=None, help="Path to SDXL diffusers dir or safetensors")
    parser.add_argument("--lora", default=None, help="LCM LoRA path or HF id")
    parser.add_argument("--steps", type=int, default=2, choices=[1, 2, 3, 4], help="Denoising steps")
    parser.add_argument("--fps-only", action="store_true", help="Skip GIF generation, bench only")
    args = parser.parse_args()

    fps = run_tests(args)
    print(f"\nFinal FPS: {fps:.1f}")
    if fps < 5:
        print("WARNING: FPS very low. Check GPU utilization and model size.")
    elif fps >= 30:
        print("EXCELLENT: 30+ FPS achieved — StreamDiffusion is working well.")
    elif fps >= 10:
        print("GOOD: 10+ FPS. Enable TRT or reduce resolution for more speed.")


if __name__ == "__main__":
    main()
