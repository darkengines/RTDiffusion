"""
StreamDiffusion Inpainting Pipeline — direct GPU-speed implementation.

Hot-path:  PIL image → VAE encode → UNet forward → VAE decode → PIL image
           No data-URL conversion, no Pydantic validation, no layer conditions.

Session state (latent buffer, prompt embeddings, stock noise) persists across
frames. Each infer() call is exactly one batched UNet forward pass.

RCFG (self-negative, default):
  One UNet pass per frame.  Stock noise approximates the uncond direction.
  Achieves 10-30+ FPS on SDXL with TinyVAE on modern GPUs.

Full CFG:
  Two UNet passes per frame (standard CFG).  Half the FPS.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger("rtdiffusion.stream")

_TINY_VAE_SDXL = "madebyollin/taesdxl"
_TINY_VAE_SD15 = "madebyollin/taesd"


def _is_cuda(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


# ──────────────────────────────────────────────────────────────────
# StreamSession
# ──────────────────────────────────────────────────────────────────

class StreamSession:
    """
    One continuous StreamDiffusion session.

    Maintains rolling latent buffer, prompt embeddings, TinyVAE, and
    all per-frame state between calls.

    Call .infer(image, mask) every frame.
    Re-create only when model / prompt / resolution / key settings change.
    """

    def __init__(
        self,
        *,
        pipe: Any,
        device: str,
        prompt: str = "",
        negative_prompt: str = "",
        t_indices: list[int] | None = None,
        frame_buffer_size: int = 1,
        cfg_type: str = "self",
        cfg_scale: float = 1.0,
        width: int = 512,
        height: int = 512,
        use_tiny_vae: bool = True,
        prompt_b: str = "",
        rcfg_delta: float = 1.0,
        controlnet: Any = None,
    ) -> None:
        import torch
        from diffusers import LCMScheduler

        self._width = width
        self._height = height
        self._cfg_type = cfg_type

        device_t = torch.device(device)
        unet = pipe.unet
        dtype = next(unet.parameters()).dtype
        self._infer_lock = threading.Lock()
        # prediction_type: "epsilon" (most SDXL/SD1.5) or "v_prediction" (SD2, some fine-tunes)
        self._prediction_type = getattr(unet.config, "prediction_type", "epsilon")

        # ── scheduler + timestep indices ───────────────────────────
        t_idx = sorted(set(max(0, min(999, i)) for i in (t_indices or [32, 45])))[:16]
        scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        n_sched = max(50, max(t_idx) + 1)
        scheduler.set_timesteps(n_sched, device=device_t)
        ts = scheduler.timesteps.to(device_t)
        timestep_values = [ts[min(i, len(ts) - 1)] for i in t_idx]
        n_steps = len(timestep_values)
        fbsz = max(1, min(4, frame_buffer_size))
        batch_size = n_steps * fbsz

        sub_t = torch.repeat_interleave(
            torch.stack(timestep_values).long().to(device_t),
            fbsz, dim=0,
        )

        # ── latent geometry ────────────────────────────────────────
        vae_sf = int(getattr(pipe, "vae_scale_factor", 8))
        lh = height // vae_sf
        lw = width // vae_sf

        unet_ch = int(getattr(unet.config, "in_channels", 4))
        vae = _load_vae(pipe, use_tiny_vae, dtype, device_t)
        lat_ch = int(getattr(vae.config, "latent_channels",
                              getattr(pipe.vae.config, "latent_channels", 4)))
        is_inpaint = (unet_ch == lat_ch + 5)  # 9-channel inpaint UNet

        # ── noise buffers ──────────────────────────────────────────
        g = torch.Generator(device=device_t).manual_seed(42)
        init_noise = torch.randn((batch_size, lat_ch, lh, lw),
                                 generator=g, device=device_t, dtype=dtype)
        stock_noise = torch.randn_like(init_noise)

        # ── latent buffer (multi-step streaming) ───────────────────
        latent_buffer: Any = None
        if n_steps > 1:
            latent_buffer = torch.zeros(
                ((n_steps - 1) * fbsz, lat_ch, lh, lw),
                device=device_t, dtype=dtype,
            )

        # ── LCM scalings ───────────────────────────────────────────
        c_skip, c_out, alpha, beta = _compute_scalings(
            scheduler, timestep_values, fbsz, dtype, device_t, torch
        )

        # ── prompt embeddings ──────────────────────────────────────
        do_full_cfg = cfg_type == "full" and cfg_scale > 1.0
        prompt_embeds, neg_embeds, pooled, neg_pooled = _encode_prompt(
            pipe, prompt, negative_prompt if do_full_cfg else "",
            do_full_cfg, device_t, dtype,
        )
        # repeat to fill batch
        prompt_embeds_batched = prompt_embeds.repeat(batch_size, 1, 1)
        if do_full_cfg and neg_embeds is not None:
            neg_batched = neg_embeds.repeat(batch_size, 1, 1)
            prompt_embeds_batched = torch.cat([neg_batched, prompt_embeds_batched], dim=0)

        # prompt B for lerp
        pb_embeds: Any = None
        pb_pooled: Any = None
        if prompt_b.strip():
            pb_e, _, pb_p, _ = _encode_prompt(
                pipe, prompt_b, "", False, device_t, dtype
            )
            pb_embeds = pb_e  # (1, seq, dim) — base, repeated per frame
            pb_pooled = pb_p

        added_cond = _build_added_cond(
            pipe, width, height, pooled, batch_size, dtype, device_t, torch
        )
        if do_full_cfg and neg_embeds is not None:
            neg_added = _build_added_cond(
                pipe, width, height,
                neg_pooled if neg_pooled is not None else pooled,
                batch_size, dtype, device_t, torch
            )
            if added_cond and neg_added:
                added_cond = {
                    k: torch.cat([neg_added[k], added_cond[k]], dim=0)
                    for k in added_cond if k in neg_added
                }

        # channels_last for throughput on CUDA
        if _is_cuda(device):
            try:
                unet.to(memory_format=torch.channels_last)
            except Exception:
                pass

        # Cache VAE device/dtype — avoids traversing model parameters every frame
        _vae_params = next(vae.parameters())
        self._vae_dev = _vae_params.device
        self._vae_dt = _vae_params.dtype
        # RTD_NOISE_DRIFT=0 (default) keeps output stable on a static canvas.
        # Set to 0.002-0.005 for subtle continuous motion; higher values cause oscillation.
        self._noise_drift = float(os.getenv("RTD_NOISE_DRIFT", "0"))

        self._s: dict[str, Any] = {
            "pipe": pipe,
            "device": device_t,
            "dtype": dtype,
            "unet": unet,
            "vae": vae,
            "vae_scale": float(getattr(vae.config, "scaling_factor", 0.18215)),
            "vae_sf": vae_sf,
            "batch_size": batch_size,
            "fbsz": fbsz,
            "n_steps": n_steps,
            "sub_t": sub_t,
            "lh": lh,
            "lw": lw,
            "lat_ch": lat_ch,
            "unet_ch": unet_ch,
            "is_inpaint": is_inpaint,
            "init_noise": init_noise,
            "stock_noise": stock_noise,
            "latent_buffer": latent_buffer,
            "c_skip": c_skip,
            "c_out": c_out,
            "alpha": alpha,
            "beta": beta,
            "prompt_embeds": prompt_embeds_batched,
            "prompt_embeds_base": prompt_embeds,      # (1, seq, dim) — for lerp
            "neg_embeds_base": neg_embeds,
            "pooled_base": pooled,
            "neg_pooled_base": neg_pooled,
            "pb_embeds": pb_embeds,
            "pb_pooled": pb_pooled,
            "added_cond": added_cond,
            "do_full_cfg": do_full_cfg,
            "cfg_scale": cfg_scale,
            "cfg_type": cfg_type,
            "rcfg_delta": rcfg_delta,
            "prediction_type": self._prediction_type,
            # mask cache
            "mask_latents": None,
            "masked_img_latents": None,
            "_mask_bytes": None,
            # ControlNet (optional)
            "cn_model": controlnet,
        }
        self._ip = getattr(pipe, "image_processor", None)

        # Per-instance override-prompt cache (class-level attributes would be
        # shared across all sessions, causing stale embeddings after rebuild).
        self._last_override_prompt: str = ""
        self._last_override_embeds: Any = None
        self._last_override_added: Any = None

    # ── public API ─────────────────────────────────────────────────

    def infer(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
        prompt_lerp: float = 0.0,
        prompt_override: str | None = None,
        denoise: float = 1.0,
        denoise_map: Image.Image | None = None,
        control_image: Image.Image | None = None,
        cn_scale: float = 1.0,
        cn_start: float = 0.0,
        cn_end: float = 1.0,
    ) -> Image.Image | None:
        """
        One StreamDiffusion step.

        image           — RGB PIL, must match session (width, height).
        mask            — L-mode PIL mask (255=generate, 0=keep input).
        prompt_override — encode this prompt on-the-fly (tagger-derived).
        denoise         — [0, 1] latent-space blend: 1.0 = full diffusion,
                          0.0 = pure input. Controls output similarity to
                          the input without session rebuild.
        denoise_map     — optional L-mode PIL map for spatial denoise control.
                  255 means fully apply denoise in that region.
        control_image   — optional preprocessed ControlNet conditioning image.
                          Only used if the session was built with a CN model.
        cn_scale        — ControlNet conditioning scale for this frame.
        Returns RGB PIL image, or None if another inference is in progress.
        """
        if not self._infer_lock.acquire(blocking=False):
            return None  # concurrent connection — skip this frame

        try:
            return self._infer_inner(image, mask, prompt_lerp, prompt_override,
                                     denoise, denoise_map, control_image, cn_scale, cn_start, cn_end)
        finally:
            self._infer_lock.release()

    def _infer_inner(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
        prompt_lerp: float = 0.0,
        prompt_override: str | None = None,
        denoise: float = 1.0,
        denoise_map: Image.Image | None = None,
        control_image: Image.Image | None = None,
        cn_scale: float = 1.0,
        cn_start: float = 0.0,
        cn_end: float = 1.0,
    ) -> Image.Image:
        import torch

        s = self._s
        dev = s["device"]
        dtype = s["dtype"]
        vae = s["vae"]
        vae_dev = self._vae_dev
        vae_dt = self._vae_dt
        vsc = s["vae_scale"]

        # ── preprocess input to (1,3,H,W) tensor ──────────────────
        with torch.inference_mode():
            if self._ip is not None:
                img_t = self._ip.preprocess(
                    image, height=self._height, width=self._width
                ).to(device=vae_dev, dtype=vae_dt)
            else:
                img_t = _pil_to_tensor(image, self._height, self._width,
                                        vae_dev, vae_dt)

            # ── VAE encode ─────────────────────────────────────────
            img_latent = _vae_encode(vae, img_t, vsc, dev, dtype)

            # ── update inpaint mask latents (9-ch UNet) ────────────
            if s["is_inpaint"]:
                _update_mask_latents(
                    s, mask, img_t, img_latent, vae, vae_dev, vae_dt,
                    dev, dtype, vsc, self._width, self._height, torch
                )

            _drift = self._noise_drift
            if _drift > 0:
                _delta = torch.randn_like(s["init_noise"])
                _new = s["init_noise"] + _drift * _delta
                # Renormalise to unit variance so noise statistics stay stable.
                s["init_noise"] = _new / (_new.std() + 1e-8)

            _d = max(0.0, min(1.0, float(denoise)))
            denoise_lat: Any = None
            if denoise_map is not None:
                import numpy as _np
                den_arr = _np.array(
                    denoise_map.convert("L").resize((self._width, self._height), Image.LANCZOS),
                    dtype="float32",
                ) / 255.0
                den_t = torch.from_numpy(den_arr).unsqueeze(0).unsqueeze(0).to(device=dev, dtype=dtype)
                denoise_lat = torch.nn.functional.interpolate(
                    den_t,
                    size=img_latent.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)

            # ── noise input at each t-index ────────────────────────
            fbsz = s["fbsz"]
            noise_term = s["init_noise"][:fbsz]
            if denoise_lat is None:
                noise_scale = _d
            else:
                noise_scale = denoise_lat.expand(fbsz, 1, denoise_lat.shape[-2], denoise_lat.shape[-1])
            x_t = (
                s["alpha"][:fbsz] * img_latent.expand(fbsz, -1, -1, -1)
                + (s["beta"][:fbsz] * noise_scale) * noise_term
            )

            # ── build effective prompt embeds (with optional A→B lerp or override) ─
            if prompt_override and prompt_override != self._last_override_prompt:
                # Re-encode only when the override prompt changes
                ov_e, _, ov_p, _ = _encode_prompt(
                    s["pipe"], prompt_override, "", False, dev, dtype
                )
                if ov_e is not None:
                    self._last_override_embeds = ov_e.repeat(s["batch_size"], 1, 1)
                    self._last_override_added = _build_added_cond(
                        s["pipe"], self._width, self._height, ov_p,
                        s["batch_size"], dtype, dev, torch
                    ) if ov_p is not None else s.get("added_cond") or {}
                self._last_override_prompt = prompt_override

            if prompt_override and self._last_override_embeds is not None:
                pe = self._last_override_embeds
                ac = self._last_override_added or {}
            else:
                pe, ac = _frame_embeds(s, prompt_lerp, fbsz, torch)

            # ── prepare ControlNet conditioning tensor ─────────────
            control_cond: Any = None
            if control_image is not None and s.get("cn_model") is not None:
                import numpy as _np
                ctrl_arr = _np.array(
                    control_image.convert("RGB").resize((self._width, self._height), Image.LANCZOS),
                    dtype="float32",
                ) / 255.0  # [0,1] range required by ControlNet
                control_cond = torch.from_numpy(ctrl_arr).permute(2, 0, 1).unsqueeze(0).to(device=dev, dtype=dtype)

            # ── StreamDiffusion forward ────────────────────────────
            x0 = _predict_x0(x_t, pe, ac, s, torch, control_cond=control_cond,
                              cn_scale=cn_scale, cn_start=cn_start, cn_end=cn_end)

            local_denoise = float(denoise_lat.max().item()) if denoise_lat is not None else _d
            if local_denoise <= 0.001:
                x0 = img_latent.expand_as(x0)

            # ── VAE decode ─────────────────────────────────────────
            decoded = vae.decode(x0 / vsc).sample        # (1, 3, H, W)
            decoded = decoded.float().clamp(-1, 1)
            decoded = ((decoded + 1) / 2)[0].cpu().permute(1, 2, 0).numpy()
            output = Image.fromarray((decoded * 255).astype("uint8"), mode="RGB")

        # ── inpainting compositing ─────────────────────────────────
        if mask is not None:
            import numpy as _np
            mask_l = mask.convert("L").resize((self._width, self._height), Image.NEAREST)
            if _np.count_nonzero(_np.asarray(mask_l) > 10) >= self._width * self._height * 0.005:
                base = image.convert("RGB").resize((self._width, self._height), Image.LANCZOS)
                return Image.composite(output, base, mask_l)
        return output

    def warmup(self, n: int = 1) -> None:
        blank = Image.new("RGB", (self._width, self._height), (128, 128, 128))
        for _ in range(n):
            try:
                self.infer(blank)
            except Exception:
                logger.exception("Warmup step failed")
        logger.info(
            "StreamSession warmed up (%d step(s)) at %dx%d", n, self._width, self._height
        )


# ──────────────────────────────────────────────────────────────────
# Module-level helpers (all receive `torch` as arg to avoid import overhead)
# ──────────────────────────────────────────────────────────────────

def _load_vae(pipe: Any, use_tiny: bool, dtype: Any, device: Any) -> Any:
    if not use_tiny:
        return pipe.vae
    is_xl = hasattr(pipe, "text_encoder_2")
    tiny_id = _TINY_VAE_SDXL if is_xl else _TINY_VAE_SD15
    try:
        from diffusers import AutoencoderTiny
        vae = AutoencoderTiny.from_pretrained(tiny_id, torch_dtype=dtype).to(device)
        logger.info("StreamSession: TinyVAE %s loaded", tiny_id)
        return vae
    except Exception:
        logger.warning("StreamSession: TinyVAE unavailable, using pipeline VAE")
        return pipe.vae


def _encode_prompt(pipe, prompt, negative_prompt, do_cfg, device, dtype):
    """Returns (prompt_embeds, neg_embeds, pooled, neg_pooled) — each (1, seq, dim) or None."""
    encoded = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt if (do_cfg and negative_prompt) else None,
    )
    if isinstance(encoded, (list, tuple)):
        pe  = encoded[0] if len(encoded) > 0 else None
        ne  = encoded[1] if len(encoded) > 1 else None
        pp  = encoded[2] if len(encoded) > 2 else None
        np_ = encoded[3] if len(encoded) > 3 else None
    else:
        pe, ne, pp, np_ = encoded, None, None, None
    if pe is not None:
        pe = pe.to(device=device, dtype=dtype)
    if ne is not None:
        ne = ne.to(device=device, dtype=dtype)
    if pp is not None:
        pp = pp.to(device=device, dtype=dtype)
    if np_ is not None:
        np_ = np_.to(device=device, dtype=dtype)
    return pe, ne, pp, np_


def _build_added_cond(pipe, width, height, pooled, batch_size, dtype, device, torch):
    """Build SDXL added_cond_kwargs dict or empty dict for SD1.5."""
    if pooled is None:
        return {}
    try:
        # Try diffusers helper first
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
            _get_add_time_ids,  # type: ignore[import]
        )
        time_ids = _get_add_time_ids(
            pipe,
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            dtype=dtype,
        ).to(device=device)
    except Exception:
        try:
            time_ids = torch.tensor(
                [[height, width, 0, 0, height, width]],
                dtype=dtype, device=device,
            )
        except Exception:
            return {}

    time_ids = time_ids.expand(batch_size, -1)
    pooled_b = pooled.expand(batch_size, -1) if pooled.ndim == 2 else pooled.repeat(batch_size, 1)
    return {"text_embeds": pooled_b, "time_ids": time_ids}


def _compute_scalings(scheduler, timestep_values, fbsz, dtype, device, torch):
    c_skip_l, c_out_l, alpha_l, beta_l = [], [], [], []
    for t in timestep_values:
        cs, co = scheduler.get_scalings_for_boundary_condition_discrete(t)
        ti = int(t.item())
        a = scheduler.alphas_cumprod[ti].sqrt()
        b = (1 - scheduler.alphas_cumprod[ti]).sqrt()
        c_skip_l.append(cs); c_out_l.append(co)
        alpha_l.append(a);   beta_l.append(b)

    def expand(vs):
        return torch.repeat_interleave(
            torch.stack(vs).view(len(vs), 1, 1, 1).to(device=device, dtype=dtype),
            fbsz, dim=0,
        )
    return expand(c_skip_l), expand(c_out_l), expand(alpha_l), expand(beta_l)


def _vae_encode(vae, img_t, vsc, device, dtype):
    """Encode image tensor to scaled latent (deterministic — uses mode, not sample)."""
    enc = vae.encode(img_t)
    if hasattr(enc, "latent_dist"):
        # Use mode (mean) instead of sample — deterministic, prevents per-frame noise oscillation
        lat = enc.latent_dist.mode()
    elif hasattr(enc, "latents"):
        lat = enc.latents
    elif isinstance(enc, (tuple, list)):
        lat = enc[0]
    else:
        lat = enc
    return (lat * vsc).to(device=device, dtype=dtype)


def _update_mask_latents(s, mask, img_t, img_latent, vae, vae_dev, vae_dt,
                          dev, dtype, vsc, width, height, torch):
    """Cache mask latents for 9-channel inpaint UNet; only recompute on change."""
    import torch.nn.functional as F

    import numpy as _np
    mask_pil = mask if mask is not None else None
    if mask_pil is not None:
        if _np.count_nonzero(_np.asarray(mask_pil.convert("L")) > 10) < width * height * 0.005:
            mask_pil = None

    mask_bytes = mask_pil.convert("L").tobytes() if mask_pil is not None else b""
    lh, lw = s["lh"], s["lw"]
    cached_ml = s.get("mask_latents")
    if (s.get("_mask_bytes") == mask_bytes
            and cached_ml is not None
            and cached_ml.shape[2] == lh
            and cached_ml.shape[3] == lw):
        return

    if mask_pil is None:
        mask_t = torch.ones((1, 1, height, width), device=vae_dev, dtype=vae_dt)
    else:
        arr = __import__("numpy").array(
            mask_pil.convert("L").resize((width, height), Image.NEAREST),
            dtype="float32"
        ) / 255.0
        mask_t = torch.from_numpy(arr).to(device=vae_dev, dtype=vae_dt).unsqueeze(0).unsqueeze(0)

    masked_img = img_t * (1.0 - mask_t)
    masked_enc = vae.encode(masked_img)
    if hasattr(masked_enc, "latent_dist"):
        masked_lat = masked_enc.latent_dist.mode()
    elif hasattr(masked_enc, "latents"):
        masked_lat = masked_enc.latents
    else:
        masked_lat = masked_enc[0] if isinstance(masked_enc, (tuple, list)) else masked_enc

    bsz = s["batch_size"]
    mask_down = F.interpolate(mask_t.to(dtype=dtype), size=(lh, lw), mode="nearest")

    s["mask_latents"] = mask_down.expand(bsz, 1, lh, lw).to(device=dev, dtype=dtype)
    s["masked_img_latents"] = (masked_lat * vsc).expand(bsz, -1, lh, lw).to(device=dev, dtype=dtype)
    s["_mask_bytes"] = mask_bytes


def _frame_embeds(s, lerp_t, fbsz, torch):
    """Build (prompt_embeds, added_cond) for this frame, with optional A→B lerp."""
    pe = s["prompt_embeds"]
    ac = s.get("added_cond") or {}
    pb = s.get("pb_embeds")
    if pb is not None and 0.0 < lerp_t < 1.0:
        base = s["prompt_embeds_base"]           # (1, seq, dim)
        lerped = (1.0 - lerp_t) * base + lerp_t * pb
        batch_sz = s["batch_size"]
        lerped_b = lerped.repeat(batch_sz, 1, 1)
        if s["do_full_cfg"] and s.get("neg_embeds_base") is not None:
            neg = s["neg_embeds_base"].repeat(batch_sz, 1, 1)
            pe = torch.cat([neg, lerped_b], dim=0)
        else:
            pe = lerped_b
        # TODO: lerp pooled for added_cond
    return pe, ac


def _predict_x0(x_t, pe, ac, s, torch,
                control_cond=None, cn_scale: float = 1.0,
                cn_start: float = 0.0, cn_end: float = 1.0):
    """
    Core StreamDiffusion forward.

    1. Concatenate x_t with latent_buffer (if multi-step).
    2. (Optional) Run ControlNet forward → inject residuals into UNet.
    3. Batch UNet forward.
    4. Apply RCFG or full CFG.
    5. LCM scheduler step → x0 batch.
    6. Update latent_buffer.
    7. Return final x0 (1, ch, lh, lw).
    """
    n = s["n_steps"]
    lb = s["latent_buffer"]

    if n > 1 and isinstance(lb, torch.Tensor):
        x_in = torch.cat([x_t, lb], dim=0)
    else:
        x_in = x_t

    bsz = int(s["sub_t"].shape[0])
    x_in = _match_batch(x_in, bsz, torch)

    model_in = x_in
    sub_t = s["sub_t"]

    if s["do_full_cfg"]:
        model_in = torch.cat([x_in, x_in], dim=0)
        sub_t = torch.cat([sub_t, sub_t], dim=0)

    if s["is_inpaint"]:
        ml = s.get("mask_latents")
        mil = s.get("masked_img_latents")
        if isinstance(ml, torch.Tensor) and isinstance(mil, torch.Tensor):
            ml2 = _match_batch(ml, model_in.shape[0], torch)
            mil2 = _match_batch(mil, model_in.shape[0], torch)
            # Guard: spatial dims must match model_in (h, w)
            if ml2.shape[2:] != model_in.shape[2:]:
                import torch.nn.functional as F
                target = model_in.shape[2:]
                logger.warning(
                    "mask_latents spatial mismatch %s vs model_in %s — resizing",
                    tuple(ml2.shape[2:]), tuple(target),
                )
                ml2 = F.interpolate(ml2.float(), size=target, mode="nearest").to(model_in.dtype)
                mil2 = F.interpolate(mil2.float(), size=target, mode="nearest").to(model_in.dtype)
            if s["do_full_cfg"]:
                ml2 = torch.cat([ml2, ml2], dim=0)
                mil2 = torch.cat([mil2, mil2], dim=0)
            model_in = torch.cat([model_in, ml2, mil2], dim=1)

    pe_in = _match_batch(pe, model_in.shape[0], torch)
    kwargs: dict[str, Any] = {
        "encoder_hidden_states": pe_in,
        "return_dict": False,
    }
    if ac:
        kwargs["added_cond_kwargs"] = {
            k: _match_batch(v, model_in.shape[0], torch) if isinstance(v, torch.Tensor) else v
            for k, v in ac.items()
        }

    t_in = sub_t
    if t_in.shape[0] != model_in.shape[0]:
        t_in = _match_batch(t_in, model_in.shape[0], torch)

    # ── ControlNet injection ───────────────────────────────────────
    cn_model = s.get("cn_model")
    if cn_model is not None and control_cond is not None:
        try:
            # Guard: SD1.5 ControlNets are architecturally incompatible with SDXL UNets.
            # The text cross-attn dim differs (768 vs 2048) AND the mid-block spatial
            # resolution differs by 2× (SD1.5 downsamples 8× more than SDXL internally).
            # Both mismatches cause crashes; there is no in-place workaround.
            _cn_cfg = getattr(cn_model, "config", None)
            cn_cross_dim: int | None = getattr(_cn_cfg, "cross_attention_dim", None)
            pe_dim: int = s["prompt_embeds_base"].shape[-1]
            if cn_cross_dim is not None and cn_cross_dim != pe_dim:
                if not s.get("_cn_arch_warned"):
                    s["_cn_arch_warned"] = True
                    logger.warning(
                        "ControlNet SKIPPED: SD1.5 CN (cross_attention_dim=%d) is not compatible "
                        "with SDXL UNet (dim=%d). Both text dims and spatial mid-block feature "
                        "maps are mismatched. Use SDXL ControlNets: "
                        "diffusers/controlnet-canny-sdxl-1.0, diffusers/controlnet-depth-sdxl-1.0",
                        cn_cross_dim, pe_dim,
                    )
            else:
                # For 9-channel inpaint UNet, ControlNet expects only the 4-channel noisy latent.
                # Use x_in (before CFG doubling) so CN sees one sample per timestep.
                cn_sample = x_in[:, :4] if s["is_inpaint"] else x_in
                cn_bsz = cn_sample.shape[0]
                cn_cond = control_cond.expand(cn_bsz, -1, -1, -1).to(cn_sample.dtype)
                # Slice embeddings to match CN batch size (avoid doubled CFG batch)
                cn_pe = s["prompt_embeds_base"].repeat(cn_bsz, 1, 1).to(s["device"])
                cn_t = sub_t[:cn_bsz]

                cn_kwargs: dict[str, Any] = {
                    "encoder_hidden_states": cn_pe,
                    "controlnet_cond": cn_cond,
                    "conditioning_scale": float(cn_scale),
                    "return_dict": False,
                }
                # SDXL ControlNet requires text_embeds + time_ids in added_cond_kwargs.
                # SD1.5 ControlNets do NOT accept this — check addition_embed_type.
                cn_is_sdxl = getattr(_cn_cfg, "addition_embed_type", None) is not None
                if cn_is_sdxl and ac and "text_embeds" in ac:
                    cn_added = {
                        k: _match_batch(v, cn_bsz, torch) if isinstance(v, torch.Tensor) else v
                        for k, v in ac.items()
                    }
                    cn_kwargs["added_cond_kwargs"] = cn_added
                down_res, mid_res = cn_model(cn_sample, cn_t, **cn_kwargs)
                if down_res is not None:
                    # Expand residuals to full CFG batch if needed
                    res_bsz = model_in.shape[0]
                    expanded = [_match_batch(r, res_bsz, torch).to(model_in.dtype) for r in down_res]
                    kwargs["down_block_additional_residuals"] = expanded
                    kwargs["mid_block_additional_residual"] = _match_batch(
                        mid_res, res_bsz, torch
                    ).to(model_in.dtype)
        except Exception as _cn_exc:
            import traceback as _tb
            logger.warning("ControlNet forward failed (skipping): %s\n%s",
                           _cn_exc, _tb.format_exc())

    unet = s["unet"]
    pred = unet(model_in, t_in, **kwargs)[0]

    # ── guidance ──────────────────────────────────────────────────
    if s["do_full_cfg"]:
        uncond, cond = pred.chunk(2)
        pred = uncond + s["cfg_scale"] * (cond - uncond)
    elif s["cfg_type"] in {"self", "initialize"} and s["cfg_scale"] > 1.0:
        bsz_x = x_in.shape[0]
        stock = s["stock_noise"][:bsz_x] * s["rcfg_delta"]
        pred_guided = stock + s["cfg_scale"] * (pred - stock)
        # Update stock_noise with the CURRENT UNet prediction so the next
        # frame's "stock" is a real previous prediction, not random noise.
        # Rolling insert: new pred[:fbsz] at front, drop last fbsz elements.
        fbsz = s["fbsz"]
        s["stock_noise"] = torch.cat(
            [pred[:fbsz].detach().clone(), s["stock_noise"][:-fbsz]], dim=0
        )
        pred = pred_guided

    # ── LCM step: pred → x0 ──────────────────────────────────────
    # Handles both epsilon prediction (most models) and v-prediction (SD2/some fine-tunes).
    if s.get("prediction_type") == "v_prediction":
        # x0 = alpha * x_t - sigma * v_pred  (EDM/v-param formula)
        f_theta = s["alpha"] * x_in - s["beta"] * pred
    else:
        # x0 = (x_t - sigma * eps_pred) / alpha  (epsilon prediction)
        f_theta = (x_in - s["beta"] * pred) / s["alpha"]
    x0_batch = s["c_out"] * f_theta + s["c_skip"] * x_in

    if n > 1:
        out = x0_batch[-1:].contiguous()
        s["latent_buffer"] = (
            s["alpha"][1:] * x0_batch[:-1]
            + s["beta"][1:] * s["init_noise"][1:]
        )
    else:
        out = x0_batch
        s["latent_buffer"] = None

    return out


def _match_batch(t, bsz, torch):
    if not isinstance(t, torch.Tensor) or t.shape[0] == bsz:
        return t
    if t.shape[0] > bsz:
        return t[:bsz].contiguous()
    tail = t[-1:].expand(bsz - t.shape[0], *t.shape[1:])
    return torch.cat([t, tail], dim=0).contiguous()


def _pil_to_tensor(image: Image.Image, height, width, device, dtype):
    import torch
    import numpy as np
    arr = np.array(image.convert("RGB").resize((width, height), Image.LANCZOS),
                   dtype="float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


# ──────────────────────────────────────────────────────────────────
# SessionManager
# ──────────────────────────────────────────────────────────────────

_REBUILD_DEBOUNCE_S = 0.35  # seconds to wait for settings to stabilise before rebuilding


def _log_sig_diff(old_sig: str | None, new_sig: str, log: Any, label: str) -> None:
    """Log which fields changed between two session signatures to aid debugging."""
    if old_sig is None:
        log.info("%s: first build", label)
        return
    try:
        old_t = eval(old_sig)  # noqa: S307 — trusted internal repr() string
        new_t = eval(new_sig)  # noqa: S307
        fields = ("model_path", "device", "prompt", "negative_prompt",
                  "timestep_indices", "frame_buffer_size", "cfg_type", "cfg",
                  "dims", "lora_paths", "prompt_b")
        diffs = [fields[i] for i, (a, b) in enumerate(zip(old_t, new_t)) if a != b]
        log.info("%s: sig changed — fields: %s", label, diffs or "<unknown>")
    except Exception:
        log.info("%s: sig changed (diff unavailable)", label)


class SessionManager:
    """
    Caches one StreamSession per settings signature.
    Rebuilds the pipeline only when model_path / device / LoRA changes.
    Debounces setting changes so that typing a prompt doesn't trigger one rebuild
    per keystroke — only rebuilds after settings have been stable for 350ms.
    """

    def __init__(self) -> None:
        self._session: StreamSession | None = None
        self._sig: str | None = None
        self._pipe: Any = None
        self._pipe_sig: str | None = None
        self.last_output: Any = None  # last PIL image — carried across reconnections
        self._generation: int = 0     # incremented on each session rebuild
        self._build_lock = threading.Lock()  # serialise session builds
        # Debounce: track the pending sig and when it first appeared
        self._pending_sig: str | None = None
        self._sig_changed_at: float = 0.0

    def get_session(self, settings: dict[str, Any]) -> StreamSession | None:
        sig = _session_sig(settings)
        # Fast path — no lock needed for a cache hit.
        if self._session is not None and self._sig == sig:
            self._pending_sig = None
            return self._session

        # Debounce: only rebuild after settings have been stable for _REBUILD_DEBOUNCE_S.
        # This prevents per-keystroke rebuilds when the user types a prompt.
        now = time.monotonic()
        if self._pending_sig != sig:
            self._pending_sig = sig
            self._sig_changed_at = now
            return self._session  # return old session while waiting for stable settings
        if now - self._sig_changed_at < _REBUILD_DEBOUNCE_S:
            return self._session  # still in debounce window

        with self._build_lock:
            # Re-check after acquiring the lock (another thread may have built it).
            if self._session is not None and self._sig == sig:
                self._pending_sig = None
                return self._session

            # Log what actually changed to help diagnose spurious rebuilds.
            _log_sig_diff(self._sig, sig, logger, "StreamSession")
            logger.info("StreamSession: rebuilding")
            pipe = self._ensure_pipe(settings)
            if pipe is None:
                return None
            try:
                w, h = _clamp_dims(
                    int(settings.get("width", 512)),
                    int(settings.get("height", 512)),
                )
                t_idx = list(settings.get("stream_timestep_indices") or [32, 45])
                device = settings.get("device") or _default_device()
                cn_model = _load_cn_for_streaming(settings, device, pipe)
                session = StreamSession(
                    pipe=pipe,
                    device=device,
                    prompt=settings.get("prompt", ""),
                    negative_prompt=settings.get("negative_prompt", ""),
                    t_indices=t_idx,
                    frame_buffer_size=int(settings.get("stream_frame_buffer_size", 1)),
                    cfg_type=settings.get("stream_cfg_type", "self"),
                    cfg_scale=float(settings.get("cfg", 1.0)),
                    width=w,
                    height=h,
                    use_tiny_vae=os.getenv("RTD_STREAM_TINY_VAE", "1").strip() not in ("0", "false", "no"),
                    prompt_b=settings.get("prompt_b", ""),
                    rcfg_delta=float(os.getenv("RTD_STREAM_RCFG_DELTA", "1.0")),
                    controlnet=cn_model,
                )
                # Always warm up at least len(t_idx) passes to fill the latent buffer.
                # Zero-initialised latent_buffer → first frames are garbage without this.
                # RTD_STREAM_WARMUP=0 skips only the *extra* quality passes.
                min_warmup = len(t_idx)
                extra_warmup = (
                    len(t_idx) * 2
                    if os.getenv("RTD_STREAM_WARMUP", "1").strip() not in ("0", "false", "no")
                    else 0
                )
                session.warmup(min_warmup + extra_warmup)
                self._session = session
                self._sig = sig
                self._pending_sig = None
                self._generation += 1
                logger.info(
                    "StreamSession ready at %dx%d (t=%s, gen=%d)",
                    w, h, t_idx, self._generation,
                )
                return session
            except Exception:
                logger.exception("StreamSession build failed")
                self._session = None
                self._sig = None
                # Reset debounce timer so the next failure waits another full window
                # before retrying, instead of hammering the GPU on every inference tick.
                self._sig_changed_at = time.monotonic()
                return None

    def _ensure_pipe(self, settings: dict[str, Any]) -> Any | None:
        model_path = settings.get("model_path") or _env_model_path()
        device = settings.get("device") or _default_device()
        loras = tuple(sorted(settings.get("lora_paths") or []))
        psig = repr((model_path, device, loras))
        if self._pipe is not None and self._pipe_sig == psig:
            return self._pipe
        pipe = _load_pipe(model_path, device, list(loras))
        if pipe is None:
            return None
        self._pipe = pipe
        self._pipe_sig = psig
        return pipe

    def reset(self) -> None:
        self._session = None
        self._sig = None


def _session_sig(s: dict[str, Any]) -> str:
    cn_model_id = _extract_cn_model_id(s)
    return repr((
        s.get("model_path") or _env_model_path(),
        s.get("device") or _default_device(),
        s.get("prompt", ""),
        s.get("negative_prompt", ""),
        tuple(s.get("stream_timestep_indices") or [32, 45]),
        int(s.get("stream_frame_buffer_size", 1)),
        s.get("stream_cfg_type", "self"),
        round(float(s.get("cfg", 1.0)), 3),
        _clamp_dims(int(s.get("width", 512)), int(s.get("height", 512))),
        tuple(sorted(s.get("lora_paths") or [])),
        s.get("prompt_b", ""),
        cn_model_id,
    ))


def _extract_cn_model_id(settings: dict[str, Any]) -> str:
    """Return a stable string identifying the ControlNet model (or '' if none)."""
    for cond in (settings.get("layer_conditions") or []):
        model = cond.get("controlnet_model", "")
        if model:
            path = cond.get("controlnet_model_path", "")
            return path or model  # use explicit path if given, else short name
    return ""


def _load_cn_for_streaming(settings: dict[str, Any], device: str, pipe: Any) -> Any | None:
    """Load the first active ControlNet model from layer_conditions, or None."""
    import torch
    from . import controlnet as _cn_mod

    for cond in (settings.get("layer_conditions") or []):
        short = cond.get("controlnet_model", "")
        if not short:
            continue
        explicit_path = cond.get("controlnet_model_path", "")
        is_sdxl = hasattr(pipe, "text_encoder_2")
        try:
            dtype = next(pipe.unet.parameters()).dtype
            model_id = _cn_mod.resolve_model_id(short, explicit_path or None, is_sdxl)
            logger.info("StreamSession: loading ControlNet %s for streaming", model_id)
            cn = _cn_mod.load_controlnet(model_id, device, dtype)
            logger.info("StreamSession: ControlNet ready (%s)", short)
            return cn
        except Exception as exc:
            logger.error("StreamSession: ControlNet load failed (%s): %s", short, exc)
    return None


def _clamp_dims(w: int, h: int) -> tuple[int, int]:
    """Clamp to max 768px on longest side, align to 64."""
    max_side = int(os.getenv("RTD_STREAM_MAX_SIDE", "768"))
    scale = min(1.0, max_side / max(w, h, 1))
    w2 = max(64, int(w * scale) // 64 * 64)
    h2 = max(64, int(h * scale) // 64 * 64)
    return w2, h2


def _default_device() -> str:
    v = os.getenv("RTD_DEVICE", "cuda").strip().lower()
    return "cuda:0" if v == "cuda" else v


def _env_model_path() -> str:
    return os.getenv("RTD_MODEL_ID") or os.getenv("RTD_MODEL_PATH") or ""


def _load_pipe(model_path: str | None, device: str, lora_paths: list[str]) -> Any | None:
    """Load a Diffusers pipeline suitable for StreamDiffusion."""
    import torch

    if not model_path:
        try:
            from .assets import default_model_path as dmp
            model_path = dmp()
        except Exception:
            pass
    if not model_path:
        logger.error("StreamSession: no model configured (set RTD_MODEL_PATH)")
        return None

    is_cu = _is_cuda(device)
    dtype = torch.float16 if is_cu else torch.float32
    path = Path(model_path)

    # navigate up from component files to the diffusers dir
    if path.is_file():
        for p in path.parents:
            if (p / "model_index.json").is_file():
                logger.info("StreamSession: resolved component → diffusers dir %s", p)
                model_path, path = str(p), p
                break

    logger.info("StreamSession: loading pipe %s on %s", model_path, device)
    t0 = time.perf_counter()

    try:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting

        kw: dict[str, Any] = {"torch_dtype": dtype}
        if path.is_dir():
            kw["local_files_only"] = True
            kw["low_cpu_mem_usage"] = False

        if path.is_file():
            pipe = _load_single_file(path, dtype)
        else:
            # prefer inpaint pipeline (9-ch UNet handles inpainting natively)
            # but fall back to img2img which works fine with compositing
            try:
                pipe = AutoPipelineForInpainting.from_pretrained(model_path, **kw)
            except Exception:
                pipe = AutoPipelineForImage2Image.from_pretrained(model_path, **kw)

        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        if is_cu:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # channels_last for better CUDA kernel utilization
        try:
            pipe.unet.to(memory_format=torch.channels_last)
        except Exception:
            pass

        # xformers memory-efficient attention — big speedup on SDXL
        xformers_ok = False
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                xformers_ok = True
                logger.info("StreamSession: xformers memory-efficient attention enabled")
            except Exception as e:
                logger.info("StreamSession: xformers not available (%s)", e)

        for lp in lora_paths:
            try:
                pipe.load_lora_weights(lp, adapter_name=Path(lp).stem)
                logger.info("StreamSession: LoRA %s loaded", lp)
            except Exception:
                logger.warning("StreamSession: LoRA load failed: %s", lp)

        # Auto-inject LCM LoRA for SDXL models that aren't already fast
        _maybe_inject_lcm(pipe, model_path, lora_paths)

        # torch.compile — requires triton (available in Docker image)
        if os.getenv("RTD_STREAM_COMPILE", "0").strip() not in ("0", "false", "no", ""):
            _compile_pipe(pipe)

        elapsed = time.perf_counter() - t0
        logger.info("StreamSession: pipe ready in %.1fs (xformers=%s)", elapsed, xformers_ok)
        return pipe

    except Exception:
        logger.exception("StreamSession: pipe load failed for %s", model_path)
        gc.collect()
        try:
            if is_cu:
                torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def _compile_pipe(pipe: Any) -> None:
    """Apply torch.compile to UNet (and optionally VAE) for triton-accelerated inference."""
    import torch
    compile_mode = os.getenv("RTD_STREAM_COMPILE_MODE", "reduce-overhead")
    t0 = time.perf_counter()
    try:
        pipe.unet = torch.compile(pipe.unet, mode=compile_mode, fullgraph=False)
        logger.info("StreamSession: torch.compile applied to UNet (mode=%s)", compile_mode)
    except Exception as e:
        logger.warning("StreamSession: torch.compile failed: %s", e)
        return
    # Optionally compile the VAE decoder for extra speed
    if os.getenv("RTD_STREAM_COMPILE_VAE", "0").strip() not in ("0", "false", "no", ""):
        try:
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "decoder"):
                pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode=compile_mode, fullgraph=False)
                logger.info("StreamSession: torch.compile applied to VAE decoder")
        except Exception:
            pass
    logger.info("StreamSession: torch.compile setup in %.1fs", time.perf_counter() - t0)


def _load_single_file(path: Path, dtype: Any) -> Any:
    name = path.name.lower()
    # Try inpainting pipelines first for models with "inpaint" in name
    if "inpaint" in name:
        for cls_name in ("StableDiffusionXLInpaintPipeline", "StableDiffusionInpaintPipeline"):
            try:
                import diffusers
                cls = getattr(diffusers, cls_name)
                return cls.from_single_file(str(path), torch_dtype=dtype)
            except Exception:
                continue
    # Standard img2img pipelines: SDXL first, then SD1.5
    for cls_name in ("StableDiffusionXLImg2ImgPipeline", "StableDiffusionImg2ImgPipeline"):
        try:
            import diffusers
            cls = getattr(diffusers, cls_name)
            return cls.from_single_file(str(path), torch_dtype=dtype)
        except Exception:
            continue
    raise RuntimeError(f"Could not load {path} with any known pipeline class")


def _maybe_inject_lcm(pipe: Any, model_path: str, lora_paths: list[str] | None = None) -> None:
    """Auto-inject LCM LoRA for SDXL models without built-in speed."""
    if os.getenv("RTD_STREAM_LCM_LORA", "1").strip() in ("0", "false", "no"):
        return
    nm = model_path.replace("\\", "/").lower()
    if any(m in nm for m in ("turbo", "lightning", "lcm", "hyper", "distill")):
        logger.info("StreamSession: LCM LoRA skip (model name implies fast): %s", Path(model_path).name)
        return
    if not hasattr(pipe, "text_encoder_2"):
        return  # not SDXL
    # Skip if the user already has an LCM/latent-consistency LoRA in their lora_paths.
    # Check the full path (directory + stem) so "lcm-lora-sdxl/pytorch_lora_weights.safetensors" is caught.
    if lora_paths:
        for lp in lora_paths:
            lp_lower = lp.replace("\\", "/").lower()
            if any(m in lp_lower for m in ("lcm", "latent_consistency", "latent-consistency")):
                # Also activate the user's LoRA adapter so it's actually used.
                adapter_name = Path(lp).stem
                try:
                    pipe.set_adapters([adapter_name], [1.0])
                except Exception:
                    pass
                logger.info("StreamSession: LCM LoRA skip (user LoRA path contains %r)", adapter_name)
                return
    try:
        is_xl = hasattr(pipe, "text_encoder_2")
        lora_id = "latent-consistency/lcm-lora-sdxl" if is_xl else "latent-consistency/lcm-lora-sdv1-5"
        logger.info("StreamSession: injecting LCM LoRA %s …", lora_id)
        pipe.load_lora_weights(lora_id, adapter_name="_lcm")
        pipe.set_adapters(["_lcm"], [1.0])
        logger.info("StreamSession: LCM LoRA injected (%s)", lora_id)
    except Exception as e:
        logger.info("StreamSession: LCM LoRA injection skipped (%s): %s", Path(model_path).name, e)
