"""StreamSession — one continuous StreamDiffusion inpainting session."""
from __future__ import annotations

import gc
import hashlib
import logging
import threading
import time
import os
from typing import Any

from PIL import Image

from .helpers import (
    _TINY_VAE_SD15,
    _TINY_VAE_SDXL,
    _TRITON_OK,
    _build_added_cond,
    _compute_scalings,
    _encode_prompt,
    _frame_embeds,
    _is_cuda,
    _load_vae,
    _match_batch,
    _predict_x0,
    _update_mask_latents,
    _vae_encode,
    auto_t_indices,
    logger,
    resolve_t_indices,
)


def _denoise_map_fingerprint(mask: Image.Image) -> bytes:
    arr = mask.convert("L")
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"{arr.width}x{arr.height}".encode("ascii"))
    digest.update(arr.tobytes())
    return digest.digest()


def _apply_denoise_output_blend(
    output: Image.Image,
    image: Image.Image,
    denoise_map: Image.Image | None,
    width: int,
    height: int,
) -> Image.Image:
    if denoise_map is None:
        return output
    denoise_blend = denoise_map.convert("L").resize((width, height), Image.Resampling.BILINEAR)
    input_rgb = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    return Image.composite(output, input_rgb, denoise_blend)


def _pil_rgb_to_tensor_gpu_resize(image: Image.Image, height: int, width: int, device: Any, dtype: Any, torch: Any):
    import numpy as _np

    arr = _np.asarray(image.convert("RGB"), dtype="float32") / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    if tensor.shape[-2:] != (height, width):
        tensor = torch.nn.functional.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.clamp(-1.0, 1.0)


def _pil_l_to_tensor_gpu_resize(image: Image.Image, height: int, width: int, device: Any, dtype: Any, torch: Any, *, mode: str = "bilinear"):
    import numpy as _np

    arr = _np.asarray(image.convert("L"), dtype="float32") / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
    if tensor.shape[-2:] != (height, width):
        if mode == "nearest":
            tensor = torch.nn.functional.interpolate(tensor, size=(height, width), mode="nearest")
        else:
            tensor = torch.nn.functional.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.clamp(0.0, 1.0)


def _tensor01_to_pil(image_tensor: Any) -> Image.Image:
    array = image_tensor[0].detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype("uint8"), mode="RGB")


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
        seed: int = 42,
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
        seed = int(seed) % (2**32)
        self._noise_seed = seed
        g = torch.Generator(device=device_t).manual_seed(seed)
        init_noise = torch.randn((batch_size, lat_ch, lh, lw),
                     generator=g, device=device_t, dtype=dtype)
        stock_noise = torch.randn(init_noise.shape, generator=g, device=device_t, dtype=dtype)

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
        self._last_override_prompt: tuple[str, bool] | None = None
        self._last_override_embeds: Any = None
        self._last_override_added: Any = None

        # Noise drift delta ring buffer — pre-generated at init to avoid
        # torch.randn_like() allocation in the hot inference path every frame.
        if self._noise_drift > 0:
            _ring_size = 32
            _g_ring = torch.Generator(device=device_t).manual_seed((seed + 137) % (2**32))
            self._noise_ring: Any = torch.stack([
                torch.randn((batch_size, lat_ch, lh, lw),
                            generator=_g_ring, device=device_t, dtype=dtype)
                for _ in range(_ring_size)
            ])
            self._noise_ring_idx: int = 0
        else:
            self._noise_ring: Any = None
            self._noise_ring_idx: int = 0

        # Denoise map tensor cache — avoids repeated PIL resize + GPU interpolate
        # when the denoise map is unchanged (common during static scenes).
        self._cached_denoise_hash: bytes | None = None
        self._cached_denoise_lat: Any = None

    def reset_noise(self, seed: int) -> None:
        """Rotate StreamDiffusion's persistent noise without rebuilding the pipeline."""
        seed = int(seed) % (2**32)
        if seed == getattr(self, "_noise_seed", None):
            return
        import torch

        s = self._s
        device = s["device"]
        dtype = s["dtype"]
        generator = torch.Generator(device=device).manual_seed(seed)
        shape = tuple(s["init_noise"].shape)
        s["init_noise"] = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        s["stock_noise"] = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latent_buffer = s.get("latent_buffer")
        if latent_buffer is not None:
            s["latent_buffer"] = torch.zeros_like(latent_buffer)
        if self._noise_ring is not None:
            ring_shape = tuple(self._noise_ring.shape)
            ring_generator = torch.Generator(device=device).manual_seed((seed + 137) % (2**32))
            self._noise_ring = torch.randn(ring_shape, generator=ring_generator, device=device, dtype=dtype)
            self._noise_ring_idx = 0
        region_buffers = getattr(self, "_region_latent_buffers", None)
        if isinstance(region_buffers, dict):
            region_buffers.clear()
        self._noise_seed = seed

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
        composite_base: Image.Image | None = None,
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
                                     denoise, denoise_map, control_image, cn_scale, cn_start, cn_end,
                                     composite_base)
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
        composite_base: Image.Image | None = None,
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
                img_t = _pil_rgb_to_tensor_gpu_resize(image, self._height, self._width,
                                                       vae_dev, vae_dt, torch)

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
                if self._noise_ring is not None:
                    _delta = self._noise_ring[self._noise_ring_idx]
                    self._noise_ring_idx = (self._noise_ring_idx + 1) % self._noise_ring.shape[0]
                else:
                    _delta = torch.randn_like(s["init_noise"])
                _new = s["init_noise"] + _drift * _delta
                # Renormalise to unit variance so noise statistics stay stable.
                s["init_noise"] = _new / (_new.std() + 1e-8)

            _d = max(0.0, min(1.0, float(denoise)))
            denoise_lat: Any = None
            denoise_full: Any = None
            if denoise_map is not None:
                _dm_hash = _denoise_map_fingerprint(denoise_map)
                if _dm_hash != self._cached_denoise_hash:
                    den_t = _pil_l_to_tensor_gpu_resize(denoise_map, self._height, self._width, dev, dtype, torch)
                    self._cached_denoise_lat = torch.nn.functional.interpolate(
                        den_t,
                        size=img_latent.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0)
                    self._cached_denoise_full = den_t
                    self._cached_denoise_hash = _dm_hash
                denoise_lat = self._cached_denoise_lat
                denoise_full = getattr(self, "_cached_denoise_full", None)

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
            override_full_cfg = bool(s.get("override_full_cfg")) and prompt_override is not None and float(s.get("cfg_scale", 1.0)) > 1.0
            override_key = (prompt_override or "", override_full_cfg)
            if prompt_override is not None and override_key != self._last_override_prompt:
                # Re-encode only when the override prompt changes
                ov_e, ov_ne, ov_p, ov_np = _encode_prompt(
                    s["pipe"], prompt_override, "", override_full_cfg, dev, dtype
                )
                if ov_e is not None:
                    pos_embeds = ov_e.repeat(s["batch_size"], 1, 1)
                    if override_full_cfg and ov_ne is not None:
                        neg_embeds = ov_ne.repeat(s["batch_size"], 1, 1)
                        self._last_override_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)
                    else:
                        self._last_override_embeds = pos_embeds
                    pos_added = _build_added_cond(
                        s["pipe"], self._width, self._height, ov_p,
                        s["batch_size"], dtype, dev, torch
                    ) if ov_p is not None else s.get("added_cond") or {}
                    if override_full_cfg and ov_np is not None:
                        neg_added = _build_added_cond(
                            s["pipe"], self._width, self._height, ov_np,
                            s["batch_size"], dtype, dev, torch
                        )
                        self._last_override_added = {
                            key: torch.cat([neg_added[key], pos_added[key]], dim=0)
                            for key in pos_added if key in neg_added
                        }
                    else:
                        self._last_override_added = pos_added
                self._last_override_prompt = override_key

            if prompt_override is not None and self._last_override_embeds is not None:
                pe = self._last_override_embeds
                ac = self._last_override_added or {}
            else:
                pe, ac = _frame_embeds(s, prompt_lerp, fbsz, torch)

            # ── prepare ControlNet conditioning tensor ─────────────
            control_cond: Any = None
            if control_image is not None and s.get("cn_model") is not None:
                control_cond = (_pil_rgb_to_tensor_gpu_resize(control_image, self._height, self._width, dev, dtype, torch) + 1.0) / 2.0

            # ── StreamDiffusion forward ────────────────────────────
            previous_do_full_cfg = s.get("do_full_cfg")
            if override_full_cfg and prompt_override is not None:
                s["do_full_cfg"] = True
            try:
                x0 = _predict_x0(x_t, pe, ac, s, torch, control_cond=control_cond,
                                  cn_scale=cn_scale, cn_start=cn_start, cn_end=cn_end)
            except RuntimeError as _rte:
                if "already recording to mempool_id" in str(_rte) or "beginAllocateToPool" in str(_rte):
                    # CUDA graph recording conflict after a batch-size change (e.g. new
                    # t-indices). Reset dynamo so the next session rebuild starts clean.
                    try:
                        import torch._dynamo as _dynamo
                        _dynamo.reset()
                    except Exception:
                        pass
                raise
            finally:
                s["do_full_cfg"] = previous_do_full_cfg

            local_denoise = float(denoise_lat.max().item()) if denoise_lat is not None else _d
            if local_denoise <= 0.001:
                x0 = img_latent.expand_as(x0)

            # ── VAE decode ─────────────────────────────────────────
            decoded = vae.decode(x0 / vsc).sample        # (1, 3, H, W)
            decoded = decoded.float().clamp(-1, 1)
            output_t = ((decoded + 1) / 2).clamp(0.0, 1.0)
            input_t = ((img_t.to(device=dev, dtype=dtype) + 1.0) / 2.0).clamp(0.0, 1.0)
            if denoise_full is not None:
                output_t = output_t * denoise_full + input_t * (1.0 - denoise_full)

            # ── inpainting compositing ─────────────────────────────
            if mask is not None:
                mask_t = _pil_l_to_tensor_gpu_resize(mask, self._height, self._width, dev, dtype, torch, mode="nearest")
                if bool((mask_t > (10.0 / 255.0)).sum().item() >= self._width * self._height * 0.005):
                    # Use composite_base (previous inference output) when available so the
                    # non-masked area shows the stable previous result, not the grey canvas.
                    base_img = composite_base if composite_base is not None else image
                    base_t = ((_pil_rgb_to_tensor_gpu_resize(base_img, self._height, self._width, dev, dtype, torch) + 1.0) / 2.0).clamp(0.0, 1.0)
                    output_t = output_t * mask_t + base_t * (1.0 - mask_t)

            output = _tensor01_to_pil(output_t)

        return output

    def warmup(self, n: int = 1, image: Image.Image | None = None) -> None:
        blank = (image.convert("RGB").resize((self._width, self._height), Image.BILINEAR)
                 if image is not None else Image.new("RGB", (self._width, self._height), (128, 128, 128)))
        for _ in range(n):
            try:
                self.infer(blank)
            except Exception:
                logger.exception("Warmup step failed")
        logger.info(
            "StreamSession warmed up (%d step(s)) at %dx%d", n, self._width, self._height
        )

