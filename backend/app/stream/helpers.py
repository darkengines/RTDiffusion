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


def _triton_available() -> bool:
    """Triton is not packaged for Windows; without it torch.compile fails at
    first forward pass with a runtime error. Detect once at import."""
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


_TRITON_OK = _triton_available()
if not _TRITON_OK:
    logger.info("StreamSession: Triton not available — torch.compile disabled.")


def auto_t_indices(quality: float, n_sched: int = 50) -> list[int]:
    """Map a 0..1 quality slider to StreamDiffusion timestep indices.

    quality=0.0 → 2 steps (fastest, sloppy); quality=1.0 → 8 steps (slowest,
    cleanest). Indices are spread evenly across the scheduler's range so the
    first step gets the noisiest latent (most creative freedom) and the last
    step gets the cleanest (closest to input). This matches the paper's
    "early noise, late refinement" intuition.

    The default `[0, 16, 32, 45]` corresponds roughly to quality≈0.4.
    """
    q = max(0.0, min(1.0, float(quality)))
    k = max(2, round(2 + q * 6))
    span = max(1, n_sched - 1)
    return [round(i * span / (k - 1)) for i in range(k)]


def resolve_t_indices(settings: dict, n_sched: int = 50) -> list[int]:
    """Pick t_indices honoring `stream_quality` override.

    Centralised so every call site (session build, signature, prebuild) agrees.
    """
    q = settings.get("stream_quality")
    if q is not None:
        return auto_t_indices(float(q), n_sched=n_sched)
    raw = settings.get("stream_timestep_indices")
    if raw:
        return list(raw)
    return [0, 16, 32, 45]


def _is_cuda(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


# ──────────────────────────────────────────────────────────────────
# StreamSession
# ──────────────────────────────────────────────────────────────────
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


def _te_device(pipe: Any) -> Any:
    """Return the actual device that the pipe's text encoder lives on.

    from_single_file can leave text encoders on a different CUDA device than
    the UNet/VAE.  We detect the real device so encode_prompt can move
    text_input_ids to the encoder's device rather than assuming the target
    device.  Results are moved to `device` by _encode_prompt after the call.
    """
    for _attr in ("text_encoder", "text_encoder_2"):
        _te = getattr(pipe, _attr, None)
        if _te is not None:
            try:
                return next(_te.parameters()).device
            except StopIteration:
                pass
    return None


def _encode_prompt(pipe, prompt, negative_prompt, do_cfg, device, dtype):
    """Returns (prompt_embeds, neg_embeds, pooled, neg_pooled) — each (1, seq, dim) or None."""
    # Pass the encoder's actual device to encode_prompt so text_input_ids land
    # on the same device as the model weights.  The results are moved to the
    # UNet device (`device`) below, so downstream code is unaffected.
    encode_device = _te_device(pipe) or device
    encoded = pipe.encode_prompt(
        prompt=prompt,
        device=encode_device,
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
    if (s.get("_mask_bytes") != mask_bytes
            or cached_ml is None
            or cached_ml.shape[2] != lh
            or cached_ml.shape[3] != lw):
        mask_down = F.interpolate(mask_t.to(dtype=dtype), size=(lh, lw), mode="nearest")
        s["mask_latents"] = mask_down.expand(bsz, 1, lh, lw).to(device=dev, dtype=dtype)
        s["_mask_bytes"] = mask_bytes

    s["masked_img_latents"] = (masked_lat * vsc).expand(bsz, -1, lh, lw).to(device=dev, dtype=dtype)


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

