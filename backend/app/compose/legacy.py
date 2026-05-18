"""Translate legacy `LayerCondition` payloads into the new `Layer` model.

The frontend still emits the legacy schema (Step 8 changes that). This
adapter lets the new composer accept today's payloads without UI changes.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from ..inference.session import ControlNetSpec
from ..schemas import LayerCondition
from .layer import AutoTagSpec, Layer


def _decode_data_url_rgba(data_url: str) -> Image.Image | None:
    if not data_url:
        return None
    try:
        _, sep, payload = str(data_url).partition(",")
        raw = base64.b64decode(payload if sep else str(data_url))
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        return None


def layer_condition_to_layer(cond: LayerCondition | dict[str, Any]) -> Layer:
    """Convert one `LayerCondition` (Pydantic model or dict) into a `Layer`.

    The legacy `mode` field is preserved on the returned Layer as the private
    `_legacy_mode` attribute so the composer can dispatch denoise blending
    identically to `resolve_conditioning_mask`.
    """
    g = (lambda k, d=None: cond.get(k, d)) if isinstance(cond, dict) else (lambda k, d=None: getattr(cond, k, d))

    layer_id = str(g("layer_id", "") or "")
    color = _decode_data_url_rgba(str(g("image", "") or ""))
    cond_image_url = g("controlnet_image", None)
    cond_image = _decode_data_url_rgba(str(cond_image_url)) if cond_image_url else None

    cn_model = g("controlnet_model", None)
    cn_list: list[ControlNetSpec] = []
    if cn_model:
        cn_list.append(
            ControlNetSpec(
                model_id=str(cn_model),
                model_path=g("controlnet_model_path", None),
                scale=float(g("controlnet_scale", 1.0) or 0.0),
                start=float(g("controlnet_start_at", 0.0) or 0.0),
                end=float(g("controlnet_end_at", 1.0) or 0.0),
            )
        )

    auto_tag = None
    if bool(g("auto_tag", False)):
        auto_tag = AutoTagSpec(
            threshold=float(g("auto_tag_threshold", 0.35) or 0.35),
            refresh_frames=int(g("auto_tag_refresh_frames", 5) or 5),
        )

    mode_raw = str(g("mode", "mask") or "mask").lower().strip()
    denoise_val = g("denoise", None)

    layer = Layer(
        layer_id=layer_id,
        role="cond_only" if mode_raw == "prompt_mix" and not cn_list else "color",
        color=color,
        cond_image=cond_image,
        denoise_strength=float(denoise_val) if denoise_val is not None else 1.0,
        mask_color=None,  # follow color alpha by default
        mask_cond=None,
        mask_denoise=None,
        blend="normal",
        schedule=(
            float(g("schedule_start", 0.0) or 0.0),
            float(g("schedule_end", 1.0) or 1.0),
        ),
        weight=float(g("weight", 1.0) or 0.0),
        cn=cn_list,
        prompt_fragment=str(g("prompt", "") or "").strip() or None,
        negative_fragment=str(g("negative_prompt", "") or "").strip() or None,
        auto_tag=auto_tag,
    )
    # Preserve the legacy operator so denoise compositing stays byte-equal.
    object.__setattr__(layer, "_legacy_mode", mode_raw)
    return layer


def layer_conditions_to_layers(conditions: list[LayerCondition | dict[str, Any]]) -> list[Layer]:
    return [layer_condition_to_layer(c) for c in conditions]
