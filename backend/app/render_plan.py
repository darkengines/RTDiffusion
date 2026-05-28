"""Wire-format -> RenderPlan decoder for the v2 layer system.

The frontend aggregates the layer stack into three flat buffers (rgba, cfg_map,
denoise_map) plus a list of per-prompt attention masks, then sends them as PNGs
(base64 inline or blob refs resolved through the session blob store). This
module decodes that payload into numpy arrays that engines consume directly.
No aggregation happens here -- the frontend is the source of truth.

See plan: ``C:\\Users\\root\\.claude\\plans\\optimized-growing-marble.md``.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image


CFG_HI = 30.0


@dataclass(frozen=True)
class Prompt:
    text: str
    negative: str
    mask: np.ndarray | None  # HxW float32 in [0, 1], None means whole canvas


@dataclass(frozen=True)
class ControlNet:
    layer_id: str
    model: str
    model_path: str | None
    scale: float
    start: float
    end: float
    preprocessor_params: dict[str, Any]


@dataclass(frozen=True)
class RenderPlan:
    rgba: np.ndarray          # HxWx4 uint8 -- aggregated by the frontend
    cfg_map: np.ndarray       # HxW float32 in [0, CFG_HI]
    denoise_map: np.ndarray   # HxW float32 in [0, 1]
    prompts: tuple[Prompt, ...]
    base_prompt: str
    base_negative_prompt: str
    base_denoise: float
    base_cfg: float
    width: int
    height: int
    controlnet: tuple[ControlNet, ...]


BlobResolver = Callable[[str], bytes | None]


def _decode_b64(s: str) -> bytes:
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def _resolve_image(
    b64: str | None,
    ref: str | None,
    blob_resolver: BlobResolver | None,
) -> Image.Image | None:
    if b64:
        return Image.open(io.BytesIO(_decode_b64(b64)))
    if ref and blob_resolver is not None:
        data = blob_resolver(ref)
        if data is not None:
            return Image.open(io.BytesIO(data))
    return None


def _to_rgba(img: Image.Image | None, width: int, height: int) -> np.ndarray:
    if img is None:
        return np.zeros((height, width, 4), dtype=np.uint8)
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.BILINEAR)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


def _to_float_map(
    img: Image.Image | None,
    width: int,
    height: int,
    scale: float,
    default: float,
) -> np.ndarray:
    if img is None:
        return np.full((height, width), float(default), dtype=np.float32)
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.BILINEAR)
    mode = img.mode
    if mode in ("I;16", "I;16B", "I;16L"):
        arr = np.asarray(img, dtype=np.float32) / 65535.0
    elif mode == "F":
        arr = np.asarray(img, dtype=np.float32)
    else:
        if mode != "L":
            img = img.convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(arr * scale, 0.0, scale)


def _to_mask(img: Image.Image | None, width: int, height: int) -> np.ndarray | None:
    if img is None:
        return None
    return _to_float_map(img, width, height, 1.0, 0.0)


def plan_from_wire(
    payload: Mapping[str, Any],
    blob_resolver: BlobResolver | None = None,
) -> RenderPlan:
    """Decode a wire payload into a RenderPlan.

    ``payload`` is the v2 portion of an InpaintFrame (see ``schemas.WirePrompt``,
    ``WireControlNet``, and the new ``rgba_*`` / ``cfg_map_*`` / ``denoise_map_*``
    / ``prompts`` / ``controlnet`` / ``base_*`` / ``width`` / ``height`` fields).
    ``blob_resolver(ref) -> bytes | None`` resolves binary blob references
    uploaded via ``POST /api/rtc/{pc_id}/blob``.
    """
    width = int(payload.get("width") or 512)
    height = int(payload.get("height") or 512)
    base_prompt = str(payload.get("base_prompt") or "")
    base_negative = str(payload.get("base_negative_prompt") or "")
    base_denoise = float(payload.get("base_denoise") if payload.get("base_denoise") is not None else 1.0)
    base_cfg = float(payload.get("base_cfg") if payload.get("base_cfg") is not None else 1.5)

    rgba_img = _resolve_image(
        payload.get("rgba_b64"), payload.get("rgba_ref"), blob_resolver,
    )
    cfg_img = _resolve_image(
        payload.get("cfg_map_b64"), payload.get("cfg_map_ref"), blob_resolver,
    )
    den_img = _resolve_image(
        payload.get("denoise_map_b64"), payload.get("denoise_map_ref"), blob_resolver,
    )

    rgba = _to_rgba(rgba_img, width, height)
    cfg_map = _to_float_map(cfg_img, width, height, CFG_HI, base_cfg)
    denoise_map = _to_float_map(den_img, width, height, 1.0, base_denoise)

    prompts: list[Prompt] = []
    for entry in payload.get("prompts") or []:
        text = str(entry.get("text") or "")
        if not text:
            continue
        mask_img = _resolve_image(
            entry.get("mask_b64"), entry.get("mask_ref"), blob_resolver,
        )
        prompts.append(Prompt(
            text=text,
            negative=str(entry.get("negative") or ""),
            mask=_to_mask(mask_img, width, height),
        ))

    controlnet: list[ControlNet] = []
    for entry in payload.get("controlnet") or []:
        controlnet.append(ControlNet(
            layer_id=str(entry.get("layer_id") or ""),
            model=str(entry.get("model") or ""),
            model_path=entry.get("model_path"),
            scale=float(entry.get("scale") if entry.get("scale") is not None else 1.0),
            start=float(entry.get("start") if entry.get("start") is not None else 0.0),
            end=float(entry.get("end") if entry.get("end") is not None else 1.0),
            preprocessor_params=dict(entry.get("preprocessor_params") or {}),
        ))

    return RenderPlan(
        rgba=rgba,
        cfg_map=cfg_map,
        denoise_map=denoise_map,
        prompts=tuple(prompts),
        base_prompt=base_prompt,
        base_negative_prompt=base_negative,
        base_denoise=base_denoise,
        base_cfg=base_cfg,
        width=width,
        height=height,
        controlnet=tuple(controlnet),
    )


__all__ = ["Prompt", "ControlNet", "RenderPlan", "plan_from_wire", "CFG_HI", "BlobResolver"]
