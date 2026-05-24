"""SANA-friendly prompt formatting.

SANA is a Diffusion Transformer with an LLM-style text encoder (Gemma-derived).
It responds better to *structured* prompts than to CSV-concatenated tags.

For a layered Scene where each layer is a region with its own prompt, we map
the bottom-up stack to a labelled, sectioned prompt that the encoder can parse:

    Background:
    <bottom-layer prompt>

    Midground:
    <middle-layer prompt>

    Foreground:
    <top-layer prompt>

    Lighting:
    <derived from scene base prompt or explicit lighting hint>

Region coverage is rendered as a coarse English locator ("upper-left",
"centre", "lower-right", ...) since SANA doesn't have a native regional-prompt
mechanism — the spatial hint helps the text encoder bias generation toward
the right area.

The module is pure: no PIL/torch beyond Image read for region locator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .composition import Scene, Layer, Region, PromptOp


# Section label heuristic — bottom layer gets "Background", top "Foreground",
# anything in between gets numbered "Midground N". Override via Layer.name.
_DEFAULT_LABELS_3 = ("Background", "Midground", "Foreground")
_DEFAULT_LABELS_2 = ("Background", "Foreground")
_DEFAULT_LABEL_1 = "Scene"


@dataclass(frozen=True)
class SanaPrompt:
    """Result of formatting a Scene for SANA."""

    positive: str
    negative: str
    warnings: tuple[str, ...] = ()


# ── Region locator ────────────────────────────────────────────────────────────

def _region_locator(mask: Image.Image | None, scene_w: int, scene_h: int) -> str:
    """Return a coarse English locator for a region mask: 'centre',
    'upper-left', 'right half', etc. Empty string when the region is uniform
    (covers everything) or has no mask."""
    if mask is None:
        return ""
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != (scene_w, scene_h):
        mask = mask.resize((scene_w, scene_h), Image.Resampling.NEAREST)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    coverage = float(arr.mean())
    if coverage > 0.85:
        return ""  # full-canvas, no locator needed
    ys, xs = np.where(arr > 0.1)
    if ys.size == 0:
        return ""
    cy = float(ys.mean()) / scene_h
    cx = float(xs.mean()) / scene_w
    vert = "upper" if cy < 0.33 else ("lower" if cy > 0.66 else "centre")
    horz = "left" if cx < 0.33 else ("right" if cx > 0.66 else "centre")
    if vert == "centre" and horz == "centre":
        return "centre"
    if vert == "centre":
        return f"{horz} side"
    if horz == "centre":
        return f"{vert} centre"
    return f"{vert}-{horz}"


# ── Layer label assignment ────────────────────────────────────────────────────

def _layer_labels(num_layers: int, layers: tuple[Layer, ...]) -> list[str]:
    """Pick a label per layer based on z-order, falling back to ``Layer.name``."""
    labels: list[str] = []
    if num_layers == 1:
        defaults = (_DEFAULT_LABEL_1,)
    elif num_layers == 2:
        defaults = _DEFAULT_LABELS_2
    elif num_layers == 3:
        defaults = _DEFAULT_LABELS_3
    else:
        # >3 layers: Background, Midground 1..N-2, Foreground
        mid = tuple(f"Midground {i + 1}" for i in range(num_layers - 2))
        defaults = (_DEFAULT_LABELS_3[0], *mid, _DEFAULT_LABELS_3[-1])

    for layer, default in zip(layers, defaults):
        if layer.name and layer.name.strip() and layer.name != layer.layer_id:
            labels.append(layer.name.strip().capitalize())
        else:
            labels.append(default)
    return labels


# ── Per-layer aggregated prompt ───────────────────────────────────────────────

def _layer_prompt(layer: Layer, scene_w: int, scene_h: int) -> tuple[str, str]:
    """Aggregate a layer's regions into one ``(positive, negative)`` string.

    Each region contributes its prompt; if the region has a locator (partial
    coverage) it gets a "(<locator>) ..." prefix. Regions sharing the same
    layer concat in declaration order. Empty regions are skipped.
    """
    pos_parts: list[str] = []
    neg_parts: list[str] = []
    for region in layer.regions:
        p = region.prompt
        if p is None:
            p = layer.default_prompt
        if p:
            locator = _region_locator(region.mask, scene_w, scene_h)
            piece = f"({locator}) {p.strip()}" if locator else p.strip()
            pos_parts.append(piece)
        n = region.negative_prompt
        if n is None:
            n = layer.default_negative_prompt
        if n:
            neg_parts.append(n.strip())
    return ", ".join(pos_parts), ", ".join(neg_parts)


# ── Main formatter ────────────────────────────────────────────────────────────

def format_for_sana(scene: Scene) -> SanaPrompt:
    """Produce a SANA-friendly sectioned prompt for the given Scene.

    The base_prompt becomes the "Scene" section (always first). Each layer
    becomes its own labelled section, in z-order (bottom = Background, etc).
    Regions inside a layer concat with optional spatial locators.

    Empty layers are skipped; if nothing remains we return the base prompt
    verbatim so SANA still receives meaningful conditioning.
    """
    warnings: list[str] = []
    sections: list[str] = []
    neg_sections: list[str] = []

    base = (scene.base_prompt or "").strip()
    base_neg = (scene.base_negative_prompt or "").strip()
    if base:
        sections.append(f"Scene:\n{base}")
    if base_neg:
        neg_sections.append(base_neg)

    non_empty_layers = tuple(
        l for l in scene.layers
        if any(
            (r.prompt or l.default_prompt) for r in l.regions
        ) or any(
            (r.negative_prompt or l.default_negative_prompt) for r in l.regions
        )
    )

    if non_empty_layers:
        labels = _layer_labels(len(non_empty_layers), non_empty_layers)
        for layer, label in zip(non_empty_layers, labels):
            pos, neg = _layer_prompt(layer, scene.width, scene.height)
            if pos:
                sections.append(f"{label}:\n{pos}")
            if neg:
                neg_sections.append(neg)

    # Flag features SANA can't honour even with structured prompts.
    for layer in scene.layers:
        for region in layer.regions:
            if region.cfg is not None or layer.default_cfg is not None:
                warnings.append("SANA: per-region CFG not supported (single scalar)")
                break
            if region.schedule != (0.0, 1.0):
                warnings.append("SANA: per-region schedule not supported")
                break
            if region.prompt_op is PromptOp.EMBED_BLEND:
                warnings.append("SANA: embed_blend prompt op falls back to structured concat")
                break
        else:
            continue
        break

    if not sections:
        # No prompts anywhere — return empty but non-None.
        return SanaPrompt(positive="", negative="", warnings=tuple(dict.fromkeys(warnings)))

    return SanaPrompt(
        positive="\n\n".join(sections),
        negative=", ".join(neg_sections),
        warnings=tuple(dict.fromkeys(warnings)),  # dedupe preserving order
    )


__all__ = ["SanaPrompt", "format_for_sana"]
