"""Layer composition — formal specification.

A Scene is an ordered stack of Layers, processed bottom-up. Each Layer holds
one or more Regions; each Region has a spatial mask and overrideable sampling
parameters (denoise, cfg, prompt, weight, ...), each with its own aggregation
operator that decides how the region combines with the accumulated state below.

This module is pure: no GPU, no PIL imports beyond Image for IO. It builds a
``CompositionPlan`` that any renderer can execute (single-pass / layered-pass
/ tiled-pass). Renderer capability gates degrade unsupported features and emit
warnings; they never silently drop user intent.

Vocabulary
==========
- **Operator** (numeric): how a per-pixel scalar map combines with another.
    REPLACE  acc = where(mask>0, value, acc)              # top wins under mask
    ADD      acc = clamp(acc + value*mask_alpha, lo, hi)
    AVERAGE  acc = acc*(1-mask_alpha) + value*mask_alpha  # smooth blend
    MULTIPLY acc = acc * (1 + (value-1)*mask_alpha)
    MAX      acc = max(acc, value*mask_alpha)

- **PromptOperator** (text): how prompt strings combine in overlap.
    REPLACE      top-region prompt wins
    CONCAT       weighted concat: "(p1:w1), (p2:w2), ..."
    EMBED_BLEND  encode each, mask-blend at embedding level (renderer-specific)

- **Schedule**: ``(start, end)`` ∈ [0,1] — fraction of diffusion timesteps when
  the region is active. Inactive regions contribute 0 to all aggregations.

- **Mask alpha**: continuous L-channel 0..255 → float [0,1]. A region's
  effective alpha = ``mask * weight * schedule_influence``.

Bottom-up rule (Photoshop semantics)
====================================
Layers and regions are processed in declared order (bottom → top). For each
region, its operator applies to the current accumulator using the region's
effective alpha. Top region's operator does NOT override previous operators —
each region's operator is its own contribution to the stack.

Renderer capability
===================
``CompositionPlan`` is renderer-agnostic. A ``CompositionWarnings`` payload
lists features that were degraded for the active renderer (e.g., per-region
CFG dropped for StreamDiffusion). Callers can surface these warnings to the UI.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("rtdiffusion.composition")


# ── Constants ─────────────────────────────────────────────────────────────────

DENOISE_LO = 0.0
DENOISE_HI = 0.999
CFG_LO = 0.0
CFG_HI = 30.0
WEIGHT_LO = 0.0
WEIGHT_HI = 4.0
MASK_ACTIVE_EPSILON = 1e-6  # below this alpha the region contributes nothing


# ── Operators ─────────────────────────────────────────────────────────────────

class NumericOp(str, enum.Enum):
    """Per-pixel aggregation for numeric maps (denoise, cfg, weight)."""

    REPLACE = "replace"
    ADD = "add"
    AVERAGE = "average"
    MULTIPLY = "multiply"
    MAX = "max"

    @classmethod
    def parse(cls, value: Any, fallback: "NumericOp") -> "NumericOp":
        if isinstance(value, cls):
            return value
        s = str(value or "").lower().strip().replace("-", "_")
        for op in cls:
            if op.value == s:
                return op
        return fallback


class PromptOp(str, enum.Enum):
    """Aggregation for text prompts when regions overlap.

    REPLACE     top region's prompt wins
    CONCAT      "(p1:w1), (p2:w2), ..." — single encode, weighted attention
    EMBED_BLEND mask-weighted blend of separately encoded embeddings (advanced)
    """

    REPLACE = "replace"
    CONCAT = "concat"
    EMBED_BLEND = "embed_blend"

    @classmethod
    def parse(cls, value: Any, fallback: "PromptOp") -> "PromptOp":
        if isinstance(value, cls):
            return value
        s = str(value or "").lower().strip().replace("-", "_")
        for op in cls:
            if op.value == s:
                return op
        return fallback


# ── Region & Layer model ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Region:
    """One spatial region inside a Layer.

    A region exposes multiple paintable channels (see ``docs/LAYER_SYSTEM.md``
    §3). At the data-model level each channel is an optional L-mode mask:

      • ``mask``         — the region's "presence" alpha. Aliases ``color_mask``
                            for backward compatibility. Treated as the default
                            fallback when a per-channel mask is missing.
      • ``color_mask``   — alpha for the visible RGB pixels of this region.
                            Defaults to ``mask`` when None.
      • ``denoise_mask`` — per-pixel weight on this region's ``denoise`` scalar.
      • ``prompt_mask``  — per-pixel weight on this region's ``prompt``.
      • ``cfg_mask``     — per-pixel weight on this region's ``cfg`` scalar.

    A missing channel mask falls back to ``mask`` so the legacy single-mask
    payload still works without per-channel UI.

    Mask alpha semantics: at each pixel ``effective_param = scalar × alpha`` —
    see §4 of the spec. There is no thresholding, no inversion, no binary
    clamping.
    """

    layer_id: str = ""
    region_id: str = ""
    mask: Image.Image | None = None  # L-mode, 0..255 — fallback for every channel
    color: Image.Image | None = None  # RGB pixels for color compositing (§5)
    color_mask: Image.Image | None = None
    denoise_mask: Image.Image | None = None
    prompt_mask: Image.Image | None = None
    cfg_mask: Image.Image | None = None

    weight: float = 1.0
    schedule: tuple[float, float] = (0.0, 1.0)

    prompt: str | None = None
    prompt_op: PromptOp = PromptOp.CONCAT
    negative_prompt: str | None = None
    negative_prompt_op: PromptOp = PromptOp.CONCAT

    denoise: float | None = None
    denoise_op: NumericOp = NumericOp.REPLACE
    cfg: float | None = None
    cfg_op: NumericOp = NumericOp.REPLACE

    mask_op: NumericOp = NumericOp.ADD  # how this region's mask joins the union

    inner_blur: int = 0
    outer_blur: int = 0
    negate: bool = False


@dataclass(frozen=True)
class Layer:
    """A z-ordered container of Regions.

    Layer-level defaults apply to all regions of the layer that don't override
    the value themselves. A region with ``mask=None`` represents a layer-wide
    "uniform" application (saves having to allocate a full-canvas mask).
    """

    layer_id: str = ""
    name: str = ""
    regions: tuple[Region, ...] = ()

    # Layer-level defaults — pushed into regions that left the field at None.
    default_denoise: float | None = None
    default_cfg: float | None = None
    default_prompt: str | None = None
    default_negative_prompt: str | None = None


@dataclass(frozen=True)
class Scene:
    """Bottom-up ordered stack of Layers + global defaults."""

    layers: tuple[Layer, ...] = ()
    base_prompt: str = ""
    base_negative_prompt: str = ""
    base_denoise: float = 1.0
    base_cfg: float = 1.0
    width: int = 512
    height: int = 512


# ── Plan structures ───────────────────────────────────────────────────────────

class RenderMode(str, enum.Enum):
    SINGLE_PASS = "single_pass"
    LAYERED_PASS = "layered_pass"
    TILED_PASS = "tiled_pass"


@dataclass
class SinglePassPlan:
    """One aggregated inference call."""

    prompt: str
    negative_prompt: str
    mask: Image.Image       # union spatial mask (L)
    denoise_map: Image.Image  # per-pixel denoise (L, 0..255 = 0..1)
    cfg_map: Image.Image    # per-pixel cfg (L, 0..255 = 0..CFG_HI)
    cfg_scalar: float       # scalar fallback when renderer can't do per-pixel CFG
    # When prompt aggregation chose EMBED_BLEND we expose the per-region
    # prompt+alpha pairs so renderers that support it can encode separately.
    regional_prompts: tuple[tuple[str, Image.Image], ...] = ()


@dataclass
class PassSpec:
    """One inference pass in a layered-pass plan."""

    layer_id: str
    region_id: str
    prompt: str
    negative_prompt: str
    mask: Image.Image
    denoise_mask: Image.Image
    denoise: float
    cfg: float
    schedule_start: float
    schedule_end: float


@dataclass
class LayeredPassPlan:
    """N sequential inference calls, composited bottom-up over composite_base."""

    passes: tuple[PassSpec, ...]
    # Aggregated maps for the bottom layer if it lacks a prompt and we want
    # to run a single "base" pass first. Empty by default per the spec
    # (last_output fallback covers the empty case).
    base_pass: SinglePassPlan | None = None


@dataclass
class TilePassSpec:
    """One inference pass for one tile in a tiled-pass plan."""

    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 in canvas coords
    target_w: int                      # SDXL-optimal width for this tile
    target_h: int                      # SDXL-optimal height
    plan: SinglePassPlan              # tile-local aggregation


@dataclass
class TiledPassPlan:
    """Tile-by-tile inference; each tile has its own aggregation."""

    tiles: tuple[TilePassSpec, ...]


@dataclass
class CompositionWarnings:
    """Per-renderer feature degradation report."""

    per_region_cfg_dropped: bool = False
    per_region_schedule_dropped: bool = False
    regional_prompts_dropped: bool = False
    unsupported_mask_channels: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()


@dataclass
class CompositionPlan:
    """Renderer-agnostic plan + degradation report."""

    mode: RenderMode
    single: SinglePassPlan | None = None
    layered: LayeredPassPlan | None = None
    tiled: TiledPassPlan | None = None
    warnings: CompositionWarnings = field(default_factory=CompositionWarnings)


# ── Capability gating ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RendererSupport:
    """Subset of RendererCaps relevant to composition decisions.

    Renderers wire this from their full caps. We keep it as a small struct so
    the composition module doesn't import caps directly (lighter test deps).
    """

    regional_prompts: bool = True
    per_region_cfg: bool = True
    per_region_schedule: bool = True
    spatial_denoise: bool = True   # per-pixel denoise map
    multi_pass: bool = True        # supports layered-pass execution


# ── Mask / alpha helpers ──────────────────────────────────────────────────────

def _mask_to_alpha(mask: Image.Image | None, width: int, height: int) -> np.ndarray:
    """Decode an L-mode mask to a [0,1] float array of (height, width).

    None or wrong-size masks are resized; a None mask returns a full-1.0 array
    (i.e., a uniform "covers everything" region).
    """
    if mask is None:
        return np.ones((height, width), dtype=np.float32)
    if mask.mode == "F":
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.BILINEAR)
        return np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if mask.mode in ("I;16", "I;16B"):
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.float32) / 65535.0
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.float32) / 255.0


def _mask_to_cfg_values(mask: Image.Image, width: int, height: int) -> np.ndarray:
    """Decode an explicit CFG mask as absolute CFG values.

    F-mode masks carry CFG units directly. 16-bit and 8-bit masks carry a
    normalized 0..CFG_HI value. This keeps older PNG masks working while the
    realtime path can use float32 without quantizing CFG into one byte.
    """
    if mask.mode == "F":
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.BILINEAR)
        return np.clip(np.asarray(mask, dtype=np.float32), CFG_LO, CFG_HI)
    if mask.mode in ("I;16", "I;16B"):
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.BILINEAR)
        return np.clip(np.asarray(mask, dtype=np.float32) / 65535.0 * CFG_HI, CFG_LO, CFG_HI)
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.BILINEAR)
    return np.clip(np.asarray(mask, dtype=np.float32) / 255.0 * CFG_HI, CFG_LO, CFG_HI)


def _schedule_influence(start: float, end: float) -> float:
    """Scalar in [0,1] — fraction of timesteps the region is active."""
    return max(0.0, min(1.0, end) - max(0.0, min(1.0, start)))


# Channel name → attribute that holds its mask on a Region. ``color`` and
# ``color_mask`` are equivalent — both refer to the alpha mask for the region's
# visible RGB pixels. The other channels modulate sampling parameters.
_CHANNEL_MASK_ATTR = {
    "color": "color_mask",
    "color_mask": "color_mask",
    "denoise": "denoise_mask",
    "denoise_mask": "denoise_mask",
    "prompt": "prompt_mask",
    "prompt_mask": "prompt_mask",
    "cfg": "cfg_mask",
    "cfg_mask": "cfg_mask",
}


def _region_key(layer_id: str, region_id: str) -> str:
    return f"{layer_id}:{region_id}"


def _region_channel_mask(region: Region, channel: str) -> Image.Image | None:
    """Resolve the mask for a specific channel of a region, with fallback.

    Lookup order: explicit per-channel attribute → this region's ``color_mask``
    → ``region.mask`` (legacy layer-level single mask). Returns ``None`` only
    if no mask is set (uniform application).

    Channels: ``color``/``color_mask``, ``denoise``/``denoise_mask``,
    ``prompt``/``prompt_mask``, ``cfg``/``cfg_mask``.
    """
    attr = _CHANNEL_MASK_ATTR.get(channel)
    if attr is not None:
        specific = getattr(region, attr, None)
        if specific is not None:
            return specific
    if channel not in ("color", "color_mask") and region.color_mask is not None:
        return region.color_mask
    return region.mask


def _region_effective_alpha(
    region: Region,
    width: int,
    height: int,
    channel: str = "color",
) -> np.ndarray:
    """Per-pixel weight of this region for a given channel.

    ``effective_alpha = channel_mask × weight × schedule``, clamped to [0, 1].

    ``channel`` picks which channel-specific mask to honour
    (``color`` / ``denoise`` / ``prompt`` / ``cfg``); falls back to
    ``region.mask`` when the channel-specific mask is missing.
    """
    mask_img = _region_channel_mask(region, channel)
    alpha = _mask_to_alpha(mask_img, width, height)
    if region.negate:
        alpha = 1.0 - alpha
    weight = max(0.0, min(WEIGHT_HI, float(region.weight)))
    schedule = _schedule_influence(*region.schedule)
    if weight == 0.0 or schedule == 0.0:
        return np.zeros_like(alpha)
    return np.clip(alpha * weight * schedule, 0.0, 1.0)


# ── Numeric aggregation ───────────────────────────────────────────────────────

def aggregate_numeric(
    base: float,
    contributions: list[tuple[np.ndarray, float, NumericOp]],
    width: int,
    height: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Bottom-up aggregation of numeric per-pixel values.

    ``base`` initialises the accumulator uniformly. Each contribution is
    (alpha_map, value, op) and is applied in order — top wins for REPLACE,
    accumulates for ADD, etc.

    Returns a (height, width) float array clamped to [lo, hi].
    """
    acc = np.full((height, width), float(base), dtype=np.float32)
    for alpha, value, op in contributions:
        if alpha.shape != acc.shape:
            raise ValueError(f"alpha shape {alpha.shape} != acc shape {acc.shape}")
        active = alpha > MASK_ACTIVE_EPSILON
        if not active.any():
            continue
        v = float(value)
        if op is NumericOp.REPLACE:
            # Smooth alpha-weighted replacement so feathered masks don't
            # produce hard binary jumps.
            new = acc * (1.0 - alpha) + v * alpha
            acc = np.where(active, new, acc)
        elif op is NumericOp.ADD:
            acc = np.where(active, acc + v * alpha, acc)
        elif op is NumericOp.AVERAGE:
            acc = np.where(active, acc * (1.0 - alpha) + v * alpha, acc)
        elif op is NumericOp.MULTIPLY:
            factor = 1.0 + (v - 1.0) * alpha
            acc = np.where(active, acc * factor, acc)
        elif op is NumericOp.MAX:
            acc = np.where(active, np.maximum(acc, v * alpha), acc)
        else:  # pragma: no cover — exhaustive enum
            raise ValueError(f"unknown operator {op!r}")
    return np.clip(acc, lo, hi)


def aggregate_cfg(
    base: float,
    contributions: list[tuple[np.ndarray, float | np.ndarray, NumericOp, bool]],
    width: int,
    height: int,
) -> np.ndarray:
    acc = np.full((height, width), float(base), dtype=np.float32)
    for alpha, value, op, absolute in contributions:
        if alpha.shape != acc.shape:
            raise ValueError(f"alpha shape {alpha.shape} != acc shape {acc.shape}")
        active = alpha > MASK_ACTIVE_EPSILON
        if not active.any():
            continue
        if absolute:
            scaled = np.asarray(value, dtype=np.float32)
            if scaled.shape != acc.shape:
                raise ValueError(f"cfg value shape {scaled.shape} != acc shape {acc.shape}")
            scaled = scaled * alpha
        else:
            scaled = float(value) * alpha
        if op is NumericOp.REPLACE:
            acc = np.where(active, scaled, acc)
        elif op is NumericOp.ADD:
            acc = np.where(active, acc + scaled, acc)
        elif op is NumericOp.AVERAGE:
            acc = np.where(active, (acc + scaled) * 0.5, acc)
        elif op is NumericOp.MULTIPLY:
            acc = np.where(active, acc * scaled, acc)
        elif op is NumericOp.MAX:
            acc = np.where(active, np.maximum(acc, scaled), acc)
        else:  # pragma: no cover — exhaustive enum
            raise ValueError(f"unknown operator {op!r}")
    return np.clip(acc, CFG_LO, CFG_HI)


# ── Mask union ────────────────────────────────────────────────────────────────

def aggregate_mask(
    contributions: list[tuple[np.ndarray, NumericOp]],
    width: int,
    height: int,
) -> np.ndarray:
    """Aggregate region alphas into a single union mask (height, width)∈[0,1].

    The mask itself is also subject to per-region operators so a region can
    e.g. erase coverage from below via MULTIPLY with a negated mask.
    """
    acc = np.zeros((height, width), dtype=np.float32)
    for alpha, op in contributions:
        if alpha.shape != acc.shape:
            raise ValueError(f"alpha shape {alpha.shape} != acc shape {acc.shape}")
        active = alpha > MASK_ACTIVE_EPSILON
        if not active.any() and op is not NumericOp.MULTIPLY:
            # MULTIPLY by 0 is still meaningful (clears acc).
            continue
        if op is NumericOp.REPLACE:
            acc = np.where(active, alpha, acc)
        elif op is NumericOp.ADD:
            acc = np.clip(acc + alpha, 0.0, 1.0)
        elif op is NumericOp.AVERAGE:
            acc = np.where(active, (acc + alpha) * 0.5, acc)
        elif op is NumericOp.MULTIPLY:
            acc = acc * alpha
        elif op is NumericOp.MAX:
            acc = np.maximum(acc, alpha)
        else:  # pragma: no cover
            raise ValueError(f"unknown operator {op!r}")
    return np.clip(acc, 0.0, 1.0)


# ── Prompt aggregation ────────────────────────────────────────────────────────

def aggregate_prompt(
    base: str,
    contributions: list[tuple[float, str, PromptOp]],
) -> str:
    """Aggregate text prompts using the per-region prompt operator.

    contributions: list of (effective_weight_scalar, prompt, op) in bottom-up
    order. effective_weight is typically the max of the region's alpha map
    (1.0 for fully-active region, 0.0 for inactive). Per-pixel prompt mixing
    is the EMBED_BLEND case — handled by the renderer via
    ``SinglePassPlan.regional_prompts``.

    Returns a single prompt string.
    """
    base = (base or "").strip()
    acc = base
    for weight, prompt, op in contributions:
        p = (prompt or "").strip()
        if not p or weight <= MASK_ACTIVE_EPSILON:
            continue
        if op is PromptOp.REPLACE:
            acc = p
        elif op is PromptOp.CONCAT:
            piece = f"({p}:{weight:.2f})" if weight != 1.0 else p
            acc = f"{acc}, {piece}" if acc else piece
        elif op is PromptOp.EMBED_BLEND:
            # Renderer will mix at the embedding level; the textual fallback
            # is the top region's prompt so single-pass renderers still get
            # something usable if they ignore regional_prompts.
            acc = p
        else:  # pragma: no cover
            raise ValueError(f"unknown prompt op {op!r}")
    return acc


# ── Single-pass plan builder ──────────────────────────────────────────────────

def build_single_pass(scene: Scene, support: RendererSupport) -> tuple[SinglePassPlan, CompositionWarnings]:
    """Aggregate all layers/regions into one inference plan + warnings."""
    w, h = scene.width, scene.height
    warnings = CompositionWarnings()
    drops: list[str] = []

    # Materialise the bottom-up region stream. We compute per-channel alphas
    # lazily below — each parameter uses its own channel mask (denoise_mask,
    # prompt_mask, cfg_mask) when provided, falling back to the region's main
    # mask (color_mask / mask) otherwise. See ``docs/LAYER_SYSTEM.md`` §6.3.
    stream: list[tuple[Region, Layer]] = []
    for layer in scene.layers:
        for region in layer.regions:
            stream.append((region, layer))

    # Mask (region presence). Uses the ``color`` channel since the union mask
    # describes "where regions exist visually", which is exactly the color
    # channel's role.
    mask_contribs = [
        (_region_effective_alpha(r, w, h, "color"), r.mask_op)
        for r, _ in stream
    ]
    mask_arr = aggregate_mask(mask_contribs, w, h)

    # Denoise: regions that left denoise=None inherit from layer.default_denoise
    # or from the global base. The accumulator initialises at base_denoise so
    # uncovered pixels keep the global value.
    denoise_contribs: list[tuple[np.ndarray, float, NumericOp]] = []
    for region, layer in stream:
        v = region.denoise
        if v is None:
            v = layer.default_denoise
        if v is None:
            continue  # region truly inherits global, no per-region contribution
        alpha = _region_effective_alpha(region, w, h, "denoise")
        denoise_contribs.append((alpha, float(v), region.denoise_op))
    denoise_arr = aggregate_numeric(
        scene.base_denoise, denoise_contribs, w, h, DENOISE_LO, DENOISE_HI,
    )

    # CFG (renderer may not support per-pixel — we still build the map but
    # also compute a scalar fallback as the max contributing CFG).
    cfg_contribs: list[tuple[np.ndarray, float | np.ndarray, NumericOp, bool]] = []
    for region, layer in stream:
        if region.cfg_mask is not None:
            cfg_values = _mask_to_cfg_values(region.cfg_mask, w, h)
            weight = max(0.0, min(WEIGHT_HI, float(region.weight)))
            schedule = _schedule_influence(*region.schedule)
            alpha = np.full((h, w), np.clip(weight * schedule, 0.0, 1.0), dtype=np.float32)
            cfg_contribs.append((alpha, cfg_values, region.cfg_op, True))
            continue
        v = region.cfg
        if v is None:
            v = layer.default_cfg
        if v is None:
            continue
        alpha = _region_effective_alpha(region, w, h, "cfg")
        cfg_contribs.append((alpha, float(v), region.cfg_op, False))
    cfg_arr = aggregate_cfg(scene.base_cfg, cfg_contribs, w, h)
    cfg_scalar = float(cfg_arr.max())

    if cfg_contribs and not support.per_region_cfg:
        drops.append("per-region CFG not supported by renderer — using scalar max")

    # Prompt — uses ``prompt`` channel mask (falls back to region.mask).
    prompt_contribs: list[tuple[float, str, PromptOp]] = []
    neg_contribs: list[tuple[float, str, PromptOp]] = []
    regional_prompts: list[tuple[str, Image.Image]] = []
    for region, layer in stream:
        prompt_alpha = _region_effective_alpha(region, w, h, "prompt")
        weight = float(prompt_alpha.max(initial=0.0))
        p = region.prompt if region.prompt is not None else layer.default_prompt
        n = region.negative_prompt if region.negative_prompt is not None else layer.default_negative_prompt
        if p:
            prompt_contribs.append((weight, p, region.prompt_op))
            if region.prompt_op is PromptOp.EMBED_BLEND:
                regional_prompts.append((
                    p, Image.fromarray((prompt_alpha * 255.0).astype(np.uint8), "L"),
                ))
        if n:
            neg_contribs.append((weight, n, region.negative_prompt_op))

    if regional_prompts and not support.regional_prompts:
        warnings = _with_warning(warnings, regional_prompts_dropped=True)
        regional_prompts = []
        drops.append("regional prompts not supported — falling back to concat/top prompt")

    final_prompt = aggregate_prompt(scene.base_prompt, prompt_contribs)
    final_negative = aggregate_prompt(scene.base_negative_prompt, neg_contribs)

    if any(r.schedule != (0.0, 1.0) for r, _ in stream) and not support.per_region_schedule:
        warnings = _with_warning(warnings, per_region_schedule_dropped=True)
        drops.append("per-region schedule not supported — applied as weight scaling only")

    if drops:
        warnings = _with_warning(warnings, per_region_cfg_dropped=("per-region CFG" in drops[0] if drops else False))
        warnings = _with_warning(warnings, messages=warnings.messages + tuple(drops))

    plan = SinglePassPlan(
        prompt=final_prompt,
        negative_prompt=final_negative,
        mask=_arr_to_l(mask_arr),
        denoise_map=_arr_to_l(denoise_arr),
        cfg_map=_arr_to_l(cfg_arr / max(CFG_HI, 1e-6)),  # normalise to 0..1 for storage
        cfg_scalar=cfg_scalar,
        regional_prompts=tuple(regional_prompts),
    )
    return plan, warnings


# ── Layered-pass plan builder ─────────────────────────────────────────────────

def build_layered_pass(
    scene: Scene,
    support: RendererSupport,
) -> tuple[LayeredPassPlan, CompositionWarnings]:
    """One inference pass per region with a non-empty prompt.

    Regions without prompts (no override + no layer default) are skipped — they
    don't generate new content. Passes are ordered bottom-up.
    """
    w, h = scene.width, scene.height
    warnings = CompositionWarnings()
    passes: list[PassSpec] = []
    drops: list[str] = []

    for layer in scene.layers:
        for region in layer.regions:
            prompt = region.prompt if region.prompt is not None else layer.default_prompt
            prompt = (prompt or "").strip()
            if not prompt:
                continue
            negative = region.negative_prompt if region.negative_prompt is not None else layer.default_negative_prompt
            denoise = region.denoise if region.denoise is not None else (
                layer.default_denoise if layer.default_denoise is not None else scene.base_denoise
            )
            cfg = region.cfg if region.cfg is not None else (
                layer.default_cfg if layer.default_cfg is not None else scene.base_cfg
            )
            if region.cfg_mask is not None:
                cfg_values = _mask_to_cfg_values(region.cfg_mask, w, h)
                active_cfg = cfg_values[cfg_values > MASK_ACTIVE_EPSILON]
                if active_cfg.size:
                    cfg = float(active_cfg.max())
            # The per-region inpaint pass is driven by the prompt channel
            # mask. Falls back to the legacy single-mask if no prompt_mask is
            # provided. Per the layered-pass semantics in §8.2 of the spec.
            alpha = _region_effective_alpha(region, w, h, "prompt")
            if not (alpha > MASK_ACTIVE_EPSILON).any():
                continue
            mask_img = _arr_to_l(alpha)
            denoise_alpha = _region_effective_alpha(region, w, h, "denoise")
            passes.append(PassSpec(
                layer_id=layer.layer_id,
                region_id=region.region_id,
                prompt=prompt,
                negative_prompt=(negative or "").strip(),
                mask=mask_img,
                denoise_mask=_arr_to_l(denoise_alpha),
                denoise=float(np.clip(denoise, DENOISE_LO, DENOISE_HI)),
                cfg=float(np.clip(cfg, CFG_LO, CFG_HI)),
                schedule_start=float(region.schedule[0]),
                schedule_end=float(region.schedule[1]),
            ))

    if not support.multi_pass and passes:
        drops.append("multi-pass not supported by renderer — falling back to single-pass aggregation")
        warnings = _with_warning(warnings, messages=warnings.messages + tuple(drops))

    if any(p.schedule_start > 0.0 or p.schedule_end < 1.0 for p in passes) and not support.per_region_schedule:
        warnings = _with_warning(warnings, per_region_schedule_dropped=True)

    return LayeredPassPlan(passes=tuple(passes)), warnings


# ── Tiled-pass plan builder ───────────────────────────────────────────────────

# SDXL "optimal" resolutions sorted by aspect ratio (w, h). Tiles snap to the
# closest aspect.
_SDXL_TILE_TARGETS: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1152, 896), (896, 1152),
    (1216, 832), (832, 1216),
    (1344, 768), (768, 1344),
    (1536, 640), (640, 1536),
)


def _snap_to_sdxl(w: int, h: int) -> tuple[int, int]:
    """Pick the SDXL-optimal (W,H) with the closest aspect ratio to (w,h)."""
    aspect = (w / h) if h > 0 else 1.0
    best = min(_SDXL_TILE_TARGETS, key=lambda t: abs((t[0] / t[1]) - aspect))
    return best


def build_tiled_pass(
    scene: Scene,
    support: RendererSupport,
    tile_size: int = 512,
    tile_overlap: int = 64,
) -> tuple[TiledPassPlan, CompositionWarnings]:
    """Split canvas into overlapping tiles; aggregate per-tile."""
    w, h = scene.width, scene.height
    warnings = CompositionWarnings()
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    step = max(1, tile_size - tile_overlap)

    tiles: list[TilePassSpec] = []
    y = 0
    while y < h:
        y1 = min(y + tile_size, h)
        x = 0
        while x < w:
            x1 = min(x + tile_size, w)
            tile_w = x1 - x
            tile_h = y1 - y
            # Build a sub-scene with each layer's regions cropped to this tile.
            tile_scene = _crop_scene(scene, x, y, x1, y1)
            tile_plan, tile_warn = build_single_pass(tile_scene, support)
            target_w, target_h = _snap_to_sdxl(tile_w, tile_h)
            tiles.append(TilePassSpec(
                bbox=(x, y, x1, y1),
                target_w=target_w,
                target_h=target_h,
                plan=tile_plan,
            ))
            warnings = _merge_warnings(warnings, tile_warn)
            x += step
        y += step

    return TiledPassPlan(tiles=tuple(tiles)), warnings


def _crop_image(img: Image.Image | None, bbox: tuple[int, int, int, int]) -> Image.Image | None:
    return img.crop(bbox) if img is not None else None


def _crop_scene(scene: Scene, x0: int, y0: int, x1: int, y1: int) -> Scene:
    """Crop every region mask and the color channel in the scene to the given bbox."""
    cropped_layers: list[Layer] = []
    box = (x0, y0, x1, y1)
    for layer in scene.layers:
        cropped_regions: list[Region] = []
        for region in layer.regions:
            cropped_regions.append(Region(
                layer_id=region.layer_id,
                region_id=region.region_id,
                mask=_crop_image(region.mask, box),
                color=_crop_image(region.color, box),
                color_mask=_crop_image(region.color_mask, box),
                denoise_mask=_crop_image(region.denoise_mask, box),
                prompt_mask=_crop_image(region.prompt_mask, box),
                cfg_mask=_crop_image(region.cfg_mask, box),
                weight=region.weight,
                schedule=region.schedule,
                prompt=region.prompt,
                prompt_op=region.prompt_op,
                negative_prompt=region.negative_prompt,
                negative_prompt_op=region.negative_prompt_op,
                denoise=region.denoise,
                denoise_op=region.denoise_op,
                cfg=region.cfg,
                cfg_op=region.cfg_op,
                mask_op=region.mask_op,
                inner_blur=region.inner_blur,
                outer_blur=region.outer_blur,
                negate=region.negate,
            ))
        cropped_layers.append(Layer(
            layer_id=layer.layer_id,
            name=layer.name,
            regions=tuple(cropped_regions),
            default_denoise=layer.default_denoise,
            default_cfg=layer.default_cfg,
            default_prompt=layer.default_prompt,
            default_negative_prompt=layer.default_negative_prompt,
        ))
    return Scene(
        layers=tuple(cropped_layers),
        base_prompt=scene.base_prompt,
        base_negative_prompt=scene.base_negative_prompt,
        base_denoise=scene.base_denoise,
        base_cfg=scene.base_cfg,
        width=x1 - x0,
        height=y1 - y0,
    )


# ── Plan entry point ──────────────────────────────────────────────────────────

def compose(
    scene: Scene,
    mode: RenderMode,
    support: RendererSupport,
    tile_size: int = 512,
    tile_overlap: int = 64,
) -> CompositionPlan:
    """Build a CompositionPlan for the requested mode + renderer.

    The caller (transport layer) is responsible for executing the plan and
    surfacing warnings to the user.
    """
    if mode is RenderMode.SINGLE_PASS:
        single, warn = build_single_pass(scene, support)
        return CompositionPlan(mode=mode, single=single, warnings=warn)
    if mode is RenderMode.LAYERED_PASS:
        if not support.multi_pass:
            # Fall back to single-pass aggregation with a warning.
            single, warn = build_single_pass(scene, support)
            warn = _with_warning(
                warn, messages=warn.messages +
                ("layered-pass requested but renderer is single-pass only — degraded",),
            )
            return CompositionPlan(mode=RenderMode.SINGLE_PASS, single=single, warnings=warn)
        layered, warn = build_layered_pass(scene, support)
        return CompositionPlan(mode=mode, layered=layered, warnings=warn)
    if mode is RenderMode.TILED_PASS:
        tiled, warn = build_tiled_pass(scene, support, tile_size, tile_overlap)
        return CompositionPlan(mode=mode, tiled=tiled, warnings=warn)
    raise ValueError(f"unknown render mode {mode!r}")


# ── Legacy bridge ─────────────────────────────────────────────────────────────

# Maps from the legacy frontend payload (settings["layer_conditions"][i]) to
# the new typed model. Required while the frontend still emits the flat list.

_LEGACY_MODE_TO_DENOISE_OP: dict[str, NumericOp] = {
    "override": NumericOp.REPLACE,
    "replace": NumericOp.REPLACE,
    "mask": NumericOp.AVERAGE,
    "add": NumericOp.ADD,
    "multiply": NumericOp.MULTIPLY,
    "max": NumericOp.MAX,
}

_LEGACY_MODE_TO_MASK_OP: dict[str, NumericOp] = {
    "override": NumericOp.REPLACE,
    "replace": NumericOp.REPLACE,
    "mask": NumericOp.ADD,
    "add": NumericOp.ADD,
    "multiply": NumericOp.MULTIPLY,
    "max": NumericOp.MAX,
}


def scene_from_legacy(
    layer_conditions: list[dict[str, Any]],
    base_prompt: str,
    base_negative_prompt: str,
    base_denoise: float,
    base_cfg: float,
    width: int,
    height: int,
    masks_by_layer: dict[str, Image.Image] | None = None,
    prompt_overrides: dict[str, str] | None = None,
    channel_masks_by_region: dict[str, dict[str, Image.Image]] | None = None,
    color_by_region: dict[str, Image.Image] | None = None,
) -> Scene:
    """Build a Scene from the existing flat ``layer_conditions`` payload.

    Each ``layer_conditions`` item becomes one Region. Regions sharing the
    same ``layer_id`` are grouped into one Layer (z-order = first appearance).

    Arguments:
      • ``masks_by_layer``: per-layer fallback L-mode mask (legacy single
        alpha channel). Used as ``Region.mask`` for every region of that layer.
            • ``channel_masks_by_region``: optional per-region per-channel masks.
                Preferred keys are ``"<layer_id>:<region_id>"`` so identical region
                IDs on different layers stay isolated; legacy region-only keys are
                still accepted for compatibility. Values are dicts ``{"color": img,
                "denoise": img, "prompt": img, "cfg": img}``. Any missing channel
                falls back to ``masks_by_layer``.
      • ``color_by_region``: optional per-region RGB color channel for color
        compositing. Falls back to None (region contributes only via masks).
    """
    masks = masks_by_layer or {}
    overrides = prompt_overrides or {}
    channel_masks = channel_masks_by_region or {}
    color_imgs = color_by_region or {}

    # Preserve order of first occurrence per layer_id.
    layer_order: list[str] = []
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for cond in layer_conditions:
        lid = str(cond.get("layer_id") or "")
        if lid not in by_layer:
            by_layer[lid] = []
            layer_order.append(lid)
        by_layer[lid].append(cond)

    layers: list[Layer] = []
    for lid in layer_order:
        regions: list[Region] = []
        for cond in by_layer[lid]:
            mode = str(cond.get("mode") or "mask").lower().strip()
            if mode == "prompt_mix":
                # Legacy "prompt_mix" → region with no spatial denoise effect.
                # We model this by setting denoise=None and using PromptOp.CONCAT.
                denoise_op_default = NumericOp.REPLACE
                mask_op_default = NumericOp.ADD
            else:
                denoise_op_default = _LEGACY_MODE_TO_DENOISE_OP.get(mode, NumericOp.AVERAGE)
                mask_op_default = _LEGACY_MODE_TO_MASK_OP.get(mode, NumericOp.ADD)

            # Explicit per-region operators take precedence over the legacy mode.
            denoise_op = NumericOp.parse(cond.get("denoise_operator"), denoise_op_default)
            mask_op = NumericOp.parse(cond.get("mask_operator"), mask_op_default)
            cfg_op = NumericOp.parse(cond.get("cfg_operator"), NumericOp.REPLACE)
            prompt_op = PromptOp.parse(cond.get("prompt_operator"), PromptOp.CONCAT)
            neg_op = PromptOp.parse(cond.get("negative_prompt_operator"), PromptOp.CONCAT)

            tagger_prompt = overrides.get(lid)
            raw_prompt = tagger_prompt or cond.get("prompt")
            prompt_val = raw_prompt if raw_prompt is not None and str(raw_prompt).strip() else None

            neg_raw = cond.get("negative_prompt")
            neg_val = neg_raw if neg_raw is not None and str(neg_raw).strip() else None

            denoise_raw = cond.get("denoise")
            denoise_val = float(denoise_raw) if denoise_raw is not None else None
            cfg_raw = cond.get("cfg")
            cfg_val = float(cfg_raw) if cfg_raw is not None else None

            sched_start = float(cond.get("schedule_start") or 0.0)
            sched_end = float(cond.get("schedule_end") or 1.0)
            weight = float(cond.get("weight") or 1.0)

            region_id = str(cond.get("region_id") or cond.get("name") or "")
            ch = channel_masks.get(_region_key(lid, region_id), channel_masks.get(region_id, {}))
            regions.append(Region(
                layer_id=lid,
                region_id=region_id,
                mask=masks.get(lid),
                color=color_imgs.get(_region_key(lid, region_id), color_imgs.get(region_id)),
                color_mask=ch.get("color"),
                denoise_mask=ch.get("denoise"),
                prompt_mask=ch.get("prompt"),
                cfg_mask=ch.get("cfg"),
                weight=weight,
                schedule=(sched_start, sched_end),
                prompt=prompt_val,
                prompt_op=prompt_op,
                negative_prompt=neg_val,
                negative_prompt_op=neg_op,
                denoise=denoise_val,
                denoise_op=denoise_op,
                cfg=cfg_val,
                cfg_op=cfg_op,
                mask_op=mask_op,
                inner_blur=int(cond.get("inner_blur") or 0),
                outer_blur=int(cond.get("outer_blur") or 0),
                negate=bool(cond.get("negate") or False),
            ))
        layers.append(Layer(
            layer_id=lid,
            name=str(by_layer[lid][0].get("name") or lid),
            regions=tuple(regions),
        ))

    return Scene(
        layers=tuple(layers),
        base_prompt=base_prompt,
        base_negative_prompt=base_negative_prompt,
        base_denoise=base_denoise,
        base_cfg=base_cfg,
        width=width,
        height=height,
    )


# ── Internal utilities ────────────────────────────────────────────────────────

def _arr_to_l(arr: np.ndarray) -> Image.Image:
    """Convert a [0,1] float array to an L-mode PIL Image."""
    clipped = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((clipped * 255.0).round().astype(np.uint8), "L")


def _with_warning(w: CompositionWarnings, **changes: Any) -> CompositionWarnings:
    return CompositionWarnings(
        per_region_cfg_dropped=changes.get("per_region_cfg_dropped", w.per_region_cfg_dropped),
        per_region_schedule_dropped=changes.get("per_region_schedule_dropped", w.per_region_schedule_dropped),
        regional_prompts_dropped=changes.get("regional_prompts_dropped", w.regional_prompts_dropped),
        unsupported_mask_channels=changes.get("unsupported_mask_channels", w.unsupported_mask_channels),
        messages=changes.get("messages", w.messages),
    )


def _merge_warnings(a: CompositionWarnings, b: CompositionWarnings) -> CompositionWarnings:
    return CompositionWarnings(
        per_region_cfg_dropped=a.per_region_cfg_dropped or b.per_region_cfg_dropped,
        per_region_schedule_dropped=a.per_region_schedule_dropped or b.per_region_schedule_dropped,
        regional_prompts_dropped=a.regional_prompts_dropped or b.regional_prompts_dropped,
        unsupported_mask_channels=tuple(sorted(set(a.unsupported_mask_channels) | set(b.unsupported_mask_channels))),
        messages=a.messages + b.messages,
    )


__all__ = [
    "NumericOp",
    "PromptOp",
    "Region",
    "Layer",
    "Scene",
    "RenderMode",
    "RendererSupport",
    "SinglePassPlan",
    "PassSpec",
    "LayeredPassPlan",
    "TilePassSpec",
    "TiledPassPlan",
    "CompositionWarnings",
    "CompositionPlan",
    "aggregate_numeric",
    "aggregate_mask",
    "aggregate_prompt",
    "build_single_pass",
    "build_layered_pass",
    "build_tiled_pass",
    "compose",
    "scene_from_legacy",
]
