"""Extended data structures for layered SDXL rendering.

Relationships to the existing model
─────────────────────────────────────
  compose.Layer         → lightweight layer model used by SceneComposer
  render.RenderLayer    → full SDXL parameter set; adapts to Layer via to_compose_layer()
  inference.FrameRequest → what a backend actually receives (post-composition)

`RenderLayer` is the *source of truth* for per-layer authoring.  The rendering
strategies convert it to whatever the downstream needs (SceneComposer.Layer for
single-pass, per-crop FrameRequest for per-layer/tiled).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from PIL.Image import Image

from ..inference.session import ControlNetSpec, CondInput, FrameResult


# ── Scalar / spatial CFG ──────────────────────────────────────────────────────

@dataclass
class CFGCond:
    """Per-pixel guidance scale.

    `scale` is the uniform fallback.  When `spatial_map` is set (L-mode,
    white = full scale), guidance varies spatially:
        cfg(x,y) = scale_min + (scale - scale_min) * spatial_map[x,y]
    where `scale_min` defaults to 1.0 (unconditional floor).
    """
    scale: float = 7.5
    spatial_map: Image | None = None
    scale_min: float = 1.0


# ── LoRA ─────────────────────────────────────────────────────────────────────

@dataclass
class LoRASpec:
    """One LoRA adapter to activate.

    `weight` is the merge coefficient.  Negative values subtract the adapter
    (useful for negative style guidance).
    """
    path: str
    weight: float = 1.0


# ── SDXL aesthetic micro-conditioning ─────────────────────────────────────────

@dataclass
class SDXLMicroCond:
    """Passed to the UNet via `add_time_ids`.

    Defaults to "native resolution" — correct for most cases.  Override when
    the diffusion crop was taken from a larger canvas (set orig_size to canvas
    dimensions and crop_coords to the bbox origin) so the UNet understands the
    spatial context it is operating in.
    """
    orig_size: tuple[int, int] = (1024, 1024)      # original image size
    target_size: tuple[int, int] = (1024, 1024)    # desired output size
    crop_coords: tuple[int, int] = (0, 0)          # top-left offset of this crop


# ── Core layer data structure ──────────────────────────────────────────────────

@dataclass
class RenderLayer:
    """Complete SDXL conditioning parameters for one composable layer.

    Design notes
    ────────────
    `color`         — what the user *sees* composited on the canvas (RGBA).
    `hidden_source` — an alternative image used *only* for conditioning
                      (e.g., a depth-map reference or a ControlNet guide image)
                      that is never blended into the color composite.  If set,
                      ControlNet preprocessors receive this instead of `color`.

    Three independent masks (all L-mode, white = active):
        mask_color   — pixels to paint over in the color composite
        mask_cond    — pixels where ControlNet conditioning applies
        mask_denoise — pixels to regenerate; defaults to color-alpha if None

    Scheduling
    ──────────
    `schedule` is a (start, end) window in [0, 1] relative to the full diffusion
    timestep range.  Outside the window the layer contributes nothing.
    `weight` linearly scales the layer's influence within the window.

    Per-layer sampler overrides
    ───────────────────────────
    Only honored by `PerLayerStrategy` (where each layer runs an independent
    diffusion pass).  `SinglePassStrategy` and `TiledStrategy` use scene-level
    `SceneSettings` for sampler params.
    """

    layer_id: str

    # ── Color source ──────────────────────────────────────────────────────
    color: Image | None = None          # RGBA visible layer
    hidden_source: Image | None = None  # RGBA conditioning-only (not composited)

    # ── Masks (L-mode, white=active) ──────────────────────────────────────
    mask_color: Image | None = None
    mask_cond: Image | None = None
    mask_denoise: Image | None = None

    # ── Denoise ───────────────────────────────────────────────────────────
    denoise: float = 1.0
    denoise_mode: str = "mask"   # "override" | "mask" | "add" | "multiply" | "prompt_mix"

    # ── Prompts ───────────────────────────────────────────────────────────
    prompt: str = ""
    negative_prompt: str = ""

    # ── Guidance ──────────────────────────────────────────────────────────
    cfg: CFGCond = field(default_factory=CFGCond)

    # ── ControlNets ───────────────────────────────────────────────────────
    cn: list[ControlNetSpec] = field(default_factory=list)

    # ── LoRAs (PerLayerStrategy only) ────────────────────────────────────
    loras: list[LoRASpec] = field(default_factory=list)

    # ── Scheduling ────────────────────────────────────────────────────────
    schedule: tuple[float, float] = (0.0, 1.0)
    weight: float = 1.0

    # ── SDXL micro-conditioning ───────────────────────────────────────────
    micro_cond: SDXLMicroCond | None = None

    # ── Per-layer sampler overrides (PerLayerStrategy only) ──────────────
    steps: int | None = None
    sampler: str | None = None
    scheduler: str | None = None
    seed: int | None = None

    def effective_source(self) -> Image | None:
        """Image to use as the conditioning source (hidden takes priority)."""
        return self.hidden_source if self.hidden_source is not None else self.color

    def bbox(self, canvas_w: int, canvas_h: int) -> tuple[int, int, int, int] | None:
        """Compute tight bounding box from mask_denoise or color alpha.

        Returns (x, y, w, h) in canvas pixels, or None if the layer is empty.
        Falls back to the full canvas when neither mask has a bbox.
        """
        from PIL import ImageChops
        import numpy as np

        src = self.mask_denoise or (
            self.color.getchannel("A")
            if self.color is not None and self.color.mode == "RGBA"
            else None
        )
        if src is None:
            return None
        src = src.convert("L").resize((canvas_w, canvas_h), Image.Resampling.NEAREST)
        box = src.getbbox()
        if box is None:
            return None
        x0, y0, x1, y1 = box
        return (x0, y0, x1 - x0, y1 - y0)


# ── Scene-level settings ──────────────────────────────────────────────────────

@dataclass
class SceneSettings:
    """Scene-level diffusion parameters shared across all layers.

    These are the defaults; `RenderLayer` fields override them per-layer in
    strategies that support per-layer passes.
    """
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg: float = 7.5
    sampler: str | None = None
    scheduler: str | None = "simple"
    seed: int | None = None
    seed_mode: str = "fixed"
    seed_variation: int = 0
    loras: list[LoRASpec] = field(default_factory=list)
    micro_cond: SDXLMicroCond = field(default_factory=SDXLMicroCond)


# ── Extension protocols ───────────────────────────────────────────────────────

@runtime_checkable
class SourceProvider(Protocol):
    """Plug in custom image sources for layers (video frames, procedural, etc.).

    The provider is called once per layer per frame before the strategy runs.
    A None return means "use the layer's own `color` field".
    """
    def get_image(self, layer_id: str, frame_index: int) -> Image | None: ...


@runtime_checkable
class ConditioningProvider(Protocol):
    """Inject additional conditioning inputs beyond what `RenderLayer.cn` declares.

    Use this to wire in auto-taggers, runtime preprocessors, or any conditioning
    that cannot be declared statically (e.g., a ControlNet fed by a live camera).
    The returned list is *appended* to the CondInputs already derived from `cn`.
    """
    def get_conditioning(self, layer: RenderLayer, frame_index: int) -> list[CondInput]: ...


@runtime_checkable
class PostProcessor(Protocol):
    """Chained post-processing after a full render pass.

    Called in registration order; each processor receives the output of the
    previous one so processors compose cleanly.
    """
    def process(self, result: Image, layers: list[RenderLayer]) -> Image: ...
