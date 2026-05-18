"""Three-channel layer model.

Every layer carries three independent channels — color, cond, denoise —
each optionally masked. Default mask for all three channels is the layer's
own color alpha. Canonical polarity: white = act on this pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL.Image import Image

from ..inference.caps import BlendMode, LayerRole
from ..inference.session import ControlNetSpec


@dataclass
class AutoTagSpec:
    model: str = "wd-eva02-large-v3"
    threshold: float = 0.35
    refresh_frames: int = 5


@dataclass
class Layer:
    """A single composable layer.

    Step 1: PIL images on entry. Step 2+: GPU tensors at compose time.
    `mask_*` of None means "follow the color alpha channel".
    """

    layer_id: str
    role: LayerRole = "color"
    color: Image | None = None  # RGBA
    cond_image: Image | None = None  # if None, falls back to color RGB
    denoise_strength: float = 1.0
    mask_color: Image | None = None  # L-mode, white = paint over
    mask_cond: Image | None = None  # L-mode, white = contribute to CN
    mask_denoise: Image | None = None  # L-mode, white = regenerate
    blend: BlendMode = "normal"
    schedule: tuple[float, float] = (0.0, 1.0)
    weight: float = 1.0
    cn: list[ControlNetSpec] = field(default_factory=list)
    prompt_fragment: str | None = None
    negative_fragment: str | None = None
    auto_tag: AutoTagSpec | None = None
    # Set by upstream when this layer's color comes from a video frame.
    is_video: bool = False
