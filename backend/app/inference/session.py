"""The single inference seam.

`FrameRequest` carries already-composed inputs (the composer has run upstream).
A backend never sees raw layers — it sees one color image, one denoise map,
a list of resolved conditioning inputs, and a prompt bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL.Image import Image

from .caps import RendererCaps


@dataclass
class RegionalPrompt:
    """A localized prompt anchored to a mask, derived from a single layer."""

    layer_id: str
    prompt: str
    negative_prompt: str = ""
    mask: Image | None = None  # white = apply
    weight: float = 1.0
    schedule: tuple[float, float] = (0.0, 1.0)


@dataclass
class PromptBundle:
    """Final prompt payload after layer composition."""

    positive: str = ""
    negative: str = ""
    regional: list[RegionalPrompt] = field(default_factory=list)
    # Optional A→B lerp for streaming tabs that animate prompts.
    positive_b: str = ""
    lerp_b: float = 0.0


@dataclass
class ControlNetSpec:
    """One ControlNet instance to apply this frame."""

    model_id: str  # canonical CN id ("canny", "depth", "openpose", ...) or HF/local path
    model_path: str | None = None
    scale: float = 1.0
    start: float = 0.0
    end: float = 1.0


@dataclass
class CondInput:
    """A resolved conditioning input.

    `image` is the final tensor/PIL the backend should feed (preprocessor has
    already run upstream). The composer is responsible for merging layer
    contributions for the same CN id into a single image.
    """

    spec: ControlNetSpec
    image: Image  # white pixels = active conditioning
    mask: Image | None = None  # per-pixel CN weight modulator (white = full)


@dataclass
class SamplerSpec:
    steps: int = 1
    cfg: float = 1.5
    sampler: str | None = None
    scheduler: str | None = "simple"
    seed: int | None = None
    seed_mode: str = "fixed"
    seed_variation: int = 0


@dataclass
class FrameRequest:
    """Single composed frame ready for a backend to denoise."""

    color: Image  # composed RGB scene at request resolution
    denoise_map: Image  # L-mode, white = regenerate, black = preserve
    width: int
    height: int
    prompt: PromptBundle
    sampler: SamplerSpec
    cond: list[CondInput] = field(default_factory=list)
    session_id: str = ""  # stateful backends (Stream) key state by this
    debug: bool = False
    # Backend-specific opaque hints (RCFG flags, SSF flags, latent reuse, etc).
    # Step 1 keeps these as a dict to avoid breaking parity; Step 2+ promotes
    # popular hints to typed fields.
    backend_hints: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameResult:
    image: Image
    latency_ms: float
    mode: str
    error: str | None = None
    debug_channels: dict[str, Image] = field(default_factory=dict)
    # Optional original encoded payload (e.g., JPEG data URL) for transports
    # that want to skip a re-encode round-trip. Adapters wrapping legacy code
    # that already produced encoded output set this; new backends may leave
    # it None and let the transport encode `image` itself.
    encoded_image: str | None = None
    fps: float = 0.0
    raw_debug_urls: dict[str, str] = field(default_factory=dict)


@dataclass
class WarmupHint:
    """Hint sent before a session goes hot so it can preload models/latents."""

    width: int = 512
    height: int = 512
    model_path: str | None = None
    lora_paths: list[str] = field(default_factory=list)
    backend_hints: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InferenceSession(Protocol):
    """A live inference backend.

    Implementations: `DiffusersSession`, `StreamInferenceSession`.
    They wrap the existing engine/stream code; Step 1 does not touch
    their hot loops.
    """

    backend: str
    caps: RendererCaps

    def step(self, request: FrameRequest) -> FrameResult: ...
    def warmup(self, hint: WarmupHint) -> None: ...
    def close(self) -> None: ...
