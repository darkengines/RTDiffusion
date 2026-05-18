"""Renderer capability contract.

Every inference backend declares what it actually supports. Transport
validates incoming payloads against the active backend's caps and rejects
out-of-cap fields explicitly (no silent ignores). The frontend reads the
aggregated caps to drive UI gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Channel = Literal["color", "cond", "denoise"]
LayerRole = Literal["color", "guidance", "mask_only", "cond_only"]
BlendMode = Literal["normal", "multiply", "screen", "overlay", "add"]

ALL_CHANNELS: frozenset[Channel] = frozenset(("color", "cond", "denoise"))
ALL_ROLES: frozenset[LayerRole] = frozenset(("color", "guidance", "mask_only", "cond_only"))


@dataclass(frozen=True)
class RendererCaps:
    """Capabilities declared by a single inference backend.

    Frozen so it can be cached and shipped to the frontend as-is.
    """

    id: str
    layers: bool = False
    layer_roles: frozenset[LayerRole] = field(default_factory=lambda: frozenset())
    masks: frozenset[Channel] = field(default_factory=lambda: frozenset())
    controlnets: frozenset[str] = field(default_factory=lambda: frozenset())
    motion_transform: bool = False
    video_layer: bool = False
    auto_tag: bool = False
    regional_prompts: bool = False
    blend_modes: frozenset[BlendMode] = field(default_factory=lambda: frozenset(("normal",)))
    max_layers: int = 0
    max_resolution: int = 1024
    realtime: bool = False
    streamable: bool = False
    transport: str = "none"
    runtime: str = ""
    native: bool = True
    notes: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "layers": self.layers,
            "layer_roles": sorted(self.layer_roles),
            "masks": sorted(self.masks),
            "controlnets": sorted(self.controlnets),
            "motion_transform": self.motion_transform,
            "video_layer": self.video_layer,
            "auto_tag": self.auto_tag,
            "regional_prompts": self.regional_prompts,
            "blend_modes": sorted(self.blend_modes),
            "max_layers": self.max_layers,
            "max_resolution": self.max_resolution,
            "realtime": self.realtime,
            "streamable": self.streamable,
            "transport": self.transport,
            "runtime": self.runtime,
            "native": self.native,
            "notes": self.notes,
        }
