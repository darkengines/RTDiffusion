"""Backend registry.

Step 1 populates this with hand-written `RendererCaps` matching today's
behavior. As each backend gains an `InferenceSession` adapter, its factory
moves here too.
"""

from __future__ import annotations

import importlib.util
import os

from .caps import ALL_CHANNELS, RendererCaps


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _sdxl_caps() -> RendererCaps:
    return RendererCaps(
        id="sdxl",
        layers=True,
        layer_roles=frozenset(("color", "guidance", "mask_only", "cond_only")),
        masks=ALL_CHANNELS,
        controlnets=frozenset(("canny", "depth", "openpose", "lineart", "softedge", "mlsd", "seg", "normal")),
        motion_transform=False,
        video_layer=True,
        auto_tag=True,
        regional_prompts=True,
        blend_modes=frozenset(("normal",)),
        max_layers=16,
        max_resolution=1536,
        realtime=True,
        streamable=True,
        transport="websocket-binary-image",
        runtime="diffusers",
        native=True,
        notes="Diffusers image renderer streamed over binary WebSocket.",
    )


def _z_image_caps() -> RendererCaps:
    return RendererCaps(
        id="z-image",
        layers=True,
        layer_roles=frozenset(("color", "guidance", "mask_only", "cond_only")),
        masks=ALL_CHANNELS,
        controlnets=frozenset(),  # Z-Image native path does not currently expose CN
        motion_transform=False,
        video_layer=True,
        auto_tag=True,
        regional_prompts=True,
        blend_modes=frozenset(("normal",)),
        max_layers=16,
        max_resolution=1536,
        realtime=True,
        streamable=True,
        transport="websocket-binary-image",
        runtime="diffusers",
        native=True,
        notes="Z-Image renderer streamed over binary WebSocket.",
    )


def _streamdiffusion_caps() -> RendererCaps:
    native = os.getenv("RTD_STREAM_NATIVE", "1") == "1"
    return RendererCaps(
        id="streamdiffusion",
        layers=True,
        layer_roles=frozenset(("color", "guidance", "mask_only", "cond_only")),
        masks=ALL_CHANNELS,
        controlnets=frozenset(("canny", "depth", "openpose", "lineart", "softedge", "mlsd", "seg")),
        motion_transform=True,
        video_layer=True,
        auto_tag=True,
        regional_prompts=False,  # streaming path does not currently apply regional prompts per-frame
        blend_modes=frozenset(("normal",)),
        max_layers=8,
        max_resolution=1024,
        realtime=True,
        streamable=True,
        transport="webrtc",
        runtime="internal-streamdiffusion" if native else "diffusers-fallback",
        native=native,
        notes="Persistent StreamDiffusion-style denoising batch over WebRTC.",
    )


def _sana_caps() -> RendererCaps:
    return RendererCaps(
        id="sana",
        layers=True,
        layer_roles=frozenset(("color", "mask_only")),
        masks=frozenset(("color", "denoise")),
        controlnets=frozenset(),  # SANA does not expose CN in current code
        motion_transform=False,
        video_layer=True,
        auto_tag=False,
        regional_prompts=False,
        blend_modes=frozenset(("normal",)),
        max_layers=4,
        max_resolution=1024,
        realtime=True,
        streamable=True,
        transport="webrtc",
        runtime="sana",
        native=True,
        notes="SANA linear-attention sampler.",
    )


def _causal_forcing_caps() -> RendererCaps:
    from .. import motion as _motion  # local import to avoid cycles

    configured = _motion.motion_adapter_configured("causal-forcing-1step")
    return RendererCaps(
        id="causal-forcing",
        layers=False,
        layer_roles=frozenset(),
        masks=frozenset(),
        controlnets=frozenset(),
        motion_transform=False,
        video_layer=False,
        auto_tag=False,
        regional_prompts=False,
        blend_modes=frozenset(("normal",)),
        max_layers=0,
        max_resolution=1024,
        realtime=False,
        streamable=configured,
        transport="task-websocket-video" if configured else "none",
        runtime="external-command",
        native=False,
        notes=(
            "Causal-Forcing external runtime configured."
            if configured
            else _motion.missing_motion_adapter_message("causal-forcing-1step")
        ),
    )


def _fastvideo_caps() -> RendererCaps:
    from .. import motion as _motion

    configured = _motion.motion_adapter_configured("fastvideo")
    return RendererCaps(
        id="fastvideo",
        layers=False,
        layer_roles=frozenset(),
        masks=frozenset(),
        controlnets=frozenset(),
        motion_transform=False,
        video_layer=False,
        auto_tag=False,
        regional_prompts=False,
        blend_modes=frozenset(("normal",)),
        max_layers=0,
        max_resolution=1536,
        realtime=False,
        streamable=configured,
        transport="task-websocket-video" if configured else "none",
        runtime="external-command",
        native=False,
        notes=(
            "WAN/FastVideo external runtime configured."
            if configured
            else _motion.missing_motion_adapter_message("fastvideo")
        ),
    )


# Registry of backend id → caps factory. Each factory is called fresh every
# time `get_backend_caps` is invoked so env-var changes take effect without
# a process restart. Adapter classes will be registered alongside in later
# steps via `register_backend`.
BACKENDS: dict[str, "callable[[], RendererCaps]"] = {
    "sdxl": _sdxl_caps,
    "z-image": _z_image_caps,
    "streamdiffusion": _streamdiffusion_caps,
    "sana": _sana_caps,
    "causal-forcing": _causal_forcing_caps,
    "fastvideo": _fastvideo_caps,
}


def register_backend(backend_id: str, caps_factory: "callable[[], RendererCaps]") -> None:
    BACKENDS[backend_id] = caps_factory


def list_backends() -> list[str]:
    return list(BACKENDS.keys())


def get_backend_caps(backend_id: str) -> RendererCaps | None:
    factory = BACKENDS.get(backend_id)
    if factory is None:
        return None
    return factory()


def all_caps() -> dict[str, RendererCaps]:
    return {bid: factory() for bid, factory in BACKENDS.items()}
