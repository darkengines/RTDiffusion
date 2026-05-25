import importlib.util
import os
from importlib import metadata

from .inference.registry import all_caps as _all_backend_caps


def renderer_capabilities() -> dict[str, object]:
    streamdiffusion_package = _module_importable("streamdiffusion")
    xformers_available = _module_available("xformers")
    tensorrt_available = _module_available("tensorrt")
    internal_stream_enabled = os.getenv("RTD_STREAM_NATIVE", "1") == "1"
    direct_stream_enabled = os.getenv("RTD_STREAM_DIRECT", "0") == "1"
    payload: dict[str, object] = {
        "accelerators": {
            "xformers": xformers_available,
            "tensorrt": tensorrt_available,
        },
        "renderers": {
            "sdxl": {
                "configured": True,
                "native": True,
                "streamable": True,
                "realtime": True,
                "runtime": "diffusers",
                "transport": "websocket-binary-image",
                "message": "Diffusers image renderer streamed over binary WebSocket.",
                "requirements": [],
            },
            "z-image": {
                "configured": True,
                "native": True,
                "streamable": True,
                "realtime": True,
                "runtime": "diffusers",
                "transport": "websocket-binary-image",
                "message": "Z-Image renderer streamed over binary WebSocket when the selected model is compatible.",
                "requirements": [],
            },
            "streamdiffusion": _streamdiffusion_capability(
                streamdiffusion_package,
                internal_stream_enabled,
                direct_stream_enabled,
                xformers_available,
                tensorrt_available,
            ),
        },
    }
    _merge_backend_caps(payload)
    return payload


def _merge_backend_caps(payload: dict[str, object]) -> None:
    """Attach `RendererCaps` from the BACKENDS registry as a `caps` field on
    each renderer. Legacy fields stay intact so the current frontend keeps
    working; the new `caps` field is what capability-driven UI will read.
    """
    renderers = payload.get("renderers")
    if not isinstance(renderers, dict):
        return
    for backend_id, caps in _all_backend_caps().items():
        entry = renderers.get(backend_id)
        if isinstance(entry, dict):
            entry["caps"] = caps.to_json()
        else:
            renderers[backend_id] = {"caps": caps.to_json()}


def _streamdiffusion_capability(package_native: bool, internal_native: bool, direct_enabled: bool, xformers: bool, tensorrt: bool) -> dict[str, object]:
    if internal_native:
        return {
            "configured": True,
            "native": True,
            "streamable": True,
            "realtime": True,
            "runtime": "internal-streamdiffusion",
            "transport": "websocket-binary-image",
            "message": "Persistent StreamDiffusion-style denoising batch is active in the RTDiffusion engine. Use LCM, Lightning, Turbo, or Hyper-SD compatible models.",
            "requirements": [],
            "accelerators": {
                "xformers": xformers,
                "tensorrt": tensorrt,
                "direct_stream_enabled": direct_enabled,
                "stream_package": package_native,
                "tiny_vae": os.getenv("RTD_STREAM_TINY_VAE", "1") == "1",
                "warmup": True,
            },
        }
    return {
        "configured": True,
        "native": False,
        "streamable": True,
        "realtime": True,
        "runtime": "diffusers-fallback",
        "transport": "websocket-binary-image",
        "message": "Using the streamed Diffusers fallback. Upstream StreamDiffusion is not active in this environment.",
        "requirements": ["compatible StreamDiffusion install", "persistent worker", "warmup", "stream batch", "tiny VAE"],
        "accelerators": {"xformers": xformers, "tensorrt": tensorrt, "direct_stream_enabled": direct_enabled},
    }

def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _module_importable(name: str) -> bool:
    if importlib.util.find_spec(name) is None:
        return False
    try:
        __import__(name)
        return True
    except Exception:
        return False


def installed_runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("diffusers", "streamdiffusion", "xformers", "tensorrt"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions