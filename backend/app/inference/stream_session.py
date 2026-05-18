"""StreamInferenceSession — `InferenceSession` adapter over `StreamSession`.

Step 5 in the refactor. Mirrors `DiffusersSession` but for the StreamDiffusion
hot loop. The underlying `StreamSession` is stateful (cascading latent buffer,
init noise, prompt embeds) so a single adapter instance must be reused for a
given session_id; the caller owns lifecycle and passes the live `StreamSession`
in at construction time.

The hot loop is NOT touched. The adapter only marshals `FrameRequest` fields
onto `StreamSession.infer`'s positional args. Step 6 collapses the rtc.py
call site onto this seam.
"""

from __future__ import annotations

from typing import Any

from .caps import RendererCaps
from .registry import get_backend_caps
from .session import FrameRequest, FrameResult, WarmupHint


class StreamInferenceSession:
    """Wraps a live `StreamSession` behind the `InferenceSession` protocol."""

    backend = "stream"

    def __init__(self, stream_session: Any, caps_id: str = "streamdiffusion") -> None:
        self.stream = stream_session
        caps = get_backend_caps(caps_id) or get_backend_caps("streamdiffusion")
        assert caps is not None
        self.caps: RendererCaps = caps

    def step(self, request: FrameRequest) -> FrameResult:
        import time

        from PIL import Image

        hints = request.backend_hints or {}
        mask = hints.get("mask")  # PIL L, white=generate, or None
        prompt_override = hints.get("prompt_override")
        denoise = float(hints.get("denoise", 1.0))
        cn_scale = float(hints.get("cn_scale", 1.0))
        cn_start = float(hints.get("cn_start", 0.0))
        cn_end = float(hints.get("cn_end", 1.0))

        # Stream takes the first CN cond's image as `control_image`; the
        # composer is expected to have merged contributions per-CN already.
        control_image = request.cond[0].image if request.cond else None

        # `denoise_map` from the composer is the canonical white=regenerate map.
        # Legacy callers that want to fall back to scalar denoise pass an
        # explicit `denoise_map=None` hint (the FrameRequest always carries a
        # placeholder map because the field is non-optional).
        if "denoise_map" in hints:
            denoise_map = hints["denoise_map"]
        else:
            denoise_map = request.denoise_map

        t0 = time.perf_counter()
        out = self.stream.infer(
            request.color,
            mask,
            prompt_lerp=float(request.prompt.lerp_b),
            prompt_override=prompt_override,
            denoise=denoise,
            denoise_map=denoise_map,
            control_image=control_image,
            cn_scale=cn_scale,
            cn_start=cn_start,
            cn_end=cn_end,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if out is None:
            # Concurrent skip: surface a non-error result with a placeholder
            # so the transport can decide to drop the frame.
            placeholder = Image.new("RGB", (request.width, request.height))
            return FrameResult(
                image=placeholder,
                latency_ms=latency_ms,
                mode="stream-skip",
                error=None,
            )

        return FrameResult(
            image=out,
            latency_ms=latency_ms,
            mode="stream",
            error=None,
        )

    def warmup(self, hint: WarmupHint) -> None:
        warmup_fn = getattr(self.stream, "warmup", None)
        if callable(warmup_fn):
            try:
                warmup_fn(1)
            except Exception:
                pass

    def close(self) -> None:
        # StreamSession lifecycle is owned by the caller (rtc.py session
        # manager); the adapter does not tear down the live pipeline.
        return None


__all__ = ["StreamInferenceSession"]
