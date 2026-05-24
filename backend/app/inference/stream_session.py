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


def _clone_latent_buffer(buffer: Any) -> Any:
    if buffer is None:
        return None
    clone = getattr(buffer, "clone", None)
    if callable(clone):
        return clone()
    return buffer


def _reset_latent_buffer(buffer: Any) -> Any:
    if buffer is None:
        return None
    zeros_like = getattr(buffer, "new_zeros", None)
    if callable(zeros_like):
        return zeros_like(buffer.shape)
    return None


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
        layer_regions: list = hints.get("layer_regions") or []
        composite_base = hints.get("composite_base")  # previous frame output — base for non-masked area

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
        base_mask = hints.get("base_mask", mask)
        base_denoise = float(hints.get("base_denoise", denoise))
        base_denoise_map = hints.get("base_denoise_map", denoise_map)

        t0 = time.perf_counter()

        if layer_regions:
            # Per-layer multi-pass: one infer per region, composited bottom-to-top.
            # Start with a global pass so renderer prompt/input edits update the
            # whole scene; regional passes then overlay prompt-specific areas.
            stream_state = getattr(self.stream, "_s", None)
            base_prompt_signature = str(prompt_override or "")
            if isinstance(stream_state, dict) and base_prompt_signature.strip():
                previous_signature = getattr(self.stream, "_single_prompt_signature", None)
                if previous_signature != base_prompt_signature:
                    stream_state["latent_buffer"] = _reset_latent_buffer(stream_state.get("latent_buffer"))
                    setattr(self.stream, "_single_prompt_signature", base_prompt_signature)
            current = request.color
            cb = composite_base
            out = None
            if base_prompt_signature.strip():
                effective_base_mask = base_mask
                try:
                    if effective_base_mask is None or effective_base_mask.convert("L").getbbox() is None:
                        effective_base_mask = Image.new("L", (request.width, request.height), 255)
                except Exception:
                    effective_base_mask = Image.new("L", (request.width, request.height), 255)
                base_result = self.stream.infer(
                    current,
                    effective_base_mask,
                    prompt_lerp=float(request.prompt.lerp_b),
                    prompt_override=prompt_override,
                    denoise=base_denoise,
                    denoise_map=base_denoise_map,
                    control_image=control_image,
                    cn_scale=cn_scale,
                    cn_start=cn_start,
                    cn_end=cn_end,
                    composite_base=cb,
                )
                if base_result is not None:
                    current = base_result
                    cb = base_result
                    out = base_result
            base_latent_buffer = _clone_latent_buffer(stream_state.get("latent_buffer")) if isinstance(stream_state, dict) else None
            region_buffers = getattr(self.stream, "_region_latent_buffers", None)
            if region_buffers is None:
                region_buffers = {}
                setattr(self.stream, "_region_latent_buffers", region_buffers)
            region_prompt_signatures = getattr(self.stream, "_region_prompt_signatures", None)
            if region_prompt_signatures is None:
                region_prompt_signatures = {}
                setattr(self.stream, "_region_prompt_signatures", region_prompt_signatures)
            for region in layer_regions:
                region_key = f"{region.get('layer_id', '')}:{region.get('region_id', '')}"
                region_prompt = f"{base_prompt_signature}\0{str(region.get('prompt') or '')}"
                previous_region_prompt = region_prompt_signatures.get(region_key)
                if previous_region_prompt != region_prompt:
                    if previous_region_prompt is None:
                        region_buffers.pop(region_key, None)
                    else:
                        region_buffers[region_key] = _reset_latent_buffer(region_buffers.get(region_key, base_latent_buffer))
                    region_prompt_signatures[region_key] = region_prompt
                if isinstance(stream_state, dict):
                    saved_buffer = region_buffers.get(region_key, base_latent_buffer)
                    stream_state["latent_buffer"] = _clone_latent_buffer(saved_buffer)
                missing_cfg_scale = object()
                previous_cfg_scale = stream_state.get("cfg_scale", missing_cfg_scale) if isinstance(stream_state, dict) else missing_cfg_scale
                if isinstance(stream_state, dict) and region.get("cfg") is not None:
                    stream_state["cfg_scale"] = float(region["cfg"])
                region_denoise_map = region.get("denoise_map", denoise_map)
                try:
                    result = self.stream.infer(
                        current,
                        region["mask"],
                        prompt_lerp=float(request.prompt.lerp_b),
                        prompt_override=region["prompt"],
                        denoise=float(region.get("denoise", denoise)),
                        denoise_map=region_denoise_map,
                        control_image=control_image,
                        cn_scale=cn_scale,
                        cn_start=cn_start,
                        cn_end=cn_end,
                        composite_base=cb,
                    )
                finally:
                    if isinstance(stream_state, dict):
                        if previous_cfg_scale is missing_cfg_scale:
                            stream_state.pop("cfg_scale", None)
                        else:
                            stream_state["cfg_scale"] = previous_cfg_scale
                if isinstance(stream_state, dict):
                    region_buffers[region_key] = _clone_latent_buffer(stream_state.get("latent_buffer"))
                if result is not None:
                    current = result
                    cb = result  # next region composites over accumulated output
                    out = result
            if isinstance(stream_state, dict):
                stream_state["latent_buffer"] = base_latent_buffer
        else:
            # Single-pass fallback: merged prompt + combined mask.
            stream_state = getattr(self.stream, "_s", None)
            prompt_signature = str(prompt_override or "")
            if isinstance(stream_state, dict):
                previous_signature = getattr(self.stream, "_single_prompt_signature", None)
                if previous_signature != prompt_signature:
                    stream_state["latent_buffer"] = _reset_latent_buffer(stream_state.get("latent_buffer"))
                    setattr(self.stream, "_single_prompt_signature", prompt_signature)
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
                composite_base=composite_base,
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
