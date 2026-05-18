"""DiffusersSession — `InferenceSession` adapter over `DiffusionEngine`.

Step 4 in the refactor: introduce the seam without changing inference
behavior. The adapter reconstitutes an `InpaintFrame` from a `FrameRequest`
and forwards to `engine.generate`. Engine internals keep doing their own
prompt-mix / mask compositing for now; Step 6 cuts that over to the shared
composer.

Output is byte-equal to a direct `engine.generate(frame)` call when the
request was produced by `inpaint_frame_to_request(frame)`.
"""

from __future__ import annotations

from .. import schemas
from ..compose.io import encode_jpeg_data_url, encode_png_data_url
from ..engine import DiffusionEngine
from .caps import RendererCaps
from .registry import get_backend_caps
from .session import FrameRequest, FrameResult, WarmupHint


def inpaint_frame_to_request(frame: schemas.InpaintFrame) -> FrameRequest:
    """Wrap a legacy `InpaintFrame` for the new seam without losing fields.

    The original frame is parked on `backend_hints["legacy_frame"]` so the
    adapter can reconstruct it byte-equal on the other side. This is the
    Step-4 shape; Steps 5/6 reduce reliance on the legacy field as the
    composer takes over.
    """
    from PIL import Image  # local: keep the adapter PIL-agnostic at top level

    width, height = frame.width, frame.height
    placeholder = Image.new("RGB", (width, height))
    return FrameRequest(
        color=placeholder,
        denoise_map=Image.new("L", (width, height), 255),
        width=width,
        height=height,
        prompt=_prompt_from_frame(frame),
        sampler=_sampler_from_frame(frame),
        cond=[],
        session_id="",
        debug=frame.debug_streams,
        backend_hints={"legacy_frame": frame},
    )


def _prompt_from_frame(frame: schemas.InpaintFrame):
    from .session import PromptBundle

    return PromptBundle(
        positive=frame.prompt,
        negative=frame.negative_prompt,
        positive_b=frame.prompt_b,
        lerp_b=frame.prompt_lerp,
    )


def _sampler_from_frame(frame: schemas.InpaintFrame):
    from .session import SamplerSpec

    return SamplerSpec(
        steps=frame.steps,
        cfg=frame.cfg,
        sampler=frame.sampler,
        scheduler=frame.scheduler,
        seed=frame.seed,
        seed_mode=frame.seed_mode,
        seed_variation=frame.seed_variation,
    )


class DiffusersSession:
    """Wraps a `DiffusionEngine` behind the `InferenceSession` protocol."""

    backend = "diffusers"

    def __init__(self, engine: DiffusionEngine, caps_id: str = "sdxl") -> None:
        self.engine = engine
        caps = get_backend_caps(caps_id) or get_backend_caps("sdxl")
        assert caps is not None
        self.caps: RendererCaps = caps

    def step(self, request: FrameRequest) -> FrameResult:
        legacy = request.backend_hints.get("legacy_frame")
        if not isinstance(legacy, schemas.InpaintFrame):
            raise RuntimeError(
                "DiffusersSession (Step 4) requires backend_hints['legacy_frame']; "
                "the composer-driven path lands in Step 6."
            )
        result: schemas.InpaintResult = self.engine.generate(legacy)
        return _inpaint_result_to_frame_result(result)

    def warmup(self, hint: WarmupHint) -> None:
        # Engine performs lazy warmup inside generate(); a dedicated hook
        # arrives in Step 5 alongside StreamSession.
        return None

    def close(self) -> None:
        try:
            self.engine.unload_runtime()
        except Exception:
            pass


def _inpaint_result_to_frame_result(result: schemas.InpaintResult) -> FrameResult:
    from PIL import Image  # local import; keep adapter import-light

    # Skip the decode round-trip; transports that need the PIL image can
    # decode `encoded_image` on demand. The placeholder keeps the dataclass
    # invariant without paying JPEG decode cost in the hot path.
    placeholder = Image.new("RGB", (1, 1))
    return FrameResult(
        image=placeholder,
        latency_ms=result.latency_ms,
        mode=result.mode,
        error=result.error,
        debug_channels={},
        encoded_image=result.image or None,
        fps=result.fps,
        raw_debug_urls=dict(result.debug_channels or {}),
    )


def _decode_data_url(data_url: str):
    import base64
    import io

    from PIL import Image

    _, _, payload = data_url.partition(",")
    raw = base64.b64decode(payload or data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# Convenience for transports that already have the encoded JPEG/PNG payload
# and want to skip the decode→encode round trip in `step()`. Not used in
# Step 4 but kept here so Step 5+ has a stable home.
__all__ = [
    "DiffusersSession",
    "encode_jpeg_data_url",
    "encode_png_data_url",
    "inpaint_frame_to_request",
]
