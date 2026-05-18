"""DiffusersSession adapter parity (Refactor Step 4).

The adapter must produce output that round-trips byte-equal to a direct
`engine.generate()` call for any `InpaintFrame` it wraps. This pins the
cutover so /ws/inpaint can keep its on-the-wire payload format identical.
"""

from __future__ import annotations

import base64
import io
import os

import pytest
from PIL import Image

os.environ.setdefault("RTD_MOCK", "1")

from backend.app.engine import DiffusionEngine, EngineConfig
from backend.app.inference import DiffusersSession, inpaint_frame_to_request
from backend.app.schemas import InpaintFrame


def _png_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def mock_engine() -> DiffusionEngine:
    return DiffusionEngine(EngineConfig(device="cpu"))


def _make_frame(width: int = 256, height: int = 256) -> InpaintFrame:
    return InpaintFrame(
        prompt="hello world",
        negative_prompt="ugly",
        image=_png_url(Image.new("RGBA", (width, height), (200, 150, 100, 255))),
        mask=_png_url(Image.new("L", (width, height), 0)),
        width=width,
        height=height,
        steps=1,
        strength=0.5,
        cfg=1.5,
    )


def test_adapter_byte_equal_to_direct(mock_engine: DiffusionEngine) -> None:
    frame = _make_frame()
    direct = mock_engine.generate(frame)
    session = DiffusersSession(mock_engine)
    result = session.step(inpaint_frame_to_request(frame))
    assert result.encoded_image == direct.image
    assert result.mode == direct.mode
    # fps/latency are measured per-call; only the image bytes must match.


def test_adapter_preserves_debug_channels(mock_engine: DiffusionEngine) -> None:
    frame = _make_frame()
    session = DiffusersSession(mock_engine)
    result = session.step(inpaint_frame_to_request(frame))
    # In mock mode there are no debug channels but the field must exist
    assert isinstance(result.raw_debug_urls, dict)


def test_adapter_caps_default_to_sdxl(mock_engine: DiffusionEngine) -> None:
    session = DiffusersSession(mock_engine)
    assert session.caps.id == "sdxl"
    assert session.backend == "diffusers"


def test_step_requires_legacy_frame_in_hints(mock_engine: DiffusionEngine) -> None:
    """Step 4 contract: composer-driven path is not wired yet."""
    from backend.app.inference import FrameRequest, PromptBundle, SamplerSpec

    session = DiffusersSession(mock_engine)
    bad_request = FrameRequest(
        color=Image.new("RGB", (8, 8)),
        denoise_map=Image.new("L", (8, 8), 255),
        width=8,
        height=8,
        prompt=PromptBundle(),
        sampler=SamplerSpec(),
    )
    with pytest.raises(RuntimeError, match="Step 4"):
        session.step(bad_request)
