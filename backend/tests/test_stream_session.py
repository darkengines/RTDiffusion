"""StreamInferenceSession adapter (Refactor Step 5).

The adapter wraps a live `StreamSession`. The hot loop is untouched; we only
verify that `FrameRequest` fields land on the right `infer(...)` arguments and
that lifecycle methods are safe no-ops from the caller's perspective.

These tests use a fake `StreamSession` (no torch / no GPU) so they stay fast.
"""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from backend.app.inference import (
    CondInput,
    ControlNetSpec,
    FrameRequest,
    PromptBundle,
    SamplerSpec,
    StreamInferenceSession,
    WarmupHint,
)


class _FakeStream:
    def __init__(self, output_size: tuple[int, int] = (32, 32)) -> None:
        self.calls: list[dict[str, Any]] = []
        self.warmup_calls: list[int] = []
        self._output_size = output_size
        self._return: Image.Image | None = Image.new("RGB", output_size, (10, 20, 30))

    def infer(self, image, mask=None, **kwargs):
        self.calls.append({"image": image, "mask": mask, **kwargs})
        return self._return

    def warmup(self, n: int = 1) -> None:
        self.warmup_calls.append(n)


def _request(width: int = 32, height: int = 32, **hints) -> FrameRequest:
    return FrameRequest(
        color=Image.new("RGB", (width, height), (200, 150, 100)),
        denoise_map=Image.new("L", (width, height), 255),
        width=width,
        height=height,
        prompt=PromptBundle(positive="hi", lerp_b=0.25),
        sampler=SamplerSpec(),
        backend_hints=hints,
    )


def test_step_forwards_args() -> None:
    fake = _FakeStream()
    session = StreamInferenceSession(fake)
    mask = Image.new("L", (32, 32), 128)
    req = _request(
        mask=mask,
        prompt_override="cyberpunk",
        denoise=0.7,
        cn_scale=0.8,
        cn_start=0.1,
        cn_end=0.9,
    )
    result = session.step(req)

    assert result.mode == "stream"
    assert result.image.size == (32, 32)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["mask"] is mask
    assert call["prompt_override"] == "cyberpunk"
    assert call["denoise"] == pytest.approx(0.7)
    assert call["prompt_lerp"] == pytest.approx(0.25)
    assert call["cn_scale"] == pytest.approx(0.8)
    assert call["cn_start"] == pytest.approx(0.1)
    assert call["cn_end"] == pytest.approx(0.9)
    assert call["denoise_map"] is req.denoise_map
    assert call["control_image"] is None


def test_step_takes_first_cond_as_control_image() -> None:
    fake = _FakeStream()
    session = StreamInferenceSession(fake)
    cn_img = Image.new("RGB", (32, 32), (1, 2, 3))
    req = _request()
    req.cond.append(
        CondInput(spec=ControlNetSpec(model_id="canny"), image=cn_img)
    )
    session.step(req)
    assert fake.calls[0]["control_image"] is cn_img


def test_step_denoise_map_hint_overrides_request() -> None:
    """Legacy callers pass denoise_map=None via hints to force scalar denoise."""
    fake = _FakeStream()
    session = StreamInferenceSession(fake)
    req = _request(denoise_map=None)  # explicit None hint
    session.step(req)
    assert fake.calls[0]["denoise_map"] is None


def test_step_handles_concurrent_skip() -> None:
    fake = _FakeStream()
    fake._return = None  # simulate "another infer in progress"
    session = StreamInferenceSession(fake)
    result = session.step(_request())
    assert result.mode == "stream-skip"
    assert result.error is None


def test_warmup_delegates() -> None:
    fake = _FakeStream()
    session = StreamInferenceSession(fake)
    session.warmup(WarmupHint(width=32, height=32))
    assert fake.warmup_calls == [1]


def test_caps_default_to_streamdiffusion() -> None:
    session = StreamInferenceSession(_FakeStream())
    assert session.caps.id == "streamdiffusion"
    assert session.backend == "stream"


def test_close_is_noop() -> None:
    fake = _FakeStream()
    session = StreamInferenceSession(fake)
    session.close()  # must not raise; does not touch fake
