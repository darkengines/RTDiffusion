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

from app.inference import (
    CondInput,
    ControlNetSpec,
    FrameRequest,
    PromptBundle,
    SamplerSpec,
    StreamInferenceSession,
    WarmupHint,
)
from app.stream.manager import SessionManager, _session_sig
from app.stream.session import _denoise_map_fingerprint, _latent_source_image


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


class _FakeLatent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.shape = (1,)

    def clone(self) -> "_FakeLatent":
        return _FakeLatent(self.name)

    def new_zeros(self, shape) -> "_FakeLatent":
        return _FakeLatent("zero")


class _FakeStatefulStream(_FakeStream):
    def __init__(self) -> None:
        super().__init__()
        self._s = {"latent_buffer": _FakeLatent("base")}

    def infer(self, image, mask=None, **kwargs):
        self.calls.append({"image": image, "mask": mask, "latent_in": self._s["latent_buffer"].name, **kwargs})
        self._s["latent_buffer"] = _FakeLatent(str(kwargs.get("prompt_override")))
        return self._return


class _FakeCfgStream(_FakeStream):
    def __init__(self) -> None:
        super().__init__()
        self._s = {"cfg_scale": 3.0}

    def infer(self, image, mask=None, **kwargs):
        self.calls.append({"image": image, "mask": mask, "cfg_scale": self._s["cfg_scale"], **kwargs})
        return self._return


def test_stream_manager_returns_none_during_rebuild_debounce() -> None:
    manager = SessionManager()
    current_settings = {
        "model_path": "model-a",
        "device": "cpu",
        "width": 512,
        "height": 512,
        "stream_timestep_indices": [32, 45],
        "stream_frame_buffer_size": 1,
        "cfg": 1.0,
    }
    next_settings = {**current_settings, "stream_timestep_indices": [20, 32, 45]}
    fake = _FakeCfgStream()
    manager._session = fake  # type: ignore[assignment]
    manager._sig = _session_sig(current_settings)

    assert manager.get_session(next_settings) is None


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


def test_layer_regions_use_region_denoise_maps() -> None:
    fake = _FakeStream()
    session = StreamInferenceSession(fake)
    mask_a = Image.new("L", (32, 32), 64)
    mask_b = Image.new("L", (32, 32), 192)
    denoise_a = Image.new("L", (32, 32), 80)
    denoise_b = Image.new("L", (32, 32), 220)
    global_denoise = Image.new("L", (32, 32), 255)
    req = _request(
        denoise_map=global_denoise,
        layer_regions=[
            {"mask": mask_a, "denoise_map": denoise_a, "prompt": "room", "denoise": 0.8},
            {"mask": mask_b, "denoise_map": denoise_b, "prompt": "bed", "denoise": 0.8},
        ],
    )
    session.step(req)
    assert len(fake.calls) == 2
    assert fake.calls[0]["mask"] is mask_a
    assert fake.calls[0]["denoise_map"] is denoise_a
    assert fake.calls[0]["prompt_override"] == "room"
    assert fake.calls[1]["mask"] is mask_b
    assert fake.calls[1]["denoise_map"] is denoise_b
    assert fake.calls[1]["prompt_override"] == "bed"


def test_layer_regions_run_base_pass_when_renderer_prompt_is_present() -> None:
        fake = _FakeStream()
        session = StreamInferenceSession(fake)
        aggregate_mask = Image.new("L", (32, 32), 64)
        base_mask = Image.new("L", (32, 32), 255)
        region_mask = Image.new("L", (32, 32), 128)
        req = _request(
            mask=aggregate_mask,
            base_mask=base_mask,
            prompt_override="whole scene",
            layer_regions=[{"layer_id": "L1", "region_id": "rgba", "mask": region_mask, "prompt": "monster"}],
        )

        session.step(req)

        assert len(fake.calls) == 2
        assert fake.calls[0]["mask"] is base_mask
        assert fake.calls[0]["prompt_override"] == "whole scene"
        assert fake.calls[1]["mask"] is region_mask
        assert fake.calls[1]["prompt_override"] == "monster"


def test_layer_regions_use_full_base_mask_when_renderer_mask_is_empty() -> None:
    fake = _FakeStream()
    session = StreamInferenceSession(fake)
    empty_base_mask = Image.new("L", (32, 32), 0)
    region_mask = Image.new("L", (32, 32), 128)
    req = _request(
        base_mask=empty_base_mask,
        prompt_override="jungle room",
        layer_regions=[{"layer_id": "L1", "region_id": "rgba", "mask": region_mask, "prompt": "monster"}],
    )

    session.step(req)

    base_call_mask = fake.calls[0]["mask"]
    assert base_call_mask is not empty_base_mask
    assert base_call_mask.getbbox() == (0, 0, 32, 32)
    assert fake.calls[0]["prompt_override"] == "jungle room"
    assert fake.calls[1]["mask"] is region_mask


def test_layer_regions_apply_region_cfg_scale() -> None:
    fake = _FakeCfgStream()
    session = StreamInferenceSession(fake)
    mask = Image.new("L", (32, 32), 255)
    req = _request(
        layer_regions=[
            {"mask": mask, "prompt": "room", "cfg": 4.5},
            {"mask": mask, "prompt": "bed", "cfg": 7.0},
        ],
    )

    session.step(req)

    assert [call["cfg_scale"] for call in fake.calls] == pytest.approx([4.5, 7.0])
    assert fake._s["cfg_scale"] == pytest.approx(3.0)


def test_layer_region_prompts_preserve_region_cfg_scale() -> None:
    fake = _FakeCfgStream()
    session = StreamInferenceSession(fake)
    mask = Image.new("L", (32, 32), 255)
    req = _request(
        layer_regions=[
            {"mask": mask, "prompt": "monster", "cfg": 1.0},
        ],
    )

    session.step(req)

    assert fake.calls[0]["cfg_scale"] == pytest.approx(1.0)
    assert fake._s["cfg_scale"] == pytest.approx(3.0)


def test_single_pass_uses_request_cfg_scale_and_restores_stream_state() -> None:
    fake = _FakeCfgStream()
    session = StreamInferenceSession(fake)
    req = _request(prompt_override="room")
    req.sampler = SamplerSpec(cfg=6.5)

    session.step(req)

    assert fake.calls[0]["cfg_scale"] == pytest.approx(6.5)
    assert fake._s["cfg_scale"] == pytest.approx(3.0)


def test_denoise_map_fingerprint_distinguishes_sparse_regions() -> None:
    mask_a = Image.new("L", (832, 1216), 0)
    mask_b = Image.new("L", (832, 1216), 0)
    mask_a.putpixel((247, 431), 255)
    mask_b.putpixel((641, 287), 255)

    assert _denoise_map_fingerprint(mask_a) != _denoise_map_fingerprint(mask_b)


def test_latent_source_prefers_previous_rendered_output() -> None:
    current = Image.new("RGB", (32, 32), (200, 150, 100))
    previous = Image.new("RGB", (32, 32), (10, 20, 30))

    assert _latent_source_image(current, None) is current
    assert _latent_source_image(current, previous) is previous


def test_layer_regions_use_isolated_latent_buffers() -> None:
    fake = _FakeStatefulStream()
    session = StreamInferenceSession(fake)
    mask = Image.new("L", (32, 32), 255)
    req = _request(
        layer_regions=[
            {"layer_id": "L1", "region_id": "green", "mask": mask, "prompt": "cat"},
            {"layer_id": "L2", "region_id": "red", "mask": mask, "prompt": "car"},
        ],
    )

    session.step(req)

    assert fake.calls[0]["latent_in"] == "base"
    assert fake.calls[1]["latent_in"] == "base"
    region_buffers = getattr(fake, "_region_latent_buffers")
    assert region_buffers["L1:green"].name == "cat"
    assert region_buffers["L2:red"].name == "car"
    assert fake._s["latent_buffer"].name == "base"


def test_layer_region_prompt_change_resets_latent_buffer() -> None:
    fake = _FakeStatefulStream()
    session = StreamInferenceSession(fake)
    mask = Image.new("L", (32, 32), 255)

    session.step(_request(layer_regions=[{"layer_id": "L1", "region_id": "rgba", "mask": mask, "prompt": "cat"}]))
    session.step(_request(layer_regions=[{"layer_id": "L1", "region_id": "rgba", "mask": mask, "prompt": "cat"}]))
    session.step(_request(layer_regions=[{"layer_id": "L1", "region_id": "rgba", "mask": mask, "prompt": "monster"}]))

    assert fake.calls[0]["latent_in"] == "base"
    assert fake.calls[1]["latent_in"] == "cat"
    assert fake.calls[2]["latent_in"] == "zero"
    region_buffers = getattr(fake, "_region_latent_buffers")
    assert region_buffers["L1:rgba"].name == "monster"


def test_single_pass_prompt_change_resets_latent_buffer() -> None:
    fake = _FakeStatefulStream()
    session = StreamInferenceSession(fake)

    session.step(_request(prompt_override="cat"))
    session.step(_request(prompt_override="cat"))
    session.step(_request(prompt_override="monster"))

    assert fake.calls[0]["latent_in"] == "zero"
    assert fake.calls[1]["latent_in"] == "cat"
    assert fake.calls[2]["latent_in"] == "zero"
    assert fake._s["latent_buffer"].name == "monster"


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
