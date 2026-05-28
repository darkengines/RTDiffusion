from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from app.inference import FrameResult
from app.rtc.session import InpaintSession


def _png_bytes(mode: str = "RGB", value: int = 128) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (2, 2), value).save(buffer, format="PNG")
    return buffer.getvalue()


def _png_data_url(mode: str = "RGB", value: int = 128) -> str:
    return f"data:image/png;base64,{base64.b64encode(_png_bytes(mode, value)).decode()}"


def _transparent_png_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _opaque_rgba_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGBA", (2, 2), (128, 128, 128, 255)).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _rgba_data_url(color: tuple[int, int, int, int]) -> str:
    buffer = io.BytesIO()
    Image.new("RGBA", (2, 2), color).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _payload_image(data_url: str) -> Image.Image:
    payload = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def test_rtc_non_stream_diffusers_uses_composed_cfg_scalar(monkeypatch):
    calls = []

    class FakeEngine:
        def compile_status(self):
            return {}

    class FakeDiffusersSession:
        engine = FakeEngine()

        def step(self, request):
            calls.append(request.backend_hints["legacy_frame"])
            return FrameResult(
                image=Image.new("RGB", (1, 1)),
                latency_ms=4.0,
                mode="diffusers",
                encoded_image=_png_data_url("RGB"),
                fps=12.5,
            )

    monkeypatch.setattr("app.api.state.get_diffusers_session", lambda device=None: FakeDiffusersSession())
    session = InpaintSession("test-peer")

    result = session._infer_sync(
        _transparent_png_data_url(),
        {
            "prompt": "room",
            "width": 512,
            "height": 512,
            "device": "cpu",
            "stream_diffusion": False,
            "mask": _png_data_url("L", 255),
            "layer_conditions": [{
                "layer_id": "L1",
                "region_id": "R1",
                "name": "cfg-only",
                "image": _opaque_rgba_data_url(),
                "cfg_mask": _png_data_url("L", 255),
            }],
        },
        {},
        None,
        None,
        0,
    )

    assert result is not None
    assert calls
    assert calls[0].cfg == pytest.approx(30.0, abs=0.05)


def test_rtc_signal_payload_includes_final_channel_maps_without_debug_overlay(monkeypatch):
    class FakeEngine:
        def compile_status(self):
            return {}

    class FakeDiffusersSession:
        engine = FakeEngine()

        def step(self, request):
            return FrameResult(
                image=Image.new("RGB", (2, 2)),
                latency_ms=4.0,
                mode="diffusers",
                encoded_image=_png_data_url("RGB"),
                fps=12.5,
            )

    monkeypatch.setattr("app.api.state.get_diffusers_session", lambda device=None: FakeDiffusersSession())
    session = InpaintSession("test-peer")

    result = session._infer_sync(
        _transparent_png_data_url(),
        {
            "prompt": "room",
            "width": 512,
            "height": 512,
            "device": "cpu",
            "stream_diffusion": False,
            "debug_streams": False,
            "mask": _png_data_url("L", 255),
            "layer_conditions": [{
                "layer_id": "L1",
                "region_id": "R1",
                "name": "promptName",
                "image": _opaque_rgba_data_url(),
                "prompt": "fire",
                "prompt_mask": _png_data_url("L", 255),
                "cfg_mask": _png_data_url("L", 255),
            }],
        },
        {},
        None,
        None,
        0,
    )

    assert result is not None
    _, _, debug_payload, _ = result
    assert "final/rgba/aggregated" in debug_payload
    assert "final/denoise/aggregated" in debug_payload
    assert "final/cfg/aggregated" in debug_payload
    assert "final/prompt/L1/promptName" in debug_payload
    assert "final/cfg/L1/promptName" not in debug_payload
    assert "final/prompt/aggregated" not in debug_payload


def test_rtc_rgba_only_signals_are_decoupled_from_prompt_color_and_cfg(monkeypatch):
    class FakeEngine:
        def compile_status(self):
            return {}

    class FakeDiffusersSession:
        engine = FakeEngine()

        def step(self, request):
            return FrameResult(
                image=Image.new("RGB", (2, 2)),
                latency_ms=4.0,
                mode="diffusers",
                encoded_image=_png_data_url("RGB"),
                fps=12.5,
            )

    monkeypatch.setattr("app.api.state.get_diffusers_session", lambda device=None: FakeDiffusersSession())
    session = InpaintSession("test-peer")

    result = session._infer_sync(
        _transparent_png_data_url(),
        {
            "prompt": "room",
            "width": 512,
            "height": 512,
            "device": "cpu",
            "stream_diffusion": False,
            "debug_streams": False,
            "mask": _png_data_url("L", 255),
            "layer_conditions": [{
                "layer_id": "L1",
                "region_id": "R1",
                "name": "Layer_1_alpha",
                "image": _rgba_data_url((173, 223, 74, 255)),
            }],
        },
        {},
        None,
        None,
        0,
    )

    assert result is not None
    _, _, debug_payload, _ = result
    assert "final/color/L1/Layer_1_alpha" not in debug_payload
    rgba_preview = _payload_image(debug_payload["final/rgba/aggregated"])
    cfg_preview = _payload_image(debug_payload["final/cfg/aggregated"])
    assert rgba_preview.getpixel((0, 0))[1] > 150
    assert max(rgba_preview.getpixel((0, 0))) > 0
    assert cfg_preview.getpixel((0, 0)) == (255, 255, 255)