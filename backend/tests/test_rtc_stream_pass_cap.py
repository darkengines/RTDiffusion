from __future__ import annotations

import asyncio
import base64
import io

import pytest
from PIL import Image

from app.rtc.session import InpaintSession


def _png_data_url(mode: str = "RGB", value: int = 128) -> str:
    buffer = io.BytesIO()
    Image.new(mode, (2, 2), value).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_rtc_streamdiffusion_caps_same_state_passes(monkeypatch):
    session = InpaintSession("test-peer")
    calls: list[tuple[int, int]] = []

    def fake_infer_sync(canvas_url, settings, layer_frames_url, last_output, motion_transform, expected_generation):
        calls.append((expected_generation, int(settings.get("stream_max_passes") or 0)))
        return (
            Image.new("RGB", (2, 2), (10, 20, 30)),
            b"jpeg",
            {},
            {"mode": "stream", "fps": 12.0, "latency_ms": 4.0},
        )

    monkeypatch.setattr(session, "_infer_sync", fake_infer_sync)

    session.start()
    session._settings = {
        "prompt": "room",
        "width": 2,
        "height": 2,
        "stream_diffusion": True,
        "stream_max_passes": 3,
    }
    session._canvas_url = _png_data_url("RGB")
    session._mark_staging_changed(semantic_reset=False)

    for _ in range(40):
        if len(calls) >= 3:
            break
        await asyncio.sleep(0.01)

    assert len(calls) == 3
    await asyncio.sleep(0.05)
    assert len(calls) == 3

    session.close()
    await asyncio.sleep(0)