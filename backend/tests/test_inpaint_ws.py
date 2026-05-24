import asyncio
import io
import threading

import pytest
from PIL import Image

from app.api.inpaint import _can_reuse_scene_result, _receive_latest_frame
from app.inference import FrameResult
from app.rtc.session import InpaintSession


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeWebSocket:
    def __init__(self, payloads: list[dict]):
        self._queue = asyncio.Queue()
        for payload in payloads:
            self._queue.put_nowait(payload)

    async def receive_json(self):
        return await self._queue.get()


@pytest.mark.anyio
async def test_receive_latest_frame_drains_queued_payloads():
    websocket = FakeWebSocket([
        {"client_input_id": 1, "image": "old"},
        {"client_input_id": 2, "image": "middle"},
        {"client_input_id": 3, "image": "latest"},
    ])

    payload = await _receive_latest_frame(websocket)  # type: ignore[arg-type]

    assert payload["client_input_id"] == 3
    assert payload["image"] == "latest"


def test_inpaint_ws_does_not_cache_streamdiffusion_scene_results():
    from app.schemas import InpaintFrame, InpaintResult

    result = InpaintResult(scene_id="same", image=_png_data_url("RGB"), fps=0, latency_ms=0, mode="cached")
    sdxl_frame = InpaintFrame(image=_png_data_url("RGB"), mask=_png_data_url("L"), width=512, height=512, scene_id="same", stream_diffusion=False)
    stream_frame = InpaintFrame(image=_png_data_url("RGB"), mask=_png_data_url("L"), width=512, height=512, scene_id="same", stream_diffusion=True)

    assert _can_reuse_scene_result(sdxl_frame, "same", result) is True
    assert _can_reuse_scene_result(stream_frame, "same", result) is False


def test_rtc_prompt_patch_with_current_input_revision_renders(monkeypatch):
    session = InpaintSession("test-peer")
    session._scene_struct = {
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "image", "mask_ref": "mask"},
        "settings": {"prompt": "jungle", "width": 2, "height": 2, "input_revision": 7},
        "layer_conditions": [],
    }
    session._scene_resources = {
        "image": ("image/png", _png_bytes("RGB"), "image-digest"),
        "mask": ("image/png", _png_bytes("L"), "mask-digest"),
    }
    assert session._materialize_scene()

    marks = []
    monkeypatch.setattr(session, "_mark_staging_changed", lambda: marks.append(session._settings.copy()))

    session.apply_scene_patch({"settings": {"prompt": "room", "input_revision": 7}}, seq=1)

    assert session._pending_live_media_revision == 0
    assert marks
    assert marks[-1]["prompt"] == "room"
    assert marks[-1]["_scene_revision"] == 1


def test_rtc_new_input_revision_waits_for_media(monkeypatch):
    session = InpaintSession("test-peer")
    session._scene_struct = {
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "image", "mask_ref": "mask"},
        "settings": {"prompt": "jungle", "width": 2, "height": 2, "input_revision": 7},
        "layer_conditions": [],
    }
    session._scene_resources = {
        "image": ("image/png", _png_bytes("RGB"), "image-digest"),
        "mask": ("image/png", _png_bytes("L"), "mask-digest"),
    }
    assert session._materialize_scene()

    marks = []
    monkeypatch.setattr(session, "_mark_staging_changed", lambda: marks.append(session._settings.copy()))

    session.apply_scene_patch({"settings": {"prompt": "jungle", "input_revision": 8}}, seq=1)

    assert session._pending_live_media_revision == 8
    assert marks == []


def test_rtc_scene_resource_refs_include_signal_resources():
    session = InpaintSession("test-peer")
    session._scene_struct = {
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "image", "mask_ref": "mask"},
        "settings": {"prompt": "room", "width": 2, "height": 2},
        "layer_conditions": [],
        "signals": [
            {
                "id": "layer.layer-a.region-a.prompt_mask",
                "type": "prompt_mask",
                "layer_id": "layer-a",
                "region_id": "region-a",
                "resource_ref": "prompt-mask",
            }
        ],
    }
    session._scene_resources = {
        "image": ("image/png", _png_bytes("RGB"), "image-digest"),
        "mask": ("image/png", _png_bytes("L"), "mask-digest"),
    }

    assert session._scene_resource_refs() == {"image", "mask", "prompt-mask"}
    assert session._scene_resources_ready() is False

    session._scene_resources["prompt-mask"] = ("image/png", _png_bytes("L"), "prompt-digest")

    assert session._scene_resources_ready() is True


@pytest.mark.anyio
async def test_rtc_restarts_process_loop_when_scene_is_queued_after_worker_exit(monkeypatch):
    session = InpaintSession("test-peer")
    session.start()
    first_task = session._process_task
    assert first_task is not None
    first_task.cancel()
    await asyncio.sleep(0)

    calls = []

    def fake_infer_sync(canvas_url, settings, layer_frames_url, last_output, motion_transform):
        calls.append((canvas_url, settings))
        return Image.new("RGB", (1, 1)), b"jpeg", {}

    monkeypatch.setattr(session, "_infer_sync", fake_infer_sync)
    session._settings = {"prompt": "room", "width": 2, "height": 2}
    session._canvas_url = _png_data_url("RGB")
    session._mark_staging_changed()

    frame, _debug = await asyncio.wait_for(session._output_queue.get(), timeout=1)

    assert frame == b"jpeg"
    assert calls
    assert session._process_task is not None
    assert session._process_task is not first_task
    session.close()


@pytest.mark.anyio
async def test_rtc_retries_latest_generation_after_transient_skip(monkeypatch):
    session = InpaintSession("test-peer")
    calls = 0

    def fake_infer_sync(canvas_url, settings, layer_frames_url, last_output, motion_transform):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return Image.new("RGB", (1, 1)), b"jpeg", {}

    monkeypatch.setattr(session, "_infer_sync", fake_infer_sync)
    session.start()
    session._settings = {"prompt": "room", "width": 2, "height": 2}
    session._canvas_url = _png_data_url("RGB")
    session._mark_staging_changed()

    frame = b""
    for _ in range(8):
        item = await asyncio.wait_for(session._output_queue.get(), timeout=1)
        assert item is not None
        frame, _debug = item
        if frame:
            break

    assert frame == b"jpeg"
    assert calls == 2
    session.close()


@pytest.mark.anyio
async def test_rtc_processes_scene_patch_queued_during_render(monkeypatch):
    session = InpaintSession("test-peer")
    first_infer_started = threading.Event()
    release_first_infer = threading.Event()
    calls = []

    def fake_infer_sync(canvas_url, settings, layer_frames_url, last_output, motion_transform):
        calls.append(settings.copy())
        if len(calls) == 1:
            first_infer_started.set()
            assert release_first_infer.wait(timeout=1)
        return Image.new("RGB", (1, 1)), b"jpeg", {}

    monkeypatch.setattr(session, "_infer_sync", fake_infer_sync)
    session.start()
    session._settings = {"prompt": "room", "width": 2, "height": 2, "stream_diffusion": True}
    session._canvas_url = _png_data_url("RGB")
    session._mark_staging_changed()

    assert await asyncio.to_thread(first_infer_started.wait, 1)
    session._settings = {"prompt": "monster", "width": 2, "height": 2, "stream_diffusion": True, "_scene_revision": 1}
    session._mark_staging_changed()
    release_first_infer.set()

    frames = []
    for _ in range(8):
        item = await asyncio.wait_for(session._output_queue.get(), timeout=1)
        assert item is not None
        frame, _debug = item
        if frame:
            frames.append(frame)
        if len(frames) >= 1 and len(calls) >= 2:
            break

    assert frames
    assert [call["prompt"] for call in calls[:2]] == ["room", "monster"]
    session.close()


def _png_bytes(mode: str = "RGB", value: int = 128) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (2, 2), value).save(buffer, format="PNG")
    return buffer.getvalue()


def _png_data_url(mode: str = "RGB") -> str:
    import base64

    return f"data:image/png;base64,{base64.b64encode(_png_bytes(mode)).decode()}"


def _mask_mean(mask: Image.Image) -> float:
    import numpy as np

    return float(np.asarray(mask.convert("L"), dtype=np.float32).mean())


def _rgb_data_url(color: tuple[int, int, int]) -> str:
    import base64

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _transparent_png_data_url() -> str:
    import base64

    buffer = io.BytesIO()
    Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _opaque_rgba_data_url() -> str:
    import base64

    buffer = io.BytesIO()
    Image.new("RGBA", (2, 2), (128, 128, 128, 255)).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def test_rtc_streamdiffusion_receives_layer_condition_prompt():
    calls = []

    class FakeStreamSession:
        def __init__(self):
            self._s = {}

        def infer(self, color, mask, **kwargs):
            calls.append({"mask": mask, **kwargs})
            return color.convert("RGB")

    class FakeManager:
        def __init__(self):
            self._pipe = None
            self._generation = 1
            self.last_output = None
            self.build_phase = "ready"
            self.build_progress = 1.0
            self.build_message = ""

        def get_session(self, settings):
            return FakeStreamSession()

    session = InpaintSession("test-peer")
    session._session_manager = FakeManager()

    result = session._infer_sync(
        _png_data_url("RGB"),
        {
            "prompt": "scene prompt",
            "negative_prompt": "",
            "width": 512,
            "height": 512,
            "stream_diffusion": True,
            "strength": 0.8,
            "layer_conditions": [{
                "layer_id": "layer-a",
                "region_id": "inherited:layer-a",
                "prompt": "monster",
                "negative_prompt": "",
                "image": _opaque_rgba_data_url(),
                "prompt_mask": _png_data_url("L"),
            }],
        },
        {},
        None,
        None,
    )

    assert result is not None
    assert len(calls) == 2
    assert calls[0]["prompt_override"] == "scene prompt"
    assert calls[1]["prompt_override"] == "monster"
    assert calls[0]["mask"] is not calls[1]["mask"]
    assert calls[1]["mask"] is not None


def test_rtc_non_stream_diffusers_builds_legacy_frame(monkeypatch):
    calls = []

    class FakeEngine:
        def compile_status(self):
            return {}

    class FakeDiffusersSession:
        engine = FakeEngine()

        def step(self, request):
            frame = request.backend_hints["legacy_frame"]
            calls.append(frame)
            assert frame.prompt == "room"
            assert frame.device == "cpu"
            assert frame.image.startswith("data:image/png;base64,")
            assert frame.mask.startswith("data:image/png;base64,")
            _, _, payload = frame.image.partition(",")
            decoded = Image.open(io.BytesIO(__import__("base64").b64decode(payload))).convert("RGB")
            assert decoded.getpixel((0, 0)) == (128, 128, 128)
            return FrameResult(
                image=Image.new("RGB", (1, 1)),
                latency_ms=4.0,
                mode="diffusers",
                encoded_image=_png_data_url("RGB"),
                fps=12.5,
                raw_debug_urls={"engine/debug": "ok"},
            )

    monkeypatch.setattr("app.api.state.get_diffusers_session", lambda device=None: FakeDiffusersSession())
    session = InpaintSession("test-peer")

    result = session._infer_sync(
        _transparent_png_data_url(),
        {
            "prompt": "room",
            "negative_prompt": "bad",
            "width": 512,
            "height": 512,
            "device": "cpu",
            "stream_diffusion": False,
            "debug_streams": True,
            "mask": _png_data_url("L"),
            "layer_conditions": [],
        },
        {},
        None,
        None,
    )

    assert result is not None
    _, _, debug_payload = result
    assert calls
    assert debug_payload["output"] == "stream:output"
    assert debug_payload["engine/debug"] == "ok"


def test_rtc_canvas_media_track_replaces_stale_scene_canvas(monkeypatch):
    calls = []

    class FakeEngine:
        def compile_status(self):
            return {}

    class FakeDiffusersSession:
        engine = FakeEngine()

        def step(self, request):
            frame = request.backend_hints["legacy_frame"]
            calls.append(frame)
            _, _, payload = frame.image.partition(",")
            decoded = Image.open(io.BytesIO(__import__("base64").b64decode(payload))).convert("RGB")
            assert decoded.getpixel((0, 0)) == (200, 10, 30)
            return FrameResult(
                image=Image.new("RGB", (1, 1)),
                latency_ms=4.0,
                mode="diffusers",
                encoded_image=_png_data_url("RGB"),
                fps=12.5,
            )

    monkeypatch.setattr("app.api.state.get_diffusers_session", lambda device=None: FakeDiffusersSession())
    session = InpaintSession("test-peer")
    session.register_media_track("track-canvas", "input/frame/canvas", channel="canvas")
    session.update_media_track_frame("track-canvas", Image.new("RGB", (2, 2), (200, 10, 30)))

    result = session._infer_sync(
        _png_data_url("RGB"),
        {
            "prompt": "room",
            "width": 512,
            "height": 512,
            "device": "cpu",
            "stream_diffusion": False,
            "layer_conditions": [],
        },
        {},
        None,
        None,
    )

    assert result is not None
    assert calls


def test_rtc_unlabelled_media_track_replaces_stale_scene_canvas(monkeypatch):
    calls = []

    class FakeEngine:
        def compile_status(self):
            return {}

    class FakeDiffusersSession:
        engine = FakeEngine()

        def step(self, request):
            frame = request.backend_hints["legacy_frame"]
            calls.append(frame)
            _, _, payload = frame.image.partition(",")
            decoded = Image.open(io.BytesIO(__import__("base64").b64decode(payload))).convert("RGB")
            assert decoded.getpixel((0, 0)) == (255, 105, 180)
            return FrameResult(
                image=Image.new("RGB", (1, 1)),
                latency_ms=4.0,
                mode="diffusers",
                encoded_image=_png_data_url("RGB"),
                fps=12.5,
            )

    monkeypatch.setattr("app.api.state.get_diffusers_session", lambda device=None: FakeDiffusersSession())
    session = InpaintSession("test-peer")
    session.register_media_track("server-track", "server-track")
    session.update_media_track_frame("server-track", Image.new("RGB", (2, 2), (255, 105, 180)))

    result = session._infer_sync(
        _png_data_url("RGB"),
        {
            "prompt": "room",
            "width": 512,
            "height": 512,
            "device": "cpu",
            "stream_diffusion": False,
            "layer_conditions": [],
        },
        {},
        None,
        None,
    )

    assert result is not None
    assert calls


def test_rtc_newest_primary_media_track_wins_over_stale_first_track(monkeypatch):
    calls = []

    class FakeEngine:
        def compile_status(self):
            return {}

    class FakeDiffusersSession:
        engine = FakeEngine()

        def step(self, request):
            frame = request.backend_hints["legacy_frame"]
            calls.append(frame)
            _, _, payload = frame.image.partition(",")
            decoded = Image.open(io.BytesIO(__import__("base64").b64decode(payload))).convert("RGB")
            assert decoded.getpixel((0, 0)) == (255, 105, 180)
            return FrameResult(
                image=Image.new("RGB", (1, 1)),
                latency_ms=4.0,
                mode="diffusers",
                encoded_image=_png_data_url("RGB"),
                fps=12.5,
            )

    monkeypatch.setattr("app.api.state.get_diffusers_session", lambda device=None: FakeDiffusersSession())
    session = InpaintSession("test-peer")
    session.register_media_track("stale-track", "stale-track")
    session.update_media_track_frame("stale-track", Image.new("RGB", (2, 2), (10, 20, 30)))
    session.register_media_track("fresh-track", "fresh-track")
    session.update_media_track_frame("fresh-track", Image.new("RGB", (2, 2), (255, 105, 180)))

    result = session._infer_sync(
        _png_data_url("RGB"),
        {
            "prompt": "room",
            "width": 512,
            "height": 512,
            "device": "cpu",
            "stream_diffusion": False,
            "layer_conditions": [],
        },
        {},
        None,
        None,
    )

    assert result is not None
    assert calls


def test_rtc_latest_primary_media_is_reused_for_scene_only_renders(monkeypatch):
    media_color = (10, 20, 30)
    scene_color = (255, 105, 180)
    calls = []

    class FakeEngine:
        def compile_status(self):
            return {}

    class FakeDiffusersSession:
        engine = FakeEngine()

        def step(self, request):
            frame = request.backend_hints["legacy_frame"]
            calls.append(frame)
            _, _, payload = frame.image.partition(",")
            decoded = Image.open(io.BytesIO(__import__("base64").b64decode(payload))).convert("RGB")
            assert decoded.getpixel((0, 0)) == media_color
            return FrameResult(
                image=Image.new("RGB", (1, 1)),
                latency_ms=4.0,
                mode="diffusers",
                encoded_image=_png_data_url("RGB"),
                fps=12.5,
            )

    monkeypatch.setattr("app.api.state.get_diffusers_session", lambda device=None: FakeDiffusersSession())
    session = InpaintSession("test-peer")
    session.register_media_track("stale-track", "stale-track")
    session.update_media_track_frame("stale-track", Image.new("RGB", (2, 2), media_color))

    settings = {
        "prompt": "room",
        "width": 512,
        "height": 512,
        "device": "cpu",
        "stream_diffusion": False,
        "layer_conditions": [],
    }
    first = session._infer_sync(_png_data_url("RGB"), settings, {}, None, None)
    second = session._infer_sync(_rgb_data_url(scene_color), settings, {}, None, None)

    assert first is not None
    assert second is not None
    assert len(calls) == 2


def test_rtc_non_stream_keeps_regional_prompt_out_of_global_prompt(monkeypatch):
    calls = []

    class FakeEngine:
        def compile_status(self):
            return {}

    class FakeDiffusersSession:
        engine = FakeEngine()

        def step(self, request):
            frame = request.backend_hints["legacy_frame"]
            calls.append(frame)
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
        _png_data_url("RGB"),
        {
            "prompt": "room",
            "negative_prompt": "bad",
            "width": 512,
            "height": 512,
            "device": "cpu",
            "stream_diffusion": False,
            "mask": _png_data_url("L"),
            "layer_conditions": [{
                "layer_id": "sprite",
                "region_id": "sprite-alpha",
                "name": "sprite alpha",
                "prompt": "red dragon",
                "image": _opaque_rgba_data_url(),
                "mode": "mask",
                "denoise": 0.5,
                "weight": 1.0,
            }],
        },
        {},
        None,
        None,
    )

    assert result is not None
    assert calls
    assert calls[0].prompt == "room"
    assert calls[0].layer_conditions[0].prompt == "red dragon"


def test_rtc_scene_materializes_latest_named_resources():
    session = InpaintSession("test-peer")
    session.store_scene_resource("res_input_image", _png_bytes("RGB"), "image/png", name="input.image")
    session.store_scene_resource("res_input_mask", _png_bytes("L"), "image/png", name="input.mask")
    session.store_scene_resource("res_layer_image", _png_bytes("RGBA"), "image/png", name="layer.layerA.regionX.image")
    session.store_scene_resource("res_color_mask", _png_bytes("L"), "image/png", name="layer.layerA.regionX.color_mask")
    session.store_scene_resource("res_prompt_mask", _png_bytes("L"), "image/png", name="layer.layerA.regionX.prompt_mask")
    session.store_scene_resource("res_cfg_mask", _png_bytes("L"), "image/png", name="layer.layerA.regionX.cfg_mask")
    session.store_scene_resource("res_denoise_mask", _png_bytes("L"), "image/png", name="layer.layerA.regionX.denoise_mask")

    session.apply_scene({
        "id": "scene_initial",
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "res_input_image", "mask_ref": "res_input_mask"},
        "settings": {"prompt": "room", "width": 512, "height": 512, "strength": 1.0},
        "layer_conditions": [{
            "name": "Layer A / X",
            "layer_id": "layerA",
            "region_id": "regionX",
            "prompt": "bed",
            "image_ref": "res_layer_image",
            "color_mask_ref_name": "res_color_mask",
            "prompt_mask_ref_name": "res_prompt_mask",
            "cfg_mask_ref_name": "res_cfg_mask",
            "denoise_mask_ref_name": "res_denoise_mask",
        }],
    }, seq=1)

    assert session._canvas_url and session._canvas_url.startswith("data:image/png;base64,")
    assert session._settings["mask"].startswith("data:image/png;base64,")
    condition = session._settings["layer_conditions"][0]
    assert condition["image"].startswith("data:image/png;base64,")
    assert condition["color_mask"].startswith("data:image/png;base64,")
    assert condition["prompt_mask"].startswith("data:image/png;base64,")
    assert condition["cfg_mask"].startswith("data:image/png;base64,")
    assert condition["denoise_mask"].startswith("data:image/png;base64,")
    assert "image_ref" not in condition
    assert "color_mask_ref_name" not in condition
    assert "prompt_mask_ref_name" not in condition
    assert "cfg_mask_ref_name" not in condition
    assert "denoise_mask_ref_name" not in condition


def test_rtc_channel_resource_snapshot_preserves_distinct_signal_values():
    session = InpaintSession("test-peer")
    session.store_scene_resource("res_input_image", _png_bytes("RGB"), "image/png", name="input.image")
    session.store_scene_resource("res_layer_image", _png_bytes("RGBA"), "image/png", name="layer.layerA.regionX.image")
    session.store_scene_resource("res_color_mask", _png_bytes("L", 32), "image/png", name="layer.layerA.regionX.color_mask")
    session.store_scene_resource("res_prompt_mask", _png_bytes("L", 96), "image/png", name="layer.layerA.regionX.prompt_mask")
    session.store_scene_resource("res_cfg_mask", _png_bytes("L", 160), "image/png", name="layer.layerA.regionX.cfg_mask")
    session.store_scene_resource("res_denoise_mask", _png_bytes("L", 224), "image/png", name="layer.layerA.regionX.denoise_mask")

    session.apply_scene({
        "id": "scene_signals",
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "res_input_image"},
        "settings": {"prompt": "room", "width": 8, "height": 8, "strength": 1.0},
        "layer_conditions": [{
            "name": "Layer A / X",
            "layer_id": "layerA",
            "region_id": "regionX",
            "prompt": "monster",
            "image_ref": "res_layer_image",
            "color_mask_ref_name": "res_color_mask",
            "prompt_mask_ref_name": "res_prompt_mask",
            "cfg_mask_ref_name": "res_cfg_mask",
            "denoise_mask_ref_name": "res_denoise_mask",
        }],
    }, seq=1)

    masks = session._build_channel_masks_by_region(session._settings["layer_conditions"], 8, 8)
    signals = masks["layerA:regionX"]
    assert _mask_mean(signals["color"]) == pytest.approx(32, abs=1)
    assert _mask_mean(signals["prompt"]) == pytest.approx(96, abs=1)
    assert _mask_mean(signals["cfg"]) == pytest.approx(160, abs=1)
    assert _mask_mean(signals["denoise"]) == pytest.approx(224, abs=1)


def test_rtc_scene_signals_materialize_layer_channel_resources():
    session = InpaintSession("test-peer")
    session.store_scene_resource("res_input_image", _png_bytes("RGB"), "image/png", name="input.image")
    session.store_scene_resource("res_layer_image", _png_bytes("RGBA"), "image/png", name="layer.layerA.regionX.image")
    session.store_scene_resource("res_color_mask", _png_bytes("L", 32), "image/png", name="layer.layerA.regionX.color_mask")
    session.store_scene_resource("res_prompt_mask", _png_bytes("L", 96), "image/png", name="layer.layerA.regionX.prompt_mask")
    session.store_scene_resource("res_cfg_mask", _png_bytes("L", 160), "image/png", name="layer.layerA.regionX.cfg_mask")
    session.store_scene_resource("res_denoise_mask", _png_bytes("L", 224), "image/png", name="layer.layerA.regionX.denoise_mask")

    session.apply_scene({
        "id": "scene_signal_index",
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "res_input_image"},
        "settings": {"prompt": "room", "width": 8, "height": 8, "strength": 1.0},
        "signals": [
            {"id": "layer.layerA.regionX.image", "type": "rgba", "channel": "color", "layer_id": "layerA", "region_id": "regionX", "resource_ref": "res_layer_image"},
            {"id": "layer.layerA.regionX.color_mask", "type": "rgba_mask", "channel": "color", "layer_id": "layerA", "region_id": "regionX", "resource_ref": "res_color_mask"},
            {"id": "layer.layerA.regionX.prompt_mask", "type": "prompt_mask", "channel": "prompt", "layer_id": "layerA", "region_id": "regionX", "resource_ref": "res_prompt_mask"},
            {"id": "layer.layerA.regionX.cfg_mask", "type": "cfg_mask", "channel": "cfg", "layer_id": "layerA", "region_id": "regionX", "resource_ref": "res_cfg_mask"},
            {"id": "layer.layerA.regionX.denoise_mask", "type": "denoise_mask", "channel": "denoise", "layer_id": "layerA", "region_id": "regionX", "resource_ref": "res_denoise_mask"},
        ],
        "layer_conditions": [{
            "name": "Layer A / X",
            "layer_id": "layerA",
            "region_id": "regionX",
            "prompt": "monster",
        }],
    }, seq=1)

    condition = session._settings["layer_conditions"][0]
    assert condition["image"].startswith("data:image/png;base64,")
    masks = session._build_channel_masks_by_region(session._settings["layer_conditions"], 8, 8)
    signals = masks["layerA:regionX"]
    assert _mask_mean(signals["color"]) == pytest.approx(32, abs=1)
    assert _mask_mean(signals["prompt"]) == pytest.approx(96, abs=1)
    assert _mask_mean(signals["cfg"]) == pytest.approx(160, abs=1)
    assert _mask_mean(signals["denoise"]) == pytest.approx(224, abs=1)


def test_rtc_scene_patch_updates_settings_without_new_resources():
    session = InpaintSession("test-peer")
    session.store_scene_resource("res_input_image", _png_bytes("RGB"), "image/png", name="input.image")

    session.apply_scene({
        "id": "scene_initial",
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "res_input_image"},
        "settings": {"prompt": "room", "width": 512, "height": 512, "strength": 1.0, "cfg": 1.0},
        "layer_conditions": [],
    }, seq=1)

    assert session._input_generation == 1
    canvas_url = session._canvas_url

    session.apply_scene_patch({"settings": {"prompt": "sunlit room", "cfg": 2.5}}, seq=2)

    assert session._input_generation == 2
    assert session._canvas_url == canvas_url
    assert session._settings["prompt"] == "sunlit room"
    assert session._settings["cfg"] == 2.5
    assert session._settings["strength"] == 1.0


def test_rtc_scene_events_update_settings_without_new_resources():
    session = InpaintSession("test-peer")
    session.store_scene_resource("res_input_image", _png_bytes("RGB"), "image/png", name="input.image")

    session.apply_scene({
        "id": "scene_initial",
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "res_input_image"},
        "settings": {"prompt": "room", "width": 512, "height": 512, "strength": 1.0, "cfg": 1.0},
        "layer_conditions": [],
    }, seq=1)

    canvas_url = session._canvas_url

    session.apply_scene_events([
        {"type": "property.set", "path": ["settings", "prompt"], "value": "sunlit room"},
        {"type": "property.set", "path": ["settings", "cfg"], "value": 2.5},
    ], seq=2)

    assert session._input_generation == 2
    assert session._canvas_url == canvas_url
    assert session._settings["prompt"] == "sunlit room"
    assert session._settings["cfg"] == 2.5
    assert session._settings["strength"] == 1.0


def test_rtc_scene_events_update_signal_resource_refs():
    session = InpaintSession("test-peer")
    session.store_scene_resource("res_input_image", _png_bytes("RGB"), "image/png", name="input.image")
    session.store_scene_resource("res_prompt_mask_a", _png_bytes("L", 64), "image/png", name="layer.layerA.regionX.prompt_mask")
    session.store_scene_resource("res_prompt_mask_b", _png_bytes("L", 192), "image/png", name="layer.layerA.regionX.prompt_mask")

    session.apply_scene({
        "id": "scene_initial",
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "res_input_image"},
        "settings": {"prompt": "room", "width": 8, "height": 8, "strength": 1.0},
        "signals": [{
            "id": "layer.layerA.regionX.prompt_mask",
            "type": "prompt_mask",
            "channel": "prompt",
            "layer_id": "layerA",
            "region_id": "regionX",
            "resource_ref": "res_prompt_mask_a",
        }],
        "layer_conditions": [{"layer_id": "layerA", "region_id": "regionX", "prompt": "monster"}],
    }, seq=1)

    session.apply_scene_events([{
        "type": "signal.update",
        "signal": {
            "id": "layer.layerA.regionX.prompt_mask",
            "type": "prompt_mask",
            "channel": "prompt",
            "layer_id": "layerA",
            "region_id": "regionX",
            "resource_ref": "res_prompt_mask_b",
        },
    }], seq=2)

    masks = session._build_channel_masks_by_region(session._settings["layer_conditions"], 8, 8)
    assert _mask_mean(masks["layerA:regionX"]["prompt"]) == pytest.approx(192, abs=1)


def test_rtc_scene_patch_updates_layer_condition_prompt_without_new_resources():
    session = InpaintSession("test-peer")
    session.store_scene_resource("res_input_image", _png_bytes("RGB"), "image/png", name="input.image")
    session.store_scene_resource("res_layer_image", _png_bytes("RGBA"), "image/png", name="layer.layerA.inherited.image")
    session.store_scene_resource("res_prompt_mask", _png_bytes("L"), "image/png", name="layer.layerA.inherited.prompt_mask")

    session.apply_scene({
        "id": "scene_initial",
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "res_input_image"},
        "settings": {"prompt": "room", "width": 512, "height": 512, "strength": 1.0, "stream_diffusion": True},
        "layer_conditions": [{
            "name": "RGBA mask",
            "layer_id": "layerA",
            "region_id": "inherited:layerA",
            "prompt": "room",
            "image_ref": "res_layer_image",
            "prompt_mask_ref_name": "res_prompt_mask",
        }],
    }, seq=1)

    assert session._input_generation == 1
    first_canvas = session._canvas_url
    first_image = session._settings["layer_conditions"][0]["image"]
    first_mask = session._settings["layer_conditions"][0]["prompt_mask"]

    session.apply_scene_patch({
        "settings": {"prompt": "room", "width": 512, "height": 512, "strength": 1.0, "stream_diffusion": True},
        "layer_conditions": [{
            "name": "RGBA mask",
            "layer_id": "layerA",
            "region_id": "inherited:layerA",
            "prompt": "monster",
            "image_ref": "res_layer_image",
            "prompt_mask_ref_name": "res_prompt_mask",
        }],
    }, seq=2)

    assert session._input_generation == 2
    assert session._settings["_scene_revision"] == 1
    assert session._canvas_url == first_canvas
    condition = session._settings["layer_conditions"][0]
    assert condition["prompt"] == "monster"
    assert condition["image"] == first_image
    assert condition["prompt_mask"] == first_mask


def test_rtc_scene_revision_outputs_are_not_dropped_as_pre_live_stale():
    assert InpaintSession._should_drop_pre_live_stale_output(
        {"prompt": "old scene"},
        {"input_revision": 8},
    ) is True
    assert InpaintSession._should_drop_pre_live_stale_output(
        {"prompt": "monster", "_scene_revision": 1},
        {"input_revision": 8},
    ) is False
    assert InpaintSession._should_drop_pre_live_stale_output(
        {"prompt": "live", "input_revision": 8},
        {"input_revision": 8},
    ) is False


def test_rtc_streamdiffusion_convergence_detection():
    session = InpaintSession("test-peer")
    first = Image.new("RGB", (2, 2), (0, 0, 0))
    changed = Image.new("RGB", (2, 2), (80, 80, 80))
    stable = Image.new("RGB", (2, 2), (80, 80, 80))

    settings = {"stream_diffusion": True, "stream_continuous_max_frames": 4}
    assert session._should_continue_stream_steps(settings, None, first, 1) is True
    assert session._should_continue_stream_steps(settings, first, changed, 1) is True
    assert session._should_continue_stream_steps(settings, changed, stable, 1) is True
    assert session._should_continue_stream_steps(settings, stable, stable, 1) is False
    assert session._should_continue_stream_steps({"stream_diffusion": False}, stable, changed, 2) is False


def test_rtc_streamdiffusion_defaults_cap_repeated_layered_frames():
    session = InpaintSession("test-peer")
    first = Image.new("RGB", (2, 2), (0, 0, 0))
    changed = Image.new("RGB", (2, 2), (80, 80, 80))
    settings = {"stream_diffusion": True, "layer_conditions": [{"layer_id": "L1"}]}

    assert session._should_continue_stream_steps(settings, None, first, 1) is True
    assert session._should_continue_stream_steps(settings, first, changed, 1) is True
    assert session._should_continue_stream_steps(settings, changed, changed, 1) is False


@pytest.mark.anyio
async def test_rtc_streamdiffusion_repeats_same_generation_until_stable(monkeypatch):
    session = InpaintSession("test-peer")
    colors = [(0, 0, 0), (80, 80, 80), (80, 80, 80), (80, 80, 80)]
    calls = 0

    def fake_infer_sync(canvas_url, settings, layer_frames_url, last_output, motion_transform):
        nonlocal calls
        color = colors[min(calls, len(colors) - 1)]
        calls += 1
        return Image.new("RGB", (2, 2), color), f"jpeg-{calls}".encode(), {}

    monkeypatch.setattr(session, "_infer_sync", fake_infer_sync)
    session.start()
    session._settings = {"prompt": "room", "width": 2, "height": 2, "stream_diffusion": True, "stream_continuous_max_frames": 4}
    session._canvas_url = _png_data_url("RGB")
    session._mark_staging_changed()

    frames: list[bytes] = []
    for _ in range(12):
        frame, _debug = await asyncio.wait_for(session._output_queue.get(), timeout=1)
        if frame:
            frames.append(frame)
        if len(frames) >= 4:
            break

    assert frames == [b"jpeg-1", b"jpeg-2", b"jpeg-3", b"jpeg-4"]
    assert calls == 4
    assert session._input_generation == 1
    session.close()


def test_rtc_scene_stages_once_when_resources_precede_scene():
    session = InpaintSession("test-peer")

    session.store_scene_resource("res_input_image", _png_bytes("RGB"), "image/png", name="input.image")
    session.store_scene_resource("res_input_mask", _png_bytes("L"), "image/png", name="input.mask")

    assert session._input_generation == 0


    session.apply_scene({
        "id": "scene_initial",
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "res_input_image", "mask_ref": "res_input_mask"},
        "settings": {"prompt": "room", "width": 512, "height": 512, "strength": 1.0},
        "layer_conditions": [],
    }, seq=1)

    assert session._input_generation == 1

    session.apply_scene({
        "id": "scene_initial",
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "res_input_image", "mask_ref": "res_input_mask"},
        "settings": {"prompt": "room", "width": 512, "height": 512, "strength": 1.0},
        "layer_conditions": [],
    }, seq=2)

    assert session._input_generation == 1


def test_rtc_scene_waits_for_all_referenced_resources_before_staging():
    session = InpaintSession("test-peer")

    session.apply_scene({
        "id": "scene_initial",
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "res_input_image", "mask_ref": "res_input_mask"},
        "settings": {"prompt": "room", "width": 512, "height": 512, "strength": 1.0},
        "layer_conditions": [],
    }, seq=1)

    assert session._input_generation == 0
    assert session._canvas_url is None

    session.store_scene_resource("res_unrelated", _png_bytes("RGB"), "image/png", name="unrelated")
    assert session._input_generation == 0

    session.store_scene_resource("res_input_image", _png_bytes("RGB"), "image/png", name="input.image")
    assert session._input_generation == 0

    session.store_scene_resource("res_input_mask", _png_bytes("L"), "image/png", name="input.mask")

    assert session._input_generation == 1
    assert session._canvas_url and session._canvas_url.startswith("data:image/png;base64,")
