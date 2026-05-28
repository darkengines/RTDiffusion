from __future__ import annotations

from PIL import Image
import pytest

from app.rtc.session import InpaintSession


def test_duplicate_primary_media_frame_does_not_advance_sequence() -> None:
    session = InpaintSession("test-peer")
    session.register_media_track("track-canvas", "input/frame/canvas", channel="canvas")

    frame = Image.new("RGB", (2, 2), (20, 40, 60))
    session.update_media_track_frame("track-canvas", frame)
    first_seq = session._media_frame_seq

    session.update_media_track_frame("track-canvas", frame.copy())

    assert session._media_frame_seq == first_seq


def test_non_stream_primary_media_waits_for_scene_change() -> None:
    session = InpaintSession("test-peer")
    session._settings = {"stream_diffusion": False}
    session.register_media_track("track-canvas", "input/frame/canvas", channel="canvas")

    session.update_media_track_frame("track-canvas", Image.new("RGB", (2, 2), (20, 40, 60)))

    assert session._media_frame_seq == 1
    assert session._input_event.is_set() is False


def test_non_stream_primary_media_consumes_pending_live_revision() -> None:
    session = InpaintSession("test-peer")
    session._settings = {"stream_diffusion": False}
    session._pending_live_media_revision = 7
    session.register_media_track("track-canvas", "input/frame/canvas", channel="canvas")

    session.update_media_track_frame("track-canvas", Image.new("RGB", (2, 2), (20, 40, 60)))

    assert session._pending_live_media_revision == 0
    assert session._input_generation == 1
    assert session._input_event.is_set() is True


def test_unlabelled_top_level_media_track_is_primary() -> None:
    session = InpaintSession("test-peer")
    session.register_media_track("server-track", "server-track")

    assert session._is_primary_media_track("server-track") is True


def test_top_level_color_media_track_is_primary() -> None:
    session = InpaintSession("test-peer")
    session.register_media_track("color-track", "color-track", channel="color")

    assert session._is_primary_media_track("color-track") is True


@pytest.mark.parametrize("apply_method", ["patch", "events"])
def test_non_stream_live_input_revision_marks_state_once_media_is_fresh(monkeypatch, apply_method: str) -> None:
    session = InpaintSession("test-peer")
    session._scene_struct = {
        "protocol": "rtd.scene.v1",
        "input": {"image_ref": "image", "mask_ref": "mask"},
        "settings": {"prompt": "room", "width": 2, "height": 2, "input_revision": 7},
        "layer_conditions": [],
    }
    session._scene_resources = {
        "image": ("image/png", b"image", "image-digest"),
        "mask": ("image/png", b"mask", "mask-digest"),
    }
    assert session._materialize_scene()
    session.register_media_track("track-canvas", "input/frame/canvas", channel="canvas")
    session.update_media_track_frame("track-canvas", Image.new("RGB", (2, 2), (20, 40, 60)))

    marks: list[bool] = []
    monkeypatch.setattr(session, "_mark_staging_changed", lambda semantic_reset=False: marks.append(bool(semantic_reset)))

    if apply_method == "patch":
        session.apply_scene_patch({"settings": {"input_revision": 8}}, seq=1)
    else:
        session.apply_scene_events([
            {"type": "property.set", "path": ["settings", "input_revision"], "value": 8},
        ], seq=1)

    assert session._pending_live_media_revision == 0
    assert marks == [False]