"""InpaintSession — one browser connection's inference session for RTC streaming."""
from __future__ import annotations

import asyncio
import base64
import copy
import io
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image, ImageChops

from ..stream import SessionManager as _SessionManager
from ..stream.helpers import _TRITON_OK as _STREAM_TRITON_OK
from ..pipeline_graph import SessionGraph, GraphResult, resolve_conditioning_mask
from ..composition import CFG_HI
from ..render_plan import plan_from_wire
from ..session_paths import ensure_session_layout, session_debug_path, session_output_path
from ..inference import (
    CondInput,
    ControlNetSpec,
    FrameRequest,
    PromptBundle,
    SamplerSpec,
    StreamInferenceSession,
)

logger = logging.getLogger("rtdiffusion.rtc")

_peer_sessions: dict[str, "InpaintSession"] = {}

_shared_session_manager = _SessionManager()

_CANVAS_ANCHOR_MOTION = 0.55
_CANVAS_ANCHOR_STATIC = 1.0

_DEFAULT_INFER_W = 512
_DEFAULT_INFER_H = 512
_MASK_F32_MIME = "application/x-rtd-mask-f32"
_MASK_F32_MAGIC = b"RTF1"

def _apply_css_affine(img: Image.Image, css: list[float]) -> Image.Image:
    a, b, c, d, e, f = (float(v) for v in css)
    det = a * d - b * c
    if abs(det) < 1e-10:
        return img
    inv = (
        d / det, -c / det, (c * f - d * e) / det,
        -b / det, a / det, (b * e - a * f) / det,
    )
    return img.transform(img.size, Image.AFFINE, inv, Image.BICUBIC)


def _decode_data_url(data_url: str) -> bytes:
    _, sep, payload = data_url.partition(",")
    return base64.b64decode(payload if sep else data_url)


def _data_url(mime: str, data: bytes) -> str:
    return f"data:{mime or 'application/octet-stream'};base64,{base64.b64encode(data).decode()}"


def _data_url_mime_and_bytes(data_url: str) -> tuple[str, bytes]:
    header, separator, payload = data_url.partition(",")
    mime = "application/octet-stream"
    if separator and header.startswith("data:"):
        mime = header[5:].split(";", 1)[0] or mime
    return mime, base64.b64decode(payload if separator else data_url)


def _decode_f32_mask(data: bytes, w: int, h: int) -> Image.Image:
    if len(data) < 12 or data[:4] != _MASK_F32_MAGIC:
        raise ValueError("invalid float32 mask header")
    src_w = int.from_bytes(data[4:8], "little", signed=False)
    src_h = int.from_bytes(data[8:12], "little", signed=False)
    if src_w <= 0 or src_h <= 0:
        raise ValueError("invalid float32 mask dimensions")
    expected = 12 + src_w * src_h * 4
    if len(data) != expected:
        raise ValueError(f"invalid float32 mask payload size: expected {expected}, got {len(data)}")
    arr = np.frombuffer(data, dtype="<f4", offset=12).reshape((src_h, src_w)).astype(np.float32, copy=True)
    img = Image.fromarray(arr, "F")
    if img.size != (w, h):
        img = img.resize((w, h), Image.Resampling.BILINEAR)
    return img


def _jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _normalize_output_transport(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"image", "png", "image/png", "frame", "frames"}:
        return "image"
    return "video"


def _neutral_rgb(img: Image.Image, neutral: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    if img.mode not in ("RGBA", "LA") and "transparency" not in img.info:
        return img.convert("RGB")
    rgba = img.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*neutral, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _images_equal(left: Image.Image | None, right: Image.Image | None) -> bool:
    if left is None or right is None:
        return False
    if left.size != right.size or left.mode != right.mode:
        return False
    try:
        return ImageChops.difference(left, right).getbbox() is None
    except Exception:
        return False


def _empty_alpha_as_denoise_mask(img: Image.Image) -> Image.Image | None:
    """Return white-act mask for transparent pixels (alpha==0 => 255).

    This mirrors the inpaint engine behavior where empty canvas regions should be
    fully denoised from noise rather than inheriting any source RGB value.
    """
    if img.mode not in ("RGBA", "LA") and "transparency" not in img.info:
        return None
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    empty = ImageChops.invert(alpha)
    return empty if empty.getbbox() else None


def _parse_debug_color(value: object, fallback: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        return fallback
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return fallback


def _tinted_mask_preview(img: Image.Image, color: object) -> Image.Image:
    rgb = np.asarray(_parse_debug_color(color), dtype=np.float32)
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = np.asarray(img.convert("RGBA"), dtype=np.float32)
        coverage = (rgba[:, :, :3].max(axis=2) / 255.0) * (rgba[:, :, 3] / 255.0)
    elif img.mode == "RGB":
        coverage = np.asarray(img, dtype=np.float32).max(axis=2) / 255.0
    else:
        coverage = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    tinted = rgb[None, None, :] * np.clip(coverage, 0.0, 1.0)[:, :, None]
    return Image.fromarray(np.clip(tinted, 0, 255).astype(np.uint8), "RGB")


def _compose_region_color_preview(scene: Any, width: int, height: int) -> Image.Image:
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for layer in scene.layers:
        for region in layer.regions:
            if region.color is None:
                continue
            mask = region.color_mask if region.color_mask is not None else region.mask
            if mask is None:
                mask = Image.new("L", (width, height), 255)
            else:
                mask = mask.convert("L")
                if mask.size != (width, height):
                    mask = mask.resize((width, height), Image.Resampling.NEAREST)
            if mask.getbbox() is None:
                continue
            color = region.color.convert("RGBA")
            if color.size != (width, height):
                color = color.resize((width, height), Image.Resampling.BILINEAR)
            alpha = ImageChops.multiply(color.getchannel("A"), mask)
            color.putalpha(alpha)
            result = Image.alpha_composite(result, color)
    return result


def _cfg_factor_preview(cfg_map: Image.Image) -> Image.Image:
    values = np.asarray(cfg_map.convert("L"), dtype=np.float32) / 255.0 * CFG_HI
    preview = np.clip(values, 0.0, 1.0) * 255.0
    preview = np.where(values >= 0.9, 255.0, preview)
    return Image.fromarray(preview.round().astype(np.uint8), "L").convert("RGB")


def _encode_debug_channels(
    debug_channels: dict[str, Image.Image],
    *,
    include_images: bool,
    always_include: set[str] | None = None,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    if debug_channels:
        payload["meta:input_resources"] = ", ".join(sorted(debug_channels))
    for ch_name, ch_img in debug_channels.items():
        if not include_images and ch_name not in (always_include or set()):
            continue
        try:
            payload[ch_name] = _data_url("image/png", _png_bytes(ch_img))
        except Exception:
            payload[ch_name] = _data_url("image/png", _png_bytes(ch_img.convert("RGB")))
    return payload


def _debug_label(value: object, fallback: str = "unnamed") -> str:
    text = str(value or fallback).strip() or fallback
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:80]


def _merge_dict(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if value is None:
            target.pop(key, None)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _set_path(target: dict[str, Any], path: list[Any], value: Any) -> None:
    if not path:
        return
    cursor = target
    for raw_key in path[:-1]:
        key = str(raw_key)
        next_item = cursor.get(key)
        if not isinstance(next_item, dict):
            next_item = {}
            cursor[key] = next_item
        cursor = next_item
    cursor[str(path[-1])] = copy.deepcopy(value)


def _unset_path(target: dict[str, Any], path: list[Any]) -> None:
    if not path:
        return
    cursor = target
    for raw_key in path[:-1]:
        key = str(raw_key)
        next_item = cursor.get(key)
        if not isinstance(next_item, dict):
            return
        cursor = next_item
    cursor.pop(str(path[-1]), None)


def _upsert_signal(target: dict[str, Any], signal: dict[str, Any]) -> None:
    signal_id = str(signal.get("id") or "").strip()
    if not signal_id:
        return
    signals = target.get("signals")
    if not isinstance(signals, list):
        signals = []
    next_signal = copy.deepcopy(signal)
    replaced = False
    for index, item in enumerate(signals):
        if isinstance(item, dict) and str(item.get("id") or "") == signal_id:
            signals[index] = next_signal
            replaced = True
            break
    if not replaced:
        signals.append(next_signal)
    target["signals"] = signals


def _remove_signal(target: dict[str, Any], signal_id: str) -> None:
    signals = target.get("signals")
    if not isinstance(signals, list):
        return
    target["signals"] = [
        item for item in signals
        if not (isinstance(item, dict) and str(item.get("id") or "") == signal_id)
    ]


def _apply_scene_event(target: dict[str, Any], event: dict[str, Any]) -> None:
    event_type = str(event.get("type") or "")
    if event_type == "property.set":
        path = event.get("path")
        if isinstance(path, list):
            _set_path(target, path, event.get("value"))
    elif event_type == "property.unset":
        path = event.get("path")
        if isinstance(path, list):
            _unset_path(target, path)
    elif event_type == "input.update":
        input_spec = event.get("input")
        if isinstance(input_spec, dict):
            target["input"] = copy.deepcopy(input_spec)
    elif event_type == "layer_conditions.replace":
        conditions = event.get("layer_conditions")
        if isinstance(conditions, list):
            target["layer_conditions"] = copy.deepcopy(conditions)
    elif event_type in {"signal.add", "signal.update"}:
        signal = event.get("signal")
        if isinstance(signal, dict):
            _upsert_signal(target, signal)
    elif event_type == "signal.remove":
        _remove_signal(target, str(event.get("id") or event.get("signal_id") or ""))


class InpaintSession:
    """
    Inference session for one browser connection.

    apply_frame / apply_settings run in the asyncio event loop — they do O(1) work only.
    _infer_sync runs in a thread pool — all PIL and GPU work happens there.
    event_generator reads pre-encoded output bytes from the output queue.
    """

    def __init__(self, pc_id: str) -> None:
        self._pc_id = pc_id
        self._tag = pc_id[:8]

        self._settings: dict[str, Any] = {}
        self._output_transport: str = "video"
        # Raw canvas URL — set by event loop (string assign = O(1)), decoded in thread
        self._canvas_url: str | None = None
        self._scene_struct: dict[str, Any] = {}
        self._scene_resources: dict[str, tuple[str, bytes, str]] = {}
        self._scene_resource_debug_names: dict[str, str] = {}
        self._media_frames: dict[str, Image.Image] = {}
        self._media_frame_versions: dict[str, int] = {}
        self._media_frame_seq: int = 0
        self._last_consumed_primary_media_version: int = 0
        self._last_rendered_media_frame_seq: int = 0
        self._pending_live_media_revision: int = 0
        self._media_debug_names: dict[str, str] = {}
        self._media_layer_ids: dict[str, str] = {}
        self._media_channels: dict[str, str] = {}
        self._last_scene_id: str = ""
        self._last_scene_seq: int = 0
        self._motion_transform: list[float] | None = None
        self._input_generation: int = 0
        self._queued_scene_id: str = ""
        self._queued_at_iso: str = ""
        self._queued_at_perf: float = 0.0
        self._input_event = asyncio.Event()
        self._process_task: asyncio.Task[None] | None = None
        self._last_settings_seq: int = 0
        self._last_frame_seq: int = 0
        self._scene_revision: int = 0
        self._stream_reset_generation: int = 0
        self._last_stream_reset_generation: int = -1
        self._last_queue_log_at: float = 0.0
        self._queue_log_interval_s: float = 0.25
        self._last_signal_event_at: float = 0.0
        self._latest_output: Image.Image | None = _shared_session_manager.last_output
        self._running = True
        self._error_count = 0
        self._log_first_frame = True

        # Mask cache — the mask rarely changes so we avoid decoding it every frame
        self._cached_mask_url: str = ""
        self._cached_mask: Image.Image | None = None

        # Canvas image cache — skip base64/image decode when the URL hasn't changed
        self._cached_canvas_url: str = ""
        self._cached_canvas_img: Image.Image | None = None
        self._cached_canvas_empty_mask: Image.Image | None = None

        # Layer frame cache — keyed by layer_id, value (url, decoded_image)
        self._layer_frames_url: dict[str, str] = {}  # set O(1) in event loop
        self._cached_layer_frames: dict[str, tuple[str, Image.Image]] = {}  # (url, img) in thread

        # Condition image cache for _build_layer_masks — keyed by layer_id: (url, w, h, rgba)
        self._cached_condition_images: dict[str, tuple[str, int, int, Image.Image]] = {}

        # Binary blob store — realtime frontend streams raw PNG bytes over the
        # WebRTC resources data channel and references the returned ID in
        # ``layer_conditions[*]_mask_ref``. This bypasses the ~33 % base64
        # overhead for high-bandwidth mask channels. LRU-evicted at 128 entries.
        self._blobs: dict[str, Image.Image] = {}
        self._blob_debug_names: dict[str, str] = {}
        self._blob_order: list[str] = []
        self._BLOB_LIMIT = 128

        # Pipeline graph — rebuilt only when pipeline_nodes signature changes
        self._session_graph: SessionGraph | None = None
        self._graph_sig: str = ""

        # Settings log dedup — only log when prompt/model/device actually changes
        self._last_logged_sig: str = ""

        # Output queue: (encoded_output_bytes, debug_payload) | None.
        # Empty output bytes with
        # ``__event__`` in debug_payload is a control event for the SSE transport.
        # debug_payload is dict[str, str]: stream refs for images, plain text for text channels
        self._output_queue: asyncio.Queue[tuple[bytes, dict[str, Any]] | None] = asyncio.Queue()
        self._output_media_queues: set[asyncio.Queue[bytes | None]] = set()
        self._last_status_signature: tuple[Any, ...] | None = None
        # Throttle debug image encoding: only encode thumbnails every N frames
        self._debug_frame_count: int = 0
        _DEBUG_IMG_EVERY = 3  # encode thumbnails every 3rd inference frame

        # Pipeline state — written in thread pool, read in event loop (GIL-safe for simple assigns)
        self._pipeline_status: str = "idle"   # idle | loading | ready | error
        self._pipeline_renderer: str = "idle"
        self._pipeline_model: str = ""
        self._pipeline_error: str = ""
        self._pipeline_fps: float = 0.0
        self._pipeline_compile_status: dict[str, object] = {}
        self._render_dispatch_count: int = 0
        self._render_success_count: int = 0
        self._render_stale_count: int = 0
        self._render_skip_count: int = 0
        self._last_completed_generation: int = 0
        self._active_inference_started_perf: float = 0.0
        self._active_inference_generation: int = 0
        self._last_queue_wait_ms: float = 0.0
        self._last_latency_ms: float = 0.0
        self._last_end_to_end_ms: float = 0.0

        self._session_manager = _shared_session_manager

    @property
    def output_transport(self) -> str:
        return self._output_transport

    def telemetry_snapshot(self) -> dict[str, Any]:
        session = getattr(self._session_manager, "_session", None)
        engine = getattr(session, "engine", None)
        step_info = engine.inpaint_step_info if engine is not None else {}
        active_for_ms = 0.0
        inference_active = self._active_inference_started_perf > 0.0
        if inference_active:
            active_for_ms = max(0.0, (time.perf_counter() - self._active_inference_started_perf) * 1000.0)
        generation_backlog = max(0, int(self._input_generation) - int(self._last_completed_generation))
        queued_trigger_count = 1 if self._input_event.is_set() else 0
        pending_live_media = 1 if self._pending_live_media_revision > 0 else 0
        inflight_count = 1 if inference_active else 0
        pending_renders = max(generation_backlog, queued_trigger_count, pending_live_media, inflight_count)
        return {
            "session_id": self._pc_id,
            "tag": self._tag,
            "running": self._running,
            "connection_kind": "unknown",
            "output_transport": self._output_transport,
            "pipeline_status": self._pipeline_status,
            "pipeline_renderer": self._pipeline_renderer,
            "pipeline_model": self._pipeline_model,
            "pipeline_error": self._pipeline_error,
            "pipeline_fps": float(self._pipeline_fps),
            "compile_status": dict(self._pipeline_compile_status),
            "build_phase": self._session_manager.build_phase,
            "build_progress": float(self._session_manager.build_progress),
            "build_message": self._session_manager.build_message,
            "generation_backlog": generation_backlog,
            "input_generation": int(self._input_generation),
            "last_completed_generation": int(self._last_completed_generation),
            "pending_renders": pending_renders,
            "queued_scene_id": self._queued_scene_id,
            "queued_at": self._queued_at_iso,
            "pending_live_media_revision": int(self._pending_live_media_revision),
            "media_tracks": len(self._media_frames),
            "output_subscribers": len(self._output_media_queues),
            "scene_resources": len(self._scene_resources),
            "signal_count": len(self._scene_struct.get("signals") or []),
            "render_dispatch_count": int(self._render_dispatch_count),
            "render_success_count": int(self._render_success_count),
            "render_stale_count": int(self._render_stale_count),
            "render_skip_count": int(self._render_skip_count),
            "render_error_count": int(self._error_count),
            "inference_active": inference_active,
            "active_inference_generation": int(self._active_inference_generation),
            "active_for_ms": active_for_ms,
            "last_queue_wait_ms": float(self._last_queue_wait_ms),
            "last_latency_ms": float(self._last_latency_ms),
            "last_end_to_end_ms": float(self._last_end_to_end_ms),
            "step_info": step_info,
        }

    @property
    def output_frame_mime(self) -> str:
        return "image/png" if self._output_transport == "image" else "image/jpeg"

    @property
    def output_file_suffix(self) -> str:
        return ".png" if self._output_transport == "image" else ".jpg"

    def set_output_transport(self, transport: object) -> None:
        self._output_transport = _normalize_output_transport(transport)

    def _encode_output_bytes(self, img: Image.Image) -> bytes:
        return _png_bytes(img) if self._output_transport == "image" else _jpeg_bytes(img)

    def start(self) -> None:
        self._ensure_process_loop()

    def _ensure_process_loop(self) -> None:
        if not self._running:
            return
        task = self._process_task
        if task is not None and not task.done():
            return
        loop = asyncio.get_running_loop()
        self._process_task = loop.create_task(self._process_loop(), name=f"rtc-proc-{self._tag}")
        self._process_task.add_done_callback(self._on_process_loop_done)

    def _on_process_loop_done(self, task: asyncio.Task[None]) -> None:
        if self._process_task is task:
            self._process_task = None
        if not self._running or task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.error(
                "RTC[%s] process loop exited unexpectedly",
                self._tag,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self._pipeline_status = "error"
            self._pipeline_error = f"{type(exc).__name__}: {exc}"
            self._notify_status()

    # ── Binary blob store ─────────────────────────────────────────────

    def store_blob(self, data: bytes, name: str | None = None, mime: str = "image/png") -> str:
        """Decode ``data`` (raw PNG bytes) into a PIL Image and cache it under
        a content-hash key. Returns the key for the frontend to reference.

        LRU-evicts the oldest entry once the cache exceeds ``_BLOB_LIMIT``.
        Returning an existing key when the same content is uploaded twice is
        intentional — the user can paint the same mask repeatedly without
        ballooning the store.
        """
        import hashlib
        key = hashlib.blake2b(data, digest_size=12).hexdigest()
        if key in self._blobs:
            if name:
                self._blob_debug_names[key] = _debug_label(name, key)
            # Bump to MRU
            try:
                self._blob_order.remove(key)
            except ValueError:
                pass
            self._blob_order.append(key)
            return key
        try:
            if (mime or "").split(";", 1)[0].lower() == _MASK_F32_MIME:
                src_w = int.from_bytes(data[4:8], "little", signed=False) if len(data) >= 12 else 0
                src_h = int.from_bytes(data[8:12], "little", signed=False) if len(data) >= 12 else 0
                img = _decode_f32_mask(data, src_w, src_h)
            else:
                img = Image.open(io.BytesIO(data))
                # Eager-load so subsequent access doesn't hit the file descriptor.
                img.load()
        except Exception as exc:
            raise ValueError(f"invalid mask/image payload: {exc}") from exc
        # Convert masks to L-mode for storage compactness; color blobs stay RGB.
        if img.mode in ("LA", "1"):
            img = img.convert("L")
        elif img.mode == "P":
            img = img.convert("RGBA")
        self._blobs[key] = img
        if name:
            self._blob_debug_names[key] = _debug_label(name, key)
        self._blob_order.append(key)
        while len(self._blob_order) > self._BLOB_LIMIT:
            oldest = self._blob_order.pop(0)
            self._blobs.pop(oldest, None)
            self._blob_debug_names.pop(oldest, None)
        return key

    def get_blob(self, blob_id: str) -> Image.Image | None:
        return self._blobs.get(blob_id)

    def store_scene_resource(self, resource_id: str, data: bytes, mime: str = "application/octet-stream", name: str = "") -> None:
        """Store an immutable scene resource blob for the normalized RTC protocol."""
        resource_key = str(resource_id or "").strip()
        if not resource_key:
            raise ValueError("resource id is required")
        import hashlib
        digest = hashlib.blake2b(data, digest_size=12).hexdigest()
        previous = self._scene_resources.get(resource_key)
        if previous and previous[2] == digest:
            if name:
                self._scene_resource_debug_names[resource_key] = _debug_label(name, resource_key)
            return
        self._scene_resources[resource_key] = (mime or "application/octet-stream", data, digest)
        if name:
            self._scene_resource_debug_names[resource_key] = _debug_label(name, resource_key)
        if resource_key in self._scene_resource_refs() and self._scene_resources_ready() and self._materialize_scene():
            # Resource streams are often high-frequency live inputs; keep latent continuity.
            if bool(self._settings.get("stream_diffusion")):
                self._stamp_render_trigger()
                self._input_event.set()
            else:
                self._mark_staging_changed(semantic_reset=False)

    def close(self) -> None:
        self._running = False
        self._input_event.set()
        if self._process_task is not None:
            self._process_task.cancel()
        try:
            self._output_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        for queue in list(self._output_media_queues):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    @property
    def has_output_media_subscribers(self) -> bool:
        return bool(self._output_media_queues)

    def subscribe_output_media(self) -> asyncio.Queue[bytes | None]:
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._output_media_queues.add(queue)
        return queue

    def unsubscribe_output_media(self, queue: asyncio.Queue[bytes | None]) -> None:
        self._output_media_queues.discard(queue)

    def _publish_output_media(self, jpeg: bytes) -> None:
        for queue in list(self._output_media_queues):
            try:
                queue.put_nowait(jpeg)
            except asyncio.QueueFull:
                pass

    def _latest_primary_media_version(self) -> int:
        latest = 0
        for track_id, version in self._media_frame_versions.items():
            if self._is_primary_media_track(track_id):
                latest = max(latest, version)
        return latest

    def _live_input_revision(self, patch: dict[str, Any]) -> int:
        settings = patch.get("settings")
        if not isinstance(settings, dict) or "input_revision" not in settings:
            return 0
        try:
            revision = int(settings.get("input_revision") or 0)
        except (TypeError, ValueError):
            return 0
        return max(0, revision)

    def _should_wait_for_live_media(self, patch: dict[str, Any]) -> int:
        revision = self._live_input_revision(patch)
        if revision <= 0:
            return 0
        if self._latest_primary_media_version() > self._last_consumed_primary_media_version:
            return 0
        return revision

    @staticmethod
    def _settings_revision(settings: dict[str, Any], key: str) -> int:
        try:
            return int(settings.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _should_drop_pre_live_stale_output(cls, rendered_settings: dict[str, Any], latest_settings: dict[str, Any]) -> bool:
        rendered_input_revision = cls._settings_revision(rendered_settings, "input_revision")
        latest_input_revision = cls._settings_revision(latest_settings, "input_revision")
        rendered_scene_revision = cls._settings_revision(rendered_settings, "_scene_revision")
        return latest_input_revision > 0 and rendered_input_revision <= 0 and rendered_scene_revision <= 0

    def _mark_staging_changed(self, *, semantic_reset: bool = False) -> None:
        try:
            self._ensure_process_loop()
        except RuntimeError:
            pass
        now = time.monotonic()
        self._input_generation += 1
        if semantic_reset:
            self._stream_reset_generation += 1
        self._stamp_render_trigger(now)
        self._input_event.set()
        if semantic_reset or (now - self._last_queue_log_at) >= self._queue_log_interval_s:
            self._last_queue_log_at = now
            logger.info(
                "RTC[%s] queued scene for rendering: scene_id=%s gen=%d queued_at=%s",
                self._tag,
                self._queued_scene_id,
                self._input_generation,
                self._queued_at_iso,
            )

    def _stamp_render_trigger(self, now: float | None = None) -> None:
        self._last_signal_event_at = now if now is not None else time.monotonic()
        self._queued_scene_id = self._last_scene_id or str(self._scene_struct.get("id") or "legacy")
        self._queued_at_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        self._queued_at_perf = time.perf_counter()

    def _resource_data_url(self, name: str) -> str:
        item = self._scene_resources.get(str(name or ""))
        if item is None:
            return ""
        mime, data, _digest = item
        return _data_url(mime, data)

    def _scene_resource_refs(self) -> set[str]:
        """Return immutable resource IDs referenced by the current scene."""
        refs: set[str] = set()

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, str) and (
                        key in {"image_ref", "mask_ref", "frame_ref"}
                        or key.endswith("_ref")
                        or key.endswith("_ref_name")
                    ):
                        ref = item.strip()
                        if ref:
                            refs.add(ref)
                    else:
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(self._scene_struct)
        return refs

    def _scene_resources_ready(self) -> bool:
        refs = self._scene_resource_refs()
        return all(ref in self._scene_resources for ref in refs)

    def _materialize_scene(self) -> bool:
        """Compose the latest scene structure and named resources into renderer inputs."""
        if not self._scene_struct or not self._scene_resources_ready():
            return False
        scene = self._scene_struct
        previous_settings = self._settings
        previous_canvas = self._canvas_url
        previous_layers = self._layer_frames_url
        previous_transform = self._motion_transform

        settings = copy.deepcopy(scene.get("settings") or {})
        ensure_session_layout(str(settings.get("session_directory") or ""))
        input_spec = scene.get("input") or {}
        canvas_url = self._resource_data_url(str(input_spec.get("image_ref") or ""))
        mask_ref = str(input_spec.get("mask_ref") or "")
        if mask_ref:
            settings["mask"] = self._resource_data_url(mask_ref)

        signal_refs: dict[tuple[str, str, str], str] = {}
        for signal in scene.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            layer_id = str(signal.get("layer_id") or "")
            region_id = str(signal.get("region_id") or "")
            ref = str(signal.get("resource_ref") or "")
            if not layer_id or not region_id or not ref:
                continue
            signal_type = str(signal.get("type") or "")
            channel = str(signal.get("channel") or "")
            if channel == "color" and signal_type == "rgba":
                channel = "image"
            elif signal_type == "prompt_mask":
                channel = "prompt"
            elif signal_type == "denoise_mask":
                channel = "denoise"
            elif signal_type == "cfg_mask":
                channel = "cfg"
            elif signal_type == "rgba_mask":
                channel = "color"
            elif signal_type == "controlnet_image":
                channel = "controlnet"
            if channel:
                signal_refs[(layer_id, region_id, channel)] = ref

        layer_frames: dict[str, str] = {}
        layer_conditions: list[dict[str, Any]] = []
        for cond in scene.get("layer_conditions") or []:
            if not isinstance(cond, dict):
                continue
            item = copy.deepcopy(cond)
            image_ref = str(item.pop("image_ref", "") or "")
            layer_id = str(item.get("layer_id") or "")
            region_id = str(item.get("region_id") or item.get("name") or "")
            if not image_ref and layer_id and region_id:
                image_ref = signal_refs.get((layer_id, region_id, "image"), "")
            if image_ref:
                item["image"] = self._resource_data_url(image_ref)
            for field, target in (
                ("denoise_mask_ref_name", "denoise_mask"),
                ("prompt_mask_ref_name", "prompt_mask"),
                ("cfg_mask_ref_name", "cfg_mask"),
                ("color_mask_ref_name", "color_mask"),
                ("color_image_ref_name", "color_image"),
                ("controlnet_image_ref_name", "controlnet_image"),
            ):
                ref_name = str(item.pop(field, "") or "")
                if not ref_name and layer_id and region_id:
                    signal_channel = {
                        "denoise_mask": "denoise",
                        "prompt_mask": "prompt",
                        "cfg_mask": "cfg",
                        "color_mask": "color",
                        "color_image": "color_image",
                        "controlnet_image": "controlnet",
                    }[target]
                    ref_name = signal_refs.get((layer_id, region_id, signal_channel), "")
                if ref_name:
                    item[target] = self._resource_data_url(ref_name)
            frame_ref = str(item.pop("frame_ref", "") or "")
            if frame_ref and layer_id:
                layer_frames[layer_id] = self._resource_data_url(frame_ref)
            layer_conditions.append(item)
        settings["layer_conditions"] = layer_conditions

        transform = input_spec.get("transform")
        motion_transform = [float(v) for v in transform] if isinstance(transform, list) and len(transform) == 6 else None
        changed = settings != previous_settings or canvas_url != previous_canvas or layer_frames != previous_layers or motion_transform != previous_transform
        self._settings = settings
        self._canvas_url = canvas_url or None
        self._layer_frames_url = layer_frames
        self._motion_transform = motion_transform
        return changed

    # ── inbound data (event loop — must be O(1)) ───────────────────

    def _message_seq(self, seq: Any) -> int | None:
        try:
            value = int(seq)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def apply_settings(self, settings: dict[str, Any], seq: Any = None) -> None:
        msg_seq = self._message_seq(seq)
        if msg_seq is not None:
            if msg_seq <= self._last_settings_seq:
                return
            self._last_settings_seq = msg_seq
        settings.setdefault("type", "settings")
        self.set_output_transport(settings.get("output_transport", self._output_transport))
        if settings != self._settings:
            self._mark_staging_changed(semantic_reset=True)
        self._settings = settings
        # Only log when the user-visible settings actually changed
        sig = f"{settings.get('prompt','')}|{settings.get('model_path','')}|{settings.get('device','')}"
        if sig != self._last_logged_sig:
            self._last_logged_sig = sig
            logger.info(
                "RTC[%s] settings changed: prompt=%r model=%r",
                self._tag,
                settings.get("prompt", "")[:60],
                (settings.get("model_path") or "")[-40:],
            )

    def apply_scene(self, scene: dict[str, Any], seq: Any = None) -> None:
        msg_seq = self._message_seq(seq)
        if msg_seq is not None:
            if msg_seq <= self._last_scene_seq:
                return
            self._last_scene_seq = msg_seq
        scene_id = str(scene.get("id") or "").strip()
        if scene_id and scene_id == self._last_scene_id:
            return
        if not scene_id and scene == self._scene_struct:
            return
        self._scene_struct = scene
        self._last_scene_id = scene_id
        if self._materialize_scene():
            self._mark_staging_changed(semantic_reset=True)
        sig = f"{self._settings.get('prompt','')}|{self._settings.get('model_path','')}|{self._settings.get('device','')}"
        if sig != self._last_logged_sig:
            self._last_logged_sig = sig
            logger.info(
                "RTC[%s] scene changed: prompt=%r model=%r resources=%d",
                self._tag,
                self._settings.get("prompt", "")[:60],
                (self._settings.get("model_path") or "")[-40:],
                len(self._scene_resources),
            )

    def apply_scene_patch(self, patch: dict[str, Any], seq: Any = None) -> None:
        msg_seq = self._message_seq(seq)
        if msg_seq is not None:
            if msg_seq <= self._last_scene_seq:
                return
            self._last_scene_seq = msg_seq
        if not self._scene_struct:
            return
        previous_input_revision = self._settings_revision(self._settings, "input_revision")
        next_scene = copy.deepcopy(self._scene_struct)
        _merge_dict(next_scene, patch)
        next_scene.pop("id", None)
        if next_scene == self._scene_struct:
            # Patch produced no net change. Acknowledge immediately so the frontend
            # is not left waiting for an input_ready that would never arrive.
            self._notify_input_ready(self._last_scene_id or "", self._input_generation, "nochange")
            return
        previous_scene = self._scene_struct
        self._scene_struct = next_scene
        self._last_scene_id = ""
        if self._materialize_scene():
            live_input_revision = self._live_input_revision(patch)
            if live_input_revision > previous_input_revision and bool(self._settings.get("stream_diffusion")):
                # In stream mode, live input revisions can arrive at brush/video rate.
                # Keep latest scene state but let media-frame arrivals drive cadence.
                self._pending_live_media_revision = max(self._pending_live_media_revision, live_input_revision)
                self._stamp_render_trigger()
                self._input_event.set()
                return
            if live_input_revision > previous_input_revision:
                if self._scene_without_live_input_revision(previous_scene) == self._scene_without_live_input_revision(next_scene):
                    pending_revision = self._should_wait_for_live_media(patch)
                    if pending_revision:
                        self._pending_live_media_revision = max(self._pending_live_media_revision, pending_revision)
                        logger.info(
                            "RTC[%s] deferring live input revision %d until next media frame",
                            self._tag,
                            pending_revision,
                        )
                        return
                    # The latest primary media frame is already available, so this
                    # input revision is immediately renderable.
                    self._mark_staging_changed(semantic_reset=False)
                    return
            pending_revision = self._should_wait_for_live_media(patch) if live_input_revision > previous_input_revision else 0
            if pending_revision:
                self._pending_live_media_revision = max(self._pending_live_media_revision, pending_revision)
                logger.info(
                    "RTC[%s] deferring live input revision %d until next media frame",
                    self._tag,
                    pending_revision,
                )
                return
            if live_input_revision > previous_input_revision:
                # Live drawing revision: refresh without resetting latent history.
                self._mark_staging_changed(semantic_reset=False)
                return
            self._scene_revision += 1
            self._settings["_scene_revision"] = self._scene_revision
            self._mark_staging_changed(semantic_reset=True)

    def apply_scene_events(self, events: list[Any], seq: Any = None) -> None:
        msg_seq = self._message_seq(seq)
        if msg_seq is not None:
            if msg_seq <= self._last_scene_seq:
                return
            self._last_scene_seq = msg_seq
        if not self._scene_struct:
            return
        previous_input_revision = self._settings_revision(self._settings, "input_revision")
        next_scene = copy.deepcopy(self._scene_struct)
        for event in events:
            if isinstance(event, dict):
                _apply_scene_event(next_scene, event)
        next_scene.pop("id", None)
        if next_scene == self._scene_struct:
            # Events produced no net change. Acknowledge immediately so the frontend
            # is not left waiting for an input_ready that would never arrive.
            self._notify_input_ready(self._last_scene_id or "", self._input_generation, "nochange")
            return
        previous_scene = self._scene_struct
        self._scene_struct = next_scene
        self._last_scene_id = ""
        if self._materialize_scene():
            patch_for_revision = {"settings": next_scene.get("settings") or {}}
            live_input_revision = self._live_input_revision(patch_for_revision)
            if live_input_revision > previous_input_revision and bool(self._settings.get("stream_diffusion")):
                # In stream mode, live input revisions can arrive at brush/video rate.
                # Keep latest scene state but let media-frame arrivals drive cadence.
                self._pending_live_media_revision = max(self._pending_live_media_revision, live_input_revision)
                self._stamp_render_trigger()
                self._input_event.set()
                return
            if live_input_revision > previous_input_revision:
                if self._scene_without_live_input_revision(previous_scene) == self._scene_without_live_input_revision(next_scene):
                    pending_revision = self._should_wait_for_live_media(patch_for_revision)
                    if pending_revision:
                        self._pending_live_media_revision = max(self._pending_live_media_revision, pending_revision)
                        logger.info(
                            "RTC[%s] deferring live input revision %d until next media frame",
                            self._tag,
                            pending_revision,
                        )
                        return
                    # The latest primary media frame is already available, so this
                    # input revision is immediately renderable.
                    self._mark_staging_changed(semantic_reset=False)
                    return
            pending_revision = self._should_wait_for_live_media(patch_for_revision) if live_input_revision > previous_input_revision else 0
            if pending_revision:
                self._pending_live_media_revision = max(self._pending_live_media_revision, pending_revision)
                logger.info(
                    "RTC[%s] deferring live input revision %d until next media frame",
                    self._tag,
                    pending_revision,
                )
                return
            if live_input_revision > previous_input_revision:
                # Live drawing revision: refresh without resetting latent history.
                self._mark_staging_changed(semantic_reset=False)
                return
            self._scene_revision += 1
            self._settings["_scene_revision"] = self._scene_revision
            self._mark_staging_changed(semantic_reset=True)

    def register_media_track(self, track_id: str, label: str = "", layer_id: str = "", channel: str = "") -> None:
        key = str(track_id or "").strip()
        if not key:
            return
        self._media_debug_names[key] = _debug_label(label or key, key)
        if layer_id:
            self._media_layer_ids[key] = str(layer_id)
        if channel:
            self._media_channels[key] = self._normalize_media_channel(str(channel))

    @staticmethod
    def _normalize_media_channel(channel: str) -> str:
        raw = str(channel or "").strip().lower()
        if not raw:
            return ""
        if raw in {"rgba", "rgb", "image", "video", "main", "primary"}:
            return "color"
        return raw

    def _is_primary_media_track(self, track_id: str) -> bool:
        channel = self._normalize_media_channel(self._media_channels.get(track_id, ""))
        layer_id = self._media_layer_ids.get(track_id, "")
        if layer_id:
            return False
        if channel in {"canvas", "input", "frame"}:
            return True
        # Backward-compat fallback for older clients that did not set channel.
        label = str(self._media_debug_names.get(track_id, "")).lower()
        if "input/frame" in label or "input:canvas" in label:
            return True
        # Some clients attach the inbound canvas track before the separate
        # `media_track` metadata message lands, leaving the track unlabeled.
        # Treat top-level color/unknown tracks as primary input so live-draw
        # revisions can still consume the freshest media frame.
        return channel in {"", "color"}

    @staticmethod
    def _scene_without_live_input_revision(scene: dict[str, Any]) -> dict[str, Any]:
        clone = copy.deepcopy(scene)
        settings = clone.get("settings")
        if isinstance(settings, dict):
            settings.pop("input_revision", None)
        return clone

    def update_media_track_frame(self, track_id: str, image: Image.Image) -> None:
        key = str(track_id or "").strip()
        if not key:
            return
        is_primary_track = self._is_primary_media_track(key)
        next_image = _neutral_rgb(image)
        previous_image = self._media_frames.get(key)
        if is_primary_track and _images_equal(previous_image, next_image):
            return
        self._media_frames[key] = next_image
        self._media_frame_seq += 1
        self._media_frame_versions[key] = self._media_frame_seq
        if self._pending_live_media_revision:
            # Deferred live scene updates become renderable once media arrives.
            self._pending_live_media_revision = 0
            if bool(self._settings.get("stream_diffusion")):
                self._stamp_render_trigger()
                self._input_event.set()
            else:
                self._mark_staging_changed(semantic_reset=False)
            return
        if is_primary_track and not bool(self._settings.get("stream_diffusion")):
            return
        # Wake the render loop, but let it coalesce multiple incoming frames into
        # a single render pass using the freshest available frame set.
        self._stamp_render_trigger()
        self._input_event.set()

    def remove_media_track(self, track_id: str) -> None:
        key = str(track_id or "").strip()
        if not key:
            return
        self._media_frames.pop(key, None)
        self._media_frame_versions.pop(key, None)
        self._media_debug_names.pop(key, None)
        self._media_layer_ids.pop(key, None)
        self._media_channels.pop(key, None)

    def apply_frame(
        self,
        img_url: str,
        transform: list[float] | None,
        layer_frames: dict[str, str] | None = None,
        seq: Any = None,
    ) -> None:
        msg_seq = self._message_seq(seq)
        if msg_seq is not None:
            if msg_seq <= self._last_frame_seq:
                return
            self._last_frame_seq = msg_seq
        # No PIL work here — just store the raw URL strings and transform.
        # Decoding happens in _infer_sync (thread pool).
        changed = False
        if img_url:
            if self._log_first_frame:
                self._log_first_frame = False
                logger.info("RTC[%s] first canvas frame received", self._tag)
            changed = img_url != self._canvas_url
            self._canvas_url = img_url
        if transform != self._motion_transform:
            changed = True
        self._motion_transform = transform
        if layer_frames and layer_frames != self._layer_frames_url:
            changed = True
            self._layer_frames_url = layer_frames
        if changed:
            # In StreamDiffusion mode, frame updates arrive at paint/video rate.
            # Do not bump generation for each frame; just wake the loop so it
            # renders the freshest snapshot and preserves latent continuity.
            if bool(self._settings.get("stream_diffusion")):
                self._stamp_render_trigger()
                self._input_event.set()
            else:
                self._mark_staging_changed(semantic_reset=False)

    # ── process loop ───────────────────────────────────────────────

    async def _process_loop(self) -> None:
        logger.info("RTC[%s] process loop started", self._tag)
        _last_generation = -1
        _last_published_input_generation = -1
        _last_stream_state_key: tuple[int, int] | None = None
        _stream_state_pass_count = 0
        while self._running:
            await self._input_event.wait()
            self._input_event.clear()
            try:
                # Staging discipline: read the LATEST canvas / motion / settings
                # *just before* dispatching to the GPU thread. The user might
                # have drawn many strokes while a previous frame was inferring;
                # this snapshot ensures the next inference uses the freshest
                # input.
                if not self._settings:
                    continue
                media_frame_seq = self._media_frame_seq
                input_generation = self._input_generation
                stream_mode = bool(self._settings.get("stream_diffusion"))
                stream_max_passes = max(1, int(self._settings.get("stream_max_passes") or 16))
                if self._input_generation == _last_published_input_generation and media_frame_seq == self._last_rendered_media_frame_seq and not stream_mode:
                    continue
                canvas_url = self._canvas_url or ""
                settings = copy.deepcopy(self._settings)
                layer_frames_url = dict(self._layer_frames_url)
                motion_transform = list(self._motion_transform) if self._motion_transform is not None else None
                last_output = self._latest_output
                settings["_stream_reset_generation"] = self._stream_reset_generation
                input_media_seq = media_frame_seq
                queued_scene_id = self._queued_scene_id
                queued_at_iso = self._queued_at_iso
                queue_wait_ms = max(0.0, (time.perf_counter() - self._queued_at_perf) * 1000.0) if self._queued_at_perf > 0 else 0.0
                render_start_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                self._render_dispatch_count += 1
                self._active_inference_started_perf = time.perf_counter()
                self._active_inference_generation = input_generation
                self._last_queue_wait_ms = queue_wait_ms
                logger.info(
                    "RTC[%s] dispatching queued scene: scene_id=%s gen=%d queued_at=%s render_start_at=%s",
                    self._tag,
                    queued_scene_id,
                    input_generation,
                    queued_at_iso,
                    render_start_at,
                )
                result = await asyncio.to_thread(
                    self._infer_sync, canvas_url, settings,
                    layer_frames_url, last_output, motion_transform, input_generation,
                )
                if not self._running:
                    break
                if result is not None:
                    img, jpeg, debug_jpgs, frame_meta = result
                    timings = frame_meta.get("timings") if isinstance(frame_meta, dict) else None
                    if isinstance(timings, dict):
                        timings.setdefault("queue_wait_ms", queue_wait_ms)
                        server_total = float(timings.get("server_total_ms") or 0.0)
                        timings["server_end_to_end_ms"] = round(queue_wait_ms + server_total, 3)
                        timings["queued_at"] = queued_at_iso
                        timings["render_started_at"] = render_start_at
                        self._last_end_to_end_ms = float(timings.get("server_end_to_end_ms") or 0.0)
                        self._last_latency_ms = float(frame_meta.get("latency_ms") or 0.0) if isinstance(frame_meta, dict) else 0.0
                    if self._input_generation != input_generation:
                        rendered_input_revision = self._settings_revision(settings, "input_revision")
                        logger.info(
                            "RTC[%s] publishing completed superseded output: rendered_gen=%d latest_gen=%d input_revision=%d",
                            self._tag,
                            input_generation,
                            self._input_generation,
                            rendered_input_revision,
                        )
                        self._latest_output = img
                        self._session_manager.last_output = img
                        try:
                            self._output_queue.put_nowait((jpeg, {**debug_jpgs, "__frame_meta__": frame_meta}))
                        except asyncio.QueueFull:
                            pass
                        self._publish_output_media(jpeg)
                        self._notify_render_done(queued_scene_id, input_generation, "stale_preview")
                        self._notify_input_ready(self._queued_scene_id, self._input_generation, "stale_preview")
                        self._notify_status()
                        self._last_rendered_media_frame_seq = input_media_seq
                        self._render_stale_count += 1
                        self._last_completed_generation = max(self._last_completed_generation, input_generation)
                        self._input_event.set()
                        _last_published_input_generation = -1
                        continue
                    cur_gen = self._session_manager._generation
                    if cur_gen != _last_generation:
                        _last_generation = cur_gen
                        logger.info("RTC[%s] new session gen=%d", self._tag, cur_gen)
                    self._latest_output = img
                    self._session_manager.last_output = img
                    try:
                        self._output_queue.put_nowait((jpeg, {**debug_jpgs, "__frame_meta__": frame_meta}))
                    except asyncio.QueueFull:
                        pass
                    self._publish_output_media(jpeg)
                    self._notify_input_ready(queued_scene_id, input_generation, "rendered")
                    self._notify_status()
                    self._last_rendered_media_frame_seq = input_media_seq
                    _last_published_input_generation = input_generation
                    self._render_success_count += 1
                    self._last_completed_generation = max(self._last_completed_generation, input_generation)
                    self._error_count = 0
                    if stream_mode:
                        stream_state_key = (input_generation, input_media_seq)
                        if _last_stream_state_key == stream_state_key:
                            _stream_state_pass_count += 1
                        else:
                            _last_stream_state_key = stream_state_key
                            _stream_state_pass_count = 1
                        if self._running and self._input_generation == input_generation and self._media_frame_seq == input_media_seq and _stream_state_pass_count < stream_max_passes:
                            self._input_event.set()
                    else:
                        _last_stream_state_key = None
                        _stream_state_pass_count = 0
                else:
                    # ``result is None`` means either:
                    #   (a) the pipeline session is still loading (build phase)
                    #   (b) the GPU lock was held by a concurrent connection
                    # Keep the latest generation alive; model loading / GPU lock
                    # release can complete without another frontend event.
                    self._notify_render_done(queued_scene_id, input_generation, "skipped")
                    self._notify_status()
                    self._last_rendered_media_frame_seq = input_media_seq
                    self._render_skip_count += 1
                    self._last_completed_generation = max(self._last_completed_generation, input_generation)
                    if self._running and self._input_generation == input_generation:
                        self._input_event.set()
            except Exception as _exc:
                self._error_count += 1
                self._pipeline_status = "error"
                self._pipeline_error = f"{type(_exc).__name__}: {_exc}"
                logger.exception("RTC[%s] inference error (count=%d)", self._tag, self._error_count)
                self._notify_render_done(self._queued_scene_id, self._input_generation, "error")
                self._notify_input_ready(self._queued_scene_id, self._input_generation, "error")
                self._notify_status()
                self._last_completed_generation = max(self._last_completed_generation, self._input_generation)
                _last_published_input_generation = self._input_generation
            finally:
                self._active_inference_started_perf = 0.0
                self._active_inference_generation = 0

    def _notify_render_done(self, scene_id: str, generation: int, status: str) -> None:
        try:
            self._output_queue.put_nowait((b"", {
                "__event__": "render_done",
                "scene_id": scene_id,
                "generation": str(generation),
                "status": status,
            }))
        except asyncio.QueueFull:
            pass

    def _notify_input_ready(self, scene_id: str, generation: int, reason: str) -> None:
        try:
            self._output_queue.put_nowait((b"", {
                "__event__": "input_ready",
                "scene_id": scene_id,
                "generation": str(generation),
                "reason": reason,
            }))
        except asyncio.QueueFull:
            pass

    def _notify_status(self) -> None:
        compile_items = tuple(sorted((str(k), str(v)) for k, v in self._pipeline_compile_status.items()))
        signature = (
            self._pipeline_status,
            self._pipeline_renderer,
            self._pipeline_model,
            round(float(self._pipeline_fps), 3),
            self._pipeline_error,
            self._session_manager.build_phase,
            round(float(self._session_manager.build_progress), 3),
            self._session_manager.build_message,
            compile_items,
        )
        if signature == self._last_status_signature:
            return
        self._last_status_signature = signature
        payload: dict[str, object] = {
            "__event__": "status",
            "phase": self._pipeline_status,
            "renderer": self._pipeline_renderer,
            "model": self._pipeline_model,
            "fps": self._pipeline_fps,
            "error": self._pipeline_error,
            "build_phase": self._session_manager.build_phase,
            "build_progress": self._session_manager.build_progress,
            "build_message": self._session_manager.build_message,
        }
        for key, value in self._pipeline_compile_status.items():
            payload[f"compile_{key}"] = value
        try:
            self._output_queue.put_nowait((b"", payload))
        except asyncio.QueueFull:
            pass

    @staticmethod
    def _round_timing_ms(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return round(max(0.0, float(value)), 3)

    @staticmethod
    def _pipe_compile_status(pipe: object | None, requested: bool) -> dict[str, object]:
        unet = getattr(pipe, "unet", None) if pipe is not None else None
        original = getattr(unet, "_orig_mod", None)
        return {
            "triton_available": _STREAM_TRITON_OK,
            "requested": requested,
            "active": original is not None,
            "unet_compiled": original is not None,
            "wrapper": type(unet).__name__ if unet is not None else "",
            "original": type(original).__name__ if original is not None else "",
        }

    @staticmethod
    def _renderer_name(mode: str, use_stream_diffusion: bool, model_path: str) -> str:
        if use_stream_diffusion:
            return "streamdiffusion"
        normalized = f"{mode} {model_path}".replace("\\", "/").lower()
        if "z-image" in normalized or "zimage" in normalized:
            return "zimage"
        if "xl" in normalized or "sdxl" in normalized:
            return "sdxl"
        if "sd/" in normalized or "stable-diffusion" in normalized or "inpaint" in normalized:
            return "sd"
        return "diffusion"

    def _decode_layer_frames(self, layer_frames_url: dict[str, str], w: int, h: int) -> dict[str, Image.Image]:
        """Decode layer frame data URLs, using cache to avoid redundant work."""
        result: dict[str, Image.Image] = {}
        for layer_id, url in layer_frames_url.items():
            cached = self._cached_layer_frames.get(layer_id)
            if cached and cached[0] == url:
                result[layer_id] = cached[1]
                continue
            try:
                img = _neutral_rgb(Image.open(io.BytesIO(_decode_data_url(url))))
                img = img.resize((w, h), Image.BILINEAR)
                if len(self._cached_layer_frames) >= 32:
                    self._cached_layer_frames.pop(next(iter(self._cached_layer_frames)))
                self._cached_layer_frames[layer_id] = (url, img)
                result[layer_id] = img
            except Exception as exc:
                logger.debug("RTC[%s] failed to decode layer frame %s: %s", self._tag, layer_id, exc)
        return result

    def _decode_channel_mask(self, url: str, w: int, h: int) -> "Image.Image | None":
        """Decode a base64 mask into an image sized to (w, h), or None.

        PNG masks decode to L. ``application/x-rtd-mask-f32`` decodes to F and
        carries absolute float values, used by CFG masks for higher precision.
        """
        if not url:
            return None
        try:
            mime, data = _data_url_mime_and_bytes(url)
            if mime.split(";", 1)[0].lower() == _MASK_F32_MIME:
                return _decode_f32_mask(data, w, h)
            img = Image.open(io.BytesIO(data))
            if img.mode not in ("L", "I;16", "I;16B", "I"):
                img = img.convert("L")
            if img.size != (w, h):
                img = img.resize((w, h), Image.NEAREST)
            return img
        except Exception as exc:
            logger.debug("RTC[%s] channel mask decode failed: %s", self._tag, exc)
            return None

    def _resize_to_canvas(self, img: "Image.Image", w: int, h: int, channel: str) -> "Image.Image":
        if channel == "cfg" and img.mode == "F":
            return img.resize((w, h), Image.BILINEAR) if img.size != (w, h) else img
        target_mode = "RGB" if channel == "color" else "L"
        if img.mode != target_mode:
            img = img.convert(target_mode)
        if img.size != (w, h):
            img = img.resize((w, h), Image.NEAREST if target_mode == "L" else Image.BILINEAR)
        return img

    def _build_channel_masks_by_region(
        self, layer_conditions: list[dict[str, Any]], w: int, h: int,
    ) -> dict[str, dict[str, "Image.Image"]]:
        """Resolve per-channel masks for each region.

        Two sources are accepted, in priority order:

             1. ``<channel>_mask_ref``: an ID returned by the session blob store.
                 The realtime frontend streams the painted PNG once over the
                 WebRTC resources data channel and stores the ID. Avoids base64
                 overhead on subsequent settings updates.
          2. ``<channel>_mask``: a base64 data URL inlined in the layer
             condition. Heavier on the wire but works without an extra
             round-trip; useful for clients that don't implement the blob
             protocol yet.

        Returns ``{region_id: {channel: PIL.Image}}`` consumed by
        ``composition.scene_from_legacy``. Missing channels are omitted from
        the inner dict; the composition module falls back to the legacy
        single-mask (alpha of ``image``).
        """
        out: dict[str, dict[str, "Image.Image"]] = {}
        for cond in layer_conditions:
            layer_id = str(cond.get("layer_id") or "")
            region_id = str(cond.get("region_id") or cond.get("name") or "")
            if not layer_id or not region_id:
                continue
            channels: dict[str, "Image.Image"] = {}
            for channel, mask_key, ref_key in (
                ("denoise", "denoise_mask", "denoise_mask_ref"),
                ("prompt", "prompt_mask", "prompt_mask_ref"),
                ("cfg", "cfg_mask", "cfg_mask_ref"),
                ("color", "color_mask", "color_mask_ref"),
            ):
                # Reference path (preferred — no base64 overhead).
                blob_id = str(cond.get(ref_key) or "")
                if blob_id:
                    blob = self.get_blob(blob_id)
                    if blob is not None:
                        channels[channel] = self._resize_to_canvas(blob, w, h, channel)
                        continue
                # Inline base64 fallback.
                url = str(cond.get(mask_key) or "")
                if not url:
                    continue
                decoded = self._decode_channel_mask(url, w, h)
                if decoded is not None:
                    channels[channel] = decoded
            if channels:
                out[f"{layer_id}:{region_id}"] = channels
        return out

    def _build_layer_masks(
        self, layer_conditions: list[dict[str, Any]], w: int, h: int
    ) -> tuple[dict[str, Image.Image], dict[str, Image.Image]]:
        """Decode condition RGBA images into per-layer alpha masks and RGB content.

        Returns:
            masks:  layer_id -> L-mode mask image (alpha channel, lighter-merged)
            images: layer_id -> RGB image (visual content of the layer, first condition wins)
        """
        masks: dict[str, Image.Image] = {}
        images: dict[str, Image.Image] = {}
        for cond in layer_conditions:
            lid = str(cond.get("layer_id") or "")
            if not lid:
                continue
            image_url = str(cond.get("image") or "")
            if not image_url:
                continue
            try:
                cached = self._cached_condition_images.get(lid)
                if cached and cached[0] == image_url and cached[1] == w and cached[2] == h:
                    rgba = cached[3]
                else:
                    rgba = Image.open(io.BytesIO(_decode_data_url(image_url))).convert("RGBA").resize((w, h), Image.BILINEAR)
                    if len(self._cached_condition_images) >= 32:
                        self._cached_condition_images.pop(next(iter(self._cached_condition_images)))
                    self._cached_condition_images[lid] = (image_url, w, h, rgba)
                alpha = rgba.getchannel("A")
                current = masks.get(lid)
                masks[lid] = alpha if current is None else ImageChops.lighter(current, alpha)
                if lid not in images:
                    images[lid] = _neutral_rgb(rgba)
            except Exception:
                continue
        return masks, images

    def _sort_layer_conditions_bottom_up(self, layer_conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not any("z_index" in cond for cond in layer_conditions):
            return layer_conditions
        return sorted(layer_conditions, key=lambda cond: float(cond.get("z_index") or 0.0))

    def _infer_sync(
        self,
        canvas_url: str,
        settings: dict[str, Any],
        layer_frames_url: dict[str, str],
        last_output: Image.Image | None,
        motion_transform: list[float] | None,
        expected_generation: int,
    ) -> tuple[Image.Image, bytes, dict[str, Any], dict[str, Any]] | None:
        """
        Runs in thread pool. All PIL and GPU work happens here.

        1. Decode canvas image (in-thread, no event loop blocking).
        2. Decode mask only if URL changed (cached).
        3. Decode layer frame images (cached by URL).
        4. Rebuild pipeline graph only when nodes change.
        5. Execute graph → prompt overrides + CN images.
        6. Compose input: blend canvas with motion-warped previous output.
        7. Run inference (StreamSession/Diffusers) with merged prompt.
        8. Encode output bytes (JPEG or PNG) in-thread.
        9. Collect debug stream labels (when debug_streams enabled).
        Returns (img, encoded_output, debug_channels) — debug_channels maps label → stream refs or text.
        """
        perf_started = time.perf_counter()
        timing_breakdown: dict[str, float] = {}

        def mark_timing(name: str, started_at: float) -> float:
            elapsed = self._round_timing_ms((time.perf_counter() - started_at) * 1000.0)
            timing_breakdown[name] = elapsed
            return elapsed

        # Fast stale bailout: if newer state arrived, don't spend time decoding
        # and preparing a frame that's already obsolete.
        if self._input_generation != expected_generation:
            return None

        w = int(settings.get("width", _DEFAULT_INFER_W))
        h = int(settings.get("height", _DEFAULT_INFER_H))

        # Decode canvas in thread — skip decode when URL unchanged (static canvas)
        stage_started = time.perf_counter()
        if canvas_url:
            if canvas_url != self._cached_canvas_url:
                try:
                    decoded_canvas = Image.open(io.BytesIO(_decode_data_url(canvas_url)))
                    self._cached_canvas_img = _neutral_rgb(decoded_canvas)
                    self._cached_canvas_empty_mask = _empty_alpha_as_denoise_mask(decoded_canvas)
                except Exception:
                    self._cached_canvas_img = Image.new("RGB", (w, h), (128, 128, 128))
                    self._cached_canvas_empty_mask = None
                self._cached_canvas_url = canvas_url
            canvas = self._cached_canvas_img if self._cached_canvas_img is not None else Image.new("RGB", (w, h), (128, 128, 128))
        else:
            canvas = Image.new("RGB", (w, h), (128, 128, 128))
            self._cached_canvas_empty_mask = None
        mark_timing("canvas_decode_ms", stage_started)

        stage_started = time.perf_counter()
        primary_media_candidates: list[tuple[int, str, Image.Image]] = []
        for track_id, media_img in self._media_frames.items():
            if self._is_primary_media_track(track_id):
                primary_media_candidates.append((self._media_frame_versions.get(track_id, 0), track_id, media_img))
        if primary_media_candidates:
            version, _track_id, media_img = max(primary_media_candidates, key=lambda item: item[0])
            media_canvas = media_img.resize((w, h), Image.BILINEAR) if media_img.size != (w, h) else media_img
            # Media frames from canvas tracks can arrive as RGB without alpha,
            # which turns fully transparent regions into black pixels.
            # Keep the latest media on opaque pixels only, and preserve the
            # neutralized canvas for transparent zones.
            if self._cached_canvas_empty_mask is not None:
                empty_mask = self._cached_canvas_empty_mask
                if empty_mask.size != (w, h):
                    empty_mask = empty_mask.resize((w, h), Image.NEAREST)
                opaque_mask = ImageChops.invert(empty_mask)
                canvas = Image.composite(media_canvas, canvas, opaque_mask)
            else:
                canvas = media_canvas
            if version > self._last_consumed_primary_media_version:
                self._last_consumed_primary_media_version = version
        mark_timing("primary_input_select_ms", stage_started)

        debug_enabled: bool = bool(settings.get("debug_streams"))
        signal_debug_enabled = bool((self._scene_struct or {}).get("signals"))
        debug_channels: dict[str, Image.Image] = {}

        stage_started = time.perf_counter()
        input_img = self._compose_input(canvas, last_output, motion_transform, w, h)
        mark_timing("input_compose_ms", stage_started)

        if debug_enabled:
            debug_channels["input/frame/canvas"] = canvas

        # Mask — decode only when URL changes (mask is usually static)
        stage_started = time.perf_counter()
        mask_url = settings.get("mask", "")
        if mask_url != self._cached_mask_url:
            if mask_url:
                try:
                    self._cached_mask = Image.open(
                        io.BytesIO(_decode_data_url(mask_url))
                    ).convert("L").resize((w, h), Image.NEAREST)
                except Exception:
                    self._cached_mask = None
            else:
                self._cached_mask = None
            self._cached_mask_url = mask_url
        mask = self._cached_mask
        if self._cached_canvas_empty_mask is not None:
            canvas_empty_mask = self._cached_canvas_empty_mask
            if canvas_empty_mask.size != (w, h):
                canvas_empty_mask = canvas_empty_mask.resize((w, h), Image.NEAREST)
            mask = canvas_empty_mask if mask is None else ImageChops.lighter(mask, canvas_empty_mask)
        mark_timing("mask_decode_ms", stage_started)

        # Decode layer frames (cached by URL)
        stage_started = time.perf_counter()
        layer_frames = self._decode_layer_frames(layer_frames_url, w, h)
        for track_id, media_img in self._media_frames.items():
            if self._media_channels.get(track_id, "color") != "color":
                continue
            layer_id = self._media_layer_ids.get(track_id, "")
            if layer_id:
                layer_frames[layer_id] = media_img.resize((w, h), Image.BILINEAR) if media_img.size != (w, h) else media_img
        mark_timing("layer_frames_decode_ms", stage_started)

        # Use video layer frame as primary SD input when available.
        # This ensures StreamDiffusion is driven by the actual video frame
        # rather than the Fabric.js canvas composite (which may lag one rAF cycle).
        stage_started = time.perf_counter()
        layer_conditions: list[dict[str, Any]] = self._sort_layer_conditions_bottom_up(settings.get("layer_conditions") or [])
        layer_masks, condition_images = self._build_layer_masks(layer_conditions, w, h)
        for _cond in layer_conditions:
            if _cond.get("primary_input") and _cond.get("layer_id") in layer_frames:
                input_img = layer_frames[_cond["layer_id"]]
                if debug_enabled:
                    debug_channels["video_frame"] = input_img
                break
        mark_timing("layer_conditions_prepare_ms", stage_started)

        if debug_enabled:
            debug_channels["input/frame/composed"] = input_img
            if not signal_debug_enabled:
                for lid, lf in layer_frames.items():
                    label = next(
                        (c.get("name", lid) for c in layer_conditions if c.get("layer_id") == lid),
                        lid,
                    )
                    debug_channels[f"input/layer_frame/{_debug_label(label, lid)}"] = lf
                for lid, layer_img in condition_images.items():
                    label = next(
                        (c.get("name", lid) for c in layer_conditions if c.get("layer_id") == lid),
                        lid,
                    )
                    debug_channels[f"input/layer/{_debug_label(label, lid)}/color"] = layer_img
                for lid, layer_mask in layer_masks.items():
                    label = next(
                        (c.get("name", lid) for c in layer_conditions if c.get("layer_id") == lid),
                        lid,
                    )
                    debug_channels[f"input/layer/{_debug_label(label, lid)}/alpha"] = layer_mask.convert("RGB")
                for blob_id, blob_img in self._blobs.items():
                    label = self._blob_debug_names.get(blob_id, blob_id)
                    debug_channels[f"input/resource/{label}"] = blob_img.convert("RGB")
                for resource_id in self._scene_resource_refs():
                    item = self._scene_resources.get(resource_id)
                    if not item:
                        continue
                    try:
                        res_img = Image.open(io.BytesIO(item[1]))
                        res_img.load()
                    except Exception:
                        continue
                    label = self._scene_resource_debug_names.get(resource_id, resource_id)
                    debug_channels[f"input/resource/{label}"] = res_img.convert("RGB")
            for track_id, media_img in self._media_frames.items():
                label = self._media_debug_names.get(track_id, track_id)
                debug_channels[f"input/media/{label}"] = media_img.convert("RGB")

        # Rebuild pipeline graph only when pipeline_nodes signature changes
        stage_started = time.perf_counter()
        nodes_raw: list[dict[str, Any]] = settings.get("pipeline_nodes") or []
        new_sig = SessionGraph.signature(nodes_raw)
        if new_sig != self._graph_sig:
            self._session_graph = SessionGraph(nodes_raw) if nodes_raw else None
            self._graph_sig = new_sig
        mark_timing("graph_build_ms", stage_started)

        # Collect primary-input layer ids (video layers whose frame is real scene content)
        primary_input_layers = {
            c["layer_id"] for c in layer_conditions
            if c.get("primary_input") and c.get("layer_id")
        }

        # Execute graph
        graph_result: GraphResult | None = None
        stage_started = time.perf_counter()
        if self._session_graph and layer_frames:
            try:
                graph_result = self._session_graph.run(
                    layer_frames,
                    layer_masks=layer_masks,
                    input_image=input_img,
                    primary_input_layers=primary_input_layers,
                    condition_images=condition_images,
                )
            except Exception as exc:
                logger.warning("RTC[%s] graph execution error: %s", self._tag, exc)
        mark_timing("graph_run_ms", stage_started)

        # Resolve conditioning mask first so resolved_strength is available for layer_regions.
        base_denoise = float(settings.get("strength", 1.0))
        resolved_mask = mask or Image.new("L", (w, h), 0)
        base_resolved_mask = resolved_mask.copy()
        resolved_strength = base_denoise
        resolved_denoise_map: Image.Image | None = None
        base_resolved_denoise_map: Image.Image | None = None
        stage_started = time.perf_counter()
        if layer_conditions:
            resolved = resolve_conditioning_mask(resolved_mask, layer_conditions, w, h, base_denoise)
            resolved_mask = resolved.mask
            resolved_strength = resolved.strength
            resolved_denoise_map = resolved.denoise_map
            resolved_mask_rgb = resolved.mask.convert("RGB")
            resolved_denoise_rgb = resolved.denoise_map.convert("RGB")
            if debug_enabled:
                debug_channels["condition_mask"] = resolved_mask_rgb
                debug_channels["condition_denoise"] = resolved_denoise_rgb
                debug_channels["input/mask/aggregated"] = resolved_mask_rgb
                debug_channels["input/mask/aggregated/denoise"] = resolved_denoise_rgb
                mark_timing("conditioning_resolve_ms", stage_started)

        # Composition plan: build a Scene from legacy layer_conditions then compose.
        # The StreamDiffusion path uses LAYERED_PASS (one infer per region with a prompt,
        # composited bottom-up); the single-pass plan supplies the merged prompt and
        # per-pixel denoise map used as the fallback when no region has its own prompt.
        base_prompt: str = settings.get("prompt") or ""
        base_neg: str = settings.get("negative_prompt") or ""
        overrides = graph_result.prompt_overrides if graph_result else {}
        # Decode per-channel masks (denoise_mask / prompt_mask / cfg_mask /
        # color_mask) so each region uses its own channel-specific weight map
        # for the corresponding parameter. Falls back to the legacy single
        # alpha mask when a channel is missing. See LAYER_SYSTEM.md §3.
        stage_started = time.perf_counter()
        channel_masks = self._build_channel_masks_by_region(layer_conditions, w, h)
        by_region: dict[str, dict[str, Any]] = {}
        for c in layer_conditions:
            layer_id = str(c.get("layer_id") or "")
            region_id = str(c.get("region_id") or c.get("name") or "")
            if region_id:
                by_region[region_id] = c
            if layer_id and region_id:
                by_region[f"{layer_id}:{region_id}"] = c
        for region_id, masks in channel_masks.items():
            cond_info: dict[str, Any] = by_region.get(region_id) or {}
            layer_id = str(cond_info.get("layer_id") or "layer")
            label = _debug_label(cond_info.get("name") or region_id, region_id)
            layer_label = _debug_label(layer_id, "layer")
            for channel, mask_img in masks.items():
                rendered = _tinted_mask_preview(mask_img, cond_info.get("mask_color")) if channel == "color" else mask_img.convert("RGB")
                if debug_enabled:
                    debug_channels[f"input/mask/{layer_label}/{label}/{channel}"] = rendered
                if channel not in {"cfg", "color"}:
                    debug_channels[f"final/{channel}/{layer_label}/{label}"] = rendered
        color_by_region: dict[str, Image.Image] = {}
        for cond in layer_conditions:
            layer_id = str(cond.get("layer_id") or "")
            region_id = str(cond.get("region_id") or cond.get("name") or "")
            image_url = str(cond.get("image") or "")
            if not layer_id or not region_id or not image_url:
                continue
            try:
                rgba = Image.open(io.BytesIO(_decode_data_url(image_url))).convert("RGBA")
                if rgba.size != (w, h):
                    rgba = rgba.resize((w, h), Image.BILINEAR)
                color_by_region[f"{layer_id}:{region_id}"] = rgba
            except Exception:
                continue
        # v2 composition: prefer the new wire format (rgba_b64 / cfg_map_b64 /
        # denoise_map_b64 / prompts[] / base_*) when present. The frontend
        # aggregates the layer stack and sends finals; we just decode + flatten
        # prompts into the legacy ``layer_regions`` shape the engine expects.
        # During the v1 -> v2 transition the old layer_conditions are still
        # decoded upstream for ControlNet / layer-frame / tagger plumbing,
        # but spatial aggregation now happens client-side.
        layer_regions: list[dict[str, Any]] = []
        merged_prompt = base_prompt
        composed_cfg = float(settings.get("base_cfg") or settings.get("cfg") or 1.0)
        if settings.get("rgba_b64") or settings.get("rgba_ref"):
            blob_resolver = lambda ref: self.get_blob(ref)  # noqa: E731
            plan = plan_from_wire(settings, blob_resolver=blob_resolver)
            composed_cfg = float(plan.base_cfg) if plan.base_cfg else composed_cfg
            for pi, p in enumerate(plan.prompts):
                if not p.text:
                    continue
                mask_arr = p.mask
                if mask_arr is not None:
                    mask_img = Image.fromarray((mask_arr * 255.0).round().astype("uint8"), "L")
                else:
                    mask_img = Image.new("L", (w, h), 255)
                # denoise_map for this region = aggregated denoise gated by mask.
                den_arr = plan.denoise_map
                if mask_arr is not None:
                    den_arr = den_arr * mask_arr
                den_img = Image.fromarray((den_arr * 255.0).round().astype("uint8"), "L")
                layer_regions.append({
                    "mask": mask_img,
                    "denoise_map": den_img,
                    "prompt": p.text,
                    "denoise": float(plan.base_denoise),
                    "cfg": composed_cfg,
                    "layer_id": f"v2:{pi}",
                    "region_id": str(pi),
                })
            if debug_enabled:
                debug_channels["final/rgba/aggregated"] = Image.fromarray(plan.rgba, "RGBA").convert("RGB")
                debug_channels["final/denoise/aggregated"] = Image.fromarray(
                    (plan.denoise_map * 255.0).round().astype("uint8"), "L"
                ).convert("RGB")
                debug_channels["final/cfg/aggregated"] = Image.fromarray(
                    (plan.cfg_map / CFG_HI * 255.0).round().clip(0, 255).astype("uint8"), "L"
                ).convert("RGB")
        prompt_override = base_prompt if layer_regions else (merged_prompt if merged_prompt is not None else base_prompt)
        mark_timing("composition_plan_ms", stage_started)

        # Route to StreamDiffusion realtime session or Diffusers single-pass session.
        model_path = settings.get("model_path", "")
        use_stream_diffusion = bool(settings.get("stream_diffusion"))
        self._pipeline_renderer = self._renderer_name("", use_stream_diffusion, str(model_path))
        self._pipeline_model = (model_path.split("/")[-1] or model_path.split("\\")[-1] or model_path)[:48]
        stream_reset_generation = int(settings.get("_stream_reset_generation", -1) or -1)
        is_new_stream_generation = (
            use_stream_diffusion
            and stream_reset_generation >= 0
            and stream_reset_generation != self._last_stream_reset_generation
        )
        if is_new_stream_generation:
            self._last_stream_reset_generation = stream_reset_generation

        # Do not launch any renderer if a newer scene/signal snapshot was staged
        # while we were preparing this request.
        if self._input_generation != expected_generation:
            return None

        if not use_stream_diffusion:
            from ..api.state import get_diffusers_session
            from ..image_io import data_url_bytes
            from ..schemas import InpaintFrame
            from ..inference import inpaint_frame_to_request

            stage_started = time.perf_counter()
            materialized_conditions = [dict(cond) for cond in layer_conditions if cond.get("image")]
            frame = InpaintFrame(
                client_frame_id=int(settings.get("client_frame_id") or self._last_frame_seq or self._input_generation),
                client_input_id=int(settings.get("client_input_id") or self._input_generation),
                scene_id=str(settings.get("scene_id") or self._last_scene_id or ""),
                prompt=base_prompt,
                negative_prompt=base_neg,
                debug_streams=debug_enabled,
                session_directory=str(settings.get("session_directory") or ""),
                model_path=settings.get("model_path"),
                device=settings.get("device"),
                lora_paths=list(settings.get("lora_paths") or []),
                image=_data_url("image/png", _png_bytes(input_img.convert("RGB"))),
                mask=_data_url("image/png", _png_bytes(resolved_mask.convert("L"))),
                width=w,
                height=h,
                steps=int(settings.get("steps") or 1),
                strength=float(resolved_strength),
                cfg=composed_cfg,
                sampler=settings.get("sampler"),
                scheduler=settings.get("scheduler"),
                residual_cfg=bool(settings.get("residual_cfg")),
                stochastic_similarity_filter=bool(settings.get("stochastic_similarity_filter")),
                stream_triton_compile=bool(settings.get("stream_triton_compile")),
                seed=settings.get("seed"),
                seed_mode=str(settings.get("seed_mode") or "fixed"),
                seed_variation=int(settings.get("seed_variation") or 0),
                reuse_previous_latent=bool(settings.get("reuse_previous_latent")),
                latent_reuse_denoise=float(settings.get("latent_reuse_denoise", 0.24)),
                latent_reuse_noise=float(settings.get("latent_reuse_noise", 0.0)),
                latent_reuse_noise_mode=str(settings.get("latent_reuse_noise_mode") or "none"),
                latent_reuse_restart=bool(settings.get("latent_reuse_restart")),
                stochastic_blur=float(settings.get("stochastic_blur", 0.0)),
                prompt_b=str(settings.get("prompt_b") or ""),
                prompt_lerp=float(settings.get("prompt_lerp", 0.0)),
                transparent_background=bool(settings.get("transparent_background")),
                transparent_background_mode=str(settings.get("transparent_background_mode") or "border"),
                transparent_background_color=str(settings.get("transparent_background_color") or "#ffffff"),
                transparent_background_tolerance=float(settings.get("transparent_background_tolerance", 34.0)),
                transparent_alpha_blur=float(settings.get("transparent_alpha_blur", 1.5)),
                transparent_alpha_threshold=int(settings.get("transparent_alpha_threshold", 10)),
                layer_conditions=materialized_conditions,
                render_strategy=str(settings.get("render_strategy") or "single"),
                tile_divisions=int(settings.get("tile_divisions") or 2),
                tile_overlap=int(settings.get("tile_overlap") or 128),
                layer_bbox_padding=int(settings.get("layer_bbox_padding") or 64),
            )
            mark_timing("request_build_ms", stage_started)
            session = get_diffusers_session(frame.device)
            self._pipeline_compile_status = session.engine.compile_status()
            stage_started = time.perf_counter()
            frame_result = session.step(inpaint_frame_to_request(frame))
            mark_timing("inference_ms", stage_started)
            self._pipeline_compile_status = session.engine.compile_status()
            if frame_result.error:
                self._pipeline_status = "error"
                self._pipeline_error = frame_result.error
                return None
            if not frame_result.encoded_image:
                return None
            mime, encoded_bytes = data_url_bytes(frame_result.encoded_image)
            try:
                img = Image.open(io.BytesIO(encoded_bytes)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (w, h), (128, 128, 128))
            self._pipeline_status = "ready"
            self._pipeline_error = ""
            self._pipeline_fps = frame_result.fps
            self._pipeline_renderer = self._renderer_name(frame_result.mode, False, str(model_path))
            debug_payload = dict(frame_result.raw_debug_urls or {})
            stage_started = time.perf_counter()
            output_bytes = self._encode_output_bytes(img)
            mark_timing("output_encode_ms", stage_started)
            stage_started = time.perf_counter()
            output_path = session_output_path(
                frame.session_directory,
                f"{frame.scene_id or 'scene'}_{frame.client_input_id:06d}",
                self.output_file_suffix,
            )
            if output_path is not None:
                output_path.write_bytes(output_bytes)
            persist_ms = mark_timing("output_persist_ms", stage_started)
            debug_persist_ms = 0.0
            # Always emit output + final/ channels so the signal picker works
            # regardless of whether the full debug overlay is enabled.
            debug_channels["output"] = img
            _signal_keys: set[str] = {"output"} | {k for k in debug_channels if k.startswith("final/")}
            debug_payload.update(_encode_debug_channels(
                {k: v for k, v in debug_channels.items() if k in _signal_keys},
                include_images=True, always_include=_signal_keys,
            ))
            if debug_enabled:
                debug_payload.update(_encode_debug_channels(debug_channels, include_images=True, always_include={"output"}))
                stage_started = time.perf_counter()
                for name, debug_image in debug_channels.items():
                    debug_path = session_debug_path(frame.session_directory, f"{frame.scene_id or 'scene'}_{frame.client_input_id:06d}_{name}", ".jpg")
                    if debug_path is not None:
                        debug_path.write_bytes(_jpeg_bytes(debug_image.convert("RGB")))
                debug_persist_ms = mark_timing("debug_persist_ms", stage_started)
            server_total_ms = self._round_timing_ms((time.perf_counter() - perf_started) * 1000.0)
            timings = {
                **timing_breakdown,
                "server_total_ms": server_total_ms,
                "model_inference_ms": self._round_timing_ms(float(frame_result.latency_ms or 0.0)),
            }
            if debug_enabled:
                timings["debug_persist_ms"] = debug_persist_ms
            frame_meta = {
                "mode": frame_result.mode,
                "fps": frame_result.fps,
                "latency_ms": float(frame_result.latency_ms or 0.0),
                "timings": timings,
            }
            return img, output_bytes, debug_payload, frame_meta
        else:
            session = self._session_manager.get_session(settings)
            self._pipeline_compile_status = self._pipe_compile_status(
                self._session_manager._pipe,
                bool(settings.get("stream_triton_compile", True)),
            )

        if session is None:
            self._pipeline_status = "loading"
            return None

        self._pipeline_status = "ready"
        self._pipeline_error = ""
        self._pipeline_renderer = self._renderer_name("streamdiffusion", True, str(model_path))
        self._pipeline_compile_status = self._pipe_compile_status(
            self._session_manager._pipe,
            bool(settings.get("stream_triton_compile", True)),
        )

        if self._input_generation != expected_generation:
            return None
        # Resolve ControlNet conditioning image: prefer graph cn_images, then
        # layer_conditions with controlnet_use_layer_frame + a decoded layer frame.
        cn_image: "Image.Image | None" = None
        cn_scale = 1.0
        cn_start = 0.0
        cn_end = 1.0
        if graph_result and graph_result.cn_images:
            # Take the first CN image produced by cn_from_layer graph nodes
            first_layer_id = next(iter(graph_result.cn_images))
            cn_image = graph_result.cn_images[first_layer_id]
            graph_cn = graph_result.cn_params.get(first_layer_id, {})
            cn_scale = float(graph_cn.get("scale", 1.0))
            cn_start = float(graph_cn.get("start", 0.0))
            cn_end = float(graph_cn.get("end", 1.0))
            if debug_enabled:
                for lid, cni in graph_result.cn_images.items():
                    lbl = next(
                        (c.get("name", lid) for c in layer_conditions if c.get("layer_id") == lid),
                        lid,
                    )
                    debug_channels[f"cn:{lbl}"] = cni
        else:
            # Static CN path: preprocess from layer_conditions
            for condition in layer_conditions:
                if not condition.get("controlnet_model"):
                    continue
                cn_scale = float(condition.get("controlnet_scale", 1.0))
                cn_start = float(condition.get("controlnet_start_at", 0.0))
                cn_end = float(condition.get("controlnet_end_at", 1.0))
                try:
                    from .. import controlnet as _cn_mod
                    from ..schemas import ControlNetPreprocessorParams
                    params = ControlNetPreprocessorParams(**(condition.get("controlnet_preprocessor_params") or {}))
                    if condition.get("controlnet_image"):
                        cn_image = Image.open(io.BytesIO(_decode_data_url(condition["controlnet_image"]))).convert("RGB").resize((w, h), Image.BILINEAR)
                    elif condition.get("controlnet_use_layer_frame"):
                        lf = layer_frames.get(condition.get("layer_id", ""))
                        if lf is not None:
                            cn_image = _cn_mod.preprocess_image(lf, condition["controlnet_model"], params) if condition.get("controlnet_preprocessor") else lf.convert("RGB")
                    elif condition.get("controlnet_preprocessor"):
                        cn_image = _cn_mod.preprocess_image(input_img, condition["controlnet_model"], params)
                except Exception as _e:
                    logger.debug("CN preprocessing failed: %s", _e)
                break  # use first active CN condition

        if debug_enabled and cn_image is not None:
            # Find CN model name from layer_conditions
            cn_label = next(
                (c.get("controlnet_model", "cn") for c in layer_conditions if c.get("controlnet_model")),
                "cn",
            )
            debug_channels[f"cn:{cn_label}"] = cn_image

        # Route the live StreamSession through the InferenceSession seam
        # (Step 6a). Composition logic above is unchanged; the adapter just
        # marshals args onto StreamSession.infer.
        adapter = StreamInferenceSession(session)
        cond_inputs: list[CondInput] = []
        if cn_image is not None:
            cond_inputs.append(
                CondInput(
                    spec=ControlNetSpec(model_id="cn", scale=cn_scale, start=cn_start, end=cn_end),
                    image=cn_image,
                )
            )
        stage_started = time.perf_counter()
        request = FrameRequest(
            color=input_img,
            denoise_map=resolved_denoise_map or Image.new("L", (w, h), 255),
            width=w,
            height=h,
            prompt=PromptBundle(lerp_b=float(settings.get("prompt_lerp", 0.0))),
            sampler=SamplerSpec(cfg=composed_cfg),
            cond=cond_inputs,
            backend_hints={
                "mask": resolved_mask,
                "prompt_override": prompt_override,
                "denoise": resolved_strength,
                "denoise_map": resolved_denoise_map,  # preserve None=use scalar
                "base_mask": base_resolved_mask,
                "base_denoise": base_denoise,
                "base_denoise_map": base_resolved_denoise_map,
                "cn_scale": cn_scale,
                "cn_start": cn_start,
                "cn_end": cn_end,
                # Per-layer regions: non-empty only when at least one layer has a prompt+mask.
                # StreamInferenceSession.step() runs one infer pass per region,
                # compositing bottom-to-top so each prompt stays in its mask area.
                "layer_regions": layer_regions,
                # Previous frame output: non-masked area composites over this instead
                # of the grey canvas so the stable output is preserved between frames.
                # For a NEW staged generation, do not carry over prior output —
                # that can visually decouple output from the latest input.
                "composite_base": None if is_new_stream_generation else last_output,
                # Reset StreamDiffusion latent state on new staged generation so
                # stale latent history does not drift into unrelated outputs.
                "reset_stream_state": is_new_stream_generation,
            },
        )
        mark_timing("request_build_ms", stage_started)
        stage_started = time.perf_counter()
        frame_result = adapter.step(request)
        mark_timing("inference_ms", stage_started)
        img = frame_result.image

        # Always emit output for the signal picker; debug_enabled also uses it.
        debug_channels["output"] = img

        self._pipeline_fps = 1000.0 / frame_result.latency_ms if frame_result.latency_ms > 0 else 0.0

        # Build debug payload: images throttled, text channels every frame
        debug_payload: dict[str, str] = {}
        session_directory = str(settings.get("session_directory") or "")
        stage_started = time.perf_counter()
        output_bytes = self._encode_output_bytes(img)
        mark_timing("output_encode_ms", stage_started)
        stage_started = time.perf_counter()
        output_path = session_output_path(
            session_directory,
            f"{self._last_scene_id or 'scene'}_{self._input_generation:06d}",
            self.output_file_suffix,
        )
        if output_path is not None:
            output_path.write_bytes(output_bytes)
        persist_ms = mark_timing("output_persist_ms", stage_started)
        debug_persist_ms = 0.0
        # Always emit output + final/ channels so the signal picker works
        # regardless of whether the full debug overlay is enabled.
        _signal_keys: set[str] = {"output"} | {k for k in debug_channels if k.startswith("final/")}
        debug_payload.update(_encode_debug_channels(
            {k: v for k, v in debug_channels.items() if k in _signal_keys},
            include_images=True, always_include=_signal_keys,
        ))
        if debug_enabled:
            self._debug_frame_count += 1
            encode_images = (self._debug_frame_count % 3 == 0)

            # Text channels (cheap — always send)
            if graph_result and graph_result.prompt_overrides:
                for layer_id, prompt in graph_result.prompt_overrides.items():
                    lbl = next(
                        (c.get("name", layer_id) for c in layer_conditions if c.get("layer_id") == layer_id),
                        layer_id,
                    )
                    debug_payload[f"tagger:{_debug_label(lbl, layer_id)}"] = prompt
            # Image channels are throttled except the final backend output, so
            # the debug panel can show outputs that are not the active preview.
            debug_payload.update(_encode_debug_channels(debug_channels, include_images=encode_images, always_include={"output"}))
            stage_started = time.perf_counter()
            for name, debug_image in debug_channels.items():
                debug_path = session_debug_path(session_directory, f"{self._last_scene_id or 'scene'}_{self._input_generation:06d}_{name}", ".jpg")
                if debug_path is not None:
                    debug_path.write_bytes(_jpeg_bytes(debug_image.convert("RGB")))
            debug_persist_ms = mark_timing("debug_persist_ms", stage_started)

        # Encode output bytes in the thread — transports only need to base64-encode bytes
        server_total_ms = self._round_timing_ms((time.perf_counter() - perf_started) * 1000.0)
        timings = {
            **timing_breakdown,
            "server_total_ms": server_total_ms,
            "model_inference_ms": self._round_timing_ms(float(frame_result.latency_ms or 0.0)),
        }
        if debug_enabled:
            timings["debug_persist_ms"] = debug_persist_ms
        frame_meta = {
            "mode": frame_result.mode,
            "fps": self._pipeline_fps,
            "latency_ms": float(frame_result.latency_ms or 0.0),
            "timings": timings,
        }
        return img, output_bytes, debug_payload, frame_meta

    @staticmethod
    def _compose_input(
        canvas: Image.Image,
        last_output: Image.Image | None,
        transform: list[float] | None,
        w: int,
        h: int,
    ) -> Image.Image:
        canvas_r = canvas.convert("RGB").resize((w, h), Image.BILINEAR)
        if last_output is None:
            return canvas_r
        prev = last_output.convert("RGB").resize((w, h), Image.BILINEAR)
        if transform is not None:
            prev = _apply_css_affine(prev, transform)
            anchor = _CANVAS_ANCHOR_MOTION
        else:
            anchor = _CANVAS_ANCHOR_STATIC
        return Image.blend(prev, canvas_r, anchor)

