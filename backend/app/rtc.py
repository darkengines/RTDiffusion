"""
HTTP streaming for StreamDiffusion inpainting.

Architecture
────────────
Browser → server (HTTP POST):
  POST /api/rtc/start                         — register session, get pc_id
  POST /api/rtc/{pc_id}/frame                — canvas JPEG + optional motion transform
  POST /api/rtc/{pc_id}/settings             — model, prompt, strength, etc.

Server → browser (Server-Sent Events):
  GET  /api/rtc/{pc_id}/stream               — SSE stream of JPEG frames (base64)

Async model
───────────
_process_loop   — background asyncio task per session, runs at GPU speed.
                  Maintains one StreamSession; rebuilds only on settings change.

Hot-path per frame
  apply_frame  (event loop)  : stores raw canvas URL string — O(1), no I/O
  _infer_sync  (thread pool) : decodes canvas JPEG, runs inference, encodes output JPEG
  event_generator (event loop): base64-encodes pre-built JPEG bytes — no PIL work
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from PIL import Image, ImageChops

logger = logging.getLogger("rtdiffusion.rtc")

router = APIRouter(prefix="/api/rtc")

_peer_sessions: dict[str, "InpaintSession"] = {}

from .stream_pipeline import SessionManager as _SessionManager
from .sana_pipeline import SanaSprintSessionManager as _SanaSessionManager, is_sana_model
from .pipeline_graph import SessionGraph, GraphResult, merge_layer_prompts, resolve_conditioning_mask
from .inference import (
    CondInput,
    ControlNetSpec,
    FrameRequest,
    PromptBundle,
    SamplerSpec,
    StreamInferenceSession,
)

_shared_session_manager = _SessionManager()
_shared_sana_manager = _SanaSessionManager()

_CANVAS_ANCHOR_MOTION = 0.55
_CANVAS_ANCHOR_STATIC = 0.85

_DEFAULT_INFER_W = 512
_DEFAULT_INFER_H = 512


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


def _jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class InpaintSession:
    """
    Inference session for one browser connection.

    apply_frame / apply_settings run in the asyncio event loop — they do O(1) work only.
    _infer_sync runs in a thread pool — all PIL and GPU work happens there.
    event_generator reads pre-encoded JPEG bytes from the output queue.
    """

    def __init__(self, pc_id: str) -> None:
        self._pc_id = pc_id
        self._tag = pc_id[:8]

        self._settings: dict[str, Any] = {}
        # Raw canvas URL — set by event loop (string assign = O(1)), decoded in thread
        self._canvas_url: str | None = None
        self._motion_transform: list[float] | None = None
        self._latest_output: Image.Image | None = _shared_session_manager.last_output
        self._running = True
        self._error_count = 0
        self._log_first_frame = True

        # Mask cache — the mask rarely changes so we avoid decoding it every frame
        self._cached_mask_url: str = ""
        self._cached_mask: Image.Image | None = None

        # Layer frame cache — keyed by layer_id, value (url, decoded_image)
        self._layer_frames_url: dict[str, str] = {}  # set O(1) in event loop
        self._cached_layer_frames: dict[str, tuple[str, Image.Image]] = {}  # (url, img) in thread

        # Pipeline graph — rebuilt only when pipeline_nodes signature changes
        self._session_graph: SessionGraph | None = None
        self._graph_sig: str = ""

        # Settings log dedup — only log when prompt/model/device actually changes
        self._last_logged_sig: str = ""

        # Output queue: (jpeg_bytes, debug_payload) | None
        # debug_payload is dict[str, str]: data-URL for images, plain text for text channels
        self._output_queue: asyncio.Queue[tuple[bytes, dict[str, str]] | None] = asyncio.Queue(maxsize=2)
        # Throttle debug image encoding: only encode thumbnails every N frames
        self._debug_frame_count: int = 0
        _DEBUG_IMG_EVERY = 3  # encode thumbnails every 3rd inference frame

        # Pipeline state — written in thread pool, read in event loop (GIL-safe for simple assigns)
        self._pipeline_status: str = "idle"   # idle | loading | ready | error
        self._pipeline_model: str = ""
        self._pipeline_error: str = ""
        self._pipeline_fps: float = 0.0
        self._fps_frames: int = 0
        self._fps_window_start: float = 0.0

        self._session_manager = _shared_session_manager
        self._sana_manager = _shared_sana_manager

    def start(self) -> None:
        asyncio.get_running_loop().create_task(
            self._process_loop(), name=f"rtc-proc-{self._tag}"
        )

    def close(self) -> None:
        self._running = False
        try:
            self._output_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    # ── inbound data (event loop — must be O(1)) ───────────────────

    def apply_settings(self, settings: dict[str, Any]) -> None:
        settings.setdefault("type", "settings")
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

    def apply_frame(
        self,
        img_url: str,
        transform: list[float] | None,
        layer_frames: dict[str, str] | None = None,
    ) -> None:
        # No PIL work here — just store the raw URL strings and transform.
        # Decoding happens in _infer_sync (thread pool).
        if img_url:
            if self._log_first_frame:
                self._log_first_frame = False
                logger.info("RTC[%s] first canvas frame received", self._tag)
            self._canvas_url = img_url
        self._motion_transform = transform
        if layer_frames:
            self._layer_frames_url = layer_frames

    # ── process loop ───────────────────────────────────────────────

    async def _process_loop(self) -> None:
        logger.info("RTC[%s] process loop started", self._tag)
        _canvas_wait_s = 0.0
        _last_generation = -1
        _skip_frames = 0
        while self._running:
            settings = self._settings
            canvas_url = self._canvas_url
            if not settings:
                await asyncio.sleep(0.05)
                continue
            if canvas_url is None:
                _canvas_wait_s += 0.05
                if _canvas_wait_s >= 2.0:
                    canvas_url = ""  # _infer_sync uses gray placeholder on empty string
                    if _canvas_wait_s < 2.1:
                        logger.warning("RTC[%s] no canvas frame after 2s — using placeholder", self._tag)
                else:
                    await asyncio.sleep(0.05)
                    continue
            try:
                result = await asyncio.to_thread(
                    self._infer_sync, canvas_url, settings,
                    self._latest_output, self._motion_transform,
                )
                if result is not None:
                    img, jpeg, debug_jpgs = result
                    model_path = settings.get("model_path", "")
                    use_sana = settings.get("model_type") == "sana_sprint" or is_sana_model(model_path)
                    cur_gen = (self._sana_manager if use_sana else self._session_manager)._generation
                    if cur_gen != _last_generation:
                        _last_generation = cur_gen
                        _skip_frames = 2
                        logger.info("RTC[%s] new session gen=%d, skipping %d warmup frames",
                                    self._tag, cur_gen, _skip_frames)
                    if _skip_frames > 0:
                        _skip_frames -= 1
                    else:
                        self._latest_output = img
                        self._session_manager.last_output = img
                        try:
                            self._output_queue.put_nowait((jpeg, debug_jpgs))
                        except asyncio.QueueFull:
                            pass
                    self._error_count = 0
                else:
                    await asyncio.sleep(1.0)
            except Exception as _exc:
                self._error_count += 1
                self._pipeline_status = "error"
                self._pipeline_error = f"{type(_exc).__name__}: {_exc}"
                backoff = min(2 ** self._error_count, 5)
                logger.exception("RTC[%s] inference error (count=%d) — backing off %.0fs",
                                 self._tag, self._error_count, backoff)
                await asyncio.sleep(backoff)

    def _decode_layer_frames(self, w: int, h: int) -> dict[str, Image.Image]:
        """Decode layer frame data URLs, using cache to avoid redundant work."""
        result: dict[str, Image.Image] = {}
        for layer_id, url in self._layer_frames_url.items():
            cached = self._cached_layer_frames.get(layer_id)
            if cached and cached[0] == url:
                result[layer_id] = cached[1]
                continue
            try:
                img = Image.open(io.BytesIO(_decode_data_url(url))).convert("RGB")
                img = img.resize((w, h), Image.BILINEAR)
                self._cached_layer_frames[layer_id] = (url, img)
                result[layer_id] = img
            except Exception as exc:
                logger.debug("RTC[%s] failed to decode layer frame %s: %s", self._tag, layer_id, exc)
        return result

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
                rgba = Image.open(io.BytesIO(_decode_data_url(image_url))).convert("RGBA").resize((w, h), Image.LANCZOS)
                alpha = rgba.getchannel("A")
                current = masks.get(lid)
                masks[lid] = alpha if current is None else ImageChops.lighter(current, alpha)
                if lid not in images:
                    images[lid] = rgba.convert("RGB")
            except Exception:
                continue
        return masks, images

    def _infer_sync(
        self,
        canvas_url: str,
        settings: dict[str, Any],
        last_output: Image.Image | None,
        motion_transform: list[float] | None,
    ) -> tuple[Image.Image, bytes, dict[str, bytes]] | None:
        """
        Runs in thread pool. All PIL and GPU work happens here.

        1. Decode canvas JPEG (in-thread, no event loop blocking).
        2. Decode mask only if URL changed (cached).
        3. Decode layer frame images (cached by URL).
        4. Rebuild pipeline graph only when nodes change.
        5. Execute graph → prompt overrides + CN images.
        6. Compose input: blend canvas with motion-warped previous output.
        7. Run inference (StreamSession or SANA) with merged prompt.
        8. Encode output JPEG (in-thread — bytes go straight to SSE queue).
        9. Collect debug channel thumbnails (when debug_streams enabled).
        Returns (img, jpeg, debug_channels) — debug_channels maps label → JPEG bytes.
        """
        w = int(settings.get("width", _DEFAULT_INFER_W))
        h = int(settings.get("height", _DEFAULT_INFER_H))

        # Decode canvas in thread (avoids blocking event loop with base64+JPEG decode)
        if canvas_url:
            try:
                canvas = Image.open(io.BytesIO(_decode_data_url(canvas_url))).convert("RGB")
            except Exception:
                canvas = Image.new("RGB", (w, h), (128, 128, 128))
        else:
            canvas = Image.new("RGB", (w, h), (128, 128, 128))

        debug_enabled: bool = bool(settings.get("debug_streams"))
        debug_channels: dict[str, Image.Image] = {}

        input_img = self._compose_input(canvas, last_output, motion_transform, w, h)

        if debug_enabled:
            debug_channels["canvas"] = canvas

        # Mask — decode only when URL changes (mask is usually static)
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

        # Decode layer frames (cached by URL)
        layer_frames = self._decode_layer_frames(w, h)

        # Use video layer frame as primary SD input when available.
        # This ensures StreamDiffusion is driven by the actual video frame
        # rather than the Fabric.js canvas composite (which may lag one rAF cycle).
        layer_conditions: list[dict[str, Any]] = settings.get("layer_conditions") or []
        layer_masks, condition_images = self._build_layer_masks(layer_conditions, w, h)
        for _cond in layer_conditions:
            if _cond.get("primary_input") and _cond.get("layer_id") in layer_frames:
                input_img = layer_frames[_cond["layer_id"]]
                if debug_enabled:
                    debug_channels["video_frame"] = input_img
                break

        if debug_enabled:
            debug_channels["input_composed"] = input_img
            for lid, lf in layer_frames.items():
                label = next(
                    (c.get("name", lid) for c in layer_conditions if c.get("layer_id") == lid),
                    lid,
                )
                debug_channels[f"layer:{label}"] = lf

        # Rebuild pipeline graph only when pipeline_nodes signature changes
        nodes_raw: list[dict[str, Any]] = settings.get("pipeline_nodes") or []
        new_sig = SessionGraph.signature(nodes_raw)
        if new_sig != self._graph_sig:
            self._session_graph = SessionGraph(nodes_raw) if nodes_raw else None
            self._graph_sig = new_sig

        # Collect primary-input layer ids (video layers whose frame is real scene content)
        primary_input_layers = {
            c["layer_id"] for c in layer_conditions
            if c.get("primary_input") and c.get("layer_id")
        }

        # Execute graph
        graph_result: GraphResult | None = None
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

        # Merge layer condition prompts (fix: layer prompts were ignored in streaming path)
        base_prompt: str = settings.get("prompt") or ""
        merged_prompt = merge_layer_prompts(layer_conditions, base_prompt, graph_result)
        # Use merged prompt as override only when it differs from the session's base prompt
        prompt_override = merged_prompt if merged_prompt != base_prompt else None

        base_denoise = float(settings.get("strength", 1.0))
        resolved_mask = mask or Image.new("L", (w, h), 0)
        resolved_strength = base_denoise
        resolved_denoise_map: Image.Image | None = None
        if layer_conditions:
            resolved = resolve_conditioning_mask(resolved_mask, layer_conditions, w, h, base_denoise)
            resolved_mask = resolved.mask
            resolved_strength = resolved.strength
            resolved_denoise_map = resolved.denoise_map
            if debug_enabled:
                debug_channels["condition_mask"] = resolved.mask.convert("RGB")
                debug_channels["condition_denoise"] = resolved.denoise_map.convert("RGB")

        # Route to SANA-Sprint or StreamDiffusion
        model_path = settings.get("model_path", "")
        use_sana = settings.get("model_type") == "sana_sprint" or is_sana_model(model_path)
        self._pipeline_model = (model_path.split("/")[-1] or model_path.split("\\")[-1] or model_path)[:48]

        if use_sana:
            session = self._sana_manager.get_session(settings)
        else:
            session = self._session_manager.get_session(settings)

        if session is None:
            self._pipeline_status = "loading"
            return None

        self._pipeline_status = "ready"
        self._pipeline_error = ""

        if use_sana:
            img = session.infer(input_img, strength=float(settings.get("strength", 0.7)))
        else:
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
                for cond in layer_conditions:
                    if not cond.get("controlnet_model"):
                        continue
                    cn_scale = float(cond.get("controlnet_scale", 1.0))
                    cn_start = float(cond.get("controlnet_start_at", 0.0))
                    cn_end = float(cond.get("controlnet_end_at", 1.0))
                    try:
                        from . import controlnet as _cn_mod
                        from .schemas import ControlNetPreprocessorParams
                        params = ControlNetPreprocessorParams(**(cond.get("controlnet_preprocessor_params") or {}))
                        if cond.get("controlnet_image"):
                            cn_image = Image.open(io.BytesIO(_decode_data_url(cond["controlnet_image"]))).convert("RGB").resize((w, h), Image.LANCZOS)
                        elif cond.get("controlnet_use_layer_frame"):
                            lf = layer_frames.get(cond.get("layer_id", ""))
                            if lf is not None:
                                cn_image = _cn_mod.preprocess_image(lf, cond["controlnet_model"], params) if cond.get("controlnet_preprocessor") else lf.convert("RGB")
                        elif cond.get("controlnet_preprocessor"):
                            cn_image = _cn_mod.preprocess_image(input_img, cond["controlnet_model"], params)
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
            cond: list[CondInput] = []
            if cn_image is not None:
                cond.append(
                    CondInput(
                        spec=ControlNetSpec(model_id="cn", scale=cn_scale, start=cn_start, end=cn_end),
                        image=cn_image,
                    )
                )
            request = FrameRequest(
                color=input_img,
                denoise_map=resolved_denoise_map or Image.new("L", (w, h), 255),
                width=w,
                height=h,
                prompt=PromptBundle(lerp_b=float(settings.get("prompt_lerp", 0.0))),
                sampler=SamplerSpec(),
                cond=cond,
                backend_hints={
                    "mask": resolved_mask,
                    "prompt_override": prompt_override,
                    "denoise": resolved_strength,
                    "denoise_map": resolved_denoise_map,  # preserve None=use scalar
                    "cn_scale": cn_scale,
                    "cn_start": cn_start,
                    "cn_end": cn_end,
                },
            )
            frame_result = adapter.step(request)
            img = frame_result.image if frame_result.mode != "stream-skip" else None

        if img is None:
            return None

        if debug_enabled:
            debug_channels["output"] = img

        # Rolling FPS over a 3-second window
        now_t = time.monotonic()
        self._fps_frames += 1
        if self._fps_window_start == 0.0:
            self._fps_window_start = now_t
        elapsed = now_t - self._fps_window_start
        if elapsed >= 3.0:
            self._pipeline_fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_window_start = now_t

        # Build debug payload: images throttled, text channels every frame
        debug_payload: dict[str, str] = {}
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
                    debug_payload[f"tagger:{lbl}"] = prompt

            # Image channels (expensive — throttle to every 3rd frame)
            if encode_images and debug_channels:
                thumb_size = (320, 180)
                for ch_name, ch_img in debug_channels.items():
                    try:
                        thumb = ch_img.convert("RGB")
                        thumb.thumbnail(thumb_size, Image.BILINEAR)
                        b64 = base64.b64encode(_jpeg_bytes(thumb, quality=65)).decode()
                        debug_payload[ch_name] = f"data:image/jpeg;base64,{b64}"
                    except Exception:
                        pass

        # Encode JPEG here in the thread — SSE generator only needs to base64-encode bytes
        return img, _jpeg_bytes(img), debug_payload

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


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@router.post("/start")
async def rtc_start(request: Request):
    pc_id = uuid.uuid4().hex
    session = InpaintSession(pc_id)
    _peer_sessions[pc_id] = session
    session.start()
    logger.info("RTC[%s] session started", pc_id[:8])
    return {"pc_id": pc_id}


@router.get("/{pc_id}/stream")
async def rtc_stream(pc_id: str):
    """SSE stream of JPEG frames (pre-encoded in inference thread)."""
    session = _peer_sessions.get(pc_id)
    if not session:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="session not found")

    initial = session._latest_output

    async def event_generator():
        if initial is not None:
            b64 = base64.b64encode(_jpeg_bytes(initial)).decode()
            yield f"data: data:image/jpeg;base64,{b64}\n\n"

        while session._running:
            try:
                item = await asyncio.wait_for(session._output_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                status_payload = json.dumps({
                    "phase": session._pipeline_status,
                    "model": session._pipeline_model,
                    "fps": round(session._pipeline_fps, 1),
                    "error": session._pipeline_error,
                })
                yield f"event: status\ndata: {status_payload}\n\n"
                continue
            if item is None:
                break
            frame, debug_payload = item
            # Main output frame — already JPEG bytes
            b64 = base64.b64encode(frame).decode()
            yield f"data: data:image/jpeg;base64,{b64}\n\n"
            # Debug payload: values are either data-URLs (images) or plain text (tagger)
            # Images were already base64-encoded in the inference thread.
            if debug_payload:
                yield f"event: debug\ndata: {json.dumps(debug_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{pc_id}/frame")
async def rtc_frame(pc_id: str, request: Request):
    session = _peer_sessions.get(pc_id)
    if not session:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="session not found")
    body = await request.json()
    t = body.get("transform")
    layers = body.get("layers")
    session.apply_frame(
        img_url=body.get("image", ""),
        transform=[float(v) for v in t] if t and len(t) == 6 else None,
        layer_frames=layers if isinstance(layers, dict) else None,
    )
    return {}


@router.post("/{pc_id}/settings")
async def rtc_settings(pc_id: str, request: Request):
    session = _peer_sessions.get(pc_id)
    if not session:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="session not found")
    settings = await request.json()
    session.apply_settings(settings)
    return {}


@router.delete("/peers/{pc_id}")
async def close_peer(pc_id: str):
    session = _peer_sessions.pop(pc_id, None)
    if session:
        session.close()
    return {"closed": True}
