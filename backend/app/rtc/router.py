"""FastAPI router for the RTC streaming endpoints."""
from __future__ import annotations

import base64
import asyncio
import json
import uuid
import struct
import io
from fractions import Fraction
from datetime import datetime, timezone

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .session import InpaintSession, _jpeg_bytes, _peer_sessions, _shared_session_manager, logger

router = APIRouter(prefix="/api/rtc")
_peer_connections: dict[str, object] = {}
_peer_media_tasks: dict[str, set[asyncio.Task]] = {}


try:
    from aiortc import VideoStreamTrack
except Exception:  # pragma: no cover - /offer reports aiortc availability separately
    VideoStreamTrack = object  # type: ignore[misc,assignment]


class _SessionOutputVideoTrack(VideoStreamTrack):  # type: ignore[misc,valid-type]
    kind = "video"

    def __init__(self, session: InpaintSession) -> None:
        super().__init__()
        self._session = session
        self._queue = session.subscribe_output_media()
        self._pts = 0
        self._time_base = Fraction(1, 90000)

    async def recv(self):
        from aiortc.mediastreams import MediaStreamError
        from av import VideoFrame
        from PIL import Image
        import numpy as np

        data = await self._queue.get()
        if data is None:
            raise MediaStreamError
        image = Image.open(io.BytesIO(data)).convert("RGB")
        frame = VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
        self._pts += 3000
        frame.pts = self._pts
        frame.time_base = self._time_base
        return frame

    def stop(self) -> None:
        self._session.unsubscribe_output_media(self._queue)
        super().stop()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def _ws_admin_log(websocket: WebSocket, pc_id: str, event: str, **payload: object) -> None:
    data = {"type": "admin_log", "event": event, "session_id": pc_id, "at": _utc_now_iso(), **payload}
    logger.info("RTC[%s] admin event: %s %s", pc_id[:8], event, payload)
    await websocket.send_json(data)


async def _close_existing_peers() -> None:
    """RTDiffusion is a single-workspace realtime app; a new peer supersedes old ones."""
    for old_id, tasks in list(_peer_media_tasks.items()):
        _peer_media_tasks.pop(old_id, None)
        for task in tasks:
            task.cancel()
    for old_id, old_session in list(_peer_sessions.items()):
        _peer_sessions.pop(old_id, None)
        old_session.close()
        logger.info("RTC[%s] superseded by a new peer — session closed", old_id[:8])
    for old_id, old_pc in list(_peer_connections.items()):
        _peer_connections.pop(old_id, None)
        close = getattr(old_pc, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                logger.debug("RTC[%s] peer close failed", old_id[:8], exc_info=True)


async def _consume_media_track(session: InpaintSession, pc_id: str, track) -> None:
    track_id = str(getattr(track, "id", "") or uuid.uuid4().hex)
    session.register_media_track(track_id, track_id)
    logger.info("RTC[%s] media track attached: id=%s kind=%s", pc_id[:8], track_id, getattr(track, "kind", ""))
    try:
        while session._running:
            frame = await track.recv()
            to_image = getattr(frame, "to_image", None)
            if to_image is None:
                continue
            session.update_media_track_frame(track_id, to_image())
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.info("RTC[%s] media track ended: id=%s reason=%s", pc_id[:8], track_id, exc)
    finally:
        session.remove_media_track(track_id)


def _decode_binary_envelope(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 4:
        raise ValueError("binary envelope too short")
    header_len = struct.unpack(">I", data[:4])[0]
    if header_len <= 0 or 4 + header_len > len(data):
        raise ValueError("invalid binary envelope header length")
    header = json.loads(data[4:4 + header_len].decode("utf-8"))
    return header, data[4 + header_len:]


async def _store_ws_binary_resource(
    session: InpaintSession,
    websocket: WebSocket,
    pc_id: str,
    pending_resource_chunks: dict[str, dict[str, object]],
    raw: bytes,
) -> None:
    header, body = _decode_binary_envelope(raw)
    msg_type = header.get("type")
    if msg_type == "resource":
        request_id = str(header.get("request_id", ""))
        resource_id = str(header.get("id") or "")
        name = str(header.get("name") or resource_id)
        mime = str(header.get("mime") or "application/octet-stream")
        session.store_scene_resource(resource_id, body, mime=mime, name=name)
        logger.info("RTC[%s] resource stored: id=%s name=%s bytes=%d", pc_id[:8], resource_id, name, len(body))
        return
    if msg_type == "resource_chunk":
        request_id = str(header.get("request_id", ""))
        resource_id = str(header.get("id") or "")
        name = str(header.get("name") or resource_id)
        mime = str(header.get("mime") or "application/octet-stream")
        index = int(header.get("index") or 0)
        total = int(header.get("total") or 0)
        if not request_id or not resource_id or total <= 0 or index < 0 or index >= total:
            raise ValueError("invalid resource chunk header")
        item = pending_resource_chunks.get(request_id)
        if item is None:
            item = {"id": resource_id, "name": name, "mime": mime, "chunks": [None] * total, "received": 0}
            pending_resource_chunks[request_id] = item
        chunks = item["chunks"]
        if not isinstance(chunks, list) or len(chunks) != total:
            pending_resource_chunks.pop(request_id, None)
            raise ValueError("inconsistent resource chunk stream")
        if chunks[index] is None:
            chunks[index] = body
            item["received"] = int(item.get("received") or 0) + 1
        if int(item.get("received") or 0) == total:
            pending_resource_chunks.pop(request_id, None)
            data = b"".join(part if isinstance(part, bytes) else b"" for part in chunks)
            session.store_scene_resource(resource_id, data, mime=mime, name=name)
            logger.info("RTC[%s] resource stored: id=%s name=%s bytes=%d chunks=%d", pc_id[:8], resource_id, name, len(data), total)
        return
    if msg_type == "blob":
        request_id = str(header.get("request_id", ""))
        blob_id = session.store_blob(body, name=str(header.get("name") or ""))
        await websocket.send_json({"type": "blob_ack", "request_id": request_id, "id": blob_id})
        return
    raise ValueError(f"unknown binary message type: {msg_type}")


@router.websocket("/ws")
async def rtc_session_socket(websocket: WebSocket) -> None:
    """Full-duplex render session transport.

    JSON messages carry structured scene/session events. Binary messages carry
    immutable resource envelopes. All render outputs and administrative events
    are sent back through this same WebSocket.
    """
    await websocket.accept()
    await _close_existing_peers()
    pc_id = uuid.uuid4().hex
    session = InpaintSession(pc_id)
    _peer_sessions[pc_id] = session
    session.start()
    pending_resource_chunks: dict[str, dict[str, object]] = {}
    logger.info("RTC[%s] full-duplex WebSocket session started", pc_id[:8])

    async def send_outputs() -> None:
        while session._running:
            item = await session._output_queue.get()
            if item is None:
                break
            frame, debug_payload = item
            if not frame and debug_payload.get("__event__"):
                event_name = str(debug_payload.get("__event__"))
                payload = {key: value for key, value in debug_payload.items() if key != "__event__"}
                await websocket.send_json({"type": event_name, **payload})
                continue
            if frame:
                if not session.has_output_media_subscribers:
                    b64 = base64.b64encode(frame).decode()
                    await websocket.send_json({"type": "frame", "mime": "image/jpeg", "data": f"data:image/jpeg;base64,{b64}"})
            if debug_payload:
                await websocket.send_json({"type": "debug", "channels": debug_payload})

    output_task = asyncio.create_task(send_outputs())
    try:
        await _ws_admin_log(websocket, pc_id, "session_started", transport="websocket")
        await websocket.send_json({"type": "hello", "session_id": pc_id, "protocol": "rtd.session.v1"})
        await websocket.send_json({"type": "input_ready", "scene_id": "", "generation": "0", "reason": "session_started"})
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                try:
                    await _store_ws_binary_resource(session, websocket, pc_id, pending_resource_chunks, message["bytes"])
                except Exception as exc:
                    logger.warning("RTC[%s] WebSocket binary message failed: %s", pc_id[:8], exc)
                    await websocket.send_json({"type": "error", "message": str(exc)})
                continue
            text = message.get("text")
            if text is None:
                continue
            payload = json.loads(text)
            msg_type = payload.get("type")
            if msg_type == "scene":
                scene = payload.get("scene")
                if isinstance(scene, dict):
                    session.apply_scene(scene, seq=payload.get("seq"))
                    logger.info("RTC[%s] scene received: id=%s seq=%s", pc_id[:8], payload.get("id"), payload.get("seq"))
            elif msg_type == "scene_patch":
                events = payload.get("events")
                patch = payload.get("patch")
                if isinstance(events, list):
                    session.apply_scene_events(events, seq=payload.get("seq"))
                elif isinstance(patch, dict):
                    session.apply_scene_patch(patch, seq=payload.get("seq"))
            elif msg_type == "media_track":
                meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                label = str(meta.get("label") or meta.get("layer_id") or payload.get("stream_id") or payload.get("track_id") or "media")
                session.register_media_track(
                    str(payload.get("track_id") or ""),
                    label,
                    layer_id=str(meta.get("layer_id") or ""),
                    channel=str(meta.get("channel") or ""),
                )
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "at": _utc_now_iso()})
            elif msg_type == "stop":
                await _ws_admin_log(websocket, pc_id, "client_stop")
                break
            else:
                await websocket.send_json({"type": "error", "message": f"unknown message type: {msg_type}"})
    except WebSocketDisconnect:
        pass
    finally:
        removed = _peer_sessions.pop(pc_id, None)
        if removed is not None:
            removed.close()
        output_task.cancel()
        logger.info("RTC[%s] full-duplex WebSocket session closed", pc_id[:8])


def _attach_data_channel(session: InpaintSession, pc_id: str, channel):
    label = str(getattr(channel, "label", "?"))
    logger.info("RTC[%s] data channel attached: %s", pc_id[:8], label)
    pending_resource_chunks: dict[str, dict[str, object]] = {}

    @channel.on("message")
    def on_message(message):
        try:
            if isinstance(message, str):
                payload = json.loads(message)
                msg_type = payload.get("type")
                if label == "scene" and msg_type == "scene":
                    scene = payload.get("scene")
                    if isinstance(scene, dict):
                        session.apply_scene(scene, seq=payload.get("seq"))
                elif msg_type == "scene_patch":
                    patch = payload.get("patch")
                    if isinstance(patch, dict):
                        session.apply_scene_patch(patch, seq=payload.get("seq"))
                elif msg_type == "settings":
                    settings = payload.get("settings")
                    if isinstance(settings, dict):
                        session.apply_settings(settings, seq=payload.get("seq"))
                elif msg_type == "frame":
                    t = payload.get("transform")
                    layers = payload.get("layers")
                    session.apply_frame(
                        img_url=payload.get("image", ""),
                        transform=[float(v) for v in t] if t and len(t) == 6 else None,
                        layer_frames=layers if isinstance(layers, dict) else None,
                        seq=payload.get("seq"),
                    )
                elif msg_type == "ping":
                    channel.send(json.dumps({"type": "pong"}))
                return

            raw = bytes(message)
            header, body = _decode_binary_envelope(raw)
            if label.startswith("resource:") or header.get("type") == "resource":
                request_id = str(header.get("request_id", ""))
                resource_id = str(header.get("id") or label.removeprefix("resource:"))
                name = str(header.get("name") or resource_id)
                mime = str(header.get("mime") or "application/octet-stream")
                session.store_scene_resource(resource_id, body, mime=mime, name=name)
            elif header.get("type") == "resource_chunk":
                request_id = str(header.get("request_id", ""))
                resource_id = str(header.get("id") or "")
                name = str(header.get("name") or resource_id)
                mime = str(header.get("mime") or "application/octet-stream")
                index = int(header.get("index") or 0)
                total = int(header.get("total") or 0)
                if not request_id or not resource_id or total <= 0 or index < 0 or index >= total:
                    raise ValueError("invalid resource chunk header")
                item = pending_resource_chunks.get(request_id)
                if item is None:
                    item = {
                        "id": resource_id,
                        "name": name,
                        "mime": mime,
                        "chunks": [None] * total,
                        "received": 0,
                    }
                    pending_resource_chunks[request_id] = item
                chunks = item["chunks"]
                if not isinstance(chunks, list) or len(chunks) != total:
                    pending_resource_chunks.pop(request_id, None)
                    raise ValueError("inconsistent resource chunk stream")
                if chunks[index] is None:
                    chunks[index] = body
                    item["received"] = int(item.get("received") or 0) + 1
                if int(item.get("received") or 0) == total:
                    pending_resource_chunks.pop(request_id, None)
                    data = b"".join(part if isinstance(part, bytes) else b"" for part in chunks)
                    session.store_scene_resource(resource_id, data, mime=mime, name=name)
            elif header.get("type") == "blob":
                request_id = str(header.get("request_id", ""))
                blob_id = session.store_blob(body, name=str(header.get("name") or ""))
                channel.send(json.dumps({"type": "blob_ack", "request_id": request_id, "id": blob_id}))
        except Exception as exc:
            logger.warning("RTC[%s] data channel message failed: %s", pc_id[:8], exc)
            try:
                channel.send(json.dumps({"type": "error", "message": str(exc)}))
            except Exception:
                pass


@router.post("/offer")
async def rtc_offer(request: Request):
    """Create a true WebRTC data-channel session.

    HTTP is used once for SDP offer/answer. After that, frontend inputs
    (settings, frames, and resource blobs) are staged through data channels.
    Output frames still use the existing SSE endpoint for now.
    """
    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"aiortc unavailable: {exc}") from None

    body = await request.json()
    offer = body.get("offer") or body
    if not isinstance(offer, dict) or not offer.get("sdp") or not offer.get("type"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="missing WebRTC offer")

    requested_session = str(body.get("session_id") or body.get("pc_id") or "").strip()
    session = _peer_sessions.get(requested_session) if requested_session else None
    if requested_session and session is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="render session not found")
    owns_session = session is None
    if session is None:
        await _close_existing_peers()
        pc_id = requested_session or uuid.uuid4().hex
        session = InpaintSession(pc_id)
        _peer_sessions[pc_id] = session
        session.start()
    else:
        pc_id = requested_session

    pc = _peer_connections.get(pc_id)
    if pc is None:
        pc = RTCPeerConnection()
        _peer_connections[pc_id] = pc

    if not getattr(pc, "_rtd_output_track_attached", False):
        output_track = _SessionOutputVideoTrack(session)
        pc.addTrack(output_track)
        setattr(pc, "_rtd_output_track", output_track)
        setattr(pc, "_rtd_output_track_attached", True)

    if not getattr(pc, "_rtd_handlers_attached", False):
        setattr(pc, "_rtd_handlers_attached", True)

        @pc.on("datachannel")
        def on_datachannel(channel):
            _attach_data_channel(session, pc_id, channel)

        @pc.on("track")
        def on_track(track):
            if getattr(track, "kind", "") != "video":
                return
            task = asyncio.create_task(_consume_media_track(session, pc_id, track))
            _peer_media_tasks.setdefault(pc_id, set()).add(task)
            task.add_done_callback(lambda done: _peer_media_tasks.get(pc_id, set()).discard(done))

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                output_track = getattr(pc, "_rtd_output_track", None)
                if output_track is not None:
                    output_track.stop()
                tasks = _peer_media_tasks.pop(pc_id, set())
                for task in tasks:
                    task.cancel()
                if owns_session:
                    removed = _peer_sessions.pop(pc_id, None)
                    if removed is not None:
                        removed.close()
                _peer_connections.pop(pc_id, None)
                await pc.close()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info("RTC[%s] WebRTC data-channel session started", pc_id[:8])
    return {"pc_id": pc_id, "answer": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}}


@router.post("/start")
async def rtc_start(request: Request):
    await _close_existing_peers()

    pc_id = uuid.uuid4().hex
    session = InpaintSession(pc_id)
    _peer_sessions[pc_id] = session
    session.start()
    logger.info("RTC[%s] session started", pc_id[:8])
    return {"pc_id": pc_id}


@router.get("/{pc_id}/stream")
async def rtc_stream(pc_id: str, request: Request):
    """SSE stream of JPEG frames (pre-encoded in inference thread)."""
    session = _peer_sessions.get(pc_id)
    if not session:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="session not found")

    initial = session._latest_output

    async def event_generator():
        try:
            if initial is not None:
                b64 = base64.b64encode(_jpeg_bytes(initial)).decode()
                yield f"data: data:image/jpeg;base64,{b64}\n\n"

            while session._running:
                if await request.is_disconnected():
                    break
                item = await session._output_queue.get()
                if item is None:
                    break
                frame, debug_payload = item
                if not frame and debug_payload.get("__event__"):
                    event_name = str(debug_payload.get("__event__"))
                    payload = {key: value for key, value in debug_payload.items() if key != "__event__"}
                    yield f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"
                    continue
                # Main output frame — already JPEG bytes
                b64 = base64.b64encode(frame).decode()
                yield f"data: data:image/jpeg;base64,{b64}\n\n"
                # Debug payload: values are either data-URLs (images) or plain text (tagger)
                # Images were already base64-encoded in the inference thread.
                if debug_payload:
                    yield f"event: debug\ndata: {json.dumps(debug_payload)}\n\n"
        except GeneratorExit:
            pass
        finally:
            # Clean up when the client disconnects (page refresh, tab close, etc.)
            # without calling DELETE /peers/{pc_id} first.
            removed = _peer_sessions.pop(pc_id, None)
            if removed is not None:
                removed.close()
                logger.info("RTC[%s] client disconnected — session cleaned up", pc_id[:8])

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


@router.post("/{pc_id}/blob")
async def rtc_upload_blob(pc_id: str, request: Request):
    """Binary mask upload — POST raw PNG bytes, get back an ID to reference
    in subsequent ``layer_conditions[*]_mask_ref`` fields.

    This avoids the ~33 % base64 overhead for the channel masks
    (color_mask / denoise_mask / prompt_mask / cfg_mask) that the user paints.
    The blob lives in the session's in-memory store (LRU 128 entries) and
    dies with the session.

    Why an upload step instead of inlining: settings JSON change rarely but
    can carry several large masks each; uploading once on paint-change and
    referencing by ID is cheaper than re-sending base64 each settings update.
    """
    session = _peer_sessions.get(pc_id)
    if not session:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="session not found")
    data = await request.body()
    if not data:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="empty body")
    try:
        blob_id = session.store_blob(data, mime=request.headers.get("content-type", "image/png"))
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"id": blob_id}


@router.delete("/peers/{pc_id}")
async def close_peer(pc_id: str):
    # May already be cleaned up by SSE disconnect — idempotent.
    session = _peer_sessions.pop(pc_id, None)
    if session:
        session.close()
    pc = _peer_connections.pop(pc_id, None)
    if pc is not None:
        close = getattr(pc, "close", None)
        if close is not None:
            await close()
    return {"closed": bool(session)}
