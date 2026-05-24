"""Realtime inpainting WebSocket endpoint (/ws/inpaint)."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..image_io import data_url_bytes
from ..inference import inpaint_frame_to_request
from ..schemas import InpaintFrame, InpaintResult
from .state import (
    _device_lock,
    engine_mode,
    get_diffusers_session,
    sampling_pause_event,
)
from . import state as _state

logger = logging.getLogger("rtdiffusion.api")

router = APIRouter()

_FRAME_DRAIN_TIMEOUT_S = 0.005
_FRAME_DRAIN_LIMIT = 64


def _result_for_frame(frame: InpaintFrame, result: InpaintResult) -> InpaintResult:
    return InpaintResult(
        client_frame_id=frame.client_frame_id,
        client_input_id=frame.client_input_id,
        scene_id=frame.scene_id,
        image=result.image,
        fps=result.fps,
        latency_ms=result.latency_ms,
        mode=result.mode,
        error=result.error,
        debug_channels=result.debug_channels,
    )


def _can_reuse_scene_result(frame: InpaintFrame, last_scene_id: str, last_scene_result: InpaintResult | None) -> bool:
    return (not frame.stream_diffusion) and bool(frame.scene_id) and frame.scene_id == last_scene_id and last_scene_result is not None


async def _receive_latest_frame(websocket: WebSocket) -> dict:
    """Block until at least one frame arrives, then drain any additional queued frames."""
    payload = await websocket.receive_json()
    drained = 0
    while drained < _FRAME_DRAIN_LIMIT:
        try:
            payload = await asyncio.wait_for(websocket.receive_json(), timeout=_FRAME_DRAIN_TIMEOUT_S)
            drained += 1
        except (asyncio.TimeoutError, asyncio.CancelledError):
            break
    return payload


@router.websocket("/ws/inpaint")
async def inpaint_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    last_scene_id = ""
    last_scene_result: InpaintResult | None = None
    try:
        while True:
            payload = await _receive_latest_frame(websocket)
            binary_frames = payload.get("response_mode") == "binary"
            frame: InpaintFrame | None = None
            try:
                frame = InpaintFrame.model_validate(payload)
                _state.latest_sampling_frame = frame
                if _can_reuse_scene_result(frame, last_scene_id, last_scene_result):
                    result = _result_for_frame(frame, last_scene_result)
                elif sampling_pause_event.is_set():
                    result = InpaintResult(
                        client_frame_id=frame.client_frame_id,
                        client_input_id=frame.client_input_id,
                        scene_id=frame.scene_id,
                        image="", fps=0, latency_ms=0, mode=engine_mode(),
                        error="Realtime sampling paused for priority layer task",
                    )
                else:
                    async with _device_lock(frame.device):
                        if sampling_pause_event.is_set():
                            result = InpaintResult(
                                client_frame_id=frame.client_frame_id,
                                client_input_id=frame.client_input_id,
                                scene_id=frame.scene_id,
                                image="", fps=0, latency_ms=0, mode=engine_mode(),
                                error="Realtime sampling paused for priority layer task",
                            )
                        else:
                            session = get_diffusers_session(frame.device)
                            request = inpaint_frame_to_request(frame)
                            frame_result = await asyncio.to_thread(session.step, request)
                            result = InpaintResult(
                                client_frame_id=frame.client_frame_id,
                                client_input_id=frame.client_input_id,
                                scene_id=frame.scene_id,
                                image=frame_result.encoded_image or "",
                                fps=frame_result.fps,
                                latency_ms=frame_result.latency_ms,
                                mode=frame_result.mode,
                                error=frame_result.error,
                                debug_channels=frame_result.raw_debug_urls,
                            )
                            if (not frame.stream_diffusion) and frame.scene_id and not result.error:
                                last_scene_id = frame.scene_id
                                last_scene_result = result
            except Exception as exc:
                logger.exception("Realtime inpaint frame failed")
                result = InpaintResult(
                    client_frame_id=frame.client_frame_id if frame is not None else int(payload.get("client_frame_id") or 0),
                    client_input_id=frame.client_input_id if frame is not None else int(payload.get("client_input_id") or 0),
                    scene_id=frame.scene_id if frame is not None else str(payload.get("scene_id") or ""),
                    image="", fps=0, latency_ms=0, mode=engine_mode(),
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            if binary_frames and not result.error and result.image:
                mime, frame_bytes = data_url_bytes(result.image)
                meta: dict = {
                    "client_frame_id": result.client_frame_id,
                    "client_input_id": result.client_input_id,
                    "scene_id": result.scene_id,
                    "fps": result.fps,
                    "latency_ms": result.latency_ms,
                    "mode": result.mode,
                    "content_type": mime,
                    "binary": True,
                }
                if result.debug_channels:
                    meta["debug_channels"] = result.debug_channels
                await websocket.send_json(meta)
                await websocket.send_bytes(frame_bytes)
            else:
                await websocket.send_json(result.model_dump())
    except WebSocketDisconnect:
        return
