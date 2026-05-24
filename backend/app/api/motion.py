"""Motion clip generation endpoints and async task management."""
from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from PIL import Image

from ..motion import (
    generate_motion_clip,
    load_motion_clip,
    missing_motion_adapter_message,
    motion_adapter_configured,
    motion_clip_path,
    motion_frame_path,
    motion_video_path,
)
from ..schemas import (
    InpaintFrame,
    MotionClipFrame,
    MotionClipRequest,
    MotionTaskProgress,
    MotionTaskStart,
)
from ..image_io import encode_data_url
from .state import (
    _device_lock,
    _request_device,
    get_engine,
    get_motion_task,
    latest_sampling_frame,
    set_motion_task,
    unload_engine_runtime,
)

logger = logging.getLogger("rtdiffusion.api")

router = APIRouter()


@router.get("/motion/clips/{clip_id}/video")
def motion_video(clip_id: str) -> FileResponse:
    if load_motion_clip(clip_id) is None:
        raise HTTPException(status_code=404, detail="Motion clip not found")
    path = motion_video_path(clip_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Motion video not found")
    return FileResponse(path, media_type="video/mp4")


@router.get("/motion/clips/{clip_id}/frames/{frame_index}")
def motion_frame(clip_id: str, frame_index: int) -> FileResponse:
    clip = load_motion_clip(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Motion clip not found")
    if frame_index < 0 or frame_index >= clip.frame_count:
        raise HTTPException(status_code=404, detail="Motion frame not found")
    path = motion_frame_path(clip_id, frame_index)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Motion frame not found")
    return FileResponse(path, media_type="image/png")


@router.get("/motion/clips/{clip_id}/previews/{preview_name}")
def motion_preview(clip_id: str, preview_name: str) -> FileResponse:
    if "/" in preview_name or "\\" in preview_name or not preview_name.endswith(".png"):
        raise HTTPException(status_code=404, detail="Motion preview not found")
    path = motion_clip_path(clip_id) / preview_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Motion preview not found")
    return FileResponse(path, media_type="image/png")


@router.post("/motion/tasks")
async def start_motion_task(frame: MotionClipRequest) -> MotionTaskStart:
    if not motion_adapter_configured(frame.model):
        raise HTTPException(status_code=409, detail=missing_motion_adapter_message(frame.model))
    task_id = uuid.uuid4().hex
    set_motion_task(MotionTaskProgress(
        task_id=task_id, status="queued", phase="queued", progress=0,
        message="Queued motion clip generation",
    ))
    asyncio.create_task(_run_motion_task(task_id, frame))
    return MotionTaskStart(task_id=task_id)


@router.get("/motion/tasks/{task_id}")
def motion_task(task_id: str) -> MotionTaskProgress:
    progress = get_motion_task(task_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Motion task not found")
    return progress


@router.websocket("/ws/motion/{task_id}")
async def motion_task_socket(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    try:
        last_payload = ""
        while True:
            progress = get_motion_task(task_id)
            if progress is None:
                progress = MotionTaskProgress(
                    task_id=task_id, status="error", phase="missing", progress=1,
                    message="Task not found", error="Task not found",
                )
            payload = progress.model_dump()
            payload_text = progress.model_dump_json()
            if payload_text != last_payload:
                await websocket.send_json(payload)
                last_payload = payload_text
            if progress.status in {"complete", "error"}:
                break
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return


async def _run_motion_task(task_id: str, frame: MotionClipRequest) -> None:
    try:
        set_motion_task(MotionTaskProgress(
            task_id=task_id, status="running", phase="keyframe", progress=0.12,
            message="Preparing SDXL motion keyframes",
        ))
        frame = await _frame_with_arrival_keyframe(frame)
        video_device = _request_device(frame.device)
        sampling_device = _request_device(latest_sampling_frame.device if latest_sampling_frame is not None else None)
        if video_device == sampling_device:
            set_motion_task(MotionTaskProgress(
                task_id=task_id, status="running", phase="unload", progress=0.18,
                message="Freeing matching GPU before video generation",
            ))
            async with _device_lock(video_device):
                await asyncio.to_thread(unload_engine_runtime, video_device)

        set_motion_task(MotionTaskProgress(
            task_id=task_id, status="running", phase="synthesize", progress=0.35,
            message="Synthesizing temporally consistent loop frames",
        ))
        preview_frames: list[MotionClipFrame] = []

        def update_native_progress(phase: str, value: float, message: str, frames: list[MotionClipFrame] | None = None) -> None:
            if frames:
                known = {item.index for item in preview_frames}
                preview_frames.extend(item for item in frames if item.index not in known)
            set_motion_task(MotionTaskProgress(
                task_id=task_id, status="running", phase=phase,
                progress=0.2 + value * 0.76, message=message,
                preview_frames=preview_frames.copy(),
            ))

        result = await asyncio.to_thread(generate_motion_clip, frame, update_native_progress)
        set_motion_task(MotionTaskProgress(
            task_id=task_id, status="complete", phase="complete", progress=1,
            message=f"Cached {result.frame_count} motion frame(s)",
            preview_frames=result.frames, result=result,
        ))
    except Exception as exc:
        logger.exception("Motion task %s failed", task_id)
        set_motion_task(MotionTaskProgress(
            task_id=task_id, status="error", phase="error", progress=1,
            message="Motion generation failed",
            error=f"{exc.__class__.__name__}: {exc}",
        ))


async def _frame_with_arrival_keyframe(frame: MotionClipRequest) -> MotionClipRequest:
    if frame.arrival_image or not frame.arrival_prompt.strip():
        return frame
    mask = Image.new("L", (frame.width, frame.height), 255)
    async with _device_lock(frame.device):
        result = await asyncio.to_thread(
            get_engine(frame.device).generate,
            InpaintFrame(
                prompt=frame.arrival_prompt,
                negative_prompt=frame.negative_prompt,
                model_path=frame.model_path,
                device=frame.device,
                lora_paths=frame.lora_paths,
                image=frame.source_image,
                mask=encode_data_url(mask, image_format="PNG"),
                width=frame.width,
                height=frame.height,
                steps=max(1, min(32, frame.arrival_steps)),
                strength=frame.arrival_strength,
                cfg=frame.arrival_cfg,
                sampler=frame.arrival_sampler,
                scheduler=frame.arrival_scheduler,
            ),
        )
    return frame.model_copy(update={"arrival_image": result.image})
