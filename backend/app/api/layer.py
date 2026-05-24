"""Layer generation endpoints and async task management."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Callable

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..schemas import (
    LayerGenerateFrame,
    LayerGenerateResult,
    LayerTaskProgress,
    LayerTaskStart,
)
from .state import (
    _device_lock,
    _get_layer_generation_lock,
    _request_device,
    engine_mode,
    get_engine,
    get_layer_task,
    latest_sampling_frame,
    sampling_pause_event,
    set_layer_task,
    unload_engine_runtime,
)

logger = logging.getLogger("rtdiffusion.api")

router = APIRouter()


@router.post("/layer/generate")
async def generate_layer(frame: LayerGenerateFrame) -> LayerGenerateResult:
    try:
        async with _get_layer_generation_lock():
            return await _run_priority_layer_generation(frame)
    except Exception as exc:
        logger.exception("Layer generation request failed")
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc


@router.post("/layer/tasks")
async def start_layer_task(frame: LayerGenerateFrame) -> LayerTaskStart:
    task_id = uuid.uuid4().hex
    set_layer_task(LayerTaskProgress(
        task_id=task_id, status="queued", phase="queued", progress=0, message="Queued layer generation",
    ))
    asyncio.create_task(_run_layer_task(task_id, frame))
    return LayerTaskStart(task_id=task_id)


@router.get("/layer/tasks/{task_id}")
def layer_task_status(task_id: str) -> LayerTaskProgress:
    progress = get_layer_task(task_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Layer task not found")
    return progress


@router.websocket("/ws/tasks/{task_id}")
async def layer_task_socket(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    try:
        last_payload = ""
        while True:
            progress = get_layer_task(task_id)
            if progress is None:
                progress = LayerTaskProgress(
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


async def _run_layer_task(task_id: str, frame: LayerGenerateFrame) -> None:
    started = time.perf_counter()

    def update(phase: str, progress: float, message: str, result: LayerGenerateResult | None = None) -> None:
        previous = get_layer_task(task_id)
        set_layer_task(LayerTaskProgress(
            task_id=task_id, status="running", phase=phase, progress=progress, message=message,
            result=result if result is not None else previous.result if previous is not None else None,
        ))

    try:
        update("queued", 0.005, "Waiting for GPU worker")
        async with _get_layer_generation_lock():
            result = await _run_priority_layer_generation(frame, update)
        set_layer_task(LayerTaskProgress(
            task_id=task_id, status="complete", phase="complete", progress=1,
            message=f"Generated {len(result.variations)} variation(s) in {round(result.latency_ms)} ms",
            result=result,
        ))
    except Exception as exc:
        logger.exception("Layer task %s failed", task_id)
        set_layer_task(LayerTaskProgress(
            task_id=task_id, status="error", phase="error", progress=1,
            message="Layer generation failed",
            error=f"{exc.__class__.__name__}: {exc}",
            result=LayerGenerateResult(
                variations=[], latency_ms=(time.perf_counter() - started) * 1000, mode=engine_mode()
            ),
        ))


async def _run_priority_layer_generation(
    frame: LayerGenerateFrame,
    update: Callable[[str, float, str, LayerGenerateResult | None], None] | None = None,
) -> LayerGenerateResult:
    def emit(phase: str, progress: float, message: str, result: LayerGenerateResult | None = None) -> None:
        if update is not None:
            update(phase, progress, message, result)

    sampling_frame = latest_sampling_frame
    layer_device = _request_device(frame.device)
    sampling_device = _request_device(sampling_frame.device if sampling_frame is not None else None)
    pause_sampling = layer_device == sampling_device
    if pause_sampling:
        emit("pause_sampling", 0.01, "Pausing realtime sampling for priority layer task")
        sampling_pause_event.set()
    try:
        async with _device_lock(frame.device):
            if pause_sampling:
                emit("unload_sampling_model", 0.03, "Unloading realtime sampling model from matching GPU")
                await asyncio.to_thread(unload_engine_runtime, frame.device)
            emit("priority_task", 0.06, "Running priority layer generation task")

            def progress_proxy(phase: str, value: float, message: str, result: LayerGenerateResult | None = None) -> None:
                emit(phase, 0.08 + value * 0.82, message, result)

            try:
                result = await asyncio.to_thread(
                    get_engine(frame.device).generate_layer,
                    frame,
                    progress_proxy if update is not None else None,
                )
                return result
            finally:
                if pause_sampling:
                    emit("load_sampling_model", 0.93, "Reloading realtime sampling model")
                    if sampling_frame is not None:
                        await asyncio.to_thread(get_engine(sampling_frame.device).prepare_for_frame, sampling_frame)
                    emit("resume_sampling", 0.98, "Resuming realtime sampling")
    finally:
        if pause_sampling:
            sampling_pause_event.clear()
