"""SANA image and video generation endpoints."""
from __future__ import annotations

import asyncio
import io
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from PIL import Image

from ..sana_video import (
    _DEFAULT_480P as _SANA_480P,
    _DEFAULT_720P as _SANA_720P,
    get_longive_stream,
    get_video_generator,
    get_wm_generator,
)
from .state import _request_device, sana_task_lock, sana_video_tasks

logger = logging.getLogger("rtdiffusion.api")

router = APIRouter()


@router.get("/sana/models")
def sana_models() -> dict[str, object]:
    return {
        "sprint": [
            {"id": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers", "name": "SANA Sprint 0.6B (fast)", "type": "sana_sprint"},
            {"id": "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", "name": "SANA Sprint 1.6B (quality)", "type": "sana_sprint"},
        ],
        "video": [
            {"id": _SANA_480P, "name": "SANA-Video 2B 480p", "type": "sana_video"},
            {"id": _SANA_720P, "name": "SANA-Video 2B 720p (HQ)", "type": "sana_video"},
            {"id": "Efficient-Large-Model/SANA-Video_2B_480p_LongLive_diffusers", "name": "SANA-Video LongLive 480p (streaming)", "type": "sana_longlive"},
        ],
        "world_model": [
            {"id": "Efficient-Large-Model/SANA-WM-2.6B", "name": "SANA-WM 2.6B (camera control)", "type": "sana_wm"},
        ],
    }


@router.post("/sana/video/tasks")
async def start_sana_video_task(request) -> dict[str, str]:
    from fastapi import Request
    body = await request.json()
    task_id = uuid.uuid4().hex
    with sana_task_lock:
        sana_video_tasks[task_id] = {"status": "queued", "progress": 0, "message": "Queued"}
    asyncio.create_task(_run_sana_video_task(task_id, body))
    return {"task_id": task_id}


@router.get("/sana/video/tasks/{task_id}")
def get_sana_video_task(task_id: str) -> dict:
    with sana_task_lock:
        task = sana_video_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="SANA video task not found")
    return task


@router.websocket("/ws/sana/video/{task_id}")
async def sana_video_task_socket(websocket: WebSocket, task_id: str) -> None:
    import json
    await websocket.accept()
    try:
        last_payload = ""
        while True:
            with sana_task_lock:
                task = sana_video_tasks.get(task_id)
            if task is None:
                await websocket.send_json({"task_id": task_id, "status": "error", "progress": 1,
                                           "message": "Task not found", "error": "Task not found"})
                break
            task_with_id = {"task_id": task_id, **task}
            payload_text = json.dumps(task_with_id, sort_keys=True)
            if payload_text != last_payload:
                await websocket.send_json(task_with_id)
                last_payload = payload_text
            if task.get("status") in {"complete", "error"}:
                break
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return


@router.get("/sana/video/tasks/{task_id}/download")
def download_sana_video(task_id: str) -> FileResponse:
    with sana_task_lock:
        task = sana_video_tasks.get(task_id)
    if task is None or task.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Video not ready")
    path = task.get("video_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Video file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"sana_{task_id[:8]}.mp4")


async def _run_sana_video_task(task_id: str, body: dict) -> None:
    def _set(update: dict) -> None:
        with sana_task_lock:
            sana_video_tasks[task_id] = {**sana_video_tasks.get(task_id, {}), **update}

    def _progress(phase: str, frac: float, message: str) -> None:
        _set({"status": "running", "phase": phase, "progress": round(frac, 3), "message": message})

    try:
        _set({"status": "running", "phase": "decode_image", "progress": 0.01, "message": "Decoding input image"})

        from ..image_io import data_url_bytes
        img_url = body.get("image", "")
        if img_url:
            _, img_bytes = data_url_bytes(img_url)
            init_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            w = int(body.get("width", 832))
            h = int(body.get("height", 480))
            init_image = Image.new("RGB", (w, h), (128, 128, 128))

        model_variant = body.get("model", "480p")
        device = body.get("device") or _request_device(None)
        prompt = body.get("prompt", "")
        negative_prompt = body.get("negative_prompt", "")
        num_frames = int(body.get("num_frames", 81))
        guidance_scale = float(body.get("guidance_scale", 6.0))
        num_inference_steps = int(body.get("num_inference_steps", 20))
        width = int(body.get("width", 832))
        height = int(body.get("height", 480))
        seed = body.get("seed")

        _set({"status": "running", "phase": "load_model", "progress": 0.03, "message": "Loading SANA model"})

        video_path: Path | None = None

        if model_variant == "wm":
            gen = get_wm_generator(device=device)
            camera_poses = body.get("camera_poses")
            video_path = await asyncio.to_thread(
                gen.generate,
                image=init_image, prompt=prompt, negative_prompt=negative_prompt,
                camera_poses=camera_poses, num_frames=num_frames,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                width=width, height=height, progress=_progress,
            )
        elif model_variant in {"longlive", "longive"}:
            stream = get_longive_stream(device=device)
            video_path = await asyncio.to_thread(
                stream.start,
                init_image=init_image, prompt=prompt, negative_prompt=negative_prompt,
                num_frames=num_frames, guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width, height=height, seed=seed if seed is not None else None,
                progress=_progress,
            )
        else:
            model_id = _SANA_720P if model_variant == "720p" else _SANA_480P
            gen = get_video_generator(model_id=model_id, device=device)
            video_path = await asyncio.to_thread(
                gen.generate,
                image=init_image, prompt=prompt, negative_prompt=negative_prompt,
                num_frames=num_frames, guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width, height=height,
                seed=int(seed) if seed is not None else None,
                progress=_progress,
            )

        if video_path is None:
            _set({"status": "error", "progress": 1, "message": "Video generation failed", "error": "No output returned"})
            return

        _set({
            "status": "complete",
            "phase": "complete",
            "progress": 1.0,
            "message": f"Generated {num_frames} frames",
            "video_path": str(video_path),
            "download_url": f"/sana/video/tasks/{task_id}/download",
        })

    except Exception as exc:
        logger.exception("SANA video task %s failed", task_id)
        _set({"status": "error", "progress": 1, "message": "Task failed", "error": f"{exc.__class__.__name__}: {exc}"})
