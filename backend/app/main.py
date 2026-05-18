import asyncio
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from .assets import list_assets
from .engine import DiffusionEngine, EngineConfig
from .image_io import data_url_bytes, encode_data_url
from .inference import inpaint_frame_to_request
from .motion import generate_motion_clip, load_motion_clip, missing_motion_adapter_message, motion_adapter_configured, motion_clip_path, motion_frame_path, motion_video_path
from .renderers import installed_runtime_versions, renderer_capabilities
from .rtc import router as rtc_router
from .sana_video import (
    get_video_generator,
    get_longive_stream,
    get_wm_generator,
    _DEFAULT_480P as _SANA_480P,
    _DEFAULT_720P as _SANA_720P,
)
from .sana_pipeline import is_sana_model as _is_sana_model
from .schemas import (
    AssetCatalog,
    InpaintFrame,
    InpaintResult,
    LayerGenerateFrame,
    LayerGenerateResult,
    LayerTaskProgress,
    LayerTaskStart,
    MotionClipRequest,
    MotionClipFrame,
    MotionTaskProgress,
    MotionTaskStart,
)

sana_video_tasks: dict[str, dict] = {}
sana_task_lock = threading.Lock()


logging.basicConfig(
    level=os.getenv("RTD_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rtdiffusion.api")

app = FastAPI(title="RTDiffusion")
app.include_router(rtc_router)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: DiffusionEngine | None = None
_engines: dict[str, DiffusionEngine] = {}
_engine_init_lock = threading.Lock()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
layer_tasks: dict[str, LayerTaskProgress] = {}
motion_tasks: dict[str, MotionTaskProgress] = {}
layer_task_lock = threading.Lock()
motion_task_lock = threading.Lock()
layer_generation_lock = asyncio.Lock()
engine_locks: dict[str, asyncio.Lock] = {}
sampling_pause_event = asyncio.Event()
latest_sampling_frame: InpaintFrame | None = None


def _request_device(device: str | None = None) -> str:
    value = (device or os.getenv("RTD_DEVICE", "cuda")).strip().lower()
    if value == "cuda":
        return "cuda:0"
    return value if value == "cpu" or value.startswith("cuda") else "cpu"


def _device_lock(device: str | None = None) -> asyncio.Lock:
    resolved = _request_device(device)
    if resolved not in engine_locks:
        engine_locks[resolved] = asyncio.Lock()
    return engine_locks[resolved]


def get_engine(device: str | None = None) -> DiffusionEngine:
    global _engine
    resolved = _request_device(device)
    if resolved in _engines:
        return _engines[resolved]
    if _engine is None and resolved == _request_device(None):
        with _engine_init_lock:
            if _engine is None:
                _engine = DiffusionEngine(EngineConfig(device=resolved))
                _engines[resolved] = _engine
        return _engine
    with _engine_init_lock:
        if resolved not in _engines:
            _engines[resolved] = DiffusionEngine(EngineConfig(device=resolved))
        return _engines[resolved]


_diffusers_sessions: dict[str, "DiffusersSession"] = {}


def get_diffusers_session(device: str | None = None) -> "DiffusersSession":
    """Return the `InferenceSession` adapter for the device's engine.

    Cached per device so the WebSocket loop doesn't allocate one per frame.
    """
    from .inference import DiffusersSession

    resolved = _request_device(device)
    engine = get_engine(resolved)
    cached = _diffusers_sessions.get(resolved)
    if cached is not None and cached.engine is engine:
        return cached
    session = DiffusersSession(engine)
    _diffusers_sessions[resolved] = session
    return session


def engine_mode() -> str:
    if _engine is not None:
        return _engine.mode
    return "lazy"


def unload_engine_runtime(device: str | None = None) -> None:
    if device is not None:
        engine = _engines.get(_request_device(device))
        if engine is not None:
            engine.unload_runtime()
        return
    for engine in list(_engines.values()):
        engine.unload_runtime()


@app.get("/system/gpus")
def system_gpus() -> dict[str, object]:
    devices: list[dict[str, object]] = [{"id": "cpu", "name": "CPU", "kind": "cpu", "available": True}]
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "id": f"cuda:{index}",
                        "name": props.name,
                        "kind": "cuda",
                        "available": True,
                        "memory_total": props.total_memory,
                    }
                )
    except Exception as exc:
        logger.warning("Failed to enumerate CUDA devices: %s", exc)
    return {"default": _request_device(None), "devices": devices}


@app.exception_handler(HTTPException)
async def log_http_exception(request: Request, exc: HTTPException):
    logger.warning(
        "HTTP %s %s -> %s: %s",
        request.method,
        request.url,
        exc.status_code,
        exc.detail,
    )
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def log_request_validation_exception(request: Request, exc: RequestValidationError):
    logger.exception("Request validation failed for %s %s", request.method, request.url)
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def log_unhandled_exception(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    raise exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": engine_mode()}


# ──────────────────────────────────────────────────────────────────
# SANA Video endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/sana/models")
def sana_models() -> dict[str, object]:
    """List available SANA model variants."""
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


@app.post("/sana/video/tasks")
async def start_sana_video_task(request: Request) -> dict[str, str]:
    """
    Start an asynchronous SANA image-to-video generation task.

    Body:
      image          — data URL (JPEG/PNG) of the input frame
      prompt         — text prompt
      negative_prompt — (optional)
    model          — "480p" | "720p" | "longlive" | "wm"  (default: "480p")
      num_frames     — number of frames (default 81 ≈ 3.4 s at 24 fps)
      guidance_scale — (default 6.0)
      num_inference_steps — (default 20)
      width / height — output resolution (default 832x480 or 1280x720)
      seed           — (optional) for reproducibility
      camera_poses   — list of [tx,ty,tz,rx,ry,rz] per frame (wm mode only)
      device         — (optional) "cuda:0"
    """
    body = await request.json()
    task_id = uuid.uuid4().hex

    with sana_task_lock:
        sana_video_tasks[task_id] = {"status": "queued", "progress": 0, "message": "Queued"}

    asyncio.create_task(_run_sana_video_task(task_id, body))
    return {"task_id": task_id}


@app.get("/sana/video/tasks/{task_id}")
def get_sana_video_task(task_id: str) -> dict:
    with sana_task_lock:
        task = sana_video_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="SANA video task not found")
    return task


@app.get("/sana/video/tasks/{task_id}/download")
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

        from .image_io import data_url_bytes
        img_url = body.get("image", "")
        if img_url:
            _, img_bytes = data_url_bytes(img_url)
            import io
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


@app.get("/controlnet/models")
def controlnet_models() -> list[dict]:
    """List locally available ControlNet models (ComfyUI directory)."""
    from .controlnet import list_local_controlnet_models
    return list_local_controlnet_models()


@app.get("/assets")
def assets() -> AssetCatalog:
    return AssetCatalog.model_validate(list_assets())


@app.get("/video/capabilities")
def video_capabilities() -> dict[str, object]:
    models = ["causal-forcing", "causal-forcing-2step", "causal-forcing-1step", "krea-realtime-video", "fastvideo"]
    return {
        "models": {
            model: {
                "configured": motion_adapter_configured(model),
                "message": "" if motion_adapter_configured(model) else missing_motion_adapter_message(model),
            }
            for model in models
        }
    }


@app.get("/renderers/capabilities")
def renderers_capabilities() -> dict[str, object]:
    capabilities = renderer_capabilities()
    capabilities["versions"] = installed_runtime_versions()
    return capabilities


@app.get("/motion/clips/{clip_id}/video")
def motion_video(clip_id: str) -> FileResponse:
    if load_motion_clip(clip_id) is None:
        raise HTTPException(status_code=404, detail="Motion clip not found")
    path = motion_video_path(clip_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Motion video not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/motion/clips/{clip_id}/frames/{frame_index}")
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


@app.get("/motion/clips/{clip_id}/previews/{preview_name}")
def motion_preview(clip_id: str, preview_name: str) -> FileResponse:
    if "/" in preview_name or "\\" in preview_name or not preview_name.endswith(".png"):
        raise HTTPException(status_code=404, detail="Motion preview not found")
    path = motion_clip_path(clip_id) / preview_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Motion preview not found")
    return FileResponse(path, media_type="image/png")


@app.post("/motion/tasks")
async def start_motion_task(frame: MotionClipRequest) -> MotionTaskStart:
    if not motion_adapter_configured(frame.model):
        raise HTTPException(status_code=409, detail=missing_motion_adapter_message(frame.model))
    task_id = uuid.uuid4().hex
    _set_motion_task(
        MotionTaskProgress(
            task_id=task_id,
            status="queued",
            phase="queued",
            progress=0,
            message="Queued motion clip generation",
        )
    )
    asyncio.create_task(_run_motion_task(task_id, frame))
    return MotionTaskStart(task_id=task_id)


@app.get("/motion/tasks/{task_id}")
def motion_task(task_id: str) -> MotionTaskProgress:
    progress = _get_motion_task(task_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Motion task not found")
    return progress


async def _run_motion_task(task_id: str, frame: MotionClipRequest) -> None:
    try:
        _set_motion_task(MotionTaskProgress(task_id=task_id, status="running", phase="keyframe", progress=0.12, message="Preparing SDXL motion keyframes"))
        frame = await _frame_with_arrival_keyframe(frame)
        video_device = _request_device(frame.device)
        sampling_device = _request_device(latest_sampling_frame.device if latest_sampling_frame is not None else None)
        if video_device == sampling_device:
            _set_motion_task(MotionTaskProgress(task_id=task_id, status="running", phase="unload", progress=0.18, message="Freeing matching GPU before video generation"))
            async with _device_lock(video_device):
                await asyncio.to_thread(unload_engine_runtime, video_device)
        _set_motion_task(MotionTaskProgress(task_id=task_id, status="running", phase="synthesize", progress=0.35, message="Synthesizing temporally consistent loop frames"))
        preview_frames: list[MotionClipFrame] = []

        def update_native_progress(phase: str, value: float, message: str, frames: list[MotionClipFrame] | None = None) -> None:
            if frames:
                known = {item.index for item in preview_frames}
                preview_frames.extend(item for item in frames if item.index not in known)
            _set_motion_task(
                MotionTaskProgress(
                    task_id=task_id,
                    status="running",
                    phase=phase,
                    progress=0.2 + value * 0.76,
                    message=message,
                    preview_frames=preview_frames.copy(),
                )
            )

        result = await asyncio.to_thread(generate_motion_clip, frame, update_native_progress)
        _set_motion_task(
            MotionTaskProgress(
                task_id=task_id,
                status="complete",
                phase="complete",
                progress=1,
                message=f"Cached {result.frame_count} motion frame(s)",
                preview_frames=result.frames,
                result=result,
            )
        )
    except Exception as exc:
        logger.exception("Motion task %s failed", task_id)
        _set_motion_task(
            MotionTaskProgress(
                task_id=task_id,
                status="error",
                phase="error",
                progress=1,
                message="Motion generation failed",
                error=f"{exc.__class__.__name__}: {exc}",
            )
        )


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


def _resolve_source(root: str, rel: str = "") -> Path:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
                logger.warning("Invalid source root: root=%r resolved=%s rel=%r", root, root_path, rel)
                raise HTTPException(status_code=400, detail="Source root does not exist or is not a folder")
    path = (root_path / rel).resolve()
    if root_path != path and root_path not in path.parents:
                logger.warning("Rejected source path outside root: root=%s rel=%r resolved=%s", root_path, rel, path)
                raise HTTPException(status_code=400, detail="Source path is outside the selected root")
    return path


@app.get("/sources")
def sources(root: str = Query(..., min_length=1), limit: int = Query(600, ge=1, le=2000)) -> dict[str, object]:
    root_path = _resolve_source(root)
    items = []
    for path in root_path.rglob("*"):
        if len(items) >= limit:
            break
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stat = path.stat()
            items.append(
                {
                    "name": path.name,
                    "rel": path.relative_to(root_path).as_posix(),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
    items.sort(key=lambda item: str(item["rel"]).lower())
    return {"root": str(root_path), "images": items}


@app.post("/sources/pick-root")
async def pick_source_root() -> dict[str, str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        logger.exception("Folder picker import failed")
        raise HTTPException(status_code=500, detail=f"Folder picker is not available: {exc}") from exc

    window = None
    try:
        window = tk.Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        window.update()
        selected = filedialog.askdirectory(parent=window, title="Choose RTDiffusion source root")
    except Exception as exc:
        logger.exception("Folder picker failed")
        raise HTTPException(status_code=500, detail=f"Folder picker failed: {exc}") from exc
    finally:
        if window is not None:
            try:
                window.destroy()
            except Exception:
                pass
    if not selected:
        return {"root": ""}
    return {"root": str(Path(selected).resolve())}


@app.get("/sources/image")
def source_image(root: str = Query(..., min_length=1), rel: str = Query(..., min_length=1)) -> FileResponse:
    path = _resolve_source(root, rel)
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.post("/layer/generate")
async def generate_layer(frame: LayerGenerateFrame) -> LayerGenerateResult:
    try:
        async with layer_generation_lock:
            return await _run_priority_layer_generation(frame)
    except Exception as exc:
        logger.exception("Layer generation request failed")
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc


@app.post("/layer/tasks")
async def start_layer_task(frame: LayerGenerateFrame) -> LayerTaskStart:
    task_id = uuid.uuid4().hex
    _set_layer_task(
        LayerTaskProgress(
            task_id=task_id,
            status="queued",
            phase="queued",
            progress=0,
            message="Queued layer generation",
        )
    )
    asyncio.create_task(_run_layer_task(task_id, frame))
    return LayerTaskStart(task_id=task_id)


async def _run_layer_task(task_id: str, frame: LayerGenerateFrame) -> None:
    started = time.perf_counter()

    def update(phase: str, progress: float, message: str, result: LayerGenerateResult | None = None) -> None:
        previous = _get_layer_task(task_id)
        _set_layer_task(
            LayerTaskProgress(
                task_id=task_id,
                status="running",
                phase=phase,
                progress=progress,
                message=message,
                result=result if result is not None else previous.result if previous is not None else None,
            )
        )

    try:
        update("queued", 0.005, "Waiting for GPU worker")
        async with layer_generation_lock:
            result = await _run_priority_layer_generation(frame, update)
        _set_layer_task(
            LayerTaskProgress(
                task_id=task_id,
                status="complete",
                phase="complete",
                progress=1,
                message=f"Generated {len(result.variations)} variation(s) in {round(result.latency_ms)} ms",
                result=result,
            )
        )
    except Exception as exc:
        logger.exception("Layer task %s failed", task_id)
        _set_layer_task(
            LayerTaskProgress(
                task_id=task_id,
                status="error",
                phase="error",
                progress=1,
                message="Layer generation failed",
                error=f"{exc.__class__.__name__}: {exc}",
                result=LayerGenerateResult(variations=[], latency_ms=(time.perf_counter() - started) * 1000, mode=engine_mode()),
            )
        )


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
                result = await asyncio.to_thread(get_engine(frame.device).generate_layer, frame, progress_proxy if update is not None else None)
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


def _set_layer_task(progress: LayerTaskProgress) -> None:
    with layer_task_lock:
        layer_tasks[progress.task_id] = progress
    print(
        f"[layer-task {progress.task_id[:8]}] {progress.status} {progress.progress * 100:5.1f}% "
        f"{progress.phase}: {progress.error or progress.message}",
        flush=True,
    )


def _get_layer_task(task_id: str) -> LayerTaskProgress | None:
    with layer_task_lock:
        return layer_tasks.get(task_id)


def _set_motion_task(progress: MotionTaskProgress) -> None:
    with motion_task_lock:
        motion_tasks[progress.task_id] = progress
    print(
        f"[motion-task {progress.task_id[:8]}] {progress.status} {progress.progress * 100:5.1f}% "
        f"{progress.phase}: {progress.error or progress.message}",
        flush=True,
    )


def _get_motion_task(task_id: str) -> MotionTaskProgress | None:
    with motion_task_lock:
        return motion_tasks.get(task_id)


@app.websocket("/ws/tasks/{task_id}")
async def layer_task_socket(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    try:
        last_payload = ""
        while True:
            progress = _get_layer_task(task_id)
            if progress is None:
                progress = LayerTaskProgress(
                    task_id=task_id,
                    status="error",
                    phase="missing",
                    progress=1,
                    message="Task not found",
                    error="Task not found",
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


@app.websocket("/ws/motion/{task_id}")
async def motion_task_socket(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    try:
        last_payload = ""
        while True:
            progress = _get_motion_task(task_id)
            if progress is None:
                progress = MotionTaskProgress(
                    task_id=task_id,
                    status="error",
                    phase="missing",
                    progress=1,
                    message="Task not found",
                    error="Task not found",
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


async def _receive_latest_frame(websocket: WebSocket) -> dict:
    """Block until at least one frame arrives, then drain any additional queued frames.
    For StreamDiffusion the client sends continuously; we always want the freshest one."""
    payload = await websocket.receive_json()
    while True:
        try:
            payload = await asyncio.wait_for(websocket.receive_json(), timeout=0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            break
    return payload


@app.websocket("/ws/inpaint")
async def inpaint_socket(websocket: WebSocket) -> None:
    global latest_sampling_frame
    await websocket.accept()
    try:
        while True:
            payload = await _receive_latest_frame(websocket)
            binary_frames = payload.get("response_mode") == "binary"
            try:
                frame = InpaintFrame.model_validate(payload)
                latest_sampling_frame = frame
                if sampling_pause_event.is_set():
                    result = InpaintResult(
                        image="",
                        fps=0,
                        latency_ms=0,
                        mode=engine_mode(),
                        error="Realtime sampling paused for priority layer task",
                    )
                else:
                    async with _device_lock(frame.device):
                        if sampling_pause_event.is_set():
                            result = InpaintResult(
                                image="",
                                fps=0,
                                latency_ms=0,
                                mode=engine_mode(),
                                error="Realtime sampling paused for priority layer task",
                            )
                        else:
                            # Cutover (refactor Step 4): /ws/inpaint goes through the
                            # InferenceSession seam. The DiffusersSession adapter forwards
                            # to engine.generate() unchanged, so payload bytes are
                            # byte-equal. Composer takes over in Step 6.
                            session = get_diffusers_session(frame.device)
                            request = inpaint_frame_to_request(frame)
                            frame_result = await asyncio.to_thread(session.step, request)
                            result = InpaintResult(
                                image=frame_result.encoded_image or "",
                                fps=frame_result.fps,
                                latency_ms=frame_result.latency_ms,
                                mode=frame_result.mode,
                                error=frame_result.error,
                                debug_channels=frame_result.raw_debug_urls,
                            )
            except Exception as exc:
                logger.exception("Realtime inpaint frame failed")
                result = InpaintResult(
                    image="",
                    fps=0,
                    latency_ms=0,
                    mode=engine_mode(),
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            if binary_frames and not result.error and result.image:
                mime, frame_bytes = data_url_bytes(result.image)
                meta: dict = {
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
