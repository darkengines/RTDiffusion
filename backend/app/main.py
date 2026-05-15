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

from .assets import list_assets
from .engine import DiffusionEngine
from .schemas import AssetCatalog, InpaintFrame, InpaintResult, LayerGenerateFrame, LayerGenerateResult, LayerTaskProgress, LayerTaskStart

logging.basicConfig(
    level=os.getenv("RTD_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rtdiffusion.api")

app = FastAPI(title="RTDiffusion")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = DiffusionEngine()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
layer_tasks: dict[str, LayerTaskProgress] = {}
layer_task_lock = threading.Lock()
layer_generation_lock = asyncio.Lock()
engine_lock = asyncio.Lock()
sampling_pause_event = asyncio.Event()
latest_sampling_frame: InpaintFrame | None = None


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
    return {"status": "ok", "mode": engine.mode}


@app.get("/assets")
def assets() -> AssetCatalog:
    return AssetCatalog.model_validate(list_assets())


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
def pick_source_root() -> dict[str, str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        logger.exception("Folder picker import failed")
        raise HTTPException(status_code=500, detail=f"Folder picker is not available: {exc}") from exc

    window = tk.Tk()
    window.withdraw()
    window.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(title="Choose RTDiffusion source root")
    finally:
        window.destroy()
    if not selected:
        raise HTTPException(status_code=400, detail="No folder selected")
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
                result=LayerGenerateResult(variations=[], latency_ms=(time.perf_counter() - started) * 1000, mode=engine.mode),
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
    emit("pause_sampling", 0.01, "Pausing realtime sampling for priority layer task")
    sampling_pause_event.set()
    try:
        async with engine_lock:
            emit("unload_sampling_model", 0.03, "Unloading realtime sampling model from VRAM")
            await asyncio.to_thread(engine.unload_runtime)
            emit("priority_task", 0.06, "Running priority layer generation task")

            def progress_proxy(phase: str, value: float, message: str, result: LayerGenerateResult | None = None) -> None:
                emit(phase, 0.08 + value * 0.82, message, result)

            try:
                result = await asyncio.to_thread(engine.generate_layer, frame, progress_proxy if update is not None else None)
                return result
            finally:
                emit("load_sampling_model", 0.93, "Reloading realtime sampling model")
                if sampling_frame is not None:
                    await asyncio.to_thread(engine.prepare_for_frame, sampling_frame)
                emit("resume_sampling", 0.98, "Resuming realtime sampling")
    finally:
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


@app.websocket("/ws/inpaint")
async def inpaint_socket(websocket: WebSocket) -> None:
    global latest_sampling_frame
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                frame = InpaintFrame.model_validate(payload)
                latest_sampling_frame = frame
                if sampling_pause_event.is_set():
                    result = InpaintResult(
                        image="",
                        fps=0,
                        latency_ms=0,
                        mode=engine.mode,
                        error="Realtime sampling paused for priority layer task",
                    )
                else:
                    async with engine_lock:
                        if sampling_pause_event.is_set():
                            result = InpaintResult(
                                image="",
                                fps=0,
                                latency_ms=0,
                                mode=engine.mode,
                                error="Realtime sampling paused for priority layer task",
                            )
                        else:
                            result = await asyncio.to_thread(engine.generate, frame)
            except Exception as exc:
                logger.exception("Realtime inpaint frame failed")
                result = InpaintResult(
                    image="",
                    fps=0,
                    latency_ms=0,
                    mode=engine.mode,
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            await websocket.send_json(result.model_dump())
    except WebSocketDisconnect:
        return