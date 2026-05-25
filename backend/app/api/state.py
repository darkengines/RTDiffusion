"""Shared mutable application state — engine registry, task stores, synchronisation primitives."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import DiffusionEngine
    from ..inference import DiffusersSession

logger = logging.getLogger("rtdiffusion.api")

# ── Engine registry ────────────────────────────────────────────────

_engine: "DiffusionEngine | None" = None
_engines: dict[str, "DiffusionEngine"] = {}
_engine_init_lock = threading.Lock()
_diffusers_sessions: dict[str, "DiffusersSession"] = {}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# ── Task stores ────────────────────────────────────────────────────

from ..schemas import LayerTaskProgress, MotionTaskProgress

layer_tasks: dict[str, LayerTaskProgress] = {}
motion_tasks: dict[str, MotionTaskProgress] = {}

layer_task_lock = threading.Lock()
motion_task_lock = threading.Lock()

# ── System tasks ───────────────────────────────────────────────────

system_tasks: dict[str, dict] = {}
system_task_lock = threading.Lock()

# ── Realtime sampling state ────────────────────────────────────────

from ..schemas import InpaintFrame

layer_generation_lock: asyncio.Lock  # initialised on first use via _get_layer_lock()
engine_locks: dict[str, asyncio.Lock] = {}
sampling_pause_event = asyncio.Event()
latest_sampling_frame: InpaintFrame | None = None


def _get_layer_generation_lock() -> asyncio.Lock:
    global layer_generation_lock
    try:
        return layer_generation_lock
    except NameError:
        layer_generation_lock = asyncio.Lock()
        return layer_generation_lock


# ── Engine access helpers ──────────────────────────────────────────

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


def get_engine(device: str | None = None) -> "DiffusionEngine":
    global _engine
    from ..engine import DiffusionEngine, EngineConfig

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


def get_diffusers_session(device: str | None = None) -> "DiffusersSession":
    from ..inference import DiffusersSession

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


# ── System task helpers ────────────────────────────────────────────

import time as _time


def set_system_task(
    task_id: str, status: str, phase: str, progress: float, message: str, error: str | None = None
) -> None:
    now = _time.monotonic()
    with system_task_lock:
        prev = system_tasks.get(task_id, {})
        system_tasks[task_id] = {
            "task_id": task_id,
            "type": "system",
            "status": status,
            "phase": phase,
            "progress": round(max(0.0, min(1.0, progress)), 3),
            "message": message,
            "error": error,
            "_started_at": prev.get("_started_at", now),
            "_completed_at": now if status in {"complete", "error"} else prev.get("_completed_at"),
        }


def remove_system_task(task_id: str) -> None:
    with system_task_lock:
        system_tasks.pop(task_id, None)


def on_stream_session_build(phase: str, progress: float, message: str) -> None:
    task_id = "sys_stream_build"
    if phase == "ready":
        set_system_task(task_id, "complete", phase, progress, message)
    elif phase == "error":
        set_system_task(task_id, "error", phase, progress, message, error=message)
    elif phase == "idle":
        remove_system_task(task_id)
    else:
        set_system_task(task_id, "running", phase, progress, message)


# ── Layer / motion task helpers ────────────────────────────────────

def set_layer_task(progress: LayerTaskProgress) -> None:
    with layer_task_lock:
        layer_tasks[progress.task_id] = progress
    print(
        f"[layer-task {progress.task_id[:8]}] {progress.status} {progress.progress * 100:5.1f}% "
        f"{progress.phase}: {progress.error or progress.message}",
        flush=True,
    )


def get_layer_task(task_id: str) -> LayerTaskProgress | None:
    with layer_task_lock:
        return layer_tasks.get(task_id)


def set_motion_task(progress: MotionTaskProgress) -> None:
    with motion_task_lock:
        motion_tasks[progress.task_id] = progress
    print(
        f"[motion-task {progress.task_id[:8]}] {progress.status} {progress.progress * 100:5.1f}% "
        f"{progress.phase}: {progress.error or progress.message}",
        flush=True,
    )


def get_motion_task(task_id: str) -> MotionTaskProgress | None:
    with motion_task_lock:
        return motion_tasks.get(task_id)
