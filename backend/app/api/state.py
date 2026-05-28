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


def _prune_completed_system_tasks(now: float | None = None) -> None:
    current = now if now is not None else _time.monotonic()
    expired = [
        tid for tid, task in system_tasks.items()
        if task.get("status") in {"complete", "error"}
        and current - (task.get("_completed_at") or current) > 5.0
    ]
    for tid in expired:
        del system_tasks[tid]


def active_system_tasks_snapshot() -> list[dict]:
    """Active system tasks plus transient inpaint-step progress entries."""
    now = _time.monotonic()
    with system_task_lock:
        _prune_completed_system_tasks(now)
        result = [
            {k: v for k, v in task.items() if not k.startswith("_")}
            for task in system_tasks.values()
        ]
    for engine in list(_engines.values()):
        info = engine.inpaint_step_info
        if info.get("active") and info.get("total", 0) > 0:
            device = str(info.get("device", "cuda"))
            step = int(info.get("step", 0))
            total = max(1, int(info.get("total", 1)))
            result.append({
                "task_id": f"sys_inpaint_{device.replace(':', '_')}",
                "type": "inpaint",
                "status": "running",
                "phase": "rendering",
                "progress": round(step / total, 3),
                "message": f"Step {step}/{total}",
            })
    return result


def snapshot_runtime_metrics() -> dict[str, object]:
    """Collect a consistent backend runtime snapshot for the terminal dashboard."""
    from ..rtc import _shared_session_manager
    from ..rtc.router import _peer_connections
    from ..rtc.session import _peer_sessions

    with layer_task_lock:
        layer = [task.model_dump() for task in layer_tasks.values()]
    with motion_task_lock:
        motion = [task.model_dump() for task in motion_tasks.values()]

    connected_ids = set(_peer_connections.keys())
    sessions = []
    for session in list(_peer_sessions.values()):
        snapshot = session.telemetry_snapshot()
        snapshot["connection_kind"] = "webrtc" if snapshot.get("session_id") in connected_ids else "websocket"
        sessions.append(snapshot)

    return {
        "engine_mode": engine_mode(),
        "stream_build": {
            "phase": _shared_session_manager.build_phase,
            "progress": float(_shared_session_manager.build_progress),
            "message": _shared_session_manager.build_message,
        },
        "connections": {
            "sessions": sessions,
            "session_count": len(sessions),
            "webrtc_count": len(connected_ids),
        },
        "tasks": {
            "system": active_system_tasks_snapshot(),
            "layer": layer,
            "motion": motion,
        },
    }


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


def get_layer_task(task_id: str) -> LayerTaskProgress | None:
    with layer_task_lock:
        return layer_tasks.get(task_id)


def set_motion_task(progress: MotionTaskProgress) -> None:
    with motion_task_lock:
        motion_tasks[progress.task_id] = progress


def get_motion_task(task_id: str) -> MotionTaskProgress | None:
    with motion_task_lock:
        return motion_tasks.get(task_id)
