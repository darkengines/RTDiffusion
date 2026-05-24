"""System endpoints: health, GPU enumeration, task progress."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from .state import (
    _engines,
    _request_device,
    engine_mode,
    system_task_lock,
    system_tasks,
)

logger = logging.getLogger("rtdiffusion.api")

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": engine_mode()}


@router.get("/system/tasks")
def get_system_tasks() -> list[dict]:
    """All active system operations. Completed/errored entries auto-expire after 5 s."""
    now = time.monotonic()
    with system_task_lock:
        expired = [
            tid for tid, t in system_tasks.items()
            if t.get("status") in {"complete", "error"}
            and now - (t.get("_completed_at") or now) > 5.0
        ]
        for tid in expired:
            del system_tasks[tid]
        result = [
            {k: v for k, v in t.items() if not k.startswith("_")}
            for t in system_tasks.values()
        ]
    for engine in list(_engines.values()):
        info = engine.inpaint_step_info
        if info.get("active") and info.get("total", 0) > 0:
            device: str = str(info.get("device", "cuda"))
            task_id = f"sys_inpaint_{device.replace(':', '_')}"
            step = int(info.get("step", 0))
            total = int(info.get("total", 1))
            result.append({
                "task_id": task_id,
                "type": "inpaint",
                "status": "running",
                "phase": "rendering",
                "progress": round(step / total, 3),
                "message": f"Step {step}/{total}",
            })
    return result


@router.get("/system/gpus")
def system_gpus() -> dict[str, object]:
    devices: list[dict[str, object]] = [{"id": "cpu", "name": "CPU", "kind": "cpu", "available": True}]
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append({
                    "id": f"cuda:{index}",
                    "name": props.name,
                    "kind": "cuda",
                    "available": True,
                    "memory_total": props.total_memory,
                })
    except Exception as exc:
        logger.warning("Failed to enumerate CUDA devices: %s", exc)
    return {"default": _request_device(None), "devices": devices}


# ── Exception handlers (registered on the app in main.py) ─────────

async def log_http_exception(request: Request, exc: HTTPException):
    logger.warning("HTTP %s %s -> %s: %s", request.method, request.url, exc.status_code, exc.detail)
    return await http_exception_handler(request, exc)


async def log_request_validation_exception(request: Request, exc: RequestValidationError):
    logger.exception("Request validation failed for %s %s", request.method, request.url)
    return await request_validation_exception_handler(request, exc)


async def log_unhandled_exception(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    raise exc
