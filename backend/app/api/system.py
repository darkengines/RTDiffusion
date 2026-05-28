"""System endpoints: health, GPU enumeration, task progress."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from .state import (
    _request_device,
    active_system_tasks_snapshot,
    engine_mode,
)

logger = logging.getLogger("rtdiffusion.api")

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": engine_mode()}


@router.get("/system/tasks")
def get_system_tasks() -> list[dict]:
    """All active system operations. Completed/errored entries auto-expire after 5 s."""
    return active_system_tasks_snapshot()


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
