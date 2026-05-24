"""RTDiffusion FastAPI application entry point.

All route handlers live in the app/api/ sub-package.
This file creates the app, wires middleware, registers exception handlers, and
mounts the routers.
"""

import logging
import os
import sys

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException

from .api import assets, inpaint, layer, motion, sana, sources, system
from .api.state import on_stream_session_build
from .config import load_local_env
from .rtc import router as rtc_router, _shared_session_manager as _stream_session_manager

load_local_env()

# ── Logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("RTD_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# ── Windows: suppress harmless WinError 10054 in asyncio SSE cleanup ──
# When a browser closes an SSE connection, asyncio tries sock.shutdown(SHUT_RDWR)
# on an already-closed socket, raising ConnectionResetError. This is a known
# Windows asyncio bug (bpo-43978) — the error is noise, not a real failure.
if sys.platform == "win32":
    import asyncio.proactor_events as _pe

    _orig_cll = _pe._ProactorBasePipeTransport._call_connection_lost  # type: ignore[attr-defined]

    def _patched_cll(self: object, exc: object) -> None:
        try:
            _orig_cll(self, exc)  # type: ignore[arg-type]
        except ConnectionResetError:
            pass

    _pe._ProactorBasePipeTransport._call_connection_lost = _patched_cll  # type: ignore[attr-defined]

# ── Wire session-build status into the system task store ──────────

_stream_session_manager.set_status_callback(on_stream_session_build)

# ── App ────────────────────────────────────────────────────────────

app = FastAPI(title="RTDiffusion")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=os.getenv("RTD_CORS_ORIGIN_REGEX", r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0):\d+"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(HTTPException, system.log_http_exception)
app.add_exception_handler(RequestValidationError, system.log_request_validation_exception)
app.add_exception_handler(Exception, system.log_unhandled_exception)

# ── Routers ────────────────────────────────────────────────────────

app.include_router(rtc_router)
app.include_router(system.router)
app.include_router(assets.router)
app.include_router(sources.router)
app.include_router(sana.router)
app.include_router(layer.router)
app.include_router(motion.router)
app.include_router(inpaint.router)
