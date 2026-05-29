"""Live terminal dashboard for backend runtime telemetry."""
from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from typing import Iterable

from .api.state import snapshot_runtime_metrics

logger = logging.getLogger("rtdiffusion.dashboard")

_FALSEY = {"0", "false", "no", "off"}
_dashboard_lock = threading.Lock()
_dashboard: "ConsoleDashboard | None" = None


def dashboard_enabled() -> bool:
    enabled = os.getenv("RTD_CONSOLE_DASHBOARD", "1").strip().lower() not in _FALSEY
    return enabled and bool(getattr(sys.stdout, "isatty", lambda: False)())


class ConsoleDashboard:
    def __init__(self, refresh_interval_s: float = 0.25) -> None:
        self._refresh_interval_s = refresh_interval_s
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._ansi_enabled = False
        self._last_line_count = 0

    def start(self) -> None:
        if self._started:
            return
        self._ansi_enabled = _enable_ansi(sys.stdout)
        if not self._ansi_enabled:
            logger.warning("Console dashboard disabled: terminal does not support in-place redraw")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="rtd-console-dashboard", daemon=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._restore_console()
        self._thread = None
        self._started = False

    def _run(self) -> None:
        last_frame = ""
        while not self._stop_event.is_set():
            try:
                frame = self._render(snapshot_runtime_metrics())
            except Exception:
                logger.exception("Failed to render backend console dashboard")
                frame = "RTDiffusion backend dashboard unavailable"
            if frame != last_frame:
                self._write_frame(frame)
                last_frame = frame
            self._stop_event.wait(self._refresh_interval_s)

    def _write_frame(self, frame: str) -> None:
        if not self._ansi_enabled:
            sys.stdout.write(frame.rstrip() + "\n")
            sys.stdout.flush()
            return
        # Hard-truncate every line to terminal width so a long row (e.g.
        # the v2 masks table with a long prompt) cannot wrap and overrun
        # the next visual row. Without this, the wrapped tail sits on the
        # next terminal row and the eraser misses it on the next refresh,
        # producing the "broken layout" the user observed.
        term_w = max(40, shutil.get_terminal_size((140, 40)).columns) - 1
        lines = frame.rstrip().splitlines() or [""]
        lines = [(line[:term_w] if len(line) > term_w else line) for line in lines]
        sys.stdout.write("\x1b[?25l\x1b[H")
        for index, line in enumerate(lines):
            if index:
                sys.stdout.write("\n")
            sys.stdout.write("\x1b[2K")
            sys.stdout.write(line)
        for _ in range(len(lines), self._last_line_count):
            sys.stdout.write("\n\x1b[2K")
        self._last_line_count = len(lines)
        sys.stdout.flush()

    def _restore_console(self) -> None:
        if self._ansi_enabled:
            sys.stdout.write("\x1b[?25h\n")
            sys.stdout.flush()

    def _render(self, snapshot: dict[str, object]) -> str:
        width = max(100, shutil.get_terminal_size((140, 40)).columns)
        engine = str(snapshot.get("engine_mode") or "lazy")
        connections = snapshot.get("connections") or {}
        tasks = snapshot.get("tasks") or {}
        build = snapshot.get("stream_build") or {}
        sessions = list(connections.get("sessions") or [])
        summary = (
            f"engine={engine} | sessions={connections.get('session_count', 0)} | "
            f"webrtc={connections.get('webrtc_count', 0)} | "
            f"build={build.get('phase', 'idle')} {_pct(build.get('progress', 0.0))}"
        )
        build_message = _truncate(str(build.get("message") or "idle"), max(24, width - 12))
        lines = [
            f"RTDiffusion backend dashboard  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            summary,
            build_message,
            "=" * width,
            "Connections",
            self._render_connections_table(sessions, width),
            "",
            "v2 masks (last frame seen by each session)",
            self._render_v2_masks_table(sessions, width),
            "",
            "Tasks",
            self._render_tasks_table(tasks, width),
        ]
        return "\n".join(lines)

    def _render_v2_masks_table(self, sessions: list[dict[str, object]], width: int) -> str:
        # Aggregate one row per (session, layer condition) plus per (session,
        # v2 prompt) so the user can see whether painted masks reach the
        # engine and through which path (legacy SDXL CLIP regional attention
        # vs the new v2 layer_regions). Empty table when no session has run
        # an inference yet.
        rows: list[list[str]] = []
        for session in sessions:
            tag = str(session.get("tag") or "-")[:8]
            diag_rows = session.get("v2_diag_rows") or []
            diag_meta = session.get("v2_diag_meta") or {}
            base_prompt = str(diag_meta.get("base_prompt") or "")[:18]
            base_cfg = float(diag_meta.get("base_cfg") or 0.0)
            base_denoise = float(diag_meta.get("base_denoise") or 0.0)
            if not isinstance(diag_rows, list) or not diag_rows:
                rows.append([tag, "(no v2 data)", "", "", "-", "", f"{base_cfg:.2f}", f"{base_denoise:.2f}", base_prompt])
                continue
            for row in diag_rows:
                kind = str(row.get("kind") or "?")
                layer = str(row.get("layer") or "?")[:18]
                region = str(row.get("region") or "")[:12]
                prompt = str(row.get("prompt") or "")[:24]
                nz_raw = row.get("mask_nz")
                if isinstance(nz_raw, int) and nz_raw >= 0:
                    nz = f"{nz_raw}"
                elif isinstance(nz_raw, int) and nz_raw == -1:
                    nz = "-"
                else:
                    nz = "?"
                status = str(row.get("pm_status") or "")
                cfg = row.get("cfg")
                den = row.get("denoise")
                cfg_str = f"{float(cfg):.2f}" if isinstance(cfg, (int, float)) else "-"
                den_str = f"{float(den):.2f}" if isinstance(den, (int, float)) else "-"
                rows.append([
                    tag, f"{kind}:{layer}", region, prompt, nz, status, cfg_str, den_str, base_prompt,
                ])
        if not rows:
            return "(no v2 mask data yet -- render at least one frame)"
        # Hard cap the prompt column so even a multi-paragraph prompt
        # can't blow the row past the terminal width. Status column gets
        # most of the leftover room because that's where the diagnostic
        # tag (``fe:no-canvas act=... chKeys=... live=...``) lives -- it
        # needs visible characters to be useful.
        prompt_col = min(24, max(16, width - 160))
        status_col = max(36, width - 100)
        columns = [
            ("session", 8),
            ("layer", 22),
            ("region", 12),
            ("prompt", prompt_col),
            ("mask_nz", 8),
            ("status", status_col),
            ("cfg", 5),
            ("denoise", 7),
            ("base", 18),
        ]
        return _format_table(columns, rows)

    def _render_connections_table(self, sessions: list[dict[str, object]], width: int) -> str:
        if not sessions:
            return "(no active connections)"
        rows = []
        for item in sessions:
            step_info = item.get("step_info") if isinstance(item.get("step_info"), dict) else {}
            active_step = bool(step_info.get("active")) and int(step_info.get("total") or 0) > 0
            if active_step:
                infer = f"{int(step_info.get('step') or 0)}/{int(step_info.get('total') or 0)} {_ms_to_s(item.get('active_for_ms'))}"
            elif item.get("inference_active"):
                infer = _ms_to_s(item.get("active_for_ms"))
            else:
                infer = "idle"
            state = str(item.get("pipeline_status") or "idle")
            build_phase = str(item.get("build_phase") or "idle")
            if build_phase not in {"idle", "ready"}:
                state = f"{state}/{build_phase}"
            rows.append([
                str(item.get("tag") or "-")[:8],
                f"{item.get('connection_kind', 'unknown')}/{item.get('output_transport', '-')}",
                state,
                f"{item.get('pipeline_renderer', 'idle')} | {item.get('pipeline_model') or 'lazy'}",
                f"{int(item.get('pending_renders') or 0)}",
                infer,
                f"{float(item.get('pipeline_fps') or 0.0):4.1f}",
                f"{float(item.get('last_queue_wait_ms') or 0.0):4.0f}",
                f"{float(item.get('last_latency_ms') or 0.0):4.0f}",
                f"{int(item.get('render_success_count') or 0)}/{int(item.get('render_stale_count') or 0)}/{int(item.get('render_skip_count') or 0)}/{int(item.get('render_error_count') or 0)}",
            ])
        columns = [
            ("session", 8),
            ("conn", 14),
            ("state", 18),
            ("pipeline", max(16, width - 86)),
            ("pend", 4),
            ("infer", 14),
            ("fps", 5),
            ("qms", 5),
            ("ims", 5),
            ("ok/st/sk/er", 12),
        ]
        return _format_table(columns, rows)

    def _render_tasks_table(self, tasks: dict[str, object], width: int) -> str:
        rows = []
        for task_type in ("system", "layer", "motion"):
            for item in list(tasks.get(task_type) or [])[:12]:
                message = str(item.get("error") or item.get("message") or "")
                rows.append([
                    task_type,
                    str(item.get("task_id") or "-")[:12],
                    str(item.get("status") or "-")[:10],
                    str(item.get("phase") or "-")[:14],
                    _pct(item.get("progress", 0.0)),
                    message,
                ])
        if not rows:
            return "(no active tasks)"
        columns = [
            ("type", 6),
            ("task", 12),
            ("status", 10),
            ("phase", 14),
            ("progress", 8),
            ("message", max(24, width - 58)),
        ]
        return _format_table(columns, rows)


def start_console_dashboard() -> None:
    global _dashboard
    if not dashboard_enabled():
        return
    with _dashboard_lock:
        if _dashboard is None:
            _dashboard = ConsoleDashboard()
        _dashboard.start()


def stop_console_dashboard() -> None:
    global _dashboard
    with _dashboard_lock:
        if _dashboard is None:
            return
        _dashboard.stop()
        _dashboard = None


def _format_table(columns: list[tuple[str, int]], rows: Iterable[list[str]]) -> str:
    header = "  ".join(_pad(name, width) for name, width in columns)
    divider = "  ".join("-" * width for _name, width in columns)
    body = []
    for row in rows:
        body.append("  ".join(_pad(_truncate(str(cell), width), width) for cell, (_name, width) in zip(row, columns)))
    return "\n".join([header, divider, *body])


def _pad(value: str, width: int) -> str:
    return value.ljust(width)[:width]


def _truncate(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _pct(value: object) -> str:
    try:
        return f"{max(0.0, min(1.0, float(value))) * 100:5.1f}%"
    except (TypeError, ValueError):
        return "  0.0%"


def _ms_to_s(value: object) -> str:
    try:
        return f"{max(0.0, float(value)) / 1000.0:4.1f}s"
    except (TypeError, ValueError):
        return "0.0s"


def _enable_ansi(stream: object) -> bool:
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        return kernel32.SetConsoleMode(handle, mode.value | 0x0004) != 0
    except Exception:
        return False