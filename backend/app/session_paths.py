"""Optional per-run filesystem layout for realtime sessions."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def session_root(path: str | None) -> Path | None:
    value = (path or "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def ensure_session_layout(path: str | None) -> dict[str, Path] | None:
    root = session_root(path)
    if root is None:
        return None
    layout = {
        "root": root,
        "cache": root / "cache",
        "pipeline_cache": root / "cache" / "pipelines",
        "inductor_cache": root / "cache" / "torchinductor",
        "trt_cache": root / "cache" / "trt-engines",
        "output": root / "output",
        "debug": root / "debug",
        "resources": root / "resources",
    }
    for directory in layout.values():
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def configure_torch_compile_cache(path: str | None, *, model_id: str, device: str, backend: str) -> Path | None:
    layout = ensure_session_layout(path)
    if layout is None:
        return None
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(layout["inductor_cache"])
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    digest = hashlib.sha256(repr((backend, model_id, device)).encode("utf-8")).hexdigest()[:16]
    manifest_path = layout["pipeline_cache"] / f"{backend}_{digest}.json"
    manifest: dict[str, Any] = {
        "backend": backend,
        "model_id": model_id,
        "device": device,
        "torchinductor_cache_dir": str(layout["inductor_cache"]),
        "binary_cache": "torch.compile/inductor generated code and binaries",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def session_output_path(path: str | None, name: str, suffix: str) -> Path | None:
    layout = ensure_session_layout(path)
    if layout is None:
        return None
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._") or "frame"
    return layout["output"] / f"{safe_name}{suffix}"


def session_debug_path(path: str | None, name: str, suffix: str) -> Path | None:
    layout = ensure_session_layout(path)
    if layout is None:
        return None
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._") or "debug"
    return layout["debug"] / f"{safe_name}{suffix}"