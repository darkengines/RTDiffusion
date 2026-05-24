"""Source image browser endpoints."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .state import IMAGE_EXTENSIONS

logger = logging.getLogger("rtdiffusion.api")

router = APIRouter()


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


@router.get("/sources")
def sources(root: str = Query(..., min_length=1), limit: int = Query(600, ge=1, le=2000)) -> dict[str, object]:
    root_path = _resolve_source(root)
    items = []
    for path in root_path.rglob("*"):
        if len(items) >= limit:
            break
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stat = path.stat()
            items.append({
                "name": path.name,
                "rel": path.relative_to(root_path).as_posix(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
    items.sort(key=lambda item: str(item["rel"]).lower())
    return {"root": str(root_path), "images": items}


@router.post("/sources/pick-root")
async def pick_source_root() -> dict[str, str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Folder picker is not available: {exc}") from exc

    window = None
    try:
        window = tk.Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        window.update()
        selected = filedialog.askdirectory(parent=window, title="Choose RTDiffusion source root")
    except Exception as exc:
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


@router.get("/sources/image")
def source_image(root: str = Query(..., min_length=1), rel: str = Query(..., min_length=1)) -> FileResponse:
    path = _resolve_source(root, rel)
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)
