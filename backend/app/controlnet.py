"""
ControlNet conditioning for layer-based inpainting.

Architecture
────────────
Each LayerCondition with controlnet_model set provides:
  • controlnet_model  — shortname: canny | depth | openpose | hed | mlsd |
                        scribble | normal | lineart | seg
  • controlnet_image  — optional pre-processed data URL (skips preprocessing)
  • controlnet_preprocessor — auto-preprocess the layer image if True
  • controlnet_scale  — conditioning scale (0–2)
  • controlnet_start/end_at — fraction of denoising steps to apply CN

The module caches ControlNetModel objects keyed by (model_id, device) and
builds a ControlNet-aware pipeline on demand alongside the existing pipeline.

SDXL renderers resolve SDXL ControlNet models by default. SD1.5 defaults are
kept only for explicitly SD1.5 pipelines.
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Any

from PIL import Image

from .image_io import decode_data_url

logger = logging.getLogger("rtdiffusion.controlnet")

_SD15_DEFAULTS: dict[str, str] = {
    "canny":    "lllyasviel/control_v11p_sd15_canny",
    "depth":    "lllyasviel/control_v11f1p_sd15_depth",
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "hed":      "lllyasviel/control_v11p_sd15_softedge",
    "mlsd":     "lllyasviel/control_v11p_sd15_mlsd",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "normal":   "lllyasviel/control_v11p_sd15_normalbae",
    "lineart":  "lllyasviel/control_v11p_sd15_lineart",
    "seg":      "lllyasviel/control_v11p_sd15_seg",
}

_SDXL_DEFAULTS: dict[str, str] = {
    # Official diffusers/Stability releases — highest quality for SDXL
    "canny":    "diffusers/controlnet-canny-sdxl-1.0",
    "depth":    "diffusers/controlnet-depth-sdxl-1.0",
    # Community SDXL ControlNets (well-tested, widely used)
    "openpose": "xinsir/controlnet-openpose-sdxl-1.0",
    "hed":      "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
    "normal":   "SargeZT/controlnet-sd-xl-1.0-depth-16bit-zoe",  # Zoe depth ≈ normal-like
    "lineart":  "xinsir/controlnet-scribble-sdxl-1.0",
    "scribble": "xinsir/controlnet-scribble-sdxl-1.0",
    "seg":      "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
    "mlsd":     "diffusers/controlnet-canny-sdxl-1.0",  # MLSD is line-like; canny is closest SDXL equivalent
}

_cn_cache: dict[str, Any] = {}  # key = model_id → ControlNetModel
_cn_cache_lock = threading.Lock()

_aux_cache: dict[str, Any] = {}  # key = preprocessor name → detector instance

# ComfyUI controlnet models directory
_COMFYUI_CN_PATH = r"C:\Users\root\Documents\comfy\ComfyUI\models\controlnet"

# Keywords used to match short model names to ComfyUI filenames / directory names
_COMFYUI_NAME_HINTS: dict[str, list[str]] = {
    "canny":    ["canny"],
    "depth":    ["depth", "zoe"],
    "openpose": ["openpose", "pose"],
    "hed":      ["hed", "softedge", "dexined"],
    "mlsd":     ["mlsd"],
    "scribble": ["scribble"],
    "normal":   ["normal", "normalbae"],
    "lineart":  ["lineart"],
    "seg":      ["seg"],
}

# Model file names that indicate a directory contains a diffusers-format ControlNet
_DIFFUSERS_CN_FILES = {"diffusion_pytorch_model.safetensors", "diffusion_pytorch_model_V2.safetensors",
                       "diffusion_pytorch_model.bin"}


def _find_local_controlnet(short_name: str, prefer_sdxl: bool = False) -> str | None:
    """
    Search the ComfyUI controlnet directory for a matching model.

    Checks in this order:
      1. SDXL/ subdirectory (if prefer_sdxl) — subdirs first (diffusers format), then single files
      2. Root controlnet directory — same order
    Returns the directory path (use with from_pretrained) or file path (use with from_single_file).
    """
    import pathlib
    cn_dir = pathlib.Path(_COMFYUI_CN_PATH)
    if not cn_dir.is_dir():
        return None
    hints = _COMFYUI_NAME_HINTS.get(short_name, [short_name])

    search_dirs: list[pathlib.Path] = []
    if prefer_sdxl:
        sdxl_dir = cn_dir / "SDXL"
        if sdxl_dir.is_dir():
            search_dirs.append(sdxl_dir)
        # Do not fall back to root ComfyUI files for SDXL pipelines: those are
        # commonly SD1.5 weights and will either fail to load or silently do nothing.
        if not search_dirs:
            return None
    else:
        search_dirs.append(cn_dir)

    for search_dir in search_dirs:
        # 1) diffusers-format subdirectories (contain diffusion_pytorch_model*.safetensors)
        try:
            subdirs = [p for p in search_dir.iterdir() if p.is_dir()]
        except OSError:
            subdirs = []
        for subdir in sorted(subdirs):
            lname = subdir.name.lower()
            if not any(h in lname for h in hints):
                continue
            if any((subdir / f).exists() for f in _DIFFUSERS_CN_FILES):
                logger.info("ControlNet: using local dir %s for '%s'", subdir, short_name)
                return str(subdir)
        # 2) single-file models
        for suffix in (".safetensors", ".pth", ".pt", ".bin"):
            try:
                files = [p for p in search_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix]
            except OSError:
                files = []
            for f in sorted(files):
                if any(h in f.stem.lower() for h in hints):
                    logger.info("ControlNet: using local file %s for '%s'", f, short_name)
                    return str(f)

    return None


def list_local_controlnet_models() -> list[dict]:
    """
    Return a catalog of all local ControlNet models found in the ComfyUI directory.
    Used by the /controlnet/models API endpoint to populate the frontend selector.
    """
    import pathlib
    cn_dir = pathlib.Path(_COMFYUI_CN_PATH)
    models: list[dict] = []
    if not cn_dir.is_dir():
        return models

    def _infer_type(name: str) -> str:
        nl = name.lower()
        for t, hints in _COMFYUI_NAME_HINTS.items():
            if any(h in nl for h in hints):
                return t
        return "unknown"

    def _infer_family(path: pathlib.Path) -> str:
        parts = [p.lower() for p in path.parts]
        if "sdxl" in parts or any("xl" in p for p in parts):
            return "sdxl"
        return "sd15"

    def _add(path: pathlib.Path, label_prefix: str = "") -> None:
        name = path.name
        display = f"{label_prefix}{name}"
        models.append({
            "name": display,
            "path": str(path),
            "type": _infer_type(name),
            "family": _infer_family(path),
        })

    # Walk top-level and one level deep
    try:
        for entry in sorted(cn_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in (".safetensors", ".pth", ".pt", ".bin"):
                _add(entry)
            elif entry.is_dir() and entry.name not in ("instantid",):
                # Check if it's a diffusers-format model dir
                if any((entry / f).exists() for f in _DIFFUSERS_CN_FILES):
                    _add(entry, f"{entry.parent.name}/")
                else:
                    # Subdirectory — recurse one level
                    try:
                        for sub in sorted(entry.iterdir()):
                            if sub.is_file() and sub.suffix.lower() in (".safetensors", ".pth", ".pt", ".bin"):
                                _add(sub, f"{entry.name}/")
                            elif sub.is_dir() and any((sub / f).exists() for f in _DIFFUSERS_CN_FILES):
                                _add(sub, f"{entry.name}/")
                    except OSError:
                        pass
    except OSError:
        pass
    return models


def resolve_model_id(short_name: str, model_path: str | None, is_sdxl: bool) -> str:
    if model_path:
        return model_path
    # Check local ComfyUI directory first — prefer SDXL subdir when using SDXL pipeline
    local = _find_local_controlnet(short_name, prefer_sdxl=is_sdxl)
    if local:
        return local
    defaults = _SDXL_DEFAULTS if is_sdxl else _SD15_DEFAULTS
    return defaults.get(short_name, short_name if is_sdxl else _SD15_DEFAULTS.get(short_name, short_name))


def load_controlnet(model_id: str, device: str, dtype: Any) -> Any:
    import os
    cache_key = f"{model_id}:{device}"
    with _cn_cache_lock:
        if cache_key in _cn_cache:
            return _cn_cache[cache_key]
    logger.info("ControlNet: loading %s on %s", model_id, device)
    from diffusers import ControlNetModel

    if os.path.isfile(model_id):
        # Single safetensors/pth file — from_single_file infers architecture from weights
        cn = ControlNetModel.from_single_file(model_id, torch_dtype=dtype).to(device)
    elif os.path.isdir(model_id):
        # Local directory — use from_pretrained only when config.json is present.
        # ComfyUI stores weights-only (no config.json), so fall back to from_single_file.
        if os.path.isfile(os.path.join(model_id, "config.json")):
            cn = ControlNetModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
        else:
            weights = None
            for fname in ("diffusion_pytorch_model_V2.safetensors",
                          "diffusion_pytorch_model.safetensors",
                          "diffusion_pytorch_model.bin"):
                candidate = os.path.join(model_id, fname)
                if os.path.isfile(candidate):
                    weights = candidate
                    break
            if weights is None:
                raise FileNotFoundError(f"No ControlNet weights found in {model_id}")
            logger.info("ControlNet: no config.json, loading weights-only from %s", weights)
            cn = ControlNetModel.from_single_file(weights, torch_dtype=dtype).to(device)
    else:
        # HuggingFace repo ID — downloads automatically
        cn = ControlNetModel.from_pretrained(model_id, torch_dtype=dtype).to(device)

    with _cn_cache_lock:
        _cn_cache[cache_key] = cn
    logger.info("ControlNet: loaded %s", model_id)
    return cn


def build_controlnet_inpaint_pipe(
    base_pipe: Any,
    controlnet: Any,
    is_sdxl: bool,
) -> Any:
    """
    Build a ControlNet inpaint pipeline by wrapping the existing pipeline's
    components. Returns a new pipeline instance.
    """
    if is_sdxl:
        from diffusers import StableDiffusionXLControlNetInpaintPipeline
        pipe = StableDiffusionXLControlNetInpaintPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            text_encoder_2=base_pipe.text_encoder_2,
            tokenizer=base_pipe.tokenizer,
            tokenizer_2=base_pipe.tokenizer_2,
            unet=base_pipe.unet,
            controlnet=controlnet,
            scheduler=base_pipe.scheduler,
        )
    else:
        from diffusers import StableDiffusionControlNetInpaintPipeline
        pipe = StableDiffusionControlNetInpaintPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            unet=base_pipe.unet,
            controlnet=controlnet,
            scheduler=base_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── Preprocessors ─────────────────────────────────────────────────────────────

# Depth model cache: loaded once, reused across frames
_depth_pipe_cache: dict[str, Any] = {}


def preprocess_canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    import numpy as np
    try:
        import cv2
        gray = np.array(image.convert("L"))
        edges = cv2.Canny(gray, low, high)
        edges_rgb = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges_rgb, "RGB")
    except ImportError:
        logger.warning("ControlNet canny: opencv not available, using PIL edge fallback")
        from PIL import ImageFilter
        import numpy as _np
        edge = image.convert("L").filter(ImageFilter.FIND_EDGES)
        return Image.fromarray(
            _np.stack([_np.array(edge)] * 3, axis=-1), "RGB"
        )


# ── Native fallbacks (no controlnet_aux required) ─────────────────────────────

def _depth_gradient_fallback(image: Image.Image) -> Image.Image:
    """
    Pseudo-depth via Sobel gradient magnitude — last resort when all depth models fail.

    Midas-style convention: near=bright, far=dark. High gradient areas (edges, foreground
    objects) are assumed closer; smooth regions (backgrounds) darker. Produces a noisy but
    structurally plausible depth-like map, better than plain luminance for CN conditioning.
    """
    import numpy as np
    try:
        import cv2
        gray = np.array(image.convert("L")).astype(np.float32)
        # Blur heavily to approximate scene depth from gradient magnitude
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
        sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=5)
        sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        # Smooth the gradient map to avoid harsh edge artifacts
        depth = cv2.GaussianBlur(mag, (0, 0), sigmaX=15)
        lo, hi = depth.min(), depth.max()
        norm = ((depth - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
        return Image.fromarray(np.stack([norm] * 3, axis=-1), "RGB")
    except ImportError:
        from PIL import ImageFilter
        # No cv2 — fall back to simple luminance (worst case)
        arr = np.array(image.convert("L").filter(ImageFilter.GaussianBlur(radius=12)), dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        norm = ((arr - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
        return Image.fromarray(np.stack([norm] * 3, axis=-1), "RGB")


_DEPTH_MODELS = [
    "depth-anything/Depth-Anything-V2-Large-hf",   # V2 Large — best quality, smooth gradients
    "depth-anything/Depth-Anything-V2-Small-hf",   # V2 Small — faster fallback
    "LiheYoung/depth-anything-small-hf",            # V1 fallback
    "Intel/dpt-hybrid-midas",                       # Last resort
]


def _native_depth(image: Image.Image) -> Image.Image:
    """Depth estimation via transformers (cascade of models, sentinel-cached on failure)."""
    import numpy as np
    import torch
    for model_id in _DEPTH_MODELS:
        if _depth_pipe_cache.get(model_id) is False:
            continue  # previously failed — skip without retrying
        try:
            if model_id not in _depth_pipe_cache:
                from transformers import pipeline as _hf_pipe
                logger.info("ControlNet depth: loading %s", model_id)
                try:
                    use_device = 0 if torch.cuda.is_available() else -1
                    pipe = _hf_pipe("depth-estimation", model=model_id, device=use_device)
                except (RuntimeError, Exception) as gpu_err:
                    err_str = str(gpu_err).lower()
                    if torch.cuda.is_available() and any(k in err_str for k in ("cuda", "out of memory", "memory")):
                        logger.warning("ControlNet depth: GPU OOM for %s, retrying on CPU", model_id)
                        torch.cuda.empty_cache()
                        pipe = _hf_pipe("depth-estimation", model=model_id, device=-1)
                    else:
                        raise
                _depth_pipe_cache[model_id] = pipe
                logger.info("ControlNet depth: %s ready", model_id)
            pipe = _depth_pipe_cache[model_id]
            result = pipe(image)
            depth_img = result["depth"]
            # PIL mode "I" = 32-bit int, "F" = 32-bit float — both need float conversion.
            # Direct .convert("L") on mode "I"/"F" only keeps the low byte; use numpy instead.
            if depth_img.mode in ("I", "F"):
                arr = np.array(depth_img, dtype=np.float32)
            else:
                arr = np.array(depth_img.convert("L"), dtype=np.float32)
            # Normalize to full 0–255 so CN sees full contrast range.
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo) * 255.0
            else:
                arr = arr * 0  # flat / uniform depth — leave as black
            arr8 = arr.clip(0, 255).astype(np.uint8)
            return Image.fromarray(np.stack([arr8] * 3, axis=-1), "RGB")
        except Exception as exc:
            logger.warning("ControlNet depth %s failed: %s — trying next", model_id, exc)
            _depth_pipe_cache[model_id] = False  # sentinel: don't retry
    logger.warning("ControlNet depth: all models failed — using gradient fallback")
    return _depth_gradient_fallback(image)


def _native_softedge(image: Image.Image) -> Image.Image:
    """Soft-edge / HED approximation via Sobel gradient magnitude."""
    import numpy as np
    try:
        import cv2
        gray = np.array(image.convert("L")).astype(np.float32)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        mag = np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
        mag = cv2.GaussianBlur(mag, (5, 5), 1.5)
        return Image.fromarray(np.stack([mag] * 3, axis=-1), "RGB")
    except ImportError:
        from PIL import ImageFilter
        e = image.convert("L").filter(ImageFilter.SMOOTH).filter(ImageFilter.FIND_EDGES)
        arr = np.array(e)
        return Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB")


def _native_lineart(image: Image.Image) -> Image.Image:
    """Lineart: white-on-black clean edges via adaptive threshold / inverted Canny."""
    import numpy as np
    try:
        import cv2
        gray = np.array(image.convert("L"))
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 30, 100)
        inverted = (255 - edges)  # dark lines on white (lineart convention)
        return Image.fromarray(np.stack([inverted] * 3, axis=-1), "RGB")
    except ImportError:
        from PIL import ImageFilter
        e = image.convert("L").filter(ImageFilter.FIND_EDGES)
        arr = 255 - np.array(e)
        return Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB")


def _native_mlsd(image: Image.Image) -> Image.Image:
    """MLSD approximation: probabilistic Hough lines via cv2, softedge as fallback."""
    import numpy as np
    try:
        import cv2
        gray = np.array(image.convert("L"))
        blurred = cv2.GaussianBlur(gray, (3, 3), 1)
        edges = cv2.Canny(blurred, 50, 150)
        canvas = np.zeros_like(edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=60, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for seg in lines:
                x1, y1, x2, y2 = seg[0]
                cv2.line(canvas, (x1, y1), (x2, y2), 255, 2)
        return Image.fromarray(np.stack([canvas] * 3, axis=-1), "RGB")
    except ImportError:
        return _native_softedge(image)


# BlazePose 33-landmark connections (mediapipe 0.10+, hard-coded to avoid mp.solutions dependency)
_BLAZEPOSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

_pose_landmarker_cache: dict[str, Any] = {}  # key = "model" → detector or False


def _native_openpose(image: Image.Image) -> Image.Image:
    """Pose skeleton using mediapipe PoseLandmarker (0.10+ API), with softedge fallback."""
    import numpy as np
    import os
    from pathlib import Path as _Path

    result = _try_mediapipe_landmarker(image)
    if result is not None:
        return result

    # Legacy mediapipe < 0.10 (mp.solutions.pose)
    try:
        import mediapipe as mp  # type: ignore[import]
        if not (hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")):
            raise ImportError("mediapipe.solutions.pose not available")
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        img_np = np.array(image.convert("RGB"))
        canvas = np.zeros_like(img_np)
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            res = pose.process(img_np)
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    canvas, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=4),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(128, 128, 192), thickness=3),
                )
                return Image.fromarray(canvas, "RGB")
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("ControlNet openpose legacy mediapipe failed: %s", exc)

    logger.info(
        "ControlNet openpose: no pose detector available "
        "(install controlnet-aux or mediapipe) - using softedge fallback"
    )
    return _native_softedge(image)


def _try_mediapipe_landmarker(image: Image.Image) -> "Image.Image | None":
    """
    Try mediapipe 0.10+ PoseLandmarker. Downloads pose_landmarker_heavy.task once.
    Returns None if unavailable or detection fails.
    """
    import numpy as np

    cache_key = "pose_landmarker"
    if _pose_landmarker_cache.get(cache_key) is False:
        return None  # previously failed — don't retry

    try:
        import mediapipe as mp  # type: ignore[import]
        from mediapipe.tasks.python import vision as _mp_vision  # type: ignore[import]
        from mediapipe.tasks.python import BaseOptions as _mp_base_opts  # type: ignore[import]
    except (ImportError, Exception):
        _pose_landmarker_cache[cache_key] = False
        return None

    try:
        import os as _os
        from pathlib import Path as _Path
        model_dir = _Path(_os.path.expanduser("~/.cache/rtdiffusion/mediapipe"))
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "pose_landmarker_heavy.task"

        if not model_path.exists():
            import urllib.request
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                "pose_landmarker_heavy.task"
            )
            logger.info("ControlNet openpose: downloading pose_landmarker_heavy.task …")
            urllib.request.urlretrieve(url, str(model_path))
            logger.info("ControlNet openpose: pose_landmarker_heavy.task downloaded")

        base_options = _mp_base_opts(model_asset_path=str(model_path))
        options = _mp_vision.PoseLandmarkerOptions(base_options=base_options)
        landmarker = _mp_vision.PoseLandmarker.create_from_options(options)
        _pose_landmarker_cache[cache_key] = landmarker
    except Exception as exc:
        logger.warning("ControlNet openpose: PoseLandmarker setup failed (%s) — trying legacy", exc)
        _pose_landmarker_cache[cache_key] = False
        return None

    try:
        import cv2  # noqa: F401 — used for drawing
        landmarker = _pose_landmarker_cache[cache_key]
        img_rgb = np.array(image.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection = landmarker.detect(mp_image)

        if not detection.pose_landmarks:
            return None  # fall through to softedge

        import cv2 as _cv
        canvas = np.zeros_like(img_rgb)
        h, w = img_rgb.shape[:2]
        for body_landmarks in detection.pose_landmarks:
            pts = [
                (int(lm.x * w), int(lm.y * h))
                for lm in body_landmarks
            ]
            for a, b in _BLAZEPOSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    _cv.line(canvas, pts[a], pts[b], (128, 128, 220), 3)
            for pt in pts:
                _cv.circle(canvas, pt, 4, (255, 255, 255), -1)

        return Image.fromarray(canvas, "RGB")
    except Exception as exc:
        logger.warning("ControlNet openpose: PoseLandmarker inference failed: %s", exc)
        return None


def _native_normal(image: Image.Image) -> Image.Image:
    """Surface normal approximation from Sobel image gradients (rough but usable)."""
    import numpy as np
    try:
        import cv2
        gray = np.array(image.convert("L")).astype(np.float32)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
        dz = np.ones_like(dx) * 64.0
        length = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + 1e-6
        nx = dx / length
        ny = dy / length
        nz = dz / length
        rgb = np.stack([
            ((nx + 1) * 0.5 * 255).astype(np.uint8),
            ((ny + 1) * 0.5 * 255).astype(np.uint8),
            ((nz + 1) * 0.5 * 255).astype(np.uint8),
        ], axis=-1)
        return Image.fromarray(rgb, "RGB")
    except ImportError:
        return _native_softedge(image)


def _native_seg(image: Image.Image) -> Image.Image:
    """Semantic segmentation via transformers (downloads on first use, ~170 MB)."""
    import numpy as np
    seg_model = "nvidia/segformer-b0-finetuned-ade-512-512"
    if _depth_pipe_cache.get(seg_model) is False:
        return _native_softedge(image)  # previously failed — don't retry
    try:
        if seg_model not in _depth_pipe_cache:
            import torch
            from transformers import pipeline as _hf_pipe
            logger.info("ControlNet seg: loading %s", seg_model)
            pipe = _hf_pipe(
                "image-segmentation", model=seg_model,
                device=0 if torch.cuda.is_available() else -1,
            )
            _depth_pipe_cache[seg_model] = pipe
            logger.info("ControlNet seg: model ready")
        pipe = _depth_pipe_cache[seg_model]
        segments = pipe(image)
        h, w = image.height, image.width
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for seg in segments:
            mask = np.array(seg["mask"].resize((w, h), Image.NEAREST))
            label_hash = hash(seg.get("label", "")) & 0xFFFFFF
            colour = np.array([(label_hash >> 16) & 0xFF,
                               (label_hash >> 8) & 0xFF,
                               label_hash & 0xFF], dtype=np.uint8)
            canvas[mask > 128] = colour
        return Image.fromarray(canvas, "RGB")
    except Exception as exc:
        logger.warning("ControlNet seg native fallback failed: %s — caching failure", exc)
        _depth_pipe_cache[seg_model] = False  # sentinel: don't retry
        return _native_softedge(image)


_NATIVE_FALLBACKS = {
    "depth":    _native_depth,
    "hed":      _native_softedge,
    "scribble": _native_softedge,
    "softedge": _native_softedge,
    "lineart":  _native_lineart,
    "mlsd":     _native_mlsd,
    "openpose": _native_openpose,
    "normal":   _native_normal,
    "seg":      _native_seg,
}


def _get_aux_detector(name: str) -> Any:
    if name in _aux_cache:
        return _aux_cache[name]
    try:
        import controlnet_aux  # noqa: F401 — presence check
    except ImportError:
        # Logged once; native fallbacks will handle each preprocessor type
        logger.info(
            "controlnet-aux not installed — using native fallbacks for preprocessors. "
            "For best quality: pip install controlnet-aux>=0.0.7"
        )
        # Mark all names as None so we skip the aux path permanently this session
        _aux_cache[name] = None
        return None
    try:
        if name == "openpose":
            # DWpose is newer and higher quality than the original OpenPose annotator.
            # It uses YOLOX + DW-Pose keypoints for more accurate skeleton detection.
            try:
                from controlnet_aux import DWposeDetector
                det = DWposeDetector()
                logger.info("ControlNet: using DWposeDetector for openpose (high quality)")
            except (ImportError, AttributeError, Exception) as _dw_err:
                logger.info("ControlNet: DWposeDetector unavailable (%s), falling back to OpenposeDetector", _dw_err)
                from controlnet_aux import OpenposeDetector
                det = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        elif name == "depth":
            # Prefer Depth-Anything V2 Large (smoother, higher quality) over legacy Midas.
            try:
                from controlnet_aux import DepthAnythingDetector
                det = DepthAnythingDetector.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
                logger.info("ControlNet: using DepthAnythingDetector V2 Large for depth (high quality)")
            except (ImportError, AttributeError, Exception) as _da_err:
                logger.info("ControlNet: DepthAnythingDetector V2 Large unavailable (%s), trying ZoeDetector", _da_err)
                try:
                    from controlnet_aux import ZoeDetector
                    det = ZoeDetector.from_pretrained("lllyasviel/Annotators")
                    logger.info("ControlNet: using ZoeDetector for depth")
                except (ImportError, AttributeError, Exception):
                    from controlnet_aux import MidasDetector
                    det = MidasDetector.from_pretrained("lllyasviel/Annotators")
        elif name == "hed":
            from controlnet_aux import HEDdetector
            det = HEDdetector.from_pretrained("lllyasviel/Annotators")
        elif name == "mlsd":
            from controlnet_aux import MLSDdetector
            det = MLSDdetector.from_pretrained("lllyasviel/Annotators")
        elif name == "lineart":
            from controlnet_aux import LineartDetector
            det = LineartDetector.from_pretrained("lllyasviel/Annotators")
        elif name == "normal":
            from controlnet_aux import NormalBaeDetector
            det = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        elif name == "seg":
            from controlnet_aux import SamDetector
            det = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
        elif name == "scribble":
            from controlnet_aux import HEDdetector
            det = HEDdetector.from_pretrained("lllyasviel/Annotators")
        else:
            logger.warning("ControlNet: no preprocessor known for model '%s'", name)
            return None
        logger.info("ControlNet: preprocessor '%s' loaded", name)
        _aux_cache[name] = det
        return det
    except Exception as exc:
        logger.error("ControlNet preprocessor '%s' failed to load: %s", name, exc)
        _aux_cache[name] = None
        return None


def preprocess_image(
    image: Image.Image,
    model: str,
    params: Any,  # ControlNetPreprocessorParams
) -> Image.Image:
    """Apply the appropriate preprocessor for the given CN model."""
    image = image.convert("RGB")

    if model == "canny":
        return preprocess_canny(image, params.canny_low, params.canny_high)

    # Try controlnet_aux first (highest quality)
    det = _get_aux_detector(model)
    if det is not None:
        try:
            det_res = max(image.width, image.height)
            if model == "openpose":
                # Pass detect_resolution so the detector works at native image resolution
                # rather than defaulting to a lower fixed resolution (better skeleton quality).
                return det(
                    image,
                    include_hand=params.openpose_hands,
                    include_face=params.openpose_face,
                    detect_resolution=det_res,
                    image_resolution=det_res,
                )
            if model == "depth":
                return det(image, detect_resolution=det_res, image_resolution=det_res)
            if model == "mlsd":
                return det(image, thr_v=params.mlsd_thr_v, thr_d=params.mlsd_thr_d,
                           detect_resolution=det_res, image_resolution=det_res)
            if model == "lineart":
                return det(image, coarse=params.lineart_coarse,
                           detect_resolution=det_res, image_resolution=det_res)
            return det(image, detect_resolution=det_res, image_resolution=det_res)
        except TypeError:
            # Some older detector versions don't accept detect_resolution / image_resolution.
            try:
                if model == "openpose":
                    return det(image, include_hand=params.openpose_hands, include_face=params.openpose_face)
                if model == "mlsd":
                    return det(image, thr_v=params.mlsd_thr_v, thr_d=params.mlsd_thr_d)
                if model == "lineart":
                    return det(image, coarse=params.lineart_coarse)
                return det(image)
            except Exception as exc:
                logger.warning("ControlNet preprocessing %s failed: %s", model, exc)
        except Exception as exc:
            logger.warning("ControlNet preprocessing %s failed: %s", model, exc)

    # Native fallback — uses cv2 / transformers / mediapipe (no controlnet_aux required)
    native = _NATIVE_FALLBACKS.get(model)
    if native is not None:
        logger.debug("ControlNet: using native fallback for '%s'", model)
        return native(image)

    logger.warning("ControlNet: no preprocessor for '%s' — using canny", model)
    return preprocess_canny(image)


def resolve_control_image(
    condition: Any,  # LayerCondition
    layer_image: Image.Image,
    width: int,
    height: int,
) -> Image.Image | None:
    """
    Resolve the control image for a LayerCondition:
    1. If controlnet_image is set (data URL), decode and use it directly.
    2. Else if controlnet_use_layer_frame, use condition.image (the layer's own live frame).
    3. Else if controlnet_preprocessor, preprocess layer_image.
    4. Else use layer_image as-is (already in control format).
    """
    if not condition.controlnet_model:
        return None

    if condition.controlnet_image:
        try:
            img = decode_data_url(condition.controlnet_image).convert("RGB")
            return img.resize((width, height), Image.LANCZOS)
        except Exception as exc:
            logger.warning("ControlNet: failed to decode controlnet_image: %s", exc)

    # controlnet_use_layer_frame: use the layer's own image field (live frame embedded
    # by the frontend) rather than the composed output passed as layer_image.
    src_image = layer_image
    if getattr(condition, "controlnet_use_layer_frame", False):
        try:
            src_image = decode_data_url(condition.image).convert("RGB")
        except Exception:
            pass  # fallback to layer_image if decode fails

    img = src_image.convert("RGB").resize((width, height), Image.LANCZOS)
    if condition.controlnet_preprocessor:
        img = preprocess_image(img, condition.controlnet_model, condition.controlnet_preprocessor_params)
    return img
