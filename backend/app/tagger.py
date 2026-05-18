"""
WD-style image tagger for automatic prompt generation.

Supports SmilingWolf/wd-eva02-large-tagger-v3 (ONNX).
Model files are downloaded lazily from HuggingFace Hub on first use.

Usage:
    from .tagger import tag_image
    tags = tag_image(pil_image, threshold=0.35)  # returns "tag1, tag2, ..."
"""

from __future__ import annotations

import csv
import io
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("rtdiffusion.tagger")

_HF_REPOS: dict[str, str] = {
    "wd-eva02-large-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-swinv2-v3":      "SmilingWolf/wd-swinv2-tagger-v3",
    "wd-convnext-v3":    "SmilingWolf/wd-convnext-tagger-v3",
}

_INPUT_SIZE = 448  # all WD v3 models use 448×448

_tagger_cache: dict[str, tuple[Any, list[str], list[int]]] = {}  # model_name → (session, tag_names, tag_categories)
_cache_lock = threading.Lock()

# Categories to INCLUDE in prompt output (WD tagger CSV category column):
#   0 = General descriptive tags  ← keep
#   4 = Character names           ← keep
#   5 = Copyright (franchise)     ← skip (not useful as prompts)
#   6 = Artist                    ← skip
#   9 = Rating (safe/explicit…)   ← skip (always score high, useless as prompts)
_INCLUDE_CATEGORIES = {0, 4}

# WD v3 meta-tags: format/medium descriptors that are not subject-matter prompts.
# These consistently appear for real-world video/photo inputs and cause SD to
# stylize toward black-and-white abstract text images rather than the actual content.
_META_TAG_BLOCKLIST: frozenset[str] = frozenset({
    "monochrome", "greyscale", "grayscale", "black_and_white",
    "text", "english_text", "traditional_media", "watercolor_(medium)",
    "3d", "cg", "realistic", "photorealistic",
    "multiple_views", "4koma", "manga",
    "lowres", "absurdres", "highres",
    "comic", "sketch", "lineart",
    "censored", "uncensored", "bar_censor", "mosaic_censoring",
})


# ── Model loading ──────────────────────────────────────────────────────────────

def _get_tagger_model(model_name: str) -> tuple[Any, list[str]]:
    """Lazy-load and cache (onnx_session, tag_names). Thread-safe."""
    with _cache_lock:
        if model_name in _tagger_cache:
            return _tagger_cache[model_name]

    logger.info("Tagger: loading %s", model_name)
    repo_id = _HF_REPOS.get(model_name, model_name)
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            f"onnxruntime and huggingface_hub are required for the auto-tagger. "
            f"Install with: pip install onnxruntime-gpu huggingface_hub\n{exc}"
        ) from exc

    model_path = hf_hub_download(repo_id, "model.onnx")
    tags_path  = hf_hub_download(repo_id, "selected_tags.csv")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    tag_names: list[str] = []
    tag_categories: list[int] = []
    with open(tags_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_names.append(row["name"])
            tag_categories.append(int(row.get("category", 0)))

    with _cache_lock:
        _tagger_cache[model_name] = (session, tag_names, tag_categories)

    general_count = sum(1 for c in tag_categories if c in _INCLUDE_CATEGORIES)
    logger.info("Tagger: loaded %s (%d tags, %d usable)", model_name, len(tag_names), general_count)
    return session, tag_names, tag_categories


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess(image: Image.Image) -> np.ndarray:
    """Resize to 448×448, convert to float32 [0,1], add batch dim."""
    img = image.convert("RGB").resize((_INPUT_SIZE, _INPUT_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    # WD models expect BGR channel order (OpenCV convention)
    arr = arr[:, :, ::-1]
    return arr[np.newaxis, ...]  # (1, H, W, C)


def _crop_masked_area(image: Image.Image, mask: Image.Image | None) -> Image.Image:
    """Crop the bounding box of non-zero mask pixels. Falls back to full image."""
    if mask is None:
        return image
    mask_l = mask.convert("L")
    bbox = mask_l.getbbox()
    if not bbox:
        return image
    return image.crop(bbox)


# ── Public API ────────────────────────────────────────────────────────────────

def tag_image(
    image: Image.Image,
    *,
    mask: Image.Image | None = None,
    threshold: float = 0.35,
    model_name: str = "wd-eva02-large-v3",
    max_tags: int = 40,
) -> str:
    """
    Run WD tagger on image (cropped to mask bounding box if provided).

    Returns comma-separated tag string, empty string on failure.
    All GPU/CPU work happens here — call only from a thread pool.
    """
    try:
        session, tag_names, tag_categories = _get_tagger_model(model_name)
    except Exception as exc:
        logger.warning("Tagger: model unavailable (%s)", exc)
        return ""

    try:
        cropped = _crop_masked_area(image, mask)
        inp = _preprocess(cropped)
        input_name = session.get_inputs()[0].name
        (raw,) = session.run(None, {input_name: inp})
        scores = raw[0]
        # WD v3 ONNX outputs raw logits — apply sigmoid to get probabilities.
        # Some model exports include sigmoid in the graph (values already in [0,1]);
        # detect this by range check and skip sigmoid if already normalised.
        if scores.min() >= -0.01 and scores.max() <= 1.01:
            probs = scores.astype(np.float32)
        else:
            probs = (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)

        tags: list[tuple[float, str]] = []
        for prob, name, cat in zip(probs, tag_names, tag_categories):
            if cat not in _INCLUDE_CATEGORIES:
                continue  # skip rating / copyright / artist tags
            if name in _META_TAG_BLOCKLIST:
                continue  # skip format meta-tags that mislead stylization
            if prob >= threshold:
                tags.append((float(prob), name))

        tags.sort(key=lambda x: -x[0])
        selected = [name.replace("_", " ") for _, name in tags[:max_tags]]
        result = ", ".join(selected)
        logger.debug("Tagger: %d tags above %.2f → %s…", len(tags), threshold, result[:80])
        return result
    except Exception as exc:
        logger.warning("Tagger: inference failed (%s)", exc)
        return ""
