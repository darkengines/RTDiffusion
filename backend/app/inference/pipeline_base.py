"""Common Diffusers pipeline base helpers.

SDXL and StreamDiffusion load from the same model source and share
core pipeline concerns (model path resolution, device defaults, diffusers
loading from folder/single-file, and dimension alignment). Keep these rules
in one place so backend-specific implementations stay thin.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol


class _LoggerLike(Protocol):
    def warning(self, msg: str, *args: object) -> None: ...


def default_device() -> str:
    value = os.getenv("RTD_DEVICE", "cuda").strip().lower()
    return "cuda:0" if value == "cuda" else value


def env_model_path() -> str:
    return os.getenv("RTD_MODEL_ID") or os.getenv("RTD_MODEL_PATH") or ""


def clamp_dims_64(width: int, height: int, *, max_side_env: str = "RTD_STREAM_MAX_SIDE") -> tuple[int, int]:
    """Preserve requested dims unless an explicit resize policy is enabled.

    Behavior:
    - Default: preserve width/height exactly (no implicit alignment/downsample).
    - If ``RTD_STREAM_MAX_SIDE`` > 0: downscale proportionally to fit that side.
    - If ``RTD_STREAM_ALIGN_64`` is truthy: align down to multiples of 64.

    This keeps client viewport resolution intact by default and makes resizing
    opt-in via explicit runtime configuration.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    max_side = int(os.getenv(max_side_env, "0"))
    if max_side > 0:
        scale = min(1.0, max_side / max(width, height, 1))
        width = int(width * scale)
        height = int(height * scale)

    align_64 = os.getenv("RTD_STREAM_ALIGN_64", "0").strip().lower() not in ("0", "false", "no", "")
    if align_64:
        return max(64, width // 64 * 64), max(64, height // 64 * 64)

    return width, height


def normalize_diffusers_model_id(model_id: str, logger: _LoggerLike | None = None) -> str:
    """If a component file is selected, resolve up to its pipeline folder."""
    path = Path(model_id)
    if not path.is_file():
        return model_id
    for parent in path.parents:
        if (parent / "model_index.json").is_file():
            if logger is not None:
                logger.warning(
                    "Selected %s is a Diffusers component file; loading pipeline folder %s instead",
                    path,
                    parent,
                )
            return str(parent)
    return model_id


def load_pretrained_pipe(
    pipeline_class: Any,
    model_id: str,
    dtype: Any,
    *,
    variant: str | None = None,
    logger: _LoggerLike | None = None,
) -> Any:
    """Load a diffusers pipeline from folder or hub id with safe local defaults."""
    model_path = Path(model_id)
    if model_path.is_dir():
        return pipeline_class.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            variant=variant,
            local_files_only=True,
            low_cpu_mem_usage=False,
        )
    try:
        return pipeline_class.from_pretrained(model_id, torch_dtype=dtype, variant=variant)
    except OSError as exc:
        message = str(exc)
        if "scheduler_config.json" not in message:
            raise
        if logger is not None:
            logger.warning(
                "Model cache is missing scheduler_config.json for %s; retrying with force_download",
                model_id,
            )
        return pipeline_class.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant=variant,
            force_download=True,
        )


def load_single_file_pipe(pipeline_class: Any, model_id: str, dtype: Any) -> Any:
    """Load a diffusers single-file checkpoint with local-only resolution."""
    model_path = Path(model_id)
    try:
        return pipeline_class.from_single_file(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=model_path.suffix.lower() == ".safetensors",
        )
    except Exception as exc:
        message = str(exc)
        if "CLIPTextModel" in message and "missing" in message:
            raise RuntimeError(
                f"{model_path.name} is not a complete Stable Diffusion image checkpoint. "
                "It is missing text encoder weights, so Diffusers cannot load it as an inpaint/img2img pipeline. "
                "Choose a full SD/SDXL checkpoint or set RTD_MODEL_PATH to a known inpaint checkpoint."
            ) from exc
        raise
