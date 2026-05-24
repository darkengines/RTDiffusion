"""Engine configuration and type aliases."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from PIL import Image


@dataclass
class EngineConfig:
    model_id: str = os.getenv("RTD_MODEL_ID") or os.getenv("RTD_MODEL_PATH") or ""
    device: str = os.getenv("RTD_DEVICE", "cuda")
    guidance_scale: float = float(os.getenv("RTD_GUIDANCE_SCALE", "1.5"))
    compile_unet: bool = os.getenv("RTD_COMPILE", "0") == "1"
    attention_slicing: bool = os.getenv("RTD_ATTENTION_SLICING", "0") == "1"
    force_mock: bool = os.getenv("RTD_MOCK", "0") == "1"
    layer_batch_size: int = max(1, int(os.getenv("RTD_LAYER_BATCH_SIZE", "4")))
    transparent_layer_batch_size: int = max(1, int(os.getenv("RTD_TRANSPARENT_LAYER_BATCH_SIZE", "1")))
    z_image_base_model: str = os.getenv("RTD_Z_IMAGE_BASE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
    z_image_local_only: bool = os.getenv("RTD_Z_IMAGE_LOCAL_ONLY", "0") == "1"
    realtime_accel: bool = os.getenv("RTD_REALTIME_ACCEL", "1") == "1"
    ssf_image_threshold: float = float(os.getenv("RTD_SSF_IMAGE_THRESHOLD", "9.0"))
    ssf_mask_threshold: float = float(os.getenv("RTD_SSF_MASK_THRESHOLD", "6.0"))
    ssf_skip_probability: float = float(os.getenv("RTD_SSF_SKIP_PROBABILITY", "0.96"))
    ssf_max_skips: int = max(1, int(os.getenv("RTD_SSF_MAX_SKIPS", "12")))
    stream_compile: bool = os.getenv("RTD_STREAM_COMPILE", "0" if os.name == "nt" else "1") == "1"
    stream_direct: bool = os.getenv("RTD_STREAM_DIRECT", "0") == "1"
    stream_native: bool = os.getenv("RTD_STREAM_NATIVE", "1") == "1"
    stream_tiny_vae: bool = os.getenv("RTD_STREAM_TINY_VAE", "1") == "1"
    stream_warmup: int = max(0, int(os.getenv("RTD_STREAM_WARMUP", "0")))
    stream_lcm_lora: bool = os.getenv("RTD_STREAM_LCM_LORA", "1") == "1"
    stream_trt: bool = os.getenv("RTD_STREAM_TRT", "0") == "1"
    stream_trt_fp16: bool = os.getenv("RTD_STREAM_TRT_FP16", "1") == "1"
    stream_trt_cache_dir: str = os.getenv("RTD_STREAM_TRT_CACHE", "outputs/trt-engines")
    session_directory: str = ""


ProgressCallback = Callable[["str", "float", "str", "object | None"], None]
ChunkCallback = Callable[["list[tuple[int, Image.Image]]", "float", "str"], None]
