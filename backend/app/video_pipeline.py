import gc
import logging
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .schemas import MotionClipFrame, MotionClipRequest

logger = logging.getLogger("rtdiffusion.video")

_PIPELINES: dict[tuple[str, str], Any] = {}
_WAN_PROMPT_EMBEDS: dict[tuple[object, ...], tuple[Any, Any]] = {}
_LTX_CONFIG_REPO = "Lightricks/LTX-Video-0.9.7-dev"
_LTX2_CONFIG_REPO = "Lightricks/LTX-Video-0.9.8-13B-distilled"


class NativeVideoPipelineError(RuntimeError):
    pass


def native_video_configured(model: str) -> bool:
    return bool(_native_model_reference(model))


def native_video_missing_message(model: str) -> str:
    env_name = _model_env_name(model)
    return (
        f"No native {model} video model configured. Set {env_name} to a Diffusers model id, "
        f"Diffusers directory, or supported single-file checkpoint. WAN 2.2 component loading supports "
        f"RTD_NATIVE_WAN_HIGH_MODEL plus RTD_NATIVE_WAN_LOW_MODEL, RTD_NATIVE_WAN_TEXT_ENCODER, and RTD_NATIVE_WAN_VAE."
    )


ProgressCallback = Callable[..., None]


def generate_native_video_loop(
    source: Image.Image,
    arrival: Image.Image | None,
    frame: MotionClipRequest,
    clip_dir: Path,
    progress: ProgressCallback | None = None,
) -> int:
    reference = frame.model_path or _native_model_reference(frame.model)
    if not reference:
        raise NativeVideoPipelineError(native_video_missing_message(frame.model))

    output_dir = clip_dir / "native-frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, "load", 0.02, f"Loading native {frame.model} pipeline")
    device = _torch_device(frame.device)
    pipeline = _load_pipeline(frame.model, reference, device)
    total_frames = _native_frame_count(frame)
    width, height = _native_dimensions(frame.width, frame.height)
    steps = int(os.getenv("RTD_NATIVE_VIDEO_STEPS", str(min(frame.arrival_steps, 8))))
    guidance = float(os.getenv("RTD_NATIVE_VIDEO_CFG", str(frame.arrival_cfg or 4.0)))
    seed = int(os.getenv("RTD_NATIVE_VIDEO_SEED", "0")) or None

    generator = None
    try:
        import torch

        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    except Exception:
        generator = None

    image = source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    prompt = _combined_prompt(frame)
    _emit(progress, "encode", 0.08, f"Encoding WAN prompt and {total_frames} frame request at {width}x{height}")
    kwargs = _call_kwargs(
        frame.model,
        image=image,
        arrival=arrival.convert("RGB").resize((width, height), Image.Resampling.LANCZOS) if arrival is not None else None,
        prompt=prompt,
        negative_prompt=frame.negative_prompt,
        width=width,
        height=height,
        total_frames=total_frames,
        fps=frame.fps,
        steps=steps,
        guidance=guidance,
        generator=generator,
    )
    if _is_wan_family(frame.model):
        kwargs = _wan_prompt_embedded_kwargs(pipeline, kwargs, progress)
        if os.getenv("RTD_NATIVE_WAN_UNLOAD_TEXT_ENCODER", "1") != "0":
            pipeline.text_encoder = None
            _cleanup_memory()
    _emit(progress, "synthesize", 0.12, f"Running {steps} WAN denoise step(s)")
    kwargs.update(_progress_kwargs(steps, progress, clip_dir, frame.fps, frame.width, frame.height))
    result = pipeline(**kwargs)
    _emit(progress, "decode", 0.92, "Decoding video frames")
    frames = _extract_frames(result)
    if not frames:
        raise NativeVideoPipelineError(f"Native {frame.model} pipeline returned no frames")
    frames = [_normalize_frame(image, frame.width, frame.height) for image in frames]
    if arrival is not None and len(frames) >= 2:
        frames[0] = source.convert("RGB").resize((frame.width, frame.height), Image.Resampling.LANCZOS)
        frames[-1] = arrival.convert("RGB").resize((frame.width, frame.height), Image.Resampling.LANCZOS)
    for index, image in enumerate(frames):
        image.save(clip_dir / f"frame-{index:04d}.png", format="PNG", optimize=False)
        preview_frame = MotionClipFrame(index=index, url=f"/motion/clips/{clip_dir.name}/frames/{index}", duration_ms=round(1000 / frame.fps))
        _emit(progress, "write", 0.94 + ((index + 1) / max(1, len(frames))) * 0.05, f"Streaming frame {index + 1}/{len(frames)}", [preview_frame])
    _cleanup_memory()
    _emit(progress, "complete", 1.0, f"Generated {len(frames)} native frame(s)")
    return len(frames)


def _wan_prompt_embedded_kwargs(pipeline: Any, kwargs: dict[str, Any], progress: ProgressCallback | None = None) -> dict[str, Any]:
    import torch

    prompt = kwargs.pop("prompt")
    negative_prompt = kwargs.pop("negative_prompt", None)
    do_classifier_free_guidance = float(kwargs.get("guidance_scale", 1.0)) > 1.0
    dtype = next(pipeline.transformer.parameters()).dtype
    device = pipeline._execution_device
    cache_key = (prompt, negative_prompt, do_classifier_free_guidance, str(dtype))
    cached = _WAN_PROMPT_EMBEDS.get(cache_key)
    if cached is not None:
        _emit(progress, "encode", 0.1, "Using cached WAN prompt embeddings")
        prompt_embeds, negative_prompt_embeds = cached
        kwargs["prompt_embeds"] = prompt_embeds.to(device=device, dtype=dtype)
        kwargs["negative_prompt_embeds"] = negative_prompt_embeds.to(device=device, dtype=dtype) if negative_prompt_embeds is not None else None
        return kwargs

    _ensure_wan_text_encoder(pipeline)
    with torch.inference_mode():
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=1,
            device=device,
            dtype=dtype,
        )
    _remember_wan_prompt_embeds(cache_key, prompt_embeds, negative_prompt_embeds)
    kwargs["prompt_embeds"] = prompt_embeds
    kwargs["negative_prompt_embeds"] = negative_prompt_embeds
    return kwargs


def _remember_wan_prompt_embeds(cache_key: tuple[object, ...], prompt_embeds: Any, negative_prompt_embeds: Any) -> None:
    max_items = max(0, int(os.getenv("RTD_NATIVE_WAN_PROMPT_CACHE", "8")))
    if max_items <= 0:
        return
    _WAN_PROMPT_EMBEDS[cache_key] = (
        prompt_embeds.detach().to("cpu"),
        negative_prompt_embeds.detach().to("cpu") if negative_prompt_embeds is not None else None,
    )
    while len(_WAN_PROMPT_EMBEDS) > max_items:
        _WAN_PROMPT_EMBEDS.pop(next(iter(_WAN_PROMPT_EMBEDS)))


def _ensure_wan_text_encoder(pipeline: Any) -> None:
    if getattr(pipeline, "text_encoder", None) is not None:
        return
    import torch

    dtype = next(pipeline.transformer.parameters()).dtype
    config_root = Path(_local_diffusers_config(os.getenv("RTD_NATIVE_WAN_CONFIG_REPO", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")))
    text_path = Path(os.getenv("RTD_NATIVE_WAN_TEXT_ENCODER", "D:/comfyui/comfy/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"))
    pipeline.text_encoder = _load_wan_text_encoder(text_path, config_root, dtype).to(pipeline._execution_device)
    if pipeline.tokenizer is None:
        from transformers import T5TokenizerFast

        pipeline.tokenizer = T5TokenizerFast.from_pretrained(str(config_root / "tokenizer"))


def _progress_kwargs(steps: int, progress: ProgressCallback | None, clip_dir: Path, fps: int, width: int, height: int) -> dict[str, Any]:
    if progress is None:
        return {}
    preview_interval = max(0, int(os.getenv("RTD_NATIVE_WAN_LATENT_PREVIEW_INTERVAL", "0")))

    def on_step_end(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        value = 0.12 + ((step + 1) / max(1, steps)) * 0.78
        preview_frames = None
        if preview_interval and ((step + 1) % preview_interval == 0 or step + 1 == steps):
            preview_frames = _write_latent_preview(callback_kwargs.get("latents"), clip_dir, step, fps, width, height)
        _emit(progress, "synthesize", value, f"WAN denoise step {step + 1}/{steps}", preview_frames)
        return callback_kwargs

    kwargs: dict[str, Any] = {"callback_on_step_end": on_step_end}
    if preview_interval:
        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
    return kwargs


def _write_latent_preview(latents: Any, clip_dir: Path, step: int, fps: int, width: int, height: int) -> list[MotionClipFrame] | None:
    if latents is None:
        return None
    try:
        import torch
        import torch.nn.functional as functional

        with torch.inference_mode():
            tensor = latents.detach()
            if tensor.ndim != 5:
                return None
            sample = tensor[0]
            if sample.shape[0] < 3:
                return None
            temporal_index = sample.shape[1] // 2
            image_tensor = sample[:3, temporal_index].float().cpu()
            low = torch.quantile(image_tensor.flatten(), 0.01)
            high = torch.quantile(image_tensor.flatten(), 0.99)
            image_tensor = (image_tensor - low) / (high - low).clamp_min(1e-6)
            image_tensor = image_tensor.clamp(0, 1).unsqueeze(0)
            image_tensor = functional.interpolate(image_tensor, size=(height, width), mode="bilinear", align_corners=False)[0]
            array = (image_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        preview_name = f"latent-preview-{step + 1:04d}.png"
        Image.fromarray(array, mode="RGB").save(clip_dir / preview_name, format="PNG", optimize=False)
        return [MotionClipFrame(index=-(step + 1), url=f"/motion/clips/{clip_dir.name}/previews/{preview_name}", duration_ms=round(1000 / fps))]
    except Exception:
        return None


def _emit(progress: ProgressCallback | None, phase: str, value: float, message: str, preview_frames: list[MotionClipFrame] | None = None) -> None:
    if progress is not None:
        progress(phase, max(0.0, min(1.0, value)), message, preview_frames)


def _native_model_reference(model: str) -> str:
    specific = os.getenv(_model_env_name(model), "").strip()
    if specific:
        return specific
    if model == "fastvideo":
        fastvideo = os.getenv("RTD_FASTVIDEO_MODEL", "").strip()
        if fastvideo:
            return fastvideo
    generic = os.getenv("RTD_NATIVE_VIDEO_MODEL", "").strip()
    if generic:
        return generic
    if model == "ltx":
        return _first_existing(
            [
                Path("D:/comfyui/comfy/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"),
                Path("D:/comfyui/comfy/models/diffusion_models/ltx2310eros_beta.safetensors"),
                Path("D:/comfyui/comfy/models/checkpoints/ltx2310eros_v1.safetensors"),
            ]
        )
    if _is_wan_family(model):
        return _first_existing(
            [
                Path(os.getenv("RTD_NATIVE_WAN_LOW_MODEL", "")),
                Path("D:/comfyui/comfy/models/diffusion_models/wan22RemixT2VI2V_i2vLowV21.safetensors"),
                Path("D:/comfyui/comfy/models/diffusion_models/smoothMixWan22I2VT2V_i2vLow.safetensors"),
            ]
        )
    return ""


def _load_pipeline(model: str, reference: str, device: str) -> Any:
    key = (model, reference, device)
    if key in _PIPELINES:
        return _PIPELINES[key]
    try:
        import torch
        from diffusers import LTX2ImageToVideoPipeline, LTXImageToVideoPipeline, WanImageToVideoPipeline
    except Exception as exc:
        raise NativeVideoPipelineError("Install the backend gpu extra so Diffusers video pipelines are available") from exc

    variant = model.lower()
    dtype = _pipeline_dtype(torch, variant, device)
    path = Path(reference)
    try:
        if _is_wan_family(variant):
            pipeline = _load_wan_pipeline(reference, dtype)
        elif variant == "ltx2":
            loader = LTX2ImageToVideoPipeline.from_single_file if path.is_file() else LTX2ImageToVideoPipeline.from_pretrained
            pipeline = loader(reference, torch_dtype=dtype, **_single_file_kwargs(path, _LTX2_CONFIG_REPO))
        elif variant == "ltx":
            loader = LTXImageToVideoPipeline.from_single_file if path.is_file() else LTXImageToVideoPipeline.from_pretrained
            pipeline = loader(reference, torch_dtype=dtype, **_single_file_kwargs(path, _LTX_CONFIG_REPO))
        else:
            raise NativeVideoPipelineError(f"No native Diffusers pipeline is registered for motion model '{model}' yet")
    except NativeVideoPipelineError:
        raise
    except Exception as exc:
        raise NativeVideoPipelineError(f"Failed to load native {model} video pipeline from {reference}: {exc}") from exc

    _place_pipeline(pipeline, device, variant)
    _enable_video_memory_options(pipeline)
    _PIPELINES[key] = pipeline
    return pipeline


def _place_pipeline(pipeline: Any, device: str, variant: str) -> None:
    prefer_offload = os.getenv("RTD_NATIVE_VIDEO_CPU_OFFLOAD", "1") != "0"
    if _is_wan_family(variant) and os.getenv("RTD_NATIVE_WAN_CPU_OFFLOAD", "1") != "0":
        prefer_offload = True
    if prefer_offload and _enable_model_cpu_offload(pipeline, device):
        return
    try:
        pipeline.to(device)
    except Exception as exc:
        if "out of memory" not in str(exc).lower():
            raise
        _cleanup_memory()
        if not _enable_model_cpu_offload(pipeline, device):
            raise
        logger.warning("Moving native video pipeline to %s ran out of memory; retrying with CPU offload", device)


def _enable_model_cpu_offload(pipeline: Any, device: str) -> bool:
    if not hasattr(pipeline, "enable_model_cpu_offload"):
        return False
    gpu_id = int(device.split(":", 1)[1]) if device.startswith("cuda:") and device.split(":", 1)[1].isdigit() else None
    try:
        if gpu_id is None:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
        return True
    except Exception:
        logger.exception("Could not enable native video CPU offload")
        return False


def _enable_video_memory_options(pipeline: Any) -> None:
    for method_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        method = getattr(pipeline, method_name, None)
        if method is None:
            continue
        try:
            method()
        except Exception:
            logger.exception("Could not enable native video memory option %s", method_name)


def _load_wan_pipeline(reference: str, dtype: Any) -> Any:
    import torch
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline, WanTransformer3DModel
    from transformers import T5TokenizerFast

    path = Path(reference)
    if not path.is_file():
        return WanImageToVideoPipeline.from_pretrained(reference, torch_dtype=dtype)
    if _is_nvfp4_quant(path):
        raise NativeVideoPipelineError(
            f"{path.name} is Dasiwa NVFP4. Diffusers cannot execute that quantized format directly yet. "
            "Use an FP8/BF16 WAN2.2 safetensors for native RTDiffusion, or add a Dasiwa/Wan-Video runtime kernel."
        )

    config_root = Path(_local_diffusers_config(os.getenv("RTD_NATIVE_WAN_CONFIG_REPO", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")))
    high_path, low_path = _wan_component_paths(path)
    if high_path is not None and _is_nvfp4_quant(high_path):
        raise NativeVideoPipelineError(
            f"{high_path.name} is Dasiwa NVFP4. Diffusers cannot execute that quantized format directly yet. "
            "Use an FP8/BF16 WAN2.2 high safetensors for native RTDiffusion."
        )
    if low_path is not None and _is_nvfp4_quant(low_path):
        raise NativeVideoPipelineError(
            f"{low_path.name} is Dasiwa NVFP4. Diffusers cannot execute that quantized format directly yet. "
            "Use an FP8/BF16 WAN2.2 low safetensors for native RTDiffusion."
        )

    transformer = None
    transformer_2 = None
    boundary_ratio = None
    if high_path is not None and high_path.exists() and low_path is not None and low_path.exists():
        logger.info("Loading WAN 2.2 high/low components: high=%s low=%s", high_path, low_path)
        transformer = WanTransformer3DModel.from_single_file(
            str(high_path),
            config=str(config_root / "transformer"),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        transformer_2 = WanTransformer3DModel.from_single_file(
            str(low_path),
            config=str(config_root / "transformer_2"),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        boundary_ratio = float(os.getenv("RTD_NATIVE_WAN_BOUNDARY_RATIO", "0.875"))
    elif low_path is not None and low_path.exists():
        logger.warning("Loading WAN 2.2 low-only fallback: %s", low_path)
        transformer = WanTransformer3DModel.from_single_file(
            str(low_path),
            config=str(config_root / "transformer_2"),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    elif high_path is not None and high_path.exists():
        logger.warning("Loading WAN 2.2 high-only fallback: %s", high_path)
        transformer = WanTransformer3DModel.from_single_file(
            str(high_path),
            config=str(config_root / "transformer"),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    else:
        raise NativeVideoPipelineError(f"WAN model file not found: {path}")
    vae_path = Path(os.getenv("RTD_NATIVE_WAN_VAE", "D:/comfyui/comfy/models/vae/wan_2.1_vae.safetensors"))
    text_path = Path(os.getenv("RTD_NATIVE_WAN_TEXT_ENCODER", "D:/comfyui/comfy/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"))
    if not vae_path.exists():
        raise NativeVideoPipelineError(f"WAN VAE not found: {vae_path}")
    if not text_path.exists():
        raise NativeVideoPipelineError(f"WAN text encoder not found: {text_path}")

    vae = AutoencoderKLWan.from_single_file(str(vae_path), config=str(config_root / "vae"), torch_dtype=dtype)
    text_encoder = _load_wan_text_encoder(text_path, config_root, dtype)
    tokenizer = T5TokenizerFast.from_pretrained(str(config_root / "tokenizer"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(config_root / "scheduler"))
    pipeline = WanImageToVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
        transformer_2=transformer_2,
        boundary_ratio=boundary_ratio,
    )
    if os.getenv("RTD_NATIVE_WAN_COMPILE", "0") == "1" and hasattr(torch, "compile"):
        pipeline.transformer = torch.compile(pipeline.transformer)
    return pipeline


def _wan_component_paths(reference: Path) -> tuple[Path | None, Path | None]:
    configured_high = _first_existing(
        [
            Path(os.getenv("RTD_NATIVE_WAN_HIGH_MODEL", "")),
            Path(os.getenv("RTD_NATIVE_FASTVIDEO_HIGH_MODEL", "")),
            Path(os.getenv("RTD_FASTVIDEO_HIGH_MODEL", "")),
        ]
    )
    configured_low = _first_existing(
        [
            Path(os.getenv("RTD_NATIVE_WAN_LOW_MODEL", "")),
            Path(os.getenv("RTD_NATIVE_FASTVIDEO_LOW_MODEL", "")),
            Path(os.getenv("RTD_FASTVIDEO_LOW_MODEL", "")),
        ]
    )
    name = reference.name.lower()
    high_path = Path(configured_high) if configured_high else None
    low_path = Path(configured_low) if configured_low else None
    if "high" in name:
        high_path = reference
        low_path = low_path or _paired_wan_path(reference, "high", "low")
    elif "low" in name:
        low_path = reference
        high_path = high_path or _paired_wan_path(reference, "low", "high")
    elif reference.exists():
        low_path = low_path or reference
    return high_path, low_path


def _paired_wan_path(path: Path, source_token: str, target_token: str) -> Path | None:
    candidates = []
    name = path.name
    for source, target in (
        (source_token, target_token),
        (source_token.capitalize(), target_token.capitalize()),
        (source_token.upper(), target_token.upper()),
    ):
        if source in name:
            candidates.append(path.with_name(name.replace(source, target)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_wan_text_encoder(text_path: Path, config_root: Path, dtype: Any) -> Any:
    from safetensors.torch import load_file
    from transformers import UMT5Config, UMT5EncoderModel

    text_encoder = UMT5EncoderModel(UMT5Config.from_pretrained(str(config_root / "text_encoder"))).to(dtype=dtype)
    state_dict = _load_scaled_state_dict(load_file(str(text_path), device="cpu"), dtype)
    if "shared.weight" in state_dict and "encoder.embed_tokens.weight" not in state_dict:
        state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]
    text_encoder.load_state_dict(state_dict, strict=False)
    del state_dict
    return text_encoder


def _pipeline_dtype(torch: Any, variant: str, device: str) -> Any:
    configured = os.getenv("RTD_NATIVE_VIDEO_DTYPE", "").strip().lower()
    if configured in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if configured in {"fp16", "float16", "half"}:
        return torch.float16
    if configured in {"fp32", "float32"}:
        return torch.float32
    if not device.startswith("cuda"):
        return torch.float32
    if _is_wan_family(variant) and _cuda_device_supports_bf16(torch, device):
        return torch.bfloat16
    return torch.float16


def _cuda_device_supports_bf16(torch: Any, device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    try:
        index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(index)
        return major >= 8 and torch.cuda.is_bf16_supported()
    except Exception:
        return torch.cuda.is_bf16_supported()


def _load_scaled_state_dict(raw_state_dict: dict[str, Any], dtype: Any) -> dict[str, Any]:
    import torch

    state_dict: dict[str, Any] = {}
    for key, value in raw_state_dict.items():
        if key.endswith(".scale_weight") or key.endswith(".scale_input"):
            continue
        scale = raw_state_dict.get(f"{key.rsplit('.', 1)[0]}.scale_weight")
        if str(value.dtype).startswith("torch.float8") and scale is not None:
            state_dict[key] = (value.to(torch.float32) * scale.to(torch.float32)).to(dtype)
        elif torch.is_floating_point(value):
            state_dict[key] = value.to(dtype)
        else:
            state_dict[key] = value
    return state_dict


def _is_nvfp4_quant(path: Path) -> bool:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
        return metadata.get("quantization.bits", "").upper() == "NVFP4"
    except Exception:
        return False


def _call_kwargs(
    model: str,
    *,
    image: Image.Image,
    arrival: Image.Image | None,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    total_frames: int,
    fps: int,
    steps: int,
    guidance: float,
    generator: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": total_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "output_type": "pil",
    }
    if generator is not None:
        kwargs["generator"] = generator
    if model == "ltx" or model == "ltx2":
        kwargs["frame_rate"] = fps
    if _is_wan_family(model) and arrival is not None:
        kwargs["last_image"] = arrival
    return kwargs


def _extract_frames(result: Any) -> list[Image.Image]:
    frames = getattr(result, "frames", None)
    if frames is None and isinstance(result, dict):
        frames = result.get("frames")
    if frames is None and isinstance(result, (list, tuple)):
        frames = result[0]
    if not frames:
        return []
    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        frames = frames[0]
    return [frame for frame in frames if isinstance(frame, Image.Image)]


def _normalize_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def _native_dimensions(width: int, height: int) -> tuple[int, int]:
    max_side = int(os.getenv("RTD_NATIVE_VIDEO_MAX_SIDE", "768"))
    scale = min(1.0, max_side / max(width, height))
    native_width = max(128, int(width * scale) // 32 * 32)
    native_height = max(128, int(height * scale) // 32 * 32)
    return native_width, native_height


def _native_frame_count(frame: MotionClipRequest) -> int:
    requested = max(2, int(frame.duration_seconds * frame.fps))
    cap = int(os.getenv("RTD_NATIVE_VIDEO_MAX_FRAMES", "33"))
    count = max(2, min(requested, cap))
    if _is_wan_family(frame.model):
        count = max(5, 1 + (((count - 1) + 3) // 4) * 4)
        if count > cap:
            count = max(5, 1 + ((cap - 1) // 4) * 4)
    return count


def _combined_prompt(frame: MotionClipRequest) -> str:
    parts = [frame.prompt.strip()]
    if frame.arrival_prompt.strip():
        parts.append(f"end state: {frame.arrival_prompt.strip()}")
    parts.append("smooth temporally consistent motion, stable identity, loopable transition")
    return ", ".join(part for part in parts if part)


def _torch_device(device: str | None = None) -> str:
    value = (device or os.getenv("RTD_NATIVE_VIDEO_DEVICE") or os.getenv("RTD_DEVICE", "cuda")).strip().lower()
    if value == "cuda":
        return "cuda:0"
    return value if value == "cpu" or value.startswith("cuda") else "cpu"


def _model_env_name(model: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in model.upper()).strip("_")
    return f"RTD_NATIVE_{normalized}_MODEL"


def _is_wan_family(model: str) -> bool:
    return model in {"wan", "fastvideo", "fastvideo-wan", "wan2.2", "wan-2.2"}


def _single_file_kwargs(path: Path, config_repo: str) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {"config": _local_diffusers_config(config_repo), "cache_dir": str(_hf_cache_root())}


def _local_diffusers_config(repo_id: str) -> str:
    configured = os.getenv("RTD_NATIVE_VIDEO_CONFIG_DIR", "").strip()
    if configured:
        return configured
    target = _hf_config_root() / _safe_repo_name(repo_id)
    if (target / "model_index.json").exists():
        return str(target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id,
            local_dir=target,
            local_dir_use_symlinks=False,
            allow_patterns=["**/*.json", "*.json", "*.txt", "**/*.txt", "**/*.model"],
        )
    except Exception as exc:
        raise NativeVideoPipelineError(
            f"Could not download Diffusers config sidecars for {repo_id} without symlinks. "
            f"Set RTD_NATIVE_VIDEO_CONFIG_DIR to a local Diffusers config folder. Original error: {exc}"
        ) from exc
    return str(target)


def _hf_config_root() -> Path:
    return Path(os.getenv("RTD_NATIVE_VIDEO_CONFIG_CACHE", "outputs/hf-configs")).resolve()


def _hf_cache_root() -> Path:
    path = Path(os.getenv("RTD_NATIVE_VIDEO_HF_CACHE", "outputs/hf-cache")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "--").replace("\\", "--")


def _first_existing(paths: list[Path]) -> str:
    for path in paths:
        if str(path) in {"", "."}:
            continue
        if path.exists():
            return str(path)
    return ""


def _cleanup_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
