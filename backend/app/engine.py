import os
import hashlib
import logging
import gc
import random
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable, cast

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, ImageStat

from .config import load_local_env

from .assets import LORA_DIRS, default_model_path
from .image_io import alpha_to_empty_mask, decode_data_url, decode_data_url_rgba, encode_data_url, mask_to_luma, rgba_to_neutral_rgb
from .layerdiffuse import LayerDiffuseDecoder
from .pipeline_graph import resolve_conditioning_mask
from .schemas import InpaintFrame, InpaintResult, LayerCondition, LayerGenerateFrame, LayerGenerateResult, LayerVariation
from . import controlnet as _cn_module

load_local_env()

logger = logging.getLogger("rtdiffusion.engine")


def is_cuda_device(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


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


ProgressCallback = Callable[[str, float, str, LayerGenerateResult | None], None]
ChunkCallback = Callable[[list[tuple[int, Image.Image]], float, str], None]


class _UNetONNXWrapper:
    """Flattens dict inputs so the UNet callable is wrappable for ONNX export.

    Not a nn.Module itself — use make_unet_onnx_module(torch, unet, has_sdxl_cond) to get an
    exportable nn.Module (defined at call-time when torch is already imported).
    """

    def __init__(self, unet: Any, has_sdxl_cond: bool) -> None:
        self._unet = unet
        self._has_sdxl_cond = has_sdxl_cond

    def __call__(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        kwargs: dict[str, Any] = {"encoder_hidden_states": encoder_hidden_states, "return_dict": False}
        if self._has_sdxl_cond:
            kwargs["added_cond_kwargs"] = {"text_embeds": text_embeds, "time_ids": time_ids}
        return (self._unet(sample, timestep, **kwargs)[0],)


class _TRTUNetRunner:
    """Drop-in replacement for a PyTorch UNet — runs inference via a TensorRT engine.

    The engine is pre-built for a fixed batch size and latent resolution.
    Output is written to a pre-allocated buffer that is reused every frame.
    """

    def __init__(self, engine_bytes: bytes, out_shape: tuple, has_sdxl_cond: bool) -> None:
        try:
            import tensorrt as trt  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("tensorrt package required for RTD_STREAM_TRT=1") from exc
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError("TRT engine deserialization failed")
        self._ctx = self._engine.create_execution_context()
        self._out_shape = out_shape
        self._has_sdxl_cond = has_sdxl_cond
        self._out_buf: Any = None

    def __call__(
        self,
        sample,
        timestep,
        encoder_hidden_states: Any = None,
        added_cond_kwargs: Any = None,
        return_dict: bool = False,
        **_: Any,
    ):
        import torch

        if self._out_buf is None or self._out_buf.shape != self._out_shape:
            self._out_buf = torch.empty(self._out_shape, device=sample.device, dtype=sample.dtype)

        tensors: dict[str, Any] = {
            "sample": sample.contiguous(),
            "timestep": timestep.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
        }
        if self._has_sdxl_cond and added_cond_kwargs:
            tensors["text_embeds"] = added_cond_kwargs["text_embeds"].contiguous()
            tensors["time_ids"] = added_cond_kwargs["time_ids"].contiguous()

        for name, t in tensors.items():
            self._ctx.set_tensor_address(name, t.data_ptr())
        self._ctx.set_tensor_address("noise_pred", self._out_buf.data_ptr())

        stream_handle = torch.cuda.current_stream().cuda_stream
        self._ctx.execute_async_v3(stream_handle=stream_handle)
        return (self._out_buf,)


def _collect_engine_debug(
    image: "Image.Image",
    mask: "Image.Image",
    output: "Image.Image",
    frame: "InpaintFrame",
) -> dict[str, str]:
    """Build 320×180 JPEG thumbnail dict for the debug overlay (engine/WebSocket path)."""
    import base64, io as _io
    from PIL import Image as _PIL
    from .image_io import encode_data_url as _enc
    thumb = (320, 180)

    def _thumb_b64(img: "Image.Image") -> str:
        t = img.convert("RGB")
        t.thumbnail(thumb, _PIL.BILINEAR)
        buf = _io.BytesIO()
        t.save(buf, format="JPEG", quality=65)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    channels: dict[str, str] = {
        "canvas": _thumb_b64(image),
        "mask": _thumb_b64(mask.convert("RGB")),
        "output": _thumb_b64(output),
    }
    # Include preprocessed CN images from layer conditions
    from . import controlnet as _cn
    for cond in (frame.layer_conditions or []):
        if not cond.controlnet_model:
            continue
        try:
            from .image_io import decode_data_url
            src = decode_data_url(cond.controlnet_image or cond.image).convert("RGB")
            from .schemas import ControlNetPreprocessorParams
            params = cond.controlnet_preprocessor_params or ControlNetPreprocessorParams()
            if cond.controlnet_preprocessor:
                preprocessed = _cn.preprocess_image(src.resize((frame.width, frame.height)), cond.controlnet_model, params)
            else:
                preprocessed = src.resize((frame.width, frame.height))
            channels[f"cn:{cond.controlnet_model}"] = _thumb_b64(preprocessed)
        except Exception:
            pass
    return channels


class DiffusionEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.mode = "mock"
        self.pipe = None
        self.layer_pipe = None
        self.pipeline_family = "none"
        self.pipeline_kind = "none"
        self.model_id = ""
        self.loaded_loras: tuple[str, ...] = ()
        self.current_sampler: tuple[str | None, str | None] | None = None
        self.layer_decoder: LayerDiffuseDecoder | None = None
        self.scheduler_config: dict[str, object] | None = None
        self._ssf_cache: dict[str, object] | None = None
        self._residual_cfg_cache: dict[str, object] | None = None
        self._latent_cache: dict[str, object] | None = None
        self._prompt_cache: dict[str, object] | None = None
        self._latent_stream: dict[str, object] | None = None
        self._stream_compiled_signature: tuple[object, ...] | None = None
        self._stream_unet_callable: Any | None = None
        self._stream_vae: Any | None = None
        self._stream_vae_signature: tuple[object, ...] | None = None
        self._lcm_lora_adapter: str | None = None
        self._lcm_lora_model_id: str | None = None
        self._trt_runners: dict[str, Any] = {}
        self._seed_state: dict[str, object] | None = None
        self._cn_pipe_cache: dict[str, Any] = {}  # key = cn_model_id → wrapped pipeline
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        if self.config.force_mock:
            return
        try:
            import torch
            from diffusers import (
                AutoPipelineForImage2Image,
                AutoPipelineForInpainting,
                StableDiffusionImg2ImgPipeline,
                StableDiffusionInpaintPipeline,
                StableDiffusionXLImg2ImgPipeline,
                StableDiffusionXLInpaintPipeline,
                ZImageImg2ImgPipeline,
                ZImageInpaintPipeline,
                ZImagePipeline,
            )

            if is_cuda_device(self.config.device):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            model_id = self._normalize_diffusers_model_id(self.config.model_id or default_model_path())
            is_z_image = self._looks_like_z_image_model(model_id)
            if is_z_image and is_cuda_device(self.config.device):
                dtype = torch.bfloat16
            elif is_cuda_device(self.config.device):
                dtype = torch.float16
            else:
                dtype = torch.float32
            variant = "fp16" if dtype == torch.float16 else None
            if is_z_image:
                if Path(model_id).is_file():
                    try:
                        pipe = self._load_z_image_single_file_pipe(ZImageInpaintPipeline, model_id, dtype)
                        self.pipeline_kind = "inpaint"
                    except Exception:
                        try:
                            pipe = self._load_z_image_single_file_pipe(ZImageImg2ImgPipeline, model_id, dtype)
                            self.pipeline_kind = "image2image"
                        except Exception:
                            pipe = self._load_z_image_single_file_pipe(ZImagePipeline, model_id, dtype)
                            self.pipeline_kind = "text2image"
                else:
                    try:
                        pipe = self._load_pipe(ZImageInpaintPipeline, model_id, dtype, None)
                        self.pipeline_kind = "inpaint"
                    except Exception:
                        try:
                            pipe = self._load_pipe(ZImageImg2ImgPipeline, model_id, dtype, None)
                            self.pipeline_kind = "image2image"
                        except Exception:
                            pipe = self._load_pipe(ZImagePipeline, model_id, dtype, None)
                            self.pipeline_kind = "text2image"
                self.pipeline_family = "z-image"
            elif Path(model_id).is_file():
                is_xl = self._looks_like_sdxl_checkpoint(Path(model_id))
                inpaint_class = StableDiffusionXLInpaintPipeline if is_xl else StableDiffusionInpaintPipeline
                img2img_class = StableDiffusionXLImg2ImgPipeline if is_xl else StableDiffusionImg2ImgPipeline
                try:
                    pipe = self._load_single_file_pipe(inpaint_class, model_id, dtype)
                    self.pipeline_kind = "inpaint"
                except Exception:
                    pipe = self._load_single_file_pipe(img2img_class, model_id, dtype)
                    self.pipeline_kind = "image2image"
                self.pipeline_family = "sdxl" if is_xl else "sd"
            else:
                try:
                    pipe = self._load_pipe(AutoPipelineForInpainting, model_id, dtype, variant)
                    self.pipeline_kind = "inpaint"
                except Exception:
                    pipe = self._load_pipe(AutoPipelineForImage2Image, model_id, dtype, variant)
                    self.pipeline_kind = "image2image"
                self.pipeline_family = "sdxl" if "XL" in type(pipe).__name__ else "sd"
            pipe = pipe.to(self.config.device)
            pipe.set_progress_bar_config(disable=True)
            if self.config.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            if is_cuda_device(self.config.device) and hasattr(pipe, "unet"):
                pipe.unet.to(memory_format=torch.channels_last)
            if self.config.compile_unet and hasattr(torch, "compile"):
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            self.pipe = pipe
            self._stream_unet_callable = None
            self.layer_pipe = None
            self.layer_decoder = LayerDiffuseDecoder(self.config.device, dtype)
            self.scheduler_config = dict(pipe.scheduler.config) if hasattr(pipe, "scheduler") else None
            self.model_id = model_id
            self.current_sampler = None
            family = f"{self.pipeline_family}/" if self.pipeline_family != "sd" else ""
            self.mode = f"{family}{self.pipeline_kind}: {model_id}"
        except Exception:
            gc.collect()
            try:
                import torch

                if is_cuda_device(self.config.device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                logger.exception("Failed while clearing CUDA memory after pipeline load failure")
            logger.exception("Failed to load diffusion pipeline")
            raise

    @staticmethod
    def _normalize_diffusers_model_id(model_id: str) -> str:
        path = Path(model_id)
        if not path.is_file():
            return model_id
        for parent in path.parents:
            if (parent / "model_index.json").is_file():
                logger.warning("Selected %s is a Diffusers component file; loading pipeline folder %s instead", path, parent)
                return str(parent)
        return model_id

    @staticmethod
    def _load_pipe(pipeline_class, model_id: str, dtype, variant: str | None):
        model_path = Path(model_id)
        if model_path.is_dir():
            return pipeline_class.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                variant=variant,
                local_files_only=True,
                low_cpu_mem_usage=False,  # prevents meta-tensor init that breaks .to(device)
            )
        try:
            return pipeline_class.from_pretrained(model_id, torch_dtype=dtype, variant=variant)
        except OSError as exc:
            message = str(exc)
            if "scheduler_config.json" not in message:
                raise
            logger.warning("Model cache is missing scheduler_config.json for %s; retrying with force_download", model_id)
            return pipeline_class.from_pretrained(model_id, torch_dtype=dtype, variant=variant, force_download=True)

    @staticmethod
    def _load_single_file_pipe(pipeline_class, model_id: str, dtype):
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

    def _load_z_image_single_file_pipe(self, pipeline_class, model_id: str, dtype):
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from transformers import Qwen2Tokenizer, Qwen3Model

        base_model = self.config.z_image_base_model
        local_files_only = self.config.z_image_local_only
        text_encoder = Qwen3Model.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only)
        tokenizer = Qwen2Tokenizer.from_pretrained(base_model, subfolder="tokenizer", local_files_only=local_files_only)
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype, local_files_only=local_files_only)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler", local_files_only=local_files_only)
        return pipeline_class.from_single_file(
            str(Path(model_id)),
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=Path(model_id).suffix.lower() == ".safetensors",
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )

    @staticmethod
    def _looks_like_sdxl_checkpoint(model_path: Path) -> bool:
        name = model_path.name.lower()
        if "xl" in name or "pony" in name or "illustrious" in name:
            return True
        return model_path.stat().st_size > 6_000_000_000

    @staticmethod
    def _looks_like_z_image_model(model_id: str) -> bool:
        normalized = model_id.replace("\\", "/").lower()
        return "z-image" in normalized or "z_image" in normalized or "zimage" in normalized

    def generate(self, frame: InpaintFrame) -> InpaintResult:
        started = time.perf_counter()
        self._ensure_runtime(frame)
        base_frame = self._frame_with_prompt_mix_conditions(frame)
        if base_frame.transparent_background:
            transparent_prompt = f"{base_frame.prompt}, isolated foreground subject, transparent background, no scenery, clean silhouette"
            transparent_negative = ", ".join(
                part for part in [base_frame.negative_prompt, "busy background, scenery, environment, frame, border, watermark"] if part
            )
            base_frame = base_frame.model_copy(update={"prompt": transparent_prompt, "negative_prompt": transparent_negative})
        source = decode_data_url_rgba(frame.image).resize((frame.width, frame.height), Image.Resampling.LANCZOS)
        image = rgba_to_neutral_rgb(source)
        manual_mask = mask_to_luma(decode_data_url(frame.mask)).resize(
            (frame.width, frame.height), Image.Resampling.LANCZOS
        )
        empty_mask = alpha_to_empty_mask(source)
        mask = ImageChops.lighter(manual_mask, empty_mask)
        mask, base_frame = self._mask_with_layer_conditions(mask, base_frame)
        image = self._apply_stochastic_blur(image, mask, base_frame.stochastic_blur if not base_frame.stream_diffusion else 0)
        reused = None if base_frame.transparent_background else self._reuse_stochastic_similarity_frame(frame, base_frame, image, mask)
        if reused is not None:
            latency_ms = (time.perf_counter() - started) * 1000
            fps = 1000 / latency_ms if latency_ms else 0
            return InpaintResult(
                image=encode_data_url(reused),
                fps=fps,
                latency_ms=latency_ms,
                mode=f"{self.mode}+ssf-skip",
            )
        self._apply_sampler(frame.sampler, frame.scheduler)
        output = self._mock_inpaint(image, mask, base_frame.prompt)
        if self.pipe is not None:
            output = self._diffusers_inpaint(image, mask, base_frame)
        output = self._apply_residual_cfg(output, mask, base_frame)
        output = self._apply_zero_denoise_layer_constraints(output, frame)
        output = self._apply_layer_region_conditions(output, frame)
        output = self._apply_transparent_background(output, base_frame)
        self._remember_stochastic_similarity_frame(frame, base_frame, image, mask, output)
        latency_ms = (time.perf_counter() - started) * 1000
        fps = 1000 / latency_ms if latency_ms else 0

        debug_channels: dict[str, str] = {}
        if frame.debug_streams:
            debug_channels = _collect_engine_debug(image, mask, output, frame)

        return InpaintResult(
            image=encode_data_url(output, image_format="PNG" if base_frame.transparent_background else "JPEG"),
            fps=fps,
            latency_ms=latency_ms,
            mode=self._frame_mode(base_frame),
            debug_channels=debug_channels,
        )

    def _reuse_stochastic_similarity_frame(
        self,
        frame: InpaintFrame,
        base_frame: InpaintFrame,
        image: Image.Image,
        mask: Image.Image,
    ) -> Image.Image | None:
        if not (frame.stochastic_similarity_filter or frame.stream_diffusion) or self._ssf_cache is None:
            return None
        if self._ssf_cache.get("signature") != self._sampling_signature(base_frame):
            return None
        previous_image = self._ssf_cache.get("image_probe")
        previous_mask = self._ssf_cache.get("mask_probe")
        previous_output = self._ssf_cache.get("output")
        if not isinstance(previous_image, bytes) or not isinstance(previous_mask, bytes) or not isinstance(previous_output, Image.Image):
            return None
        image_delta = self._mean_abs_delta(previous_image, self._probe_bytes(image, "RGB"))
        mask_delta = self._mean_abs_delta(previous_mask, self._probe_bytes(mask, "L"))
        image_threshold = self.config.ssf_image_threshold
        mask_threshold = self.config.ssf_mask_threshold
        if frame.stream_diffusion:
            image_threshold = max(0.0, (1.0 - frame.stream_similarity_threshold) * 255.0)
            mask_threshold = image_threshold
        if image_delta > image_threshold or mask_delta > mask_threshold:
            return None
        skips_value = self._ssf_cache.get("skips", 0)
        skips = skips_value if isinstance(skips_value, int) else 0
        max_skips = frame.stream_max_skip_frames if frame.stream_diffusion else self.config.ssf_max_skips
        if skips >= max_skips:
            self._ssf_cache["skips"] = 0
            return None
        if not frame.stream_diffusion and random.random() > self.config.ssf_skip_probability:
            self._ssf_cache["skips"] = 0
            return None
        self._ssf_cache["skips"] = skips + 1
        return self._compose_reused_output(previous_output, image, mask)

    def _remember_stochastic_similarity_frame(
        self,
        frame: InpaintFrame,
        base_frame: InpaintFrame,
        image: Image.Image,
        mask: Image.Image,
        output: Image.Image,
    ) -> None:
        if not (frame.stochastic_similarity_filter or frame.stream_diffusion):
            self._ssf_cache = None
            return
        self._ssf_cache = {
            "signature": self._sampling_signature(base_frame),
            "image_probe": self._probe_bytes(image, "RGB"),
            "mask_probe": self._probe_bytes(mask, "L"),
            "output": output.copy(),
            "skips": 0,
        }

    def _apply_residual_cfg(self, output: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image:
        if not frame.residual_cfg:
            self._residual_cfg_cache = None
            return output
        signature = self._sampling_signature(frame)
        previous_output = self._residual_cfg_cache.get("output") if self._residual_cfg_cache and self._residual_cfg_cache.get("signature") == signature else None
        if isinstance(previous_output, Image.Image):
            residual_mask = mask.filter(ImageFilter.GaussianBlur(4)).point(lambda value: int(value * 0.32))
            output = Image.composite(previous_output.convert("RGB"), output.convert("RGB"), residual_mask)
        self._residual_cfg_cache = {"signature": signature, "output": output.copy()}
        return output

    @staticmethod
    def _compose_reused_output(previous_output: Image.Image, image: Image.Image, mask: Image.Image) -> Image.Image:
        reuse_mask = mask.filter(ImageFilter.GaussianBlur(2))
        return Image.composite(previous_output.convert("RGB"), image.convert("RGB"), reuse_mask)

    def _residual_cfg_warm(self, frame: InpaintFrame) -> bool:
        return bool(
            frame.residual_cfg
            and self._residual_cfg_cache
            and self._residual_cfg_cache.get("signature") == self._sampling_signature(frame)
            and isinstance(self._residual_cfg_cache.get("output"), Image.Image)
        )

    def _realtime_accel_enabled(self, frame: InpaintFrame) -> bool:
        return bool(
            self.config.realtime_accel
            and (
                frame.stream_diffusion
                or frame.residual_cfg
                or frame.stochastic_similarity_filter
                or frame.stochastic_blur > 0
                or frame.reuse_previous_latent
            )
        )

    def _frame_mode(self, frame: InpaintFrame) -> str:
        if not self._realtime_accel_enabled(frame):
            return self.mode
        active = []
        if frame.stream_diffusion:
            active.append(f"streamdiffusion:{frame.stream_cfg_type}")
        if frame.residual_cfg:
            active.append("rcfg")
        if frame.stochastic_similarity_filter:
            active.append("ssf")
        if frame.stochastic_blur > 0 and not frame.stream_diffusion:
            active.append("blur")
        if frame.seed_mode in {"random", "increment", "decrement"} and not frame.stream_diffusion:
            active.append(f"seed:{frame.seed_mode}")
        if frame.reuse_previous_latent and not frame.stream_diffusion:
            active.append(f"reuse:{frame.latent_reuse_denoise:.2f}")
            if frame.latent_reuse_noise > 0 and frame.latent_reuse_noise_mode in {"fixed", "random"}:
                active.append(f"perturb:{frame.latent_reuse_noise_mode}:{frame.latent_reuse_noise:.2f}")
        return f"{self.mode}+stream({'+'.join(active)})"

    def _realtime_generator(self, torch, frame: InpaintFrame):
        seed = self._realtime_seed(frame)
        try:
            return torch.Generator(device=self.config.device).manual_seed(seed)
        except Exception:
            return torch.Generator().manual_seed(seed)

    def _realtime_seed(self, frame: InpaintFrame) -> int:
        seed_parts = (
            self.model_id,
            self.pipeline_family,
            self.pipeline_kind,
            frame.prompt,
            frame.negative_prompt,
            tuple(frame.lora_paths),
            frame.width,
            frame.height,
            frame.steps,
            round(frame.strength, 4),
            round(frame.cfg, 4),
            frame.sampler,
            frame.scheduler,
            tuple(
                (
                    condition.name,
                    condition.prompt,
                    condition.negative_prompt,
                    condition.weight,
                    condition.mode,
                    condition.denoise,
                    condition.schedule,
                    condition.schedule_start,
                    condition.schedule_end,
                )
                for condition in frame.layer_conditions
            ),
        )
        digest = hashlib.sha256(repr(seed_parts).encode("utf-8")).digest()
        base_seed = frame.seed if frame.seed is not None else int.from_bytes(digest[:4], "big")
        seed_mode = frame.seed_mode if frame.seed_mode in {"fixed", "random", "increment", "decrement"} else "fixed"
        if frame.stream_diffusion:
            return base_seed % (2**32)
        if seed_mode == "random":
            return random.randint(0, (2**32) - 1)
        if seed_mode in {"increment", "decrement"}:
            state_key = repr((self.model_id, self.pipeline_family, self.pipeline_kind, base_seed, seed_mode, seed_parts))
            if not self._seed_state or self._seed_state.get("key") != state_key:
                self._seed_state = {"key": state_key, "step": 0}
            raw_step = self._seed_state.get("step", 0)
            step = raw_step if isinstance(raw_step, int) else 0
            self._seed_state["step"] = step + 1
            offset = step if seed_mode == "increment" else -step
            return (base_seed + offset) % (2**32)
        seed = base_seed % (2**32)
        if frame.seed_variation and not frame.stream_diffusion:
            seed = (seed + random.randint(0, frame.seed_variation)) % (2**32)
        return seed

    def _clear_realtime_stream_state(self) -> None:
        self._latent_cache = None
        self._latent_stream = None
        self._prompt_cache = None
        self._stream_compiled_signature = None
        self._stream_unet_callable = None
        self._stream_vae = None
        self._stream_vae_signature = None
        self._lcm_lora_adapter = None
        self._lcm_lora_model_id = None
        self._trt_runners.clear()

    @staticmethod
    def _apply_stochastic_blur(image: Image.Image, mask: Image.Image, radius: float) -> Image.Image:
        if radius <= 0:
            return image
        blurred = image.filter(ImageFilter.GaussianBlur(radius))
        blur_mask = mask.filter(ImageFilter.GaussianBlur(max(1.0, radius * 0.5)))
        return Image.composite(blurred, image, blur_mask)

    def _sampling_signature(self, frame: InpaintFrame) -> tuple[object, ...]:
        conditions = tuple(
            (
                condition.name,
                condition.prompt,
                condition.negative_prompt,
                condition.weight,
                condition.mode,
                condition.denoise,
                condition.schedule,
                condition.schedule_start,
                condition.schedule_end,
                hash(condition.image),
            )
            for condition in frame.layer_conditions
        )
        return (
            frame.prompt,
            frame.negative_prompt,
            frame.model_path,
            tuple(frame.lora_paths),
            frame.width,
            frame.height,
            frame.steps,
            round(frame.strength, 4),
            round(frame.cfg, 4),
            frame.sampler,
            frame.scheduler,
            frame.stream_diffusion,
            tuple(frame.stream_timestep_indices),
            frame.stream_frame_buffer_size,
            frame.stream_cfg_type,
            round(frame.stream_similarity_threshold, 4),
            frame.stream_max_skip_frames,
            frame.seed if not frame.stream_diffusion else None,
            frame.seed_mode if not frame.stream_diffusion else "fixed",
            frame.seed_variation if not frame.stream_diffusion else 0,
            frame.reuse_previous_latent if not frame.stream_diffusion else False,
            round(frame.latent_reuse_denoise, 4) if not frame.stream_diffusion else 0,
            round(frame.latent_reuse_noise, 4) if not frame.stream_diffusion else 0,
            frame.latent_reuse_noise_mode if not frame.stream_diffusion else "none",
            round(frame.stochastic_blur, 4) if not frame.stream_diffusion else 0,
            conditions,
        )

    @staticmethod
    def _probe_bytes(image: Image.Image, mode: str) -> bytes:
        return image.convert(mode).resize((32, 32), Image.Resampling.BILINEAR).tobytes()

    @staticmethod
    def _mean_abs_delta(left: bytes, right: bytes) -> float:
        if len(left) != len(right) or not left:
            return 255.0
        return sum(abs(a - b) for a, b in zip(left, right)) / len(left)

    def _frame_with_prompt_mix_conditions(self, frame: InpaintFrame) -> InpaintFrame:
        if not frame.layer_conditions:
            return frame
        prompt_parts = [frame.prompt.strip()] if frame.prompt.strip() else []
        negative_parts = [frame.negative_prompt.strip()] if frame.negative_prompt.strip() else []
        z_image_layout_parts: list[str] = []
        for condition in sorted(frame.layer_conditions, key=lambda item: (item.schedule_start, item.schedule_end)):
            if not condition.prompt.strip() and not condition.negative_prompt.strip():
                continue
            if self._schedule_influence(condition.schedule, condition.schedule_start, condition.schedule_end) <= 0:
                continue
            alpha = None
            try:
                condition_image = decode_data_url_rgba(condition.image).resize((frame.width, frame.height), Image.Resampling.LANCZOS)
                alpha = condition_image.getchannel("A")
                coverage = sum(value * count for value, count in enumerate(alpha.histogram())) / max(1, frame.width * frame.height * 255)
            except Exception:
                logger.exception("Failed to decode regional layer condition image for prompt conditioning")
                coverage = 0.0
            influence = max(0.0, min(1.0, coverage * condition.weight))
            if influence <= 0.002:
                continue
            if condition.prompt.strip() and alpha is not None:
                region_prompt = self._regional_prompt_text(condition.prompt.strip(), alpha, frame.width, frame.height, self.pipeline_family == "z-image")
                if self.pipeline_family == "z-image":
                    z_image_layout_parts.append(region_prompt)
                else:
                    repeats = max(1, min(3, round(influence * 5)))
                    prompt_parts.extend([region_prompt] * repeats)
            if condition.negative_prompt.strip():
                negative_parts.append(condition.negative_prompt.strip())
        if z_image_layout_parts:
            prompt_parts.append(self._z_image_layout_prompt(z_image_layout_parts))
        return frame.model_copy(
            update={
                "prompt": ", ".join(prompt_parts) if prompt_parts else frame.prompt,
                "negative_prompt": ", ".join(negative_parts) if negative_parts else frame.negative_prompt,
            }
        )

    def _mask_with_layer_conditions(self, mask: Image.Image, frame: InpaintFrame) -> tuple[Image.Image, InpaintFrame]:
        if not frame.layer_conditions:
            return mask, frame
        resolved = resolve_conditioning_mask(mask, list(frame.layer_conditions), frame.width, frame.height, frame.strength)
        if abs(resolved.strength - frame.strength) <= 0.0005:
            return resolved.mask, frame
        return resolved.mask, frame.model_copy(update={"strength": resolved.strength})

    @staticmethod
    def _regional_prompt_text(prompt: str, alpha: Image.Image, width: int, height: int, z_image: bool = False) -> str:
        bbox = alpha.getbbox()
        if not bbox:
            return prompt
        left, top, right, bottom = bbox
        center_x = (left + right) / 2 / max(1, width)
        center_y = (top + bottom) / 2 / max(1, height)
        horizontal = "left" if center_x < 0.33 else "right" if center_x > 0.67 else "center"
        vertical = "top" if center_y < 0.33 else "bottom" if center_y > 0.67 else "middle"
        if z_image:
            if horizontal == "center" and vertical == "middle":
                placement = "near the center"
            elif horizontal == "center":
                placement = f"near the {vertical} area"
            elif vertical == "middle":
                placement = f"near the {horizontal} side"
            else:
                placement = f"near the {vertical} {horizontal} area"
            return f"place {prompt} {placement} within the same scene"
        if horizontal == "center" and vertical == "middle":
            placement = "center region"
        elif horizontal == "center":
            placement = f"{vertical} region"
        elif vertical == "middle":
            placement = f"{horizontal} region"
        else:
            placement = f"{vertical} {horizontal} region"
        return f"{placement}: {prompt}"

    @staticmethod
    def _z_image_layout_prompt(parts: list[str]) -> str:
        if not parts:
            return ""
        if len(parts) == 1:
            layout = parts[0]
        else:
            layout = ", ".join(parts[:-1]) + f", and {parts[-1]}"
        return (
            "Create one continuous unified scene in a single camera view with normal perspective and consistent lighting. "
            f"In that one scene, {layout}."
        )

    def _negative_prompt(self, negative_prompt: str | None) -> str | None:
        parts = [negative_prompt.strip()] if negative_prompt and negative_prompt.strip() else []
        if self.pipeline_family == "z-image":
            parts.append("collage, split screen, multiple panels, grid layout, tiled image, contact sheet, picture-in-picture, separate images")
        return ", ".join(parts) if parts else None

    def _get_controlnet_pipe(self, condition: "LayerCondition") -> Any | None:
        """Load or retrieve a cached ControlNet-wrapped inpaint pipeline for this condition."""
        if self.pipe is None:
            return None
        if self.pipeline_family == "z-image":
            logger.warning("ControlNet: Z-Image does not expose a Diffusers ControlNet inpaint pipeline; skipping CN layer")
            return None
        is_sdxl = "xl" in self.pipeline_family.lower()
        import torch
        dtype = torch.bfloat16 if is_sdxl else torch.float16
        device = self.config.device
        model_id = _cn_module.resolve_model_id(
            condition.controlnet_model or "",
            condition.controlnet_model_path,
            is_sdxl,
        )
        if not model_id:
            return None
        cache_key = f"{model_id}:{device}"
        if cache_key not in self._cn_pipe_cache:
            try:
                cn_model = _cn_module.load_controlnet(model_id, device, dtype)
                cn_pipe = _cn_module.build_controlnet_inpaint_pipe(self.pipe, cn_model, is_sdxl)
                self._cn_pipe_cache[cache_key] = cn_pipe
            except Exception:
                logger.exception("ControlNet: failed to build pipeline for %s", model_id)
                return None
        return self._cn_pipe_cache.get(cache_key)

    def _diffusers_controlnet_inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        control_image: Image.Image,
        condition: "LayerCondition",
        frame: InpaintFrame,
    ) -> Image.Image:
        """Run ControlNet inpaint for one layer region."""
        import torch

        cn_pipe = self._get_controlnet_pipe(condition)
        if cn_pipe is None:
            logger.warning("ControlNet: no pipeline available, falling back to plain inpaint")
            return self._diffusers_inpaint(image, mask, frame)

        guidance_scale = frame.cfg
        effective_steps = max(frame.steps, 2)
        negative_prompt = frame.negative_prompt or None
        generator = torch.Generator(device=self.config.device).manual_seed(0)

        try:
            with torch.inference_mode():
                result = cn_pipe(
                    prompt=frame.prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    mask_image=mask,
                    control_image=control_image,
                    controlnet_conditioning_scale=condition.controlnet_scale,
                    control_guidance_start=condition.controlnet_start_at,
                    control_guidance_end=condition.controlnet_end_at,
                    strength=frame.strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=effective_steps,
                    width=frame.width,
                    height=frame.height,
                    generator=generator,
                )
            return result.images[0].convert("RGB")
        except Exception:
            logger.exception("ControlNet inpaint failed, falling back")
            return self._diffusers_inpaint(image, mask, frame)

    def _apply_layer_region_conditions(self, output: Image.Image, frame: InpaintFrame) -> Image.Image:
        if not frame.layer_conditions:
            return output
        current = output.convert("RGB")
        for condition in frame.layer_conditions:
            mode = condition.mode if condition.mode in {"mask", "add", "multiply", "override"} else "prompt_mix"
            if mode == "prompt_mix":
                continue
            if mode == "mask":
                denoise = condition.denoise if condition.denoise is not None else frame.strength
                denoise = max(0.0, min(0.999, denoise))
                # Plain mask denoise is resolved in the shared conditioning graph.
                # A CN-enabled layer still needs a regional pass because Diffusers
                # ControlNet is a separate pipeline in the non-streaming renderer.
                if not condition.controlnet_model:
                    continue
                if denoise <= 0:
                    continue
            try:
                condition_image = decode_data_url_rgba(condition.image).resize((frame.width, frame.height), Image.Resampling.LANCZOS)
            except Exception:
                logger.exception("Failed to decode regional layer condition image")
                continue
            alpha = condition_image.getchannel("A")
            if not alpha.getbbox():
                continue
            schedule_active = self._schedule_influence(condition.schedule, condition.schedule_start, condition.schedule_end) > 0
            if not schedule_active:
                continue

            def scale_alpha(value: int) -> int:
                return int(max(0.0, min(1.0, (value / 255) * condition.weight)) * 255)

            influence_mask = alpha.point(scale_alpha)
            if not influence_mask.getbbox():
                continue
            condition_source = current
            prompt = condition.prompt.strip()
            negative_prompt = condition.negative_prompt.strip()
            if mode != "override":
                prompt = ", ".join(part for part in [frame.prompt.strip(), prompt] if part)
                negative_prompt = ", ".join(part for part in [frame.negative_prompt.strip(), negative_prompt] if part)
            elif not prompt:
                prompt = frame.prompt
            denoise = condition.denoise if condition.denoise is not None else frame.strength
            condition_frame = frame.model_copy(
                update={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model_path": condition.model_path or frame.model_path,
                    "lora_paths": condition.lora_paths or frame.lora_paths,
                    "cfg": condition.cfg if condition.cfg is not None else frame.cfg,
                    "steps": condition.steps if condition.steps is not None else frame.steps,
                    "strength": denoise if mode == "mask" else denoise,
                    "sampler": condition.sampler or frame.sampler,
                    "scheduler": condition.scheduler or frame.scheduler,
                    "layer_conditions": [],
                }
            )
            self._ensure_runtime(condition_frame)
            self._apply_sampler(condition_frame.sampler, condition_frame.scheduler)
            regional = self._mock_inpaint(condition_source, influence_mask, condition_frame.prompt)
            if self.pipe is not None:
                if condition.controlnet_model:
                    # Resolve ControlNet guidance image
                    control_image = _cn_module.resolve_control_image(
                        condition,
                        condition_source,
                        frame.width,
                        frame.height,
                    )
                    if control_image is not None:
                        regional = self._diffusers_controlnet_inpaint(
                            condition_source, influence_mask, control_image, condition, condition_frame
                        )
                    else:
                        regional = self._diffusers_inpaint(condition_source, influence_mask, condition_frame)
                else:
                    regional = self._diffusers_inpaint(condition_source, influence_mask, condition_frame)
            current = self._composite_region_condition(current, regional, influence_mask, mode)
        return current

    def _apply_zero_denoise_layer_constraints(self, output: Image.Image, frame: InpaintFrame) -> Image.Image:
        if not frame.layer_conditions:
            return output
        current = output.convert("RGB")
        for condition in frame.layer_conditions:
            mode = condition.mode if condition.mode in {"mask", "add", "multiply", "override"} else "prompt_mix"
            if mode != "mask" or condition.denoise != 0:
                continue
            schedule_influence = self._schedule_influence(condition.schedule, condition.schedule_start, condition.schedule_end)
            if schedule_influence <= 0:
                continue
            try:
                condition_image = decode_data_url_rgba(condition.image).resize((frame.width, frame.height), Image.Resampling.LANCZOS)
            except Exception:
                logger.exception("Failed to decode zero-denoise layer condition image")
                continue
            alpha = condition_image.getchannel("A")
            if not alpha.getbbox():
                continue
            mask_scale = max(0.0, min(1.0, condition.weight * schedule_influence))
            if mask_scale <= 0:
                continue

            def scale_constraint_mask(value: int) -> int:
                return int(max(0.0, min(1.0, (value / 255) * mask_scale)) * 255)

            constraint_mask = alpha.point(scale_constraint_mask)
            current = Image.composite(condition_image.convert("RGB"), current, constraint_mask)
        return current

    @staticmethod
    def _schedule_influence(schedule: str, start: float, end: float) -> float:
        duration = max(0.0, min(1.0, end) - max(0.0, start))
        if duration <= 0:
            return 0.0
        if schedule == "ease_in":
            return duration * 0.82
        if schedule == "ease_out":
            return min(1.0, duration * 1.08)
        if schedule == "ease_in_out":
            return duration * 0.94
        return duration

    @staticmethod
    def _composite_region_condition(base: Image.Image, regional: Image.Image, mask: Image.Image, mode: str) -> Image.Image:
        base = base.convert("RGB")
        regional = regional.convert("RGB")
        if mode == "add":
            mixed = ImageChops.add(base, regional, scale=2.0)
        elif mode == "multiply":
            mixed = ImageChops.multiply(base, regional)
        else:
            mixed = regional
        return Image.composite(mixed, base, mask)

    def generate_layer(self, frame: LayerGenerateFrame, progress: ProgressCallback | None = None) -> LayerGenerateResult:
        started = time.perf_counter()
        self._emit_progress(progress, "setup", 0.02, "Preparing layer generation")
        runtime_frame = InpaintFrame(
            prompt=frame.prompt,
            negative_prompt=frame.negative_prompt,
            model_path=frame.model_path,
            lora_paths=frame.lora_paths,
            image=encode_data_url(Image.new("RGB", (frame.width, frame.height), (0, 0, 0)), image_format="PNG"),
            mask=encode_data_url(Image.new("L", (frame.width, frame.height), 255), image_format="PNG"),
            width=frame.width,
            height=frame.height,
            steps=min(frame.steps, 32),
            strength=0.999,
            cfg=frame.cfg,
            sampler=frame.sampler,
            scheduler=frame.scheduler,
        )
        self._ensure_runtime(runtime_frame)
        self._emit_progress(progress, "setup", 0.08, "Applying sampler and LoRA settings")
        self._apply_sampler(frame.sampler, frame.scheduler)
        variations: list[LayerVariation] = []
        base_seed = frame.seed if frame.seed is not None else int(time.time_ns() % 9_000_000_000_000_000)
        seeds = [base_seed + index for index in range(frame.variations)]

        def publish_variations(chunk: list[tuple[int, Image.Image]], value: float, message: str) -> None:
            for seed, image in chunk:
                variations.append(LayerVariation(image=encode_data_url(image, image_format="PNG"), seed=seed))
            partial = LayerGenerateResult(
                variations=list(variations),
                latency_ms=(time.perf_counter() - started) * 1000,
                mode=self.mode,
            )
            self._emit_progress(progress, "variations", value, message, partial)

        if self.pipe is not None:
            images = self._diffusers_layers(frame, seeds, progress, publish_variations if progress is not None else None)
            if progress is None:
                for seed, image in zip(seeds, images):
                    variations.append(LayerVariation(image=encode_data_url(image, image_format="PNG"), seed=seed))
        else:
            for index, seed in enumerate(seeds):
                publish_variations(
                    [(seed, self._mock_layer(frame, seed))],
                    0.1 + ((index + 1) / max(1, len(seeds))) * 0.84,
                    f"Generated {index + 1}/{len(seeds)} variation(s)",
                )
        latency_ms = (time.perf_counter() - started) * 1000
        result = LayerGenerateResult(variations=variations, latency_ms=latency_ms, mode=self.mode)
        self._emit_progress(progress, "complete", 1.0, f"Generated {len(variations)} variation(s)", result)
        return result

    @staticmethod
    def _emit_progress(progress: ProgressCallback | None, phase: str, value: float, message: str, result: LayerGenerateResult | None = None) -> None:
        if progress is not None:
            progress(phase, max(0.0, min(1.0, value)), message, result)

    def _ensure_runtime(self, frame: InpaintFrame) -> None:
        next_model = self._normalize_diffusers_model_id(frame.model_path or self.model_id)
        if next_model and (next_model != self.model_id or self.pipe is None):
            old_model_id = self.model_id
            old_config_model = self.config.model_id
            self.unload_runtime()
            self.model_id = ""
            self.mode = "mock"
            self.pipeline_family = "none"
            self.pipeline_kind = "none"
            self.scheduler_config = None
            self.config.model_id = next_model
            try:
                self._load_pipeline()
            except Exception as exc:
                self.model_id = old_model_id
                self.config.model_id = old_config_model
                raise RuntimeError(f"Could not load model: {next_model}") from exc
        next_loras = tuple(path for path in frame.lora_paths if path)
        if next_loras != self.loaded_loras and self.pipe is not None:
            self._apply_loras(next_loras)

    def prepare_for_frame(self, frame: InpaintFrame) -> None:
        self._ensure_runtime(frame)
        self._apply_sampler(frame.sampler, frame.scheduler)

    def unload_runtime(self) -> None:
        logger.info("Unloading active diffusion runtime: %s", self.mode)
        self.pipe = None
        self.layer_pipe = None
        self.layer_decoder = None
        self.loaded_loras = ()
        self.current_sampler = None
        self._clear_realtime_stream_state()
        gc.collect()
        try:
            import torch

            if is_cuda_device(self.config.device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            logger.exception("Failed while clearing CUDA memory")

    def _apply_loras(self, lora_paths: tuple[str, ...]) -> None:
        if not hasattr(self.pipe, "load_lora_weights"):
            self.loaded_loras = ()
            return
        previous_loras = self.loaded_loras
        try:
            self._load_lora_set(lora_paths)
            self.loaded_loras = lora_paths
            self.layer_pipe = None
            self.current_sampler = None
            self._clear_realtime_stream_state()
        except Exception as exc:
            try:
                self._load_lora_set(previous_loras)
                self.loaded_loras = previous_loras
            except Exception:
                self.loaded_loras = ()
            self.current_sampler = None
            self._clear_realtime_stream_state()
            self.mode = f"{self.pipeline_kind}: {self.model_id} | lora error: {exc.__class__.__name__}"
            raise RuntimeError(f"Could not load LoRA set: {exc}") from exc

    def _apply_sampler(self, sampler: str | None, scheduler: str | None = None) -> None:
        if not sampler or self.pipe is None or not hasattr(self.pipe, "scheduler"):
            return
        requested_sampler = (sampler, scheduler)
        if requested_sampler == self.current_sampler:
            return
        try:
            from diffusers import (
                DDIMScheduler,
                DDPMScheduler,
                DEISMultistepScheduler,
                DPMSolverMultistepScheduler,
                DPMSolverSinglestepScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
                FlowMatchEulerDiscreteScheduler,
                FlowMatchHeunDiscreteScheduler,
                HeunDiscreteScheduler,
                IPNDMScheduler,
                KDPM2AncestralDiscreteScheduler,
                KDPM2DiscreteScheduler,
                LCMScheduler,
                LMSDiscreteScheduler,
                UniPCMultistepScheduler,
            )
        except Exception as exc:
            raise RuntimeError(f"Sampler support is not available: {exc}") from exc

        if self.pipeline_family == "z-image":
            scheduler_classes = {
                "euler": FlowMatchEulerDiscreteScheduler,
                "euler_cfg_pp": FlowMatchEulerDiscreteScheduler,
                "euler_a": FlowMatchEulerDiscreteScheduler,
                "euler_ancestral": FlowMatchEulerDiscreteScheduler,
                "euler_ancestral_cfg_pp": FlowMatchEulerDiscreteScheduler,
                "heun": FlowMatchHeunDiscreteScheduler,
                "heunpp2": FlowMatchHeunDiscreteScheduler,
            }
        else:
            scheduler_classes = {
                "euler": EulerDiscreteScheduler,
                "euler_cfg_pp": EulerDiscreteScheduler,
                "euler_a": EulerAncestralDiscreteScheduler,
                "euler_ancestral": EulerAncestralDiscreteScheduler,
                "euler_ancestral_cfg_pp": EulerAncestralDiscreteScheduler,
                "heun": HeunDiscreteScheduler,
                "heunpp2": HeunDiscreteScheduler,
                "dpm_2": KDPM2DiscreteScheduler,
                "dpm_2_ancestral": KDPM2AncestralDiscreteScheduler,
                "lms": LMSDiscreteScheduler,
                "dpm_fast": DPMSolverMultistepScheduler,
                "dpm_adaptive": DPMSolverMultistepScheduler,
                "dpmpp_2s_ancestral": DPMSolverSinglestepScheduler,
                "dpmpp_sde": DPMSolverSinglestepScheduler,
                "dpmpp_sde_gpu": DPMSolverSinglestepScheduler,
                "dpmpp_2m": DPMSolverMultistepScheduler,
                "dpmpp_2m_sde": DPMSolverMultistepScheduler,
                "dpmpp_3m_sde": DPMSolverMultistepScheduler,
                "ddpm": DDPMScheduler,
                "lcm": LCMScheduler,
                "ipndm": IPNDMScheduler,
                "deis": DEISMultistepScheduler,
                "res_multistep": DPMSolverMultistepScheduler,
                "res_multistep_ancestral": DPMSolverMultistepScheduler,
                "gradient_estimation": DPMSolverMultistepScheduler,
                "er_sde": DPMSolverSinglestepScheduler,
                "seeds_2": DPMSolverMultistepScheduler,
                "seeds_3": DPMSolverMultistepScheduler,
                "sa_solver": DPMSolverMultistepScheduler,
                "ddim": DDIMScheduler,
                "unipc": UniPCMultistepScheduler,
                "uni_pc": UniPCMultistepScheduler,
            }
        scheduler_class = scheduler_classes.get(sampler)
        if scheduler_class is None:
            return
        config_overrides = self._scheduler_overrides(scheduler, scheduler_class)
        base_config = self.scheduler_config or self.pipe.scheduler.config
        try:
            self.pipe.scheduler = scheduler_class.from_config(base_config, **config_overrides)
        except TypeError:
            self.pipe.scheduler = scheduler_class.from_config(base_config)
        self.current_sampler = requested_sampler
        self._clear_realtime_stream_state()
        self.layer_pipe = None

    @staticmethod
    def _scheduler_overrides(scheduler: str | None, scheduler_class) -> dict[str, object]:
        overrides: dict[str, object] = {}
        if getattr(scheduler_class, "__name__", "").startswith("FlowMatch"):
            overrides.update({"use_karras_sigmas": False, "use_exponential_sigmas": False, "use_beta_sigmas": False})
            if scheduler in {"karras", "exponential", "beta"}:
                return overrides
        if "DPMSolver" in getattr(scheduler_class, "__name__", ""):
            overrides.update({"lower_order_final": True, "final_sigmas_type": "zero"})
        match scheduler:
            case "karras":
                overrides["use_karras_sigmas"] = True
            case "exponential":
                overrides["use_exponential_sigmas"] = True
            case "beta":
                overrides["use_beta_sigmas"] = True
            case "sgm_uniform":
                overrides["timestep_spacing"] = "trailing"
            case "ddim_uniform":
                overrides["timestep_spacing"] = "leading"
            case "normal" | "linear_quadratic" | "kl_optimal" | "simple" | None:
                pass
            case _:
                pass
        return overrides

    def _load_lora_set(self, lora_paths: tuple[str, ...]) -> None:
        pipe = cast(Any, self.pipe)
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
            # unload_lora_weights removes all adapters including _rtd_lcm
            self._lcm_lora_adapter = None
            self._lcm_lora_model_id = None
        adapter_names: list[str] = []
        for index, lora_path in enumerate(lora_paths):
            adapter_name = f"lora_{index}"
            resolved = self._resolve_lora_path(lora_path)
            try:
                try:
                    pipe.load_lora_weights(str(resolved), adapter_name=adapter_name)
                except Exception:
                    if not resolved.is_file():
                        raise
                    pipe.load_lora_weights(str(resolved.parent), weight_name=resolved.name, adapter_name=adapter_name)
            except ValueError as exc:
                # Diffusers Z-Image converter rejects some LoRA formats (e.g. alpha-only keys).
                # Skip the LoRA and continue rather than crashing the whole session.
                logger.warning("Skipping LoRA %s — format not recognised by Diffusers: %s", resolved.name, exc)
                continue
            adapter_names.append(adapter_name)
        if adapter_names and hasattr(pipe, "set_adapters"):
            pipe.set_adapters(adapter_names, adapter_weights=[1.0] * len(adapter_names))

    @staticmethod
    def _resolve_lora_path(lora_path: str) -> Path:
        path = Path(lora_path).expanduser()
        if path.exists():
            return path
        name = path.name
        if not name:
            raise FileNotFoundError(lora_path)
        for root in LORA_DIRS:
            candidate = root / name
            if candidate.exists():
                return candidate
            if root.exists():
                matches = list(root.rglob(name))
                if matches:
                    return matches[0]
        raise FileNotFoundError(f"{lora_path} (also searched configured LoRA directories for {name})")

    def _is_sdxl_pipe(self) -> bool:
        pipe = getattr(self, "pipe", None)
        return pipe is not None and "xl" in type(pipe).__name__.lower()

    def _streamdiffusion_lcm_lora_id(self) -> str:
        return "latent-consistency/lcm-lora-sdxl" if self._is_sdxl_pipe() else "latent-consistency/lcm-lora-sdv1-5"

    def _streamdiffusion_needs_lcm_lora(self, frame: InpaintFrame) -> bool:
        if not self.config.stream_lcm_lora:
            return False
        normalized = f"{self.model_id} {' '.join(self.loaded_loras)} {' '.join(frame.lora_paths)}".replace("\\", "/").lower()
        return not any(m in normalized for m in ("lcm", "lightning", "turbo", "hyper-sd", "hypersd"))

    def _streamdiffusion_ensure_lcm_lora(self, frame: InpaintFrame) -> str | None:
        if not self._streamdiffusion_needs_lcm_lora(frame):
            return None
        pipe = cast(Any, self.pipe)
        if not hasattr(pipe, "load_lora_weights") or not hasattr(pipe, "set_adapters"):
            return None
        adapter_name = "_rtd_lcm"
        if self._lcm_lora_model_id == self.model_id and self._lcm_lora_adapter == adapter_name:
            return adapter_name
        lora_id = self._streamdiffusion_lcm_lora_id()
        try:
            pipe.load_lora_weights(lora_id, adapter_name=adapter_name)
            self._lcm_lora_adapter = adapter_name
            self._lcm_lora_model_id = self.model_id
            logger.info("Injected LCM LoRA %s for StreamDiffusion on non-Turbo model", lora_id)
            return adapter_name
        except Exception:
            logger.exception("LCM LoRA load failed (%s); StreamDiffusion will use base model weights", lora_id)
            return None

    def _streamdiffusion_set_lcm_lora_scale(self, adapter_name: str, scale: float) -> None:
        pipe = cast(Any, self.pipe)
        if not hasattr(pipe, "set_adapters"):
            return
        try:
            peft_config = getattr(pipe.unet, "peft_config", {})
            if adapter_name not in peft_config:
                return
            user_adapters = [a for a in peft_config if not a.startswith("_rtd_")]
            all_adapters = user_adapters + [adapter_name]
            pipe.set_adapters(all_adapters, adapter_weights=[1.0] * len(user_adapters) + [scale])
        except Exception:
            pass

    def _streamdiffusion_direct(self, image: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image | None:
        if self.pipeline_family == "z-image":
            self.mode = f"{self.pipeline_kind}: {self.model_id} | streamdiffusion unavailable for z-image transformer"
            return None
        pipe = cast(Any, self.pipe)
        if not self.config.stream_native:
            self.mode = f"{self.pipeline_kind}: {self.model_id} | stream native disabled"
            return None
        if not (self.config.stream_direct or frame.stream_direct):
            self.mode = f"{self.pipeline_kind}: {self.model_id} | stream direct disabled"
            return None
        if not self._streamdiffusion_direct_model_supported(frame):
            self.mode = f"{self.pipeline_kind}: {self.model_id} | stream direct unavailable for inpaint-only checkpoints without LCM/Turbo LoRA"
            return None
        if pipe is None or not all(hasattr(pipe, name) for name in ("vae", "unet", "encode_prompt")):
            return None
        try:
            import torch
            from diffusers import LCMScheduler
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
        except Exception:
            return None

        session = self._streamdiffusion_session(frame, LCMScheduler, torch)
        if session is None:
            return None
        device = session["device"]
        dtype = session["dtype"]
        image_tensor = pipe.image_processor.preprocess(image, height=frame.height, width=frame.width).to(device=device, dtype=dtype)
        mask_tensor = self._streamdiffusion_mask_tensor(mask, session, torch)
        generator = self._realtime_generator(torch, frame)
        vae = session.get("vae") or pipe.vae
        vae_dtype = next(vae.parameters()).dtype
        vae_device = next(vae.parameters()).device
        with torch.inference_mode():
            encoded = vae.encode(image_tensor.to(device=vae_device, dtype=vae_dtype))
            image_latent = retrieve_latents(encoded, generator=generator).to(device=device, dtype=dtype)
            image_latent = image_latent * getattr(vae.config, "scaling_factor", 0.18215)
            frame_buffer_size = int(session["frame_buffer_size"])
            image_latent_batch = image_latent.expand(frame_buffer_size, -1, -1, -1)
            x_t_latent = session["alpha"][:frame_buffer_size] * image_latent_batch + session["beta"][:frame_buffer_size] * session["init_noise"][:frame_buffer_size]
            if session["channels"] == 9:
                # In streaming mode with no painted mask, treat the full image as the region to
                # generate — this enables continuous evolution driven by the prompt and motion driver.
                effective_mask = mask
                mask_pixel_count = frame.width * frame.height
                mask_raw = mask.convert("L").tobytes()
                mask_coverage = sum(mask.convert("L").point(lambda v: 1 if v > 10 else 0).getdata())
                if mask_coverage < mask_pixel_count * 0.005:
                    effective_mask = Image.new("L", mask.size, 255)
                full_mask_tensor = self._streamdiffusion_full_mask_tensor(effective_mask, frame.width, frame.height, device, dtype, torch)
                masked_image_tensor = image_tensor * (1 - full_mask_tensor)
                # Cache mask latents: re-encode only when the mask actually changed.
                # In streaming the mask is usually static — this saves one full VAE encode per frame.
                if session.get("_mask_raw_cache") != mask_raw or not isinstance(session.get("masked_image_latents"), torch.Tensor):
                    masked_encoded = vae.encode(masked_image_tensor.to(device=vae_device, dtype=vae_dtype))
                    masked_latent = retrieve_latents(masked_encoded, generator=generator).to(device=device, dtype=dtype)
                    session["mask_latents"] = mask_tensor.expand(session["batch_size"], 1, session["latent_height"], session["latent_width"])
                    session["masked_image_latents"] = (masked_latent * getattr(vae.config, "scaling_factor", 0.18215)).expand(session["batch_size"], -1, -1, -1)
                    session["_mask_raw_cache"] = mask_raw
                session["_mask_coverage"] = mask_coverage
            # Per-frame prompt interpolation: lerp embeddings between A and B each frame
            lerp_t = float(frame.prompt_lerp)
            prompt_b_base = session.get("prompt_b_embeds_base")
            if prompt_b_base is not None and 0.0 < lerp_t < 1.0:
                a = session["prompt_embeds_base"]
                lerped = (1.0 - lerp_t) * a + lerp_t * prompt_b_base
                lerped_batched = lerped.repeat(session["batch_size"], 1, 1)
                if session["do_full_cfg"] and session.get("neg_embeds_base") is not None:
                    neg = session["neg_embeds_base"].repeat(session["batch_size"], 1, 1)
                    combined = torch.cat([neg, lerped_batched], dim=0)
                else:
                    combined = lerped_batched
                interp_added = None
                if session.get("pooled_b_base") is not None and session.get("pooled_embeds_base") is not None:
                    lerped_pooled = (1.0 - lerp_t) * session["pooled_embeds_base"] + lerp_t * session["pooled_b_base"]
                    interp_added = self._streamdiffusion_added_cond_kwargs(pipe, frame, lerped_pooled, session["batch_size"], dtype, device)
                    if session["do_full_cfg"] and session.get("pooled_neg_base") is not None and interp_added:
                        neg_added = self._streamdiffusion_added_cond_kwargs(pipe, frame, session["pooled_neg_base"], session["batch_size"], dtype, device)
                        if neg_added:
                            interp_added = {k: torch.cat([neg_added[k], interp_added[k]], dim=0) for k in interp_added if k in neg_added}
                session = {**session, "prompt_embeds": combined, "added_cond_kwargs": interp_added or session["added_cond_kwargs"]}
            x0 = self._streamdiffusion_predict_x0(x_t_latent, session, torch)
            decoded = self._decode_latents_to_tensor(x0, vae=vae)[0].detach().cpu()
        output = self._tensor_to_pil(decoded).resize(image.size, Image.Resampling.LANCZOS)
        if self._streamdiffusion_collapsed(output, mask):
            logger.warning("StreamDiffusion direct output collapsed; resetting latent buffer and falling back for this frame")
            if self._latent_stream:
                self._latent_stream["latent_buffer"] = None
                self._latent_stream["warmed"] = False
            return None
        output_rgb = output.convert("RGB")
        mask_luma = mask.convert("L")
        mask_coverage = session.get("_mask_coverage") if session["channels"] == 9 else None
        if mask_coverage is None:
            mask_coverage = sum(mask_luma.point(lambda v: 1 if v > 10 else 0).getdata())
        # When no mask is painted, return the full generated image regardless of UNet type.
        if mask_coverage < frame.width * frame.height * 0.005:
            return output_rgb
        return Image.composite(output_rgb, image.convert("RGB"), mask_luma)

    def _streamdiffusion_direct_model_supported(self, frame: InpaintFrame) -> bool:
        normalized = f"{self.model_id} {self.mode} {' '.join(self.loaded_loras)} {' '.join(frame.lora_paths)}".replace("\\", "/").lower()
        incompatible_markers = ("inpaint", "inpainting")
        compatible_markers = ("lcm", "lightning", "turbo", "hyper-sd", "hypersd", "sdxl-turbo")
        has_incompatible = any(marker in normalized for marker in incompatible_markers)
        has_compatible = any(marker in normalized for marker in compatible_markers)
        if has_incompatible and not has_compatible:
            # Inpainting checkpoints work in the direct path — LCM LoRA is auto-injected
            # when stream_lcm_lora is enabled, giving fast multi-step denoising quality.
            # Block only when there's no LCM auto-injection available.
            if not (self.config.stream_lcm_lora and self.pipeline_family in {"sdxl", "sd"}):
                return False
        return self.pipeline_family in {"sdxl", "sd"} or has_compatible

    @staticmethod
    def _streamdiffusion_collapsed(output: Image.Image, mask: Image.Image) -> bool:
        region = output.convert("RGB")
        bbox = mask.convert("L").getbbox()
        if bbox:
            region = region.crop(bbox)
        stat = ImageStat.Stat(region)
        mean = sum(stat.mean) / max(1, len(stat.mean))
        spread = max(stat.stddev) if stat.stddev else 0.0
        extrema = cast(tuple[tuple[int, int], tuple[int, int], tuple[int, int]], region.getextrema())
        channel_range = max(float(high) - float(low) for low, high in extrema)
        return mean < 2.0 or (spread < 1.2 and channel_range < 6)

    def _streamdiffusion_session(self, frame: InpaintFrame, scheduler_class, torch):
        pipe = cast(Any, self.pipe)
        if not hasattr(pipe, "scheduler") or not hasattr(pipe, "unet") or not hasattr(pipe, "vae"):
            return None
        t_indices = [max(0, min(999, int(value))) for value in frame.stream_timestep_indices] or [32, 45]
        t_indices = t_indices[:16]
        cfg_type = frame.stream_cfg_type if frame.stream_cfg_type in {"none", "self", "initialize", "full"} else "self"
        frame_buffer_size = max(1, min(4, frame.stream_frame_buffer_size))
        needs_lcm = self._streamdiffusion_needs_lcm_lora(frame)
        signature = repr((self.model_id, self.pipeline_family, self.pipeline_kind, tuple(self.loaded_loras), self._sampling_signature(frame), self.config.stream_tiny_vae, needs_lcm, frame.prompt_b))
        if self._latent_stream and self._latent_stream.get("signature") == signature and self._latent_stream.get("kind") == "direct-streamdiffusion":
            return self._latent_stream

        lcm_adapter = self._streamdiffusion_ensure_lcm_lora(frame) if needs_lcm else None
        device = getattr(pipe, "_execution_device", None) or torch.device(self.config.device)
        unet = pipe.unet
        dtype = next(unet.parameters()).dtype
        scheduler = scheduler_class.from_config(pipe.scheduler.config)
        num_inference_steps = max(50, max(t_indices) + 1)
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps.to(device)
        timestep_values = [timesteps[min(index, len(timesteps) - 1)] for index in t_indices]
        batch_size = len(timestep_values) * frame_buffer_size
        sub_timesteps = torch.repeat_interleave(torch.stack(timestep_values).to(device=device, dtype=torch.long), frame_buffer_size, dim=0)
        latent_height = frame.height // getattr(pipe, "vae_scale_factor", 8)
        latent_width = frame.width // getattr(pipe, "vae_scale_factor", 8)
        channels = int(getattr(unet.config, "in_channels", 4))
        stream_vae = self._streamdiffusion_vae(dtype, device)
        latent_channels = int(getattr(stream_vae.config, "latent_channels", getattr(pipe.vae.config, "latent_channels", 4)))
        if channels not in {latent_channels, latent_channels + 5}:
            return None

        do_full_cfg = cfg_type == "full" and frame.cfg > 1.0
        encoded = pipe.encode_prompt(
            prompt=frame.prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_full_cfg,
            negative_prompt=self._negative_prompt(frame.negative_prompt) if do_full_cfg else None,
        )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._normalize_stream_prompt_embeds(encoded)
        prompt_embeds_base = prompt_embeds.to(device=device, dtype=dtype)
        pooled_embeds_base = pooled_prompt_embeds.to(device=device, dtype=dtype) if pooled_prompt_embeds is not None else None
        prompt_embeds = prompt_embeds_base.repeat(batch_size, 1, 1)
        added_cond_kwargs = self._streamdiffusion_added_cond_kwargs(
            pipe,
            frame,
            pooled_prompt_embeds,
            batch_size,
            dtype,
            device,
        )
        neg_embeds_base = None
        pooled_neg_base = None
        if do_full_cfg and negative_prompt_embeds is not None:
            neg_embeds_base = negative_prompt_embeds.to(device=device, dtype=dtype)
            pooled_neg_base = negative_pooled_prompt_embeds.to(device=device, dtype=dtype) if negative_pooled_prompt_embeds is not None else None
            neg_repeated = neg_embeds_base.repeat(batch_size, 1, 1)
            prompt_embeds = torch.cat([neg_repeated, prompt_embeds], dim=0)
            negative_added = self._streamdiffusion_added_cond_kwargs(
                pipe,
                frame,
                negative_pooled_prompt_embeds if negative_pooled_prompt_embeds is not None else pooled_prompt_embeds,
                batch_size,
                dtype,
                device,
            )
            if added_cond_kwargs and negative_added:
                added_cond_kwargs = {
                    key: torch.cat([negative_added[key], added_cond_kwargs[key]], dim=0)
                    for key in added_cond_kwargs.keys()
                    if key in negative_added
                }

        # Encode prompt B for per-frame interpolation
        prompt_b_embeds_base = None
        pooled_b_base = None
        if frame.prompt_b.strip():
            try:
                encoded_b = pipe.encode_prompt(
                    prompt=frame.prompt_b,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=None,
                )
                b_embeds, _, pooled_b, _ = self._normalize_stream_prompt_embeds(encoded_b)
                prompt_b_embeds_base = b_embeds.to(device=device, dtype=dtype)
                pooled_b_base = pooled_b.to(device=device, dtype=dtype) if pooled_b is not None else None
            except Exception:
                logger.exception("Failed to encode prompt B for StreamDiffusion interpolation")

        generator = self._realtime_generator(torch, frame)
        init_noise = torch.randn((batch_size, latent_channels, latent_height, latent_width), generator=generator, device=device, dtype=dtype)
        stock_noise = torch.randn_like(init_noise)
        latent_buffer = torch.zeros(((len(timestep_values) - 1) * frame_buffer_size, latent_channels, latent_height, latent_width), device=device, dtype=dtype) if len(timestep_values) > 1 else None
        c_skip, c_out, alpha, beta = self._streamdiffusion_scalings(scheduler, timestep_values, frame_buffer_size, dtype, device, torch)
        self._latent_stream = {
            "kind": "direct-streamdiffusion",
            "signature": signature,
            "device": device,
            "dtype": dtype,
            "scheduler": scheduler,
            "sub_timesteps": sub_timesteps,
            "init_noise": init_noise,
            "stock_noise": stock_noise,
            "latent_buffer": latent_buffer,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_base": prompt_embeds_base,
            "pooled_embeds_base": pooled_embeds_base,
            "neg_embeds_base": neg_embeds_base,
            "pooled_neg_base": pooled_neg_base,
            "prompt_b_embeds_base": prompt_b_embeds_base,
            "pooled_b_base": pooled_b_base,
            "added_cond_kwargs": added_cond_kwargs,
            "guidance_scale": frame.cfg if (do_full_cfg or cfg_type in {"self", "initialize"}) else 1.0,
            "do_full_cfg": do_full_cfg,
            "cfg_type": cfg_type,
            "rcfg_delta": float(os.getenv("RTD_STREAM_RCFG_DELTA", "1.0")),
            "c_skip": c_skip,
            "c_out": c_out,
            "alpha": alpha,
            "beta": beta,
            "frame_buffer_size": frame_buffer_size,
            "batch_size": batch_size,
            "channels": channels,
            "latent_channels": latent_channels,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "steps": len(timestep_values),
            "unet": pipe.unet,
            "vae": stream_vae,
            "warmup_count": self._streamdiffusion_warmup_count(len(timestep_values), frame_buffer_size),
            "warmed": False,
            "lcm_adapter": lcm_adapter,
        }
        if lcm_adapter:
            self._streamdiffusion_set_lcm_lora_scale(lcm_adapter, 1.0)
        self._compile_streamdiffusion_unet(frame, self._latent_stream, torch)
        self._warm_streamdiffusion_session(frame, self._latent_stream, torch)
        return self._latent_stream

    def _streamdiffusion_vae(self, dtype, device):
        pipe = cast(Any, self.pipe)
        # Inpainting UNets (9-channel) still output 4-channel latents — TinyVAE decodes them
        # correctly and is ~14× faster than the full VAE for encode, ~4× faster for decode.
        if not self.config.stream_tiny_vae:
            return pipe.vae
        configured = os.getenv("RTD_STREAM_TINY_VAE_ID", "").strip()
        model_text = self.model_id.replace("\\", "/").lower()
        tiny_id = configured or ("madebyollin/taesdxl" if "xl" in model_text or "sdxl" in model_text else "madebyollin/taesd")
        signature = (tiny_id, str(dtype), str(device))
        if self._stream_vae is not None and self._stream_vae_signature == signature:
            return self._stream_vae
        try:
            from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

            self._stream_vae = AutoencoderTiny.from_pretrained(tiny_id).to(device=device, dtype=dtype)
            self._stream_vae_signature = signature
            logger.info("Loaded StreamDiffusion Tiny VAE: %s", tiny_id)
            return self._stream_vae
        except Exception:
            logger.exception("Failed to load StreamDiffusion Tiny VAE; using pipeline VAE")
            self._stream_vae = None
            self._stream_vae_signature = None
            return pipe.vae

    def _streamdiffusion_warmup_count(self, step_count: int, frame_buffer_size: int) -> int:
        if self.config.stream_warmup > 0:
            return self.config.stream_warmup
        return max(1, step_count * frame_buffer_size)

    def _warm_streamdiffusion_session(self, frame: InpaintFrame, session: dict[str, Any], torch) -> None:
        if session.get("warmed"):
            return
        warmup_count = int(session.get("warmup_count") or 0)
        if warmup_count <= 0:
            session["warmed"] = True
            return
        frame_buffer_size = int(session["frame_buffer_size"])
        device = session["device"]
        dtype = session["dtype"]
        if session["channels"] == 9:
            session["mask_latents"] = torch.zeros((session["batch_size"], 1, session["latent_height"], session["latent_width"]), device=device, dtype=dtype)
            session["masked_image_latents"] = torch.zeros((session["batch_size"], session["latent_channels"], session["latent_height"], session["latent_width"]), device=device, dtype=dtype)
        with torch.inference_mode():
            for _ in range(warmup_count):
                x_t_latent = session["beta"][:frame_buffer_size] * session["init_noise"][:frame_buffer_size]
                self._streamdiffusion_predict_x0(x_t_latent, session, torch)
            if is_cuda_device(self.config.device):
                torch.cuda.synchronize()
        session["warmed"] = True
        logger.info("Warmed StreamDiffusion session with %s step(s) for %sx%s", warmup_count, frame.width, frame.height)

    @staticmethod
    def _normalize_stream_prompt_embeds(encoded):
        if len(encoded) >= 4:
            return encoded[0], encoded[1], encoded[2], encoded[3]
        return encoded[0], encoded[1] if len(encoded) > 1 else None, None, None

    @staticmethod
    def _streamdiffusion_added_cond_kwargs(pipe, frame: InpaintFrame, pooled_prompt_embeds, batch_size: int, dtype, device):
        if pooled_prompt_embeds is None or not hasattr(pipe, "_get_add_time_ids"):
            return None
        projection_dim = pooled_prompt_embeds.shape[-1]
        time_ids = pipe._get_add_time_ids(
            (frame.height, frame.width),
            (0, 0),
            (frame.height, frame.width),
            6.0,
            2.5,
            (frame.height, frame.width),
            (0, 0),
            (frame.height, frame.width),
            dtype,
            text_encoder_projection_dim=projection_dim,
        )
        if isinstance(time_ids, tuple):
            time_ids = time_ids[0]
        time_ids = time_ids.to(device=device)
        return {
            "text_embeds": pooled_prompt_embeds.to(device=device, dtype=dtype).repeat(batch_size, 1),
            "time_ids": time_ids.repeat(batch_size, 1),
        }

    @staticmethod
    def _streamdiffusion_mask_tensor(mask: Image.Image, session: dict[str, Any], torch):
        mask_image = mask.convert("L").resize((session["latent_width"], session["latent_height"]), Image.Resampling.LANCZOS)
        raw = mask_image.tobytes()
        cached_key = session.get("_mask_raw_key")
        if cached_key == raw:
            return session["_mask_tensor_cache"]
        values = torch.frombuffer(bytearray(raw), dtype=torch.uint8).float() / 255.0
        result = values.view(1, 1, session["latent_height"], session["latent_width"]).to(device=session["device"], dtype=session["dtype"])
        session["_mask_raw_key"] = raw
        session["_mask_tensor_cache"] = result
        return result

    @staticmethod
    def _streamdiffusion_full_mask_tensor(mask: Image.Image, width: int, height: int, device, dtype, torch):
        mask_image = mask.convert("L").resize((width, height), Image.Resampling.LANCZOS)
        values = torch.frombuffer(bytearray(mask_image.tobytes()), dtype=torch.uint8).float() / 255.0
        return values.view(1, 1, height, width).to(device=device, dtype=dtype)

    def _streamdiffusion_compile_trt(self, frame: InpaintFrame, session: dict[str, Any], torch) -> None:
        """Export the UNet to ONNX then build (or load) a TensorRT FP16 engine and install it as session["unet"]."""
        import hashlib, os, tempfile

        batch = session["batch_size"] * (2 if session["do_full_cfg"] else 1)
        latent_h = frame.height // 8
        latent_w = frame.width // 8
        channels = session["channels"]
        has_sdxl_cond = bool(session.get("added_cond_kwargs"))
        fp16 = self.config.stream_trt_fp16

        # Cache key — any change triggers a rebuild
        key_str = "|".join([
            self.model_id,
            str(batch),
            str(channels),
            str(latent_h),
            str(latent_w),
            str(has_sdxl_cond),
            str(fp16),
            ",".join(self.loaded_loras),
        ])
        cache_key = hashlib.sha256(key_str.encode()).hexdigest()[:16]

        if cache_key in self._trt_runners:
            session["unet"] = self._trt_runners[cache_key]
            return

        cache_dir = self.config.stream_trt_cache_dir
        engine_path = os.path.join(cache_dir, f"unet_{cache_key}.trt")

        engine_bytes: bytes | None = None

        if os.path.exists(engine_path):
            with open(engine_path, "rb") as fh:
                engine_bytes = fh.read()
            logger.info("Loaded TRT engine from %s", engine_path)
        else:
            try:
                import tensorrt as trt  # type: ignore[import-untyped]
            except ImportError as exc:
                logger.warning("TRT/ONNX not available (%s); falling back to eager", exc)
                return

            logger.info("Building TRT engine for UNet (batch=%d h=%d w=%d fp16=%s) — first run takes several minutes …", batch, latent_h, latent_w, fp16)

            dtype = torch.float16 if fp16 else torch.float32
            device = session.get("device", "cuda")

            sample_shape = (batch, channels, latent_h, latent_w)
            dummy_sample = torch.zeros(sample_shape, dtype=dtype, device=device)
            dummy_ts = torch.zeros((batch,), dtype=dtype, device=device)
            seq_len = session["prompt_embeds"].shape[1] if isinstance(session["prompt_embeds"], torch.Tensor) else 77
            dummy_enc = torch.zeros((batch, seq_len, self.pipe.unet.config.cross_attention_dim), dtype=dtype, device=device)
            # text_embeds = pooled CLIP output (e.g. 1280 for SDXL), NOT projection_class_embeddings_input_dim.
            # Read actual shapes from the live session tensors.
            if has_sdxl_cond and isinstance(session.get("added_cond_kwargs"), dict):
                cond = session["added_cond_kwargs"]
                pool_dim = cond["text_embeds"].shape[-1] if isinstance(cond.get("text_embeds"), torch.Tensor) else 1280
                time_ids_dim = cond["time_ids"].shape[-1] if isinstance(cond.get("time_ids"), torch.Tensor) else 6
                dummy_text_embeds = torch.zeros((batch, pool_dim), dtype=dtype, device=device)
                dummy_time_ids = torch.zeros((batch, time_ids_dim), dtype=dtype, device=device)
            else:
                dummy_text_embeds = torch.zeros((1,), dtype=dtype, device=device)
                dummy_time_ids = torch.zeros((1,), dtype=dtype, device=device)

            # nn.Module wrapper — must be defined here where torch is in scope.
            _unet_ref = self.pipe.unet
            _has_cond = has_sdxl_cond

            class _ONNXModule(torch.nn.Module):
                def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
                    kwargs: dict = {"encoder_hidden_states": encoder_hidden_states, "return_dict": False}
                    if _has_cond:
                        kwargs["added_cond_kwargs"] = {"text_embeds": text_embeds, "time_ids": time_ids}
                    return (_unet_ref(sample, timestep, **kwargs)[0],)

            wrapper = _ONNXModule()

            try:
                with tempfile.TemporaryDirectory() as tmp:
                    onnx_path = os.path.join(tmp, "unet.onnx")
                    input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
                    output_names = ["noise_pred"]
                    # PEFT LoRA weights have requires_grad=True; freeze them for ONNX tracing.
                    unet_grad_states = [(p, p.requires_grad) for p in self.pipe.unet.parameters()]
                    for p, _ in unet_grad_states:
                        p.requires_grad_(False)
                    try:
                        with torch.no_grad():
                            torch.onnx.export(
                                wrapper,
                                (dummy_sample, dummy_ts, dummy_enc, dummy_text_embeds, dummy_time_ids),
                                onnx_path,
                                input_names=input_names,
                                output_names=output_names,
                                opset_version=17,
                                do_constant_folding=True,
                            )
                    finally:
                        for p, grad in unet_grad_states:
                            p.requires_grad_(grad)

                    # parse_from_file resolves external-data weight files relative to the ONNX directory.
                    trt_logger = trt.Logger(trt.Logger.WARNING)
                    builder = trt.Builder(trt_logger)
                    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                    network = builder.create_network(network_flags)
                    parser = trt.OnnxParser(network, trt_logger)
                    if not parser.parse_from_file(onnx_path):
                        for i in range(parser.num_errors):
                            logger.error("TRT ONNX parse error: %s", parser.get_error(i))
                        raise RuntimeError("TRT ONNX parsing failed")

                    config = builder.create_builder_config()
                    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GiB
                    if fp16 and builder.platform_has_fast_fp16:
                        config.set_flag(trt.BuilderFlag.FP16)

                    raw = builder.build_serialized_network(network, config)
                    if raw is None:
                        raise RuntimeError("TRT engine build returned None")
                    engine_bytes = bytes(raw)

                os.makedirs(cache_dir, exist_ok=True)
                with open(engine_path, "wb") as fh:
                    fh.write(engine_bytes)
                logger.info("Saved TRT engine to %s", engine_path)

            except Exception as exc:
                logger.warning("TRT engine build failed (%s); falling back to eager/torch.compile", exc)
                return

        out_shape = (batch, session["latent_channels"], latent_h, latent_w)
        runner = _TRTUNetRunner(engine_bytes, out_shape, has_sdxl_cond)
        self._trt_runners[cache_key] = runner
        session["unet"] = runner
        logger.info("TRT UNet runner installed (key=%s)", cache_key)

    def _compile_streamdiffusion_unet(self, frame: InpaintFrame, session: dict[str, Any], torch) -> None:
        # TRT path takes priority over torch.compile
        if self.config.stream_trt and is_cuda_device(self.config.device):
            self._streamdiffusion_compile_trt(frame, session, torch)
            if session.get("unet") is not self.pipe.unet:
                return  # TRT runner installed — skip torch.compile

        if not self.config.stream_compile or not hasattr(torch, "compile") or not is_cuda_device(self.config.device):
            return
        signature = (
            self.model_id,
            self.pipeline_family,
            self.pipeline_kind,
            tuple(self.loaded_loras),
            frame.width,
            frame.height,
            session["batch_size"] * (2 if session["do_full_cfg"] else 1),
            session["channels"],
            frame.stream_cfg_type,
        )
        if self._stream_compiled_signature == signature:
            return
        try:
            self._stream_unet_callable = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=False)
            self._stream_compiled_signature = signature
            session["unet"] = self._stream_unet_callable
            logger.info("Compiled StreamDiffusion U-Net with TorchInductor/Triton for %s", signature)
        except Exception:
            logger.exception("Failed to compile StreamDiffusion U-Net; continuing eager")
            self._stream_compiled_signature = signature
            self._stream_unet_callable = None
            session["unet"] = self.pipe.unet

    @staticmethod
    def _streamdiffusion_scalings(scheduler, timesteps, frame_buffer_size: int, dtype, device, torch):
        c_skip_list = []
        c_out_list = []
        alpha_list = []
        beta_list = []
        for timestep in timesteps:
            c_skip, c_out = scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            timestep_index = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
            alpha = scheduler.alphas_cumprod[timestep_index].sqrt()
            beta = (1 - scheduler.alphas_cumprod[timestep_index]).sqrt()
            alpha_list.append(alpha)
            beta_list.append(beta)
        def expand(values):
            return torch.repeat_interleave(torch.stack(values).view(len(values), 1, 1, 1).to(device=device, dtype=dtype), frame_buffer_size, dim=0)
        return expand(c_skip_list), expand(c_out_list), expand(alpha_list), expand(beta_list)

    def _streamdiffusion_predict_x0(self, x_t_latent, session: dict[str, Any], torch):
        latent_buffer = session.get("latent_buffer")
        steps = int(session["steps"])
        if steps > 1 and isinstance(latent_buffer, torch.Tensor):
            x_in = torch.cat([x_t_latent, latent_buffer], dim=0)
        else:
            x_in = x_t_latent
        x_in = self._match_stream_batch(x_in, int(session["sub_timesteps"].shape[0]), torch)
        model_latent = x_in
        timesteps = session["sub_timesteps"]
        if session["do_full_cfg"]:
            model_latent = torch.cat([x_in, x_in], dim=0)
            timesteps = torch.cat([timesteps, timesteps], dim=0)
        if session["channels"] != session["latent_channels"]:
            mask_latents = session.get("mask_latents")
            masked_image_latents = session.get("masked_image_latents")
            if not isinstance(mask_latents, torch.Tensor) or not isinstance(masked_image_latents, torch.Tensor):
                raise RuntimeError("StreamDiffusion inpaint session is missing mask latents")
            mask_latents = mask_latents[:model_latent.shape[0]]
            masked_image_latents = masked_image_latents[:model_latent.shape[0]]
            if session["do_full_cfg"]:
                mask_latents = torch.cat([mask_latents, mask_latents], dim=0)
                masked_image_latents = torch.cat([masked_image_latents, masked_image_latents], dim=0)
            model_latent = torch.cat([model_latent, mask_latents, masked_image_latents], dim=1)
        prompt_embeds = self._match_stream_batch(session["prompt_embeds"], model_latent.shape[0], torch)
        model_kwargs = {
            "encoder_hidden_states": prompt_embeds,
            "return_dict": False,
        }
        if session.get("added_cond_kwargs"):
            model_kwargs["added_cond_kwargs"] = {
                key: self._match_stream_batch(value, model_latent.shape[0], torch) if isinstance(value, torch.Tensor) else value
                for key, value in session["added_cond_kwargs"].items()
            }
        if timesteps.shape[0] != model_latent.shape[0]:
            timesteps = self._match_stream_batch(timesteps, model_latent.shape[0], torch)
        unet = session.get("unet") or self.pipe.unet
        try:
            model_pred = unet(model_latent, timesteps, **model_kwargs)[0]
        except Exception:
            if unet is self.pipe.unet:
                raise
            logger.exception("Compiled StreamDiffusion U-Net failed; retrying eager")
            session["unet"] = self.pipe.unet
            self._stream_unet_callable = None
            self._stream_compiled_signature = None
            model_pred = self.pipe.unet(model_latent, timesteps, **model_kwargs)[0]
        if session["do_full_cfg"]:
            noise_uncond, noise_text = model_pred.chunk(2)
            model_pred = noise_uncond + session["guidance_scale"] * (noise_text - noise_uncond)
        elif session.get("cfg_type") in {"self", "initialize"} and session["guidance_scale"] > 1.0:
            bsz = x_in.shape[0]
            delta = session.get("rcfg_delta", 1.0)
            noise_uncond = session["stock_noise"][:bsz] * delta
            model_pred = noise_uncond + session["guidance_scale"] * (model_pred - noise_uncond)
            frame_buffer_size = int(session.get("frame_buffer_size", 1))
            session["stock_noise"] = torch.cat([
                session["init_noise"][:frame_buffer_size],
                session["stock_noise"][:-frame_buffer_size],
            ], dim=0)
        x0_batch = self._streamdiffusion_scheduler_step(model_pred, x_in, session)
        if steps > 1:
            output = x0_batch[-1:].contiguous()
            session["latent_buffer"] = session["alpha"][1:] * x0_batch[:-1] + session["beta"][1:] * session["init_noise"][1:]
        else:
            output = x0_batch
            session["latent_buffer"] = None
        return output

    @staticmethod
    def _match_stream_batch(tensor, batch_size: int, torch):
        if not isinstance(tensor, torch.Tensor) or tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] > batch_size:
            return tensor[:batch_size].contiguous()
        if tensor.shape[0] == 0:
            raise RuntimeError("StreamDiffusion tensor has empty batch")
        repeat_shape = [batch_size - tensor.shape[0], *tensor.shape[1:]]
        tail = tensor[-1:].expand(*repeat_shape)
        return torch.cat([tensor, tail], dim=0).contiguous()

    @staticmethod
    def _streamdiffusion_scheduler_step(model_pred, x_t_latent, session: dict[str, Any]):
        f_theta = (x_t_latent - session["beta"] * model_pred) / session["alpha"]
        return session["c_out"] * f_theta + session["c_skip"] * x_t_latent

    @staticmethod
    def _tensor_to_pil(image_tensor) -> Image.Image:
        array = (image_tensor.permute(1, 2, 0).float().numpy() * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(array)

    def _diffusers_inpaint(self, image: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image:
        import torch

        if frame.stream_diffusion:
            streamed = self._streamdiffusion_direct(image, mask, frame)
            if streamed is not None:
                return streamed
            # Deactivate LCM LoRA before falling back to regular pipeline inference
            if self._lcm_lora_adapter:
                self._streamdiffusion_set_lcm_lora_scale(self._lcm_lora_adapter, 0.0)
            frame = frame.model_copy(update={"stream_diffusion": False, "cfg": max(frame.cfg, 1.5)})

        accelerated = self._realtime_accel_enabled(frame)
        effective_steps = max(1, frame.steps) if accelerated else max(frame.steps, ceil(1 / max(frame.strength, 0.01)))
        guidance_scale = 0.0 if "turbo" in self.mode.lower() else frame.cfg
        if frame.stream_diffusion and frame.stream_cfg_type in {"none", "self", "initialize"}:
            guidance_scale = 1.0
        base_inpaint_strength = max(frame.strength, min(0.999, 1 / max(1, effective_steps))) if accelerated else frame.strength
        inpaint_strength = self._realtime_inpaint_strength(frame, base_inpaint_strength)
        negative_prompt = None if guidance_scale <= 1.0 else self._negative_prompt(frame.negative_prompt)
        generator = self._realtime_generator(torch, frame)
        prompt_kwargs = self._realtime_prompt_kwargs(frame, negative_prompt, guidance_scale)
        self._refresh_flowmatch_scheduler(self.pipe)
        with torch.inference_mode():
            if getattr(self, "pipeline_kind", "image2image") == "inpaint":
                generated = self._call_realtime_pipe(
                    {
                        **prompt_kwargs,
                        "image": image,
                        "mask_image": mask,
                        "strength": inpaint_strength,
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": effective_steps,
                        "width": frame.width,
                        "height": frame.height,
                        "generator": generator,
                    },
                    frame,
                ).images[0]
                return generated.convert("RGB")

            if getattr(self, "pipeline_kind", "image2image") == "text2image":
                generated = self._call_realtime_pipe(
                    {
                        **prompt_kwargs,
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": effective_steps,
                        "width": frame.width,
                        "height": frame.height,
                        "generator": generator,
                    },
                    frame,
                ).images[0]
                return Image.composite(generated.convert("RGB"), image, mask)

            generated = self._call_realtime_pipe(
                {
                    **prompt_kwargs,
                    "image": image,
                    "strength": inpaint_strength,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": effective_steps,
                    "width": frame.width,
                    "height": frame.height,
                    "generator": generator,
                },
                frame,
            ).images[0]
        return Image.composite(generated.convert("RGB"), image, mask)

    def _realtime_prompt_kwargs(self, frame: InpaintFrame, negative_prompt: str | None, guidance_scale: float) -> dict[str, Any]:
        pipe = cast(Any, self.pipe)
        if not hasattr(pipe, "encode_prompt"):
            return {"prompt": frame.prompt, "negative_prompt": negative_prompt}
        do_cfg = guidance_scale > 1.0
        cache_key = repr((self.model_id, self.pipeline_family, self.pipeline_kind, tuple(self.loaded_loras), frame.prompt, negative_prompt, do_cfg))
        if self._prompt_cache and self._prompt_cache.get("key") == cache_key:
            cached = self._prompt_cache.get("kwargs")
            if isinstance(cached, dict):
                return dict(cached)
        try:
            encoded = pipe.encode_prompt(
                prompt=frame.prompt,
                device=getattr(pipe, "_execution_device", self.config.device),
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt,
            )
        except TypeError:
            try:
                encoded = pipe.encode_prompt(
                    prompt=frame.prompt,
                    device=getattr(pipe, "_execution_device", self.config.device),
                    do_classifier_free_guidance=do_cfg,
                    negative_prompt=negative_prompt,
                )
            except Exception:
                return {"prompt": frame.prompt, "negative_prompt": negative_prompt}
        except Exception:
            return {"prompt": frame.prompt, "negative_prompt": negative_prompt}
        if not isinstance(encoded, tuple):
            return {"prompt": frame.prompt, "negative_prompt": negative_prompt}
        kwargs: dict[str, Any]
        if len(encoded) >= 4:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoded[:4]
            kwargs = {
                "prompt": None,
                "negative_prompt": None,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds if do_cfg else None,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds if do_cfg else None,
            }
        elif len(encoded) >= 2:
            prompt_embeds, negative_prompt_embeds = encoded[:2]
            kwargs = {
                "prompt": None,
                "negative_prompt": None,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds if do_cfg else None,
            }
        else:
            return {"prompt": frame.prompt, "negative_prompt": negative_prompt}
        self._prompt_cache = {"key": cache_key, "kwargs": kwargs}
        return dict(kwargs)

    def _call_realtime_pipe(self, kwargs: dict[str, Any], frame: InpaintFrame):
        import torch

        pipe = cast(Any, self.pipe)
        call_kwargs = dict(kwargs)
        saved_latents = None
        cached_latents = self._reusable_latents(frame)
        if cached_latents is not None:
            call_kwargs["latents"] = cached_latents

        def on_step_end(*callback_args, **callback_kwargs):
            nonlocal saved_latents
            tensor_payload = None
            if callback_args and isinstance(callback_args[-1], dict):
                tensor_payload = callback_args[-1]
            elif isinstance(callback_kwargs.get("callback_kwargs"), dict):
                tensor_payload = callback_kwargs["callback_kwargs"]
            elif isinstance(callback_kwargs, dict):
                tensor_payload = callback_kwargs
            latents = tensor_payload.get("latents") if isinstance(tensor_payload, dict) else None
            if isinstance(latents, torch.Tensor):
                saved_latents = latents.detach().clone()
            return tensor_payload if isinstance(tensor_payload, dict) else None

        try:
            result = pipe(
                **call_kwargs,
                callback_on_step_end=on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        except TypeError as exc:
            if "latents" not in str(exc) and "callback" not in str(exc):
                raise
            call_kwargs.pop("latents", None)
            try:
                result = pipe(
                    **call_kwargs,
                    callback_on_step_end=on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
            except TypeError as retry_exc:
                if "callback" not in str(retry_exc):
                    raise
                result = pipe(**call_kwargs)
        self._remember_realtime_latents(frame, saved_latents)
        return result

    def _latent_signature(self, frame: InpaintFrame) -> tuple[object, ...]:
        return (self.model_id, self.pipeline_family, self.pipeline_kind, self._sampling_signature(frame))

    @staticmethod
    def _latent_stream_length(frame: InpaintFrame) -> int:
        return max(1, min(4, max(1, frame.steps)))

    def _reusable_latents(self, frame: InpaintFrame):
        if not self._reuse_latent_cache_available(frame):
            return None
        cached = cast(Any, self._latent_cache.get("latents") if self._latent_cache else None)
        try:
            latents = cached.detach().clone()
            return self._perturb_reusable_latents(latents, frame)
        except Exception:
            self._latent_cache = None
            return None

    def _perturb_reusable_latents(self, latents, frame: InpaintFrame):
        amount = max(0.0, min(1.0, frame.latent_reuse_noise))
        mode = frame.latent_reuse_noise_mode if frame.latent_reuse_noise_mode in {"none", "fixed", "random"} else "none"
        if amount <= 0 or mode == "none":
            return latents
        try:
            import torch

            if mode == "random":
                seed = random.randint(0, (2**32) - 1)
            else:
                signature = repr((self._latent_signature(frame), "latent-reuse-perturbator"))
                digest = hashlib.sha256(signature.encode("utf-8")).digest()
                seed = int.from_bytes(digest[:4], "big")
            try:
                generator = torch.Generator(device=latents.device).manual_seed(seed)
            except Exception:
                generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            return latents + noise * amount
        except Exception:
            logger.exception("Failed to perturb reused latent; continuing with cached latent")
            return latents

    def _reuse_latent_cache_available(self, frame: InpaintFrame) -> bool:
        if frame.stream_diffusion or not frame.reuse_previous_latent:
            return False
        if frame.latent_reuse_restart:
            self._latent_cache = None
            return False
        if not self._latent_cache or self._latent_cache.get("signature") != self._latent_signature(frame):
            return False
        cached = self._latent_cache.get("latents")
        return hasattr(cached, "detach")

    def _realtime_inpaint_strength(self, frame: InpaintFrame, base_strength: float) -> float:
        if self._reuse_latent_cache_available(frame):
            return max(0.0, min(0.999, frame.latent_reuse_denoise))
        if frame.reuse_previous_latent and not frame.stream_diffusion:
            return frame.strength
        return base_strength

    def _remember_realtime_latents(self, frame: InpaintFrame, latents) -> None:
        if frame.stream_diffusion:
            self._latent_cache = None
            self._latent_stream = None
            return
        if not frame.reuse_previous_latent:
            self._latent_cache = None
            self._latent_stream = None
            return
        try:
            import torch

            if isinstance(latents, torch.Tensor):
                signature = self._latent_signature(frame)
                saved = latents.detach().clone()
                self._latent_cache = {"signature": signature, "latents": saved}
                stream_latents = []
                if self._latent_stream and self._latent_stream.get("signature") == signature:
                    existing = self._latent_stream.get("latents")
                    if isinstance(existing, list):
                        stream_latents = [item for item in existing if isinstance(item, torch.Tensor)]
                stream_latents.append(saved)
                stream_latents = stream_latents[-self._latent_stream_length(frame) :]
                self._latent_stream = {"signature": signature, "latents": stream_latents}
                return
        except Exception:
            pass
        self._latent_cache = None
        self._latent_stream = None

    def _diffusers_layers(
        self,
        frame: LayerGenerateFrame,
        seeds: list[int],
        progress: ProgressCallback | None = None,
        on_chunk: ChunkCallback | None = None,
    ) -> list[Image.Image]:
        import torch

        if frame.transparent_background:
            prompt = f"{frame.prompt}, isolated foreground subject, transparent background, no scenery, no frame"
            negative_prompt = ", ".join(
                part for part in [frame.negative_prompt, "busy background, scenery, environment, opaque background, border"] if part
            )
        else:
            prompt = frame.prompt
            negative_prompt = frame.negative_prompt
        guidance_scale = 0.0 if "turbo" in self.mode.lower() else frame.cfg
        effective_steps = max(frame.steps, ceil(1 / 0.999))
        total = max(1, len(seeds))
        batch_size_limit = max(
            1,
            self.config.transparent_layer_batch_size if frame.transparent_background else self.config.layer_batch_size,
        )
        results: list[Image.Image] = []
        text_pipe = self._text_to_image_pipe()
        self._emit_progress(progress, "queued", 0.1, f"Generating {len(seeds)} variation(s) in batches of {batch_size_limit}")

        for start in range(0, len(seeds), batch_size_limit):
            chunk_seeds = seeds[start : start + batch_size_limit]
            chunk_size = len(chunk_seeds)
            generators = [torch.Generator(device=self.config.device).manual_seed(seed) for seed in chunk_seeds]
            chunk_start = start / total
            chunk_span = chunk_size / total
            diffusion_start = 0.1 + chunk_start * 0.72
            diffusion_span = chunk_span * 0.72
            decode_start = diffusion_start + diffusion_span
            decode_span = chunk_span * 0.18
            self._emit_progress(progress, "sampling", diffusion_start, f"Sampling variations {start + 1}-{start + chunk_size}")

            def step_progress(step: int) -> None:
                step_value = (step + 1) / max(1, effective_steps)
                self._emit_progress(
                    progress,
                    "sampling",
                    diffusion_start + diffusion_span * step_value,
                    f"Sampling {start + chunk_size}/{total} | step {step + 1}/{effective_steps}",
                )

            transparent_method = self._transparent_layer_method(frame)
            use_layerdiffuse = frame.transparent_background and transparent_method == "layerdiffuse" and self.pipeline_family != "z-image"
            output_type = "latent" if use_layerdiffuse else "pil"
            with torch.inference_mode():
                if text_pipe is not None:
                    generated_output = self._call_layer_pipe(
                        text_pipe,
                        {
                            "prompt": [prompt] * chunk_size,
                            "negative_prompt": [negative_prompt] * chunk_size if negative_prompt else None,
                            "guidance_scale": guidance_scale,
                            "num_inference_steps": effective_steps,
                            "width": frame.width,
                            "height": frame.height,
                            "generator": generators,
                            "output_type": output_type,
                        },
                        step_progress,
                    ).images
                elif getattr(self, "pipeline_kind", "image2image") == "inpaint":
                    neutral = Image.new("RGB", (frame.width, frame.height), (127, 127, 127))
                    mask = Image.new("L", (frame.width, frame.height), 255)
                    generated_output = self._call_layer_pipe(
                        self.pipe,
                        {
                            "prompt": [prompt] * chunk_size,
                            "negative_prompt": [negative_prompt] * chunk_size if negative_prompt else None,
                            "image": [neutral] * chunk_size,
                            "mask_image": [mask] * chunk_size,
                            "strength": 0.999,
                            "guidance_scale": guidance_scale,
                            "num_inference_steps": effective_steps,
                            "width": frame.width,
                            "height": frame.height,
                            "generator": generators,
                            "output_type": output_type,
                        },
                        step_progress,
                    ).images
                else:
                    neutral = Image.new("RGB", (frame.width, frame.height), (127, 127, 127))
                    generated_output = self._call_layer_pipe(
                        self.pipe,
                        {
                            "prompt": [prompt] * chunk_size,
                            "negative_prompt": [negative_prompt] * chunk_size if negative_prompt else None,
                            "image": [neutral] * chunk_size,
                            "strength": 0.999,
                            "guidance_scale": guidance_scale,
                            "num_inference_steps": effective_steps,
                            "width": frame.width,
                            "height": frame.height,
                            "generator": generators,
                            "output_type": output_type,
                        },
                        step_progress,
                    ).images
            if use_layerdiffuse:
                latents = generated_output if isinstance(generated_output, torch.Tensor) else generated_output[0]
                if latents.ndim == 3:
                    latents = latents.unsqueeze(0)
                self._emit_progress(progress, "decoding", decode_start, f"Decoding variations {start + 1}-{start + chunk_size}")
                image_tensor = self._decode_latents_to_tensor(latents)
                decoded_rgb = [
                    Image.fromarray((item.permute(1, 2, 0).detach().float().cpu().numpy() * 255).round().astype("uint8"), mode="RGB")
                    for item in image_tensor
                ]
                if self.layer_decoder is None:
                    raise RuntimeError("LayerDiffuse transparent decoder is not available")
                transparent = self.layer_decoder.decode_rgba_batch(image_tensor, latents, augmented=True)
                chunk_results = [self._repair_layer_result(rgba, rgb) for rgba, rgb in zip(transparent, decoded_rgb)]
                del image_tensor, latents
            else:
                self._emit_progress(progress, "decoding", decode_start, f"Preparing opaque variations {start + 1}-{start + chunk_size}")
                pil_images = generated_output if isinstance(generated_output, list) else [generated_output]
                if frame.transparent_background:
                    chunk_results = [self._apply_transparent_background(image.convert("RGB"), frame) for image in pil_images]
                else:
                    chunk_results = [image.convert("RGBA") for image in pil_images]
            results.extend(chunk_results)
            self._emit_progress(progress, "decoding", decode_start + decode_span, f"Decoded {start + chunk_size}/{total} variation(s)")
            if on_chunk is not None:
                on_chunk(
                    list(zip(chunk_seeds, chunk_results)),
                    decode_start + decode_span,
                    f"Displaying {start + chunk_size}/{total} variation(s)",
                )
            if is_cuda_device(self.config.device):
                torch.cuda.empty_cache()
        return results

    def _transparent_layer_method(self, frame: LayerGenerateFrame) -> str:
        method = frame.transparent_background_method.lower().strip()
        if method == "layerdiffuse":
            return "layerdiffuse"
        if method == "fast":
            return "fast"
        if self.pipeline_family == "z-image" or not is_cuda_device(self.config.device):
            return "fast"
        try:
            import torch

            device = torch.device(self.config.device)
            index = device.index if device.index is not None else torch.cuda.current_device()
            total_vram = torch.cuda.get_device_properties(index).total_memory
            if total_vram >= 16 * 1024**3:
                return "layerdiffuse"
        except Exception:
            logger.exception("Could not inspect CUDA memory for transparent layer method")
        return "fast"

    @staticmethod
    def _flowmatch_scheduler_class_name(pipe) -> str:
        return pipe.scheduler.__class__.__name__ if pipe is not None and hasattr(pipe, "scheduler") else ""

    def _refresh_flowmatch_scheduler(self, pipe) -> None:
        if pipe is None or not hasattr(pipe, "scheduler"):
            return
        if not self._flowmatch_scheduler_class_name(pipe).startswith("FlowMatch"):
            return
        scheduler_class = pipe.scheduler.__class__
        config = dict(pipe.scheduler.config)
        config.update({"use_karras_sigmas": False, "use_exponential_sigmas": False, "use_beta_sigmas": False})
        try:
            pipe.scheduler = scheduler_class.from_config(config)
        except Exception:
            logger.exception("Could not refresh FlowMatch scheduler before generation")

    def _call_layer_pipe(self, pipe, kwargs: dict[str, object], step_progress: Callable[[int], None]):
        import torch

        def on_step_end(_pipeline, step: int, _timestep, callback_kwargs: dict[str, object]):
            step_progress(step)
            return callback_kwargs

        if self.pipeline_family == "z-image" and getattr(pipe, "__class__", object).__name__.endswith("InpaintPipeline"):
            kwargs["strength"] = 1.0
        self._refresh_flowmatch_scheduler(pipe)
        with torch.inference_mode():
            try:
                return pipe(
                    **kwargs,
                    callback_on_step_end=on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
            except TypeError:
                result = pipe(**kwargs)
                step_progress(int(kwargs.get("num_inference_steps", 1)) - 1)
                return result

    def _text_to_image_pipe(self):
        if self.pipe is None or getattr(self, "pipeline_kind", "none") == "inpaint":
            return None
        if self.layer_pipe is not None:
            return self.layer_pipe
        try:
            if self.pipeline_family == "z-image":
                from diffusers import ZImagePipeline

                self.layer_pipe = ZImagePipeline.from_pipe(self.pipe)
                self.layer_pipe.set_progress_bar_config(disable=True)
                return self.layer_pipe
            from diffusers import AutoPipelineForText2Image

            self.layer_pipe = AutoPipelineForText2Image.from_pipe(self.pipe)
            self.layer_pipe.set_progress_bar_config(disable=True)
            return self.layer_pipe
        except Exception:
            logger.exception("Could not create text-to-image pipeline from active pipeline")
            self.layer_pipe = None
            return None

    def _repair_layer_result(self, rgba: Image.Image, rgb: Image.Image) -> Image.Image:
        alpha = rgba.getchannel("A")
        extrema = alpha.getextrema()
        if extrema[0] == 255 and extrema[1] == 255:
            return self._cleanup_sprite_alpha(self._remove_light_background(rgb), rgb)
        bounds = alpha.getbbox()
        if bounds is None:
            return self._cleanup_sprite_alpha(self._remove_light_background(rgb), rgb)
        return self._cleanup_sprite_alpha(rgba, rgb)

    def _cleanup_sprite_alpha(self, rgba: Image.Image, rgb: Image.Image) -> Image.Image:
        try:
            import numpy as np
        except Exception:
            logger.exception("NumPy is unavailable for sprite alpha cleanup")
            return rgba

        rgba_array = np.asarray(rgba.convert("RGBA")).copy()
        rgb_array = np.asarray(rgb.convert("RGB"), dtype=np.float32)
        alpha = rgba_array[:, :, 3].astype(np.float32)
        height, width = alpha.shape
        border = max(4, min(16, min(width, height) // 24))
        border_pixels = np.concatenate(
            [
                rgb_array[:border, :, :].reshape(-1, 3),
                rgb_array[-border:, :, :].reshape(-1, 3),
                rgb_array[:, :border, :].reshape(-1, 3),
                rgb_array[:, -border:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(border_pixels, axis=0)
        distance = np.linalg.norm(rgb_array - background, axis=2)
        lightness = rgb_array.mean(axis=2)
        chroma = rgb_array.max(axis=2) - rgb_array.min(axis=2)
        border_like = (distance < 34) & (alpha < 248)
        pale_haze = (lightness > 214) & (chroma < 48) & (alpha < 235)
        weak_alpha = alpha < 28
        fade_mask = border_like | pale_haze | weak_alpha
        fade_strength = np.ones_like(alpha)
        fade_strength = np.where(distance < 16, np.minimum(fade_strength, distance / 16), fade_strength)
        fade_strength = np.where(pale_haze, np.minimum(fade_strength, np.clip((255 - lightness) / 42, 0, 1)), fade_strength)
        fade_strength = np.where(weak_alpha, 0, fade_strength)
        alpha = np.where(fade_mask, alpha * fade_strength, alpha)
        alpha = np.where(alpha < 10, 0, alpha)
        rgba_array[:, :, 3] = np.clip(alpha, 0, 255).astype("uint8")
        cleaned = Image.fromarray(rgba_array, mode="RGBA")
        cleaned_alpha = cleaned.getchannel("A")
        if cleaned_alpha.getbbox() is None:
            return rgba
        return cleaned

    def _apply_transparent_background(self, image: Image.Image, frame: InpaintFrame | LayerGenerateFrame) -> Image.Image:
        if not frame.transparent_background:
            return image
        try:
            import numpy as np
        except Exception:
            logger.exception("NumPy is unavailable for transparent background extraction")
            return image.convert("RGBA")

        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width, _channels = rgb.shape
        mode = frame.transparent_background_mode.lower().strip()
        if mode == "color":
            background = np.asarray(self._parse_hex_color(frame.transparent_background_color), dtype=np.float32)
        else:
            border = max(4, min(24, min(width, height) // 20))
            border_pixels = np.concatenate(
                [
                    rgb[:border, :, :].reshape(-1, 3),
                    rgb[-border:, :, :].reshape(-1, 3),
                    rgb[:, :border, :].reshape(-1, 3),
                    rgb[:, -border:, :].reshape(-1, 3),
                ],
                axis=0,
            )
            background = np.median(border_pixels, axis=0)
        distance = np.linalg.norm(rgb - background, axis=2)
        tolerance = max(1.0, float(frame.transparent_background_tolerance))
        alpha = np.clip((distance - tolerance * 0.45) / max(1.0, tolerance * 1.4), 0, 1) * 255
        alpha = np.where(alpha < frame.transparent_alpha_threshold, 0, alpha)
        alpha_image = Image.fromarray(alpha.astype("uint8"), mode="L")
        if frame.transparent_alpha_blur > 0:
            alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(frame.transparent_alpha_blur))
        rgba = image.convert("RGBA")
        rgba.putalpha(alpha_image)
        return self._cleanup_sprite_alpha(rgba, image.convert("RGB"))

    @staticmethod
    def _parse_hex_color(value: str) -> tuple[int, int, int]:
        text = value.strip().lstrip("#")
        if len(text) == 3:
            text = "".join(character * 2 for character in text)
        if len(text) != 6:
            return (255, 255, 255)
        try:
            return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
        except ValueError:
            return (255, 255, 255)

    def _decode_latents_to_tensor(self, latents, vae=None):
        import torch

        vae = vae or self.pipe.vae
        needs_upcast = getattr(vae.config, "force_upcast", False) and vae.dtype == torch.float16
        if needs_upcast and vae is self.pipe.vae and hasattr(self.pipe, "upcast_vae"):
            self.pipe.upcast_vae()
        vae_dtype = next(vae.parameters()).dtype
        vae_device = next(vae.parameters()).device
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
        latents = latents.to(device=vae_device, dtype=vae_dtype)
        decoded = vae.decode(latents / scaling_factor, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        if needs_upcast and vae is self.pipe.vae:
            vae.to(dtype=torch.float16)
        return decoded

    def _mock_layer(self, frame: LayerGenerateFrame, seed: int) -> Image.Image:
        image = Image.new("RGBA", (frame.width, frame.height), (0, 0, 0, 0))
        hue = (sum(ord(character) for character in frame.prompt) + seed) % 360
        color = self._hsl_to_hex(hue)
        if not frame.transparent_background:
            return Image.new("RGBA", image.size, color)
        mask = Image.new("L", (frame.width, frame.height), 0)
        center = (frame.width // 2, frame.height // 2)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            [center[0] - frame.width // 5, center[1] - frame.height // 3, center[0] + frame.width // 5, center[1] + frame.height // 3],
            fill=255,
        )
        layer = Image.new("RGBA", image.size, color)
        layer.putalpha(mask.filter(ImageFilter.GaussianBlur(2)))
        return layer

    def _remove_light_background(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        data = rgba.load()
        width, height = rgba.size
        for y in range(height):
            for x in range(width):
                red, green, blue, alpha = data[x, y]
                lightness = (red + green + blue) / 3
                chroma = max(red, green, blue) - min(red, green, blue)
                if lightness > 222 and chroma < 42:
                    fade = max(0, min(255, int((255 - lightness) * 7)))
                    data[x, y] = (red, green, blue, min(alpha, fade))
        return rgba

    def _mock_inpaint(self, image: Image.Image, mask: Image.Image, prompt: str) -> Image.Image:
        prompt_hash = sum(ord(character) for character in prompt) % 360
        hue_layer = ImageOps.colorize(mask, black="#000000", white=self._hsl_to_hex(prompt_hash))
        softened_mask = mask.filter(ImageFilter.GaussianBlur(10))
        detailed = ImageEnhance.Sharpness(image).enhance(1.4)
        tinted = Image.blend(detailed, hue_layer, 0.32)
        return Image.composite(tinted, image, softened_mask)

    @staticmethod
    def _hsl_to_hex(hue: int) -> str:
        chroma = 0.55
        x_value = chroma * (1 - abs((hue / 60) % 2 - 1))
        match int(hue / 60):
            case 0:
                red, green, blue = chroma, x_value, 0
            case 1:
                red, green, blue = x_value, chroma, 0
            case 2:
                red, green, blue = 0, chroma, x_value
            case 3:
                red, green, blue = 0, x_value, chroma
            case 4:
                red, green, blue = x_value, 0, chroma
            case _:
                red, green, blue = chroma, 0, x_value
        lightness_offset = 0.28
        return "#{:02x}{:02x}{:02x}".format(
            int((red + lightness_offset) * 255),
            int((green + lightness_offset) * 255),
            int((blue + lightness_offset) * 255),
        )