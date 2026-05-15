import os
import logging
import gc
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Callable

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

from .assets import default_model_path
from .image_io import alpha_to_empty_mask, decode_data_url, decode_data_url_rgba, encode_data_url, mask_to_luma, rgba_to_neutral_rgb
from .layerdiffuse import LayerDiffuseDecoder
from .schemas import InpaintFrame, InpaintResult, LayerGenerateFrame, LayerGenerateResult, LayerVariation

logger = logging.getLogger("rtdiffusion.engine")


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


ProgressCallback = Callable[[str, float, str, LayerGenerateResult | None], None]
ChunkCallback = Callable[[list[tuple[int, Image.Image]], float, str], None]


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
        self.layer_decoder: LayerDiffuseDecoder | None = None
        self.scheduler_config: dict[str, object] | None = None
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

            if self.config.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            model_id = self.config.model_id or default_model_path()
            is_z_image = self._looks_like_z_image_model(model_id)
            if is_z_image and self.config.device == "cuda":
                dtype = torch.bfloat16
            elif self.config.device == "cuda":
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
                self.pipeline_family = "sd"
            else:
                try:
                    pipe = self._load_pipe(AutoPipelineForInpainting, model_id, dtype, variant)
                    self.pipeline_kind = "inpaint"
                except Exception:
                    pipe = self._load_pipe(AutoPipelineForImage2Image, model_id, dtype, variant)
                    self.pipeline_kind = "image2image"
                self.pipeline_family = "sd"
            pipe = pipe.to(self.config.device)
            pipe.set_progress_bar_config(disable=True)
            if self.config.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            if self.config.device == "cuda" and hasattr(pipe, "unet"):
                pipe.unet.to(memory_format=torch.channels_last)
            if self.config.compile_unet and hasattr(torch, "compile"):
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            self.pipe = pipe
            self.layer_pipe = None
            self.layer_decoder = LayerDiffuseDecoder(self.config.device, dtype)
            self.scheduler_config = dict(pipe.scheduler.config) if hasattr(pipe, "scheduler") else None
            self.model_id = model_id
            family = f"{self.pipeline_family}/" if self.pipeline_family != "sd" else ""
            self.mode = f"{family}{self.pipeline_kind}: {model_id}"
        except Exception:
            logger.exception("Failed to load diffusion pipeline")
            raise

    @staticmethod
    def _load_pipe(pipeline_class, model_id: str, dtype, variant: str | None):
        model_path = Path(model_id)
        if model_path.is_dir():
            return pipeline_class.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                variant=variant,
                local_files_only=True,
            )
        return pipeline_class.from_pretrained(model_id, torch_dtype=dtype, variant=variant)

    @staticmethod
    def _load_single_file_pipe(pipeline_class, model_id: str, dtype):
        model_path = Path(model_id)
        return pipeline_class.from_single_file(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=model_path.suffix.lower() == ".safetensors",
        )

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
        self._apply_sampler(frame.sampler, frame.scheduler)
        base_frame = self._frame_with_prompt_mix_conditions(frame)
        source = decode_data_url_rgba(frame.image).resize((frame.width, frame.height), Image.Resampling.LANCZOS)
        image = rgba_to_neutral_rgb(source)
        manual_mask = mask_to_luma(decode_data_url(frame.mask)).resize(
            (frame.width, frame.height), Image.Resampling.LANCZOS
        )
        empty_mask = alpha_to_empty_mask(source)
        mask = ImageChops.lighter(manual_mask, empty_mask)
        mask, base_frame = self._mask_with_layer_conditions(mask, base_frame)
        output = self._mock_inpaint(image, mask, base_frame.prompt)
        if self.pipe is not None:
            output = self._diffusers_inpaint(image, mask, base_frame)
        output = self._apply_zero_denoise_layer_constraints(output, frame)
        output = self._apply_layer_region_conditions(output, frame)
        latency_ms = (time.perf_counter() - started) * 1000
        fps = 1000 / latency_ms if latency_ms else 0
        return InpaintResult(
            image=encode_data_url(output),
            fps=fps,
            latency_ms=latency_ms,
            mode=self.mode,
        )

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
        mask_conditions = []
        effective_strength = frame.strength
        for condition in frame.layer_conditions:
            mode = condition.mode if condition.mode in {"mask", "add", "multiply", "override"} else "prompt_mix"
            if mode != "mask":
                continue
            schedule_influence = self._schedule_influence(condition.schedule, condition.schedule_start, condition.schedule_end)
            if schedule_influence <= 0:
                continue
            denoise = condition.denoise if condition.denoise is not None else frame.strength
            denoise = max(0.0, min(0.999, denoise))
            if denoise <= 0:
                continue
            try:
                condition_image = decode_data_url_rgba(condition.image).resize((frame.width, frame.height), Image.Resampling.LANCZOS)
            except Exception:
                logger.exception("Failed to decode layer mask condition image")
                continue
            alpha = condition_image.getchannel("A")
            if not alpha.getbbox():
                continue
            effective_strength = max(effective_strength, denoise)
            mask_conditions.append((alpha, condition.weight, schedule_influence, denoise))
        if not mask_conditions:
            return mask, frame
        effective_strength = max(effective_strength, 0.01)
        base_scale = frame.strength / effective_strength if effective_strength > 0 else 0.0

        def scale_base_mask(value: int) -> int:
            return int(max(0.0, min(1.0, (value / 255) * base_scale)) * 255)

        merged = mask.convert("L").point(scale_base_mask)
        for alpha, weight, schedule_influence, denoise in mask_conditions:
            scale = max(0.0, min(1.0, weight * schedule_influence * (denoise / effective_strength)))
            if scale <= 0:
                continue
            condition_mask = alpha.point(lambda value, scale=scale: int(max(0.0, min(1.0, (value / 255) * scale)) * 255))
            merged = ImageChops.lighter(merged, condition_mask)
        if effective_strength == frame.strength:
            return merged, frame
        return merged, frame.model_copy(update={"strength": effective_strength})

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

    def _apply_layer_region_conditions(self, output: Image.Image, frame: InpaintFrame) -> Image.Image:
        if not frame.layer_conditions:
            return output
        current = output.convert("RGB")
        for condition in frame.layer_conditions:
            mode = condition.mode if condition.mode in {"mask", "add", "multiply", "override"} else "prompt_mix"
            if mode in {"prompt_mix", "mask"}:
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
            condition_frame = frame.model_copy(
                update={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model_path": condition.model_path or frame.model_path,
                    "lora_paths": condition.lora_paths or frame.lora_paths,
                    "cfg": condition.cfg if condition.cfg is not None else frame.cfg,
                    "steps": condition.steps if condition.steps is not None else frame.steps,
                    "strength": condition.denoise if condition.denoise is not None else frame.strength,
                    "sampler": condition.sampler or frame.sampler,
                    "scheduler": condition.scheduler or frame.scheduler,
                }
            )
            self._ensure_runtime(condition_frame)
            self._apply_sampler(condition_frame.sampler, condition_frame.scheduler)
            regional = self._mock_inpaint(condition_source, influence_mask, condition_frame.prompt)
            if self.pipe is not None:
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
        next_model = frame.model_path or self.model_id
        if next_model and (next_model != self.model_id or self.pipe is None):
            old_pipe = self.pipe
            old_model_id = self.model_id
            old_mode = self.mode
            old_family = self.pipeline_family
            old_kind = self.pipeline_kind
            old_config_model = self.config.model_id
            old_loras = self.loaded_loras
            self.config.model_id = next_model
            self.pipe = None
            self.layer_pipe = None
            self.loaded_loras = ()
            try:
                self._load_pipeline()
            except Exception as exc:
                self.pipe = old_pipe
                self.layer_pipe = None
                self.model_id = old_model_id
                self.mode = old_mode
                self.pipeline_family = old_family
                self.pipeline_kind = old_kind
                self.scheduler_config = dict(old_pipe.scheduler.config) if old_pipe is not None and hasattr(old_pipe, "scheduler") else None
                self.config.model_id = old_config_model
                self.loaded_loras = old_loras
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
        gc.collect()
        try:
            import torch

            if self.config.device == "cuda":
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
        except Exception as exc:
            try:
                self._load_lora_set(previous_loras)
                self.loaded_loras = previous_loras
            except Exception:
                self.loaded_loras = ()
            self.mode = f"{self.pipeline_kind}: {self.model_id} | lora error: {exc.__class__.__name__}"
            raise RuntimeError(f"Could not load LoRA set: {exc}") from exc

    def _apply_sampler(self, sampler: str | None, scheduler: str | None = None) -> None:
        if not sampler or self.pipe is None or not hasattr(self.pipe, "scheduler"):
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
        if hasattr(self.pipe, "unload_lora_weights"):
            self.pipe.unload_lora_weights()
        adapter_names: list[str] = []
        for index, lora_path in enumerate(lora_paths):
            adapter_name = f"lora_{index}"
            self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            adapter_names.append(adapter_name)
        if adapter_names and hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters(adapter_names, adapter_weights=[1.0] * len(adapter_names))

    def _diffusers_inpaint(self, image: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image:
        import torch

        effective_steps = max(frame.steps, ceil(1 / max(frame.strength, 0.01)))
        guidance_scale = 0.0 if "turbo" in self.mode.lower() else frame.cfg
        inpaint_strength = frame.strength
        self._refresh_flowmatch_scheduler(self.pipe)
        with torch.inference_mode():
            if getattr(self, "pipeline_kind", "image2image") == "inpaint":
                generated = self.pipe(
                    prompt=frame.prompt,
                    negative_prompt=self._negative_prompt(frame.negative_prompt),
                    image=image,
                    mask_image=mask,
                    strength=inpaint_strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=effective_steps,
                    width=frame.width,
                    height=frame.height,
                ).images[0]
                return generated.convert("RGB")

            if getattr(self, "pipeline_kind", "image2image") == "text2image":
                generated = self.pipe(
                    prompt=frame.prompt,
                    negative_prompt=self._negative_prompt(frame.negative_prompt),
                    guidance_scale=guidance_scale,
                    num_inference_steps=effective_steps,
                    width=frame.width,
                    height=frame.height,
                ).images[0]
                return Image.composite(generated.convert("RGB"), image, mask)

            generated = self.pipe(
                prompt=frame.prompt,
                negative_prompt=self._negative_prompt(frame.negative_prompt),
                image=image,
                strength=frame.strength,
                guidance_scale=guidance_scale,
                num_inference_steps=effective_steps,
                width=frame.width,
                height=frame.height,
            ).images[0]
        return Image.composite(generated.convert("RGB"), image, mask)

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

            use_layerdiffuse = frame.transparent_background and self.pipeline_family != "z-image"
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
                    chunk_results = [self._cleanup_sprite_alpha(self._remove_light_background(image.convert("RGB")), image.convert("RGB")) for image in pil_images]
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
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
        return results

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

    def _decode_latents_to_tensor(self, latents):
        import torch

        vae = self.pipe.vae
        needs_upcast = getattr(vae.config, "force_upcast", False) and vae.dtype == torch.float16
        if needs_upcast and hasattr(self.pipe, "upcast_vae"):
            self.pipe.upcast_vae()
        vae_dtype = next(vae.parameters()).dtype
        vae_device = next(vae.parameters()).device
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
        latents = latents.to(device=vae_device, dtype=vae_dtype)
        decoded = vae.decode(latents / scaling_factor, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        if needs_upcast:
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