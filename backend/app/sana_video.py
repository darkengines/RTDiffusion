"""
SANA-Video image-to-video pipeline.

Three modes:
────────────
1. SANA-Video standard  — high-quality short clips (5-10 s) from image + prompt
                          Uses SanaImageToVideoPipeline (I2V).
2. SANA-Video LongLive  — autoregressive streaming video: each chunk conditions
                          the next, enabling minute-long coherent generation.
                          Also uses SanaImageToVideoPipeline with a causal model.
3. SANA-WM scaffold     — world model with 6-DoF camera control (ready for
                          when weights are publicly available).

Pipeline API notes (diffusers):
  • Correct parameter is `frames=` NOT `num_frames=`.
  • I2V pipeline: SanaImageToVideoPipeline — takes `image=` as first argument.
  • T2V pipeline: SanaVideoPipeline — no image conditioning.
  • LongLive model config uses SanaVideoCausalTransformer3DModel which is
    aliased to SanaVideoTransformer3DModel in the current diffusers version.

HuggingFace model IDs
─────────────────────
  SANA-Video 480p  (fast, lower res)
    Efficient-Large-Model/SANA-Video_2B_480p_diffusers
  SANA-Video 720p  (high quality)
    Efficient-Large-Model/SANA-Video_2B_720p_diffusers
  SANA-Video LongLive 480p  (streaming, causal)
    Efficient-Large-Model/SANA-Video_2B_480p_LongLive_diffusers

Typical latency (RTX 6000 Ada, BF16):
  480p  5 s clip  ≈ 40-70 s
  720p  5 s clip  ≈ 90-150 s
  LongLive (chunked)  ≈ 30-50 s/chunk
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

logger = logging.getLogger("rtdiffusion.sana_video")

_DEFAULT_480P  = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
_DEFAULT_720P  = "Efficient-Large-Model/SANA-Video_2B_720p_diffusers"
_LONGIVE_480P  = "Efficient-Large-Model/SANA-Video_2B_480p_LongLive_diffusers"


def _patch_diffusers_sana_aliases() -> None:
    """
    The LongLive model config references SanaVideoCausalTransformer3DModel
    which was renamed to SanaVideoTransformer3DModel in current diffusers.
    Patch the top-level diffusers module so from_pretrained resolves the class
    via getattr(diffusers, classname) — patching diffusers.models alone is not enough.
    """
    try:
        import diffusers as _diffusers_module
        transformer_cls = getattr(_diffusers_module, "SanaVideoTransformer3DModel", None)
        if transformer_cls is None:
            try:
                from diffusers.models import SanaVideoTransformer3DModel as transformer_cls
            except Exception:
                from diffusers.models.transformers.sana_transformer import SanaVideoTransformer3DModel as transformer_cls
        for module_name in (
            "diffusers",
            "diffusers.models",
            "diffusers.models.transformers",
            "diffusers.models.transformers.sana_transformer",
        ):
            try:
                import importlib
                module = importlib.import_module(module_name)
                if not hasattr(module, "SanaVideoCausalTransformer3DModel"):
                    setattr(module, "SanaVideoCausalTransformer3DModel", transformer_cls)
            except Exception:
                continue
        logger.debug("Patched SanaVideoCausalTransformer3DModel alias in diffusers")
    except Exception as exc:
        logger.debug("Could not patch SanaVideoCausalTransformer3DModel alias: %s", exc)


def _nearest_valid32(value: int) -> int:
    """Round to the closest positive multiple of 32 for SANA video pipelines."""
    value = max(32, int(value))
    lower = max(32, (value // 32) * 32)
    upper = ((value + 31) // 32) * 32
    if value - lower <= upper - value:
        return lower
    return upper


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.size == (width, height):
        return image.convert("RGB")
    return ImageOps.fit(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _sana_geometry(width: int, height: int) -> tuple[int, int]:
    return _nearest_valid32(width), _nearest_valid32(height)


def _fit_frames(frames: list[Image.Image], width: int, height: int) -> list[Image.Image]:
    return [_fit_image(frame, width, height) for frame in frames]


# ──────────────────────────────────────────────────────────────────
# SANA-Video standard clip generation (image-to-video)
# ──────────────────────────────────────────────────────────────────

class SanaVideoGenerator:
    """
    One-shot img-to-video generation using SanaImageToVideoPipeline.

    generate() runs in a thread pool; call it via asyncio.to_thread().
    Progress callbacks fire with (phase: str, fraction: float, message: str).
    """

    def __init__(self, model_id: str = _DEFAULT_480P, device: str = "cuda:0") -> None:
        self._model_id = model_id
        self._device = device
        self._pipe: Any = None
        self._lock = threading.Lock()

    def ensure_loaded(self) -> bool:
        if self._pipe is not None:
            return True
        self._pipe = _load_i2v_pipe(self._model_id, self._device)
        return self._pipe is not None

    def generate(
        self,
        *,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 81,        # 81 = ~3.4s at 24fps; 121 = ~5s
        fps: int = 24,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 20,
        width: int = 832,
        height: int = 480,
        seed: int | None = None,
        motion_score: int | None = 30,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> Path | None:
        """
        Generate a video clip from an image and return the path to the saved MP4.
        Returns None on failure.

        motion_score: optional hint embedded in prompt (1-100, higher = more motion).
        """
        with self._lock:
            if not self.ensure_loaded():
                return None

            import torch

            device = self._device
            generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

            full_prompt = prompt
            if motion_score is not None:
                full_prompt = f"{prompt} motion score: {motion_score}.".strip()

            def _step_callback(pipe, step, timestep, kwargs):
                if progress:
                    frac = step / max(1, num_inference_steps)
                    progress("denoise", frac, f"Denoising step {step}/{num_inference_steps}")
                return kwargs

            sana_w, sana_h = _sana_geometry(width, height)
            sana_image = _fit_image(image, sana_w, sana_h)
            if sana_w != width or sana_h != height:
                logger.debug("SanaVideoGenerator: fitting %dx%d through valid SANA geometry %dx%d", width, height, sana_w, sana_h)

            if progress:
                progress("encode", 0.02, "Encoding image to video latent space")

            try:
                result = self._pipe(
                    image=sana_image,
                    prompt=full_prompt,
                    negative_prompt=negative_prompt or None,
                    frames=num_frames,          # NOTE: 'frames' not 'num_frames'
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=sana_w,
                    height=sana_h,
                    generator=generator,
                    callback_on_step_end=_step_callback,
                    output_type="pil",
                )
            except Exception:
                logger.exception("SanaVideoGenerator: generation failed")
                return None

            if progress:
                progress("encode_video", 0.9, "Encoding frames to video")

            raw_frames: list[Image.Image] = result.frames[0] if result.frames else []
            if not raw_frames:
                logger.error("SanaVideoGenerator: no frames returned")
                return None

            frames = _fit_frames(raw_frames, width, height)
            return _save_mp4(frames, fps=fps, progress=progress)


# ──────────────────────────────────────────────────────────────────
# SANA-Video LongLive: autoregressive streaming
# ──────────────────────────────────────────────────────────────────

class SanaLongLiveStream:
    """
    Continuous autoregressive video streaming using SANA-Video LongLive.

    Each call to next_chunk() generates the next segment conditioned on
    the last frame of the previous segment, creating seamless long-form
    video guided by a single initial image + evolving prompt.

    Usage:
        stream = SanaLongLiveStream(device="cuda:0")
        stream.start(init_image=canvas, prompt="...")
        while streaming:
            mp4_path = stream.next_chunk(prompt=updated_prompt)
            # send mp4_path to browser via HTTP or WebSocket
    """

    def __init__(self, model_id: str = _LONGIVE_480P, device: str = "cuda:0") -> None:
        self._model_id = model_id
        self._device = device
        self._pipe: Any = None
        self._last_frames: list[Any] | None = None  # last N frames for conditioning
        self._lock = threading.Lock()

    def ensure_loaded(self) -> bool:
        if self._pipe is not None:
            return True
        self._pipe = _load_i2v_pipe(self._model_id, self._device)
        return self._pipe is not None

    def start(
        self,
        *,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 81,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 20,
        width: int = 832,
        height: int = 480,
        seed: int | None = None,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> Path | None:
        """Generate first chunk from initial image. Returns path to MP4."""
        with self._lock:
            self._last_frames = None
            return self._generate_chunk(
                image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                seed=seed,
                progress=progress,
            )

    def next_chunk(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 81,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 20,
        width: int = 832,
        height: int = 480,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> Path | None:
        """Generate next chunk conditioned on last frame. Returns path to MP4."""
        with self._lock:
            if self._last_frames is None:
                logger.warning("SanaLongLiveStream: call start() before next_chunk()")
                return None
            last_frame = self._last_frames[-1] if self._last_frames else None
            if last_frame is None:
                return None
            return self._generate_chunk(
                image=last_frame,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                seed=None,
                progress=progress,
            )

    def _generate_chunk(
        self,
        *,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        guidance_scale: float,
        num_inference_steps: int,
        width: int,
        height: int,
        seed: int | None,
        progress: Callable[[str, float, str], None] | None,
    ) -> Path | None:
        if not self.ensure_loaded():
            return None

        import torch

        device = self._device
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        def _step_callback(pipe, step, timestep, kwargs):
            if progress:
                frac = step / max(1, num_inference_steps)
                progress("denoise", frac, f"Step {step}/{num_inference_steps}")
            return kwargs

        sana_w, sana_h = _sana_geometry(width, height)
        sana_image = _fit_image(image, sana_w, sana_h)
        if sana_w != width or sana_h != height:
            logger.debug("SanaLongLiveStream: fitting %dx%d through valid SANA geometry %dx%d", width, height, sana_w, sana_h)

        try:
            result = self._pipe(
                image=sana_image,
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                frames=num_frames,              # NOTE: 'frames' not 'num_frames'
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=sana_w,
                height=sana_h,
                generator=generator,
                callback_on_step_end=_step_callback,
                output_type="pil",
            )
        except Exception:
            logger.exception("SanaLongLiveStream: chunk generation failed")
            return None

        raw_frames: list[Image.Image] = result.frames[0] if result.frames else []
        if not raw_frames:
            return None

        frames = _fit_frames(raw_frames, width, height)

        # Cache last 8 frames for next chunk conditioning (use last frame as init)
        self._last_frames = frames[-8:]

        return _save_mp4(frames, fps=24, progress=progress)


# ──────────────────────────────────────────────────────────────────
# SANA-WM scaffold (World Model)
# ──────────────────────────────────────────────────────────────────

class SanaWMGenerator:
    """
    SANA World Model — 2.6B parameters with 6-DoF camera control.

    Generates video with explicit camera trajectory (pan, tilt, roll,
    x/y/z translation). Enables interactive world exploration from a
    single image.

    Model: Efficient-Large-Model/SANA-WM-2.6B (available May 2026+)

    Camera pose format: list of [tx, ty, tz, rx, ry, rz] per frame,
    where translations are in scene units and rotations are in radians.
    """

    _WM_MODEL = "Efficient-Large-Model/SANA-WM-2.6B"

    def __init__(self, device: str = "cuda:0") -> None:
        self._device = device
        self._pipe: Any = None

    def ensure_loaded(self) -> bool:
        if self._pipe is not None:
            return True
        # Try I2V pipeline first (likely WM is also I2V-based), then T2V
        self._pipe = _load_i2v_pipe(self._WM_MODEL, self._device)
        if self._pipe is None:
            self._pipe = _load_t2v_pipe(self._WM_MODEL, self._device)
        return self._pipe is not None

    def generate(
        self,
        *,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        camera_poses: list[list[float]] | None = None,
        num_frames: int = 81,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 20,
        width: int = 832,
        height: int = 480,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> Path | None:
        """
        Generate video with camera control.

        camera_poses: list of [tx, ty, tz, rx, ry, rz] (one per frame).
                      None = no explicit camera control (free generation).
        """
        sana_w, sana_h = _sana_geometry(width, height)
        sana_image = _fit_image(image, sana_w, sana_h)
        if sana_w != width or sana_h != height:
            logger.debug("SanaWMGenerator: fitting %dx%d through valid SANA geometry %dx%d", width, height, sana_w, sana_h)

        if not self.ensure_loaded():
            logger.warning("SanaWMGenerator: model unavailable — falling back to standard SANA-Video")
            fallback = SanaVideoGenerator(model_id=_DEFAULT_480P, device=self._device)
            return fallback.generate(
                image=image, prompt=prompt, negative_prompt=negative_prompt,
                num_frames=num_frames, guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width, height=height, progress=progress,
            )

        import torch

        def _step_callback(pipe, step, timestep, kwargs):
            if progress:
                progress("denoise", step / max(1, num_inference_steps), f"Step {step}")
            return kwargs

        # Build kwargs — try I2V signature first, then T2V
        has_image_param = _pipe_accepts_image(self._pipe)
        kw: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "frames": num_frames,           # NOTE: 'frames' not 'num_frames'
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "width": sana_w,
            "height": sana_h,
            "callback_on_step_end": _step_callback,
            "output_type": "pil",
        }
        if has_image_param:
            kw["image"] = sana_image

        # Inject camera poses if the pipeline supports them
        if camera_poses is not None:
            cam_tensor = torch.tensor(camera_poses, dtype=torch.bfloat16, device=self._device)
            kw["camera_poses"] = cam_tensor

        try:
            result = self._pipe(**kw)
        except TypeError:
            # Pipeline may not support camera_poses or image yet — retry stripped down
            kw.pop("camera_poses", None)
            kw.pop("image", None)
            try:
                result = self._pipe(**kw)
            except Exception:
                logger.exception("SanaWMGenerator: generation failed")
                return None
        except Exception:
            logger.exception("SanaWMGenerator: generation failed")
            return None

        raw_frames: list[Image.Image] = result.frames[0] if result.frames else []
        frames = _fit_frames(raw_frames, width, height)
        return _save_mp4(frames, fps=24, progress=progress) if frames else None


# ──────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────

def _pipe_accepts_image(pipe: Any) -> bool:
    """Return True if the pipeline's __call__ has an 'image' parameter."""
    try:
        import inspect
        sig = inspect.signature(pipe.__call__)
        return "image" in sig.parameters
    except Exception:
        return False


def _load_i2v_pipe(model_id: str, device: str) -> Any | None:
    """Load SanaImageToVideoPipeline (I2V). Patches class aliases first."""
    _patch_diffusers_sana_aliases()
    import torch

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    logger.info("SanaVideo I2V: loading %s on %s …", model_id, device)
    t0 = time.perf_counter()

    try:
        from diffusers import SanaImageToVideoPipeline

        pipe = SanaImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
        pipe.set_progress_bar_config(disable=True)

        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True

        elapsed = time.perf_counter() - t0
        logger.info("SanaVideo I2V: pipe ready in %.1fs", elapsed)
        return pipe

    except ImportError:
        logger.error(
            "SanaVideo: SanaImageToVideoPipeline not found — upgrade diffusers: pip install -U diffusers"
        )
    except Exception:
        logger.exception("SanaVideo: I2V load failed for %s", model_id)
        _cleanup_gpu(device)
    return None


def _load_t2v_pipe(model_id: str, device: str) -> Any | None:
    """Load SanaVideoPipeline (T2V, text-to-video, no image conditioning)."""
    _patch_diffusers_sana_aliases()
    import torch

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    logger.info("SanaVideo T2V: loading %s on %s …", model_id, device)
    t0 = time.perf_counter()

    try:
        from diffusers import SanaVideoPipeline

        pipe = SanaVideoPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
        pipe.set_progress_bar_config(disable=True)

        elapsed = time.perf_counter() - t0
        logger.info("SanaVideo T2V: pipe ready in %.1fs", elapsed)
        return pipe

    except ImportError:
        logger.error(
            "SanaVideo: SanaVideoPipeline not found — upgrade diffusers: pip install -U diffusers"
        )
    except Exception:
        logger.exception("SanaVideo: T2V load failed for %s", model_id)
        _cleanup_gpu(device)
    return None


def _cleanup_gpu(device: str) -> None:
    gc.collect()
    try:
        import torch
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass


def _save_mp4(
    frames: list[Image.Image],
    fps: int = 24,
    progress: Callable[[str, float, str], None] | None = None,
) -> Path | None:
    if not frames:
        return None

    out_dir = Path(os.getenv("RTD_VIDEO_DIR", tempfile.gettempdir())) / "rtdiffusion_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    import time as _t
    out_path = out_dir / f"sana_{int(_t.time() * 1000)}.mp4"

    try:
        import av

        container = av.open(str(out_path), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}

        for i, frame in enumerate(frames):
            if progress and i % 10 == 0:
                progress("encode_video", 0.9 + 0.09 * i / len(frames), f"Encoding frame {i}/{len(frames)}")
            av_frame = av.VideoFrame.from_image(frame.convert("RGB"))
            av_frame = av_frame.reformat(format="yuv420p")
            for pkt in stream.encode(av_frame):
                container.mux(pkt)

        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

        logger.info("SanaVideo: saved %d frames → %s", len(frames), out_path)
        if progress:
            progress("done", 1.0, f"Saved {len(frames)} frames")
        return out_path

    except Exception:
        logger.exception("SanaVideo: MP4 encoding failed")
        return None


# ──────────────────────────────────────────────────────────────────
# Module-level singletons (lazy — only load on first request)
# ──────────────────────────────────────────────────────────────────

_video_generator: SanaVideoGenerator | None = None
_longive_stream: SanaLongLiveStream | None = None
_wm_generator: SanaWMGenerator | None = None
_singleton_lock = threading.Lock()


def get_video_generator(model_id: str | None = None, device: str = "cuda:0") -> SanaVideoGenerator:
    global _video_generator
    with _singleton_lock:
        mid = model_id or _DEFAULT_480P
        if _video_generator is None or _video_generator._model_id != mid:
            _video_generator = SanaVideoGenerator(model_id=mid, device=device)
        return _video_generator


def get_longive_stream(device: str = "cuda:0") -> SanaLongLiveStream:
    global _longive_stream
    with _singleton_lock:
        if _longive_stream is None:
            _longive_stream = SanaLongLiveStream(device=device)
        return _longive_stream


def get_wm_generator(device: str = "cuda:0") -> SanaWMGenerator:
    global _wm_generator
    with _singleton_lock:
        if _wm_generator is None:
            _wm_generator = SanaWMGenerator(device=device)
        return _wm_generator
