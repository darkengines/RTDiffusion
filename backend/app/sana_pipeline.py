"""
SANA-Sprint realtime img2img streaming session.

Architecture
────────────
Input image (PIL RGB)
  → DC-AE encode          (32× spatial compression → 16×16 latents at 512px)
  → Flow-matching noise   (adds noise at `strength`, so t=strength*1000)
  → SANA Linear Transformer (O(N) attention, 1-4 inference steps)
  → DC-AE decode
  → Output PIL image

Speed targets on RTX 6000 Ada (BF16, 512px, 2 steps):
  Sana_Sprint_0.6B  ≈ 0.10-0.15 s/frame → 7-10 FPS
  Sana_Sprint_1.6B  ≈ 0.20-0.30 s/frame → 3-5 FPS
  With torch.compile ≈ 1.5-2× faster

Key advantages over StreamDiffusion:
  • No rolling latent buffer — clean stateless img2img each frame
  • 16× fewer latent tokens (DC-AE 32× vs VAE 8×)
  • Linear attention O(N) — scales to higher res cheaply
  • Built-in img2img pipeline in diffusers (SanaSprintImg2ImgPipeline)

Models (all auto-downloaded from HuggingFace on first use):
  Sana_Sprint_0.6B_1024px_diffusers  — fastest, 0.6B params
  Sana_Sprint_1.6B_1024px_diffusers  — higher quality, 1.6B params

Usage in settings dict:
  model_type: "sana_sprint"
  model_path: "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
              (or any HuggingFace model ID that starts with "Efficient-Large-Model/Sana")
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Any

from PIL import Image

logger = logging.getLogger("rtdiffusion.sana")

_DEFAULT_MODEL_0_6B = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
_DEFAULT_MODEL_1_6B = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"

_REBUILD_DEBOUNCE_S = 0.35


def _log_sig_diff(old_sig: str | None, new_sig: str, log: Any, label: str) -> None:
    if old_sig is None:
        log.info("%s: first build", label)
        return
    try:
        old_t = eval(old_sig)  # noqa: S307
        new_t = eval(new_sig)  # noqa: S307
        fields = ("model", "device", "prompt", "negative_prompt", "cfg", "sana_steps", "dims")
        diffs = [fields[i] for i, (a, b) in enumerate(zip(old_t, new_t)) if a != b]
        log.info("%s: sig changed — fields: %s", label, diffs or "<unknown>")
    except Exception:
        log.info("%s: sig changed (diff unavailable)", label)


def is_sana_model(model_path: str) -> bool:
    """Return True if model_path looks like a SANA HuggingFace model ID."""
    nm = model_path.replace("\\", "/").lower()
    return (
        "sana_sprint" in nm
        or "sana-sprint" in nm
        or "/sana_" in nm
        or "/sana-" in nm
    )


# ──────────────────────────────────────────────────────────────────
# SanaSprintSession
# ──────────────────────────────────────────────────────────────────

class SanaSprintSession:
    """
    One SANA-Sprint img2img session.

    Wraps SanaSprintImg2ImgPipeline; caches prompt embeddings so only
    the transformer forward pass runs per frame (not text encoding).
    """

    def __init__(
        self,
        *,
        pipe: Any,
        device: str,
        prompt: str = "",
        negative_prompt: str = "",
        cfg_scale: float = 4.5,
        num_steps: int = 2,
        width: int = 512,
        height: int = 512,
    ) -> None:
        import torch

        self._pipe = pipe
        self._device = torch.device(device)
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._cfg_scale = cfg_scale
        self._num_steps = max(1, min(8, num_steps))
        self._width = width
        self._height = height
        self._lock = threading.Lock()

        logger.info(
            "SanaSprintSession: ready on %s  steps=%d  size=%dx%d  prompt=%r",
            device, self._num_steps, width, height, prompt[:60],
        )

    # ── public API ─────────────────────────────────────────────────

    def infer(self, image: Image.Image, strength: float = 0.7) -> Image.Image | None:
        """Single img2img inference. Returns None if busy (non-blocking lock)."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._infer_inner(image, strength)
        finally:
            self._lock.release()

    def _infer_inner(self, image: Image.Image, strength: float) -> Image.Image:
        import torch

        image_rgb = image.convert("RGB").resize(
            (self._width, self._height), Image.LANCZOS
        )

        with torch.inference_mode():
            call_kwargs: dict[str, Any] = dict(
                prompt=self._prompt,
                image=image_rgb,
                strength=float(strength),
                num_inference_steps=self._num_steps,
                guidance_scale=self._cfg_scale,
                width=self._width,
                height=self._height,
                output_type="pil",
            )
            # SanaSprintImg2ImgPipeline does not accept negative_prompt
            import inspect
            if "negative_prompt" in inspect.signature(self._pipe.__call__).parameters:
                call_kwargs["negative_prompt"] = self._negative_prompt or None
            result = self._pipe(**call_kwargs)
        return result.images[0]

    def warmup(self) -> None:
        blank = Image.new("RGB", (self._width, self._height), (128, 128, 128))
        try:
            self._infer_inner(blank, strength=0.7)
            logger.info("SanaSprintSession: warmup done at %dx%d", self._width, self._height)
        except Exception:
            logger.exception("SanaSprintSession: warmup failed")


# ──────────────────────────────────────────────────────────────────
# SanaSprintSessionManager
# ──────────────────────────────────────────────────────────────────

class SanaSprintSessionManager:
    """
    Caches one SanaSprintSession per settings signature.
    Pipeline is shared across sessions — only the session params change.
    Debounces rebuilds (350 ms) so typing a prompt doesn't cause per-keystroke reloads.
    """

    def __init__(self) -> None:
        self._session: SanaSprintSession | None = None
        self._sig: str | None = None
        self._pipe: Any = None
        self._pipe_model: str | None = None
        self.last_output: Any = None
        self._generation: int = 0
        self._build_lock = threading.Lock()
        self._pending_sig: str | None = None
        self._sig_changed_at: float = 0.0

    def get_session(self, settings: dict[str, Any]) -> SanaSprintSession | None:
        sig = _session_sig(settings)

        if self._session is not None and self._sig == sig:
            self._pending_sig = None
            return self._session

        now = time.monotonic()
        if self._pending_sig != sig:
            self._pending_sig = sig
            self._sig_changed_at = now
            return self._session
        if now - self._sig_changed_at < _REBUILD_DEBOUNCE_S:
            return self._session

        with self._build_lock:
            if self._session is not None and self._sig == sig:
                self._pending_sig = None
                return self._session

            _log_sig_diff(self._sig, sig, logger, "SanaSprintSession")
            logger.info("SanaSprintSession: rebuilding")
            pipe = self._ensure_pipe(settings)
            if pipe is None:
                return None
            try:
                w, h = _clamp_dims(
                    int(settings.get("width", 512)),
                    int(settings.get("height", 512)),
                )
                session = SanaSprintSession(
                    pipe=pipe,
                    device=settings.get("device") or _default_device(),
                    prompt=settings.get("prompt", ""),
                    negative_prompt=settings.get("negative_prompt", ""),
                    cfg_scale=float(settings.get("cfg", 4.5)),
                    num_steps=int(settings.get("sana_steps", 2)),
                    width=w,
                    height=h,
                )
                session.warmup()
                self._session = session
                self._sig = sig
                self._pending_sig = None
                self._generation += 1
                logger.info("SanaSprintSession: ready at %dx%d (gen=%d)", w, h, self._generation)
                return session
            except Exception:
                logger.exception("SanaSprintSession: build failed")
                self._session = None
                self._sig = None
                self._sig_changed_at = time.monotonic()
                return None

    def _ensure_pipe(self, settings: dict[str, Any]) -> Any | None:
        model_id = (
            settings.get("model_path")
            or os.getenv("RTD_SANA_MODEL", _DEFAULT_MODEL_1_6B)
        )
        device = settings.get("device") or _default_device()
        if self._pipe is not None and self._pipe_model == model_id:
            return self._pipe
        pipe = _load_pipe(model_id, device)
        if pipe is None:
            return None
        self._pipe = pipe
        self._pipe_model = model_id
        return pipe

    def reset(self) -> None:
        self._session = None
        self._sig = None


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _session_sig(s: dict[str, Any]) -> str:
    model = s.get("model_path") or os.getenv("RTD_SANA_MODEL", _DEFAULT_MODEL_1_6B)
    device = s.get("device") or _default_device()
    return repr((
        model, device,
        s.get("prompt", ""),
        s.get("negative_prompt", ""),
        round(float(s.get("cfg", 4.5)), 3),
        int(s.get("sana_steps", 2)),
        _clamp_dims(int(s.get("width", 512)), int(s.get("height", 512))),
    ))


def _clamp_dims(w: int, h: int) -> tuple[int, int]:
    max_side = int(os.getenv("RTD_STREAM_MAX_SIDE", "768"))
    scale = min(1.0, max_side / max(w, h, 1))
    w2 = max(64, int(w * scale) // 64 * 64)
    h2 = max(64, int(h * scale) // 64 * 64)
    return w2, h2


def _default_device() -> str:
    v = os.getenv("RTD_DEVICE", "cuda").strip().lower()
    return "cuda:0" if v == "cuda" else v


def _load_pipe(model_id: str, device: str) -> Any | None:
    import torch

    logger.info("SanaSprintSession: loading %s on %s", model_id, device)
    t0 = time.perf_counter()
    is_cu = device.startswith("cuda")
    dtype = torch.bfloat16 if is_cu else torch.float32

    try:
        from diffusers import SanaSprintImg2ImgPipeline

        pipe = SanaSprintImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        if is_cu:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Channels-last for transformer (mild speedup)
            try:
                pipe.transformer.to(memory_format=torch.channels_last)
            except Exception:
                pass
            # xformers memory-efficient attention
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("SanaSprintSession: xformers enabled")
                except Exception as e:
                    logger.info("SanaSprintSession: xformers not available (%s)", e)

        # torch.compile for extra speed (requires Triton, available in Docker)
        if os.getenv("RTD_SANA_COMPILE", "0").strip() not in ("0", "false", "no", ""):
            try:
                mode = os.getenv("RTD_STREAM_COMPILE_MODE", "reduce-overhead")
                pipe.transformer = torch.compile(pipe.transformer, mode=mode, fullgraph=False)
                logger.info("SanaSprintSession: torch.compile applied to transformer")
            except Exception as e:
                logger.warning("SanaSprintSession: torch.compile failed: %s", e)

        elapsed = time.perf_counter() - t0
        logger.info("SanaSprintSession: pipe ready in %.1fs", elapsed)
        return pipe

    except ImportError:
        logger.error(
            "SanaSprintSession: SanaSprintImg2ImgPipeline not found — "
            "upgrade diffusers: pip install -U diffusers"
        )
    except Exception:
        logger.exception("SanaSprintSession: pipe load failed for %s", model_id)
        gc.collect()
        try:
            if is_cu:
                import torch
                torch.cuda.empty_cache()
        except Exception:
            pass
    return None
