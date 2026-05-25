"""SessionManager — caches and rebuilds StreamSession instances."""
from __future__ import annotations

import gc
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from ..session_paths import configure_torch_compile_cache
from ..inference.pipeline_base import (
    clamp_dims_64,
    default_device,
    env_model_path,
    load_pretrained_pipe,
    normalize_diffusers_model_id,
)

from .helpers import (
    _TRITON_OK,
    _is_cuda,
    logger,
    resolve_t_indices,
)
from .session import StreamSession

# ──────────────────────────────────────────────────────────────────
# SessionManager
# ──────────────────────────────────────────────────────────────────

_REBUILD_DEBOUNCE_S = 0.15  # seconds to wait for settings to stabilise before rebuilding


def _log_sig_diff(old_sig: str | None, new_sig: str, log: Any, label: str) -> None:
    """Log which fields changed between two session signatures to aid debugging."""
    if old_sig is None:
        log.info("%s: first build", label)
        return
    try:
        old_t = eval(old_sig)  # noqa: S307 — trusted internal repr() string
        new_t = eval(new_sig)  # noqa: S307
        fields = ("model_path", "device", "negative_prompt",
                  "timestep_indices", "frame_buffer_size", "cfg_type", "do_full_cfg",
                  "dims", "lora_paths", "prompt_b", "cn_model_id", "triton_compile", "session_directory", "vae_mode")
        diffs = [fields[i] for i, (a, b) in enumerate(zip(old_t, new_t)) if a != b]
        log.info("%s: sig changed — fields: %s", label, diffs or "<unknown>")
    except Exception:
        log.info("%s: sig changed (diff unavailable)", label)


class SessionManager:
    """
    Caches one StreamSession per settings signature.
    Rebuilds the pipeline only when model_path / device / LoRA changes.
    Debounces setting changes so that typing a prompt doesn't trigger one rebuild
    per keystroke — only rebuilds after settings have been stable for 350ms.
    """

    def __init__(self) -> None:
        self._session: StreamSession | None = None
        self._sig: str | None = None
        self._pipe: Any = None
        self._pipe_sig: str | None = None
        self.last_output: Any = None  # last PIL image — carried across reconnections
        self._generation: int = 0     # incremented on each session rebuild
        self._build_lock = threading.Lock()  # serialise session builds
        # Debounce: track the pending sig and when it first appeared
        self._pending_sig: str | None = None
        self._sig_changed_at: float = 0.0
        self._debounce_logged: bool = False  # suppress repeated debounce log spam
        # Build status — readable externally and propagated via optional callback
        self.build_phase: str = "idle"
        self.build_progress: float = 0.0
        self.build_message: str = ""
        self._status_cb: Any = None

    def set_status_callback(self, cb: Any) -> None:
        """Register a callable(phase, progress, message) called on each build phase change."""
        self._status_cb = cb

    def _emit_build(self, phase: str, progress: float, message: str) -> None:
        self.build_phase = phase
        self.build_progress = progress
        self.build_message = message
        logger.info("StreamSession [%s] %.0f%% — %s", phase, progress * 100, message)
        if self._status_cb is not None:
            try:
                self._status_cb(phase, progress, message)
            except Exception:
                pass

    def get_session(self, settings: dict[str, Any]) -> StreamSession | None:
        sig = _session_sig(settings)
        # Fast path — no lock needed for a cache hit.
        if self._session is not None and self._sig == sig:
            self._pending_sig = None
            self._debounce_logged = False
            # Live-update runtime scalars that don't require a session rebuild.
            cfg_scale = float(settings.get("cfg", 1.0))
            if self._session._s.get("cfg_scale") != cfg_scale:
                self._session._s["cfg_scale"] = cfg_scale
            self._apply_live_seed(settings, self._session)
            return self._session

        # Debounce: only rebuild after settings have been stable for _REBUILD_DEBOUNCE_S.
        # This prevents per-keystroke rebuilds when the user types a prompt.
        now = time.monotonic()
        if self._pending_sig != sig:
            self._pending_sig = sig
            self._sig_changed_at = now
            # Log only on first entry into debounce to avoid log spam when settings
            # keep changing (e.g. while typing or live slider adjustments).
            if not self._debounce_logged:
                self._debounce_logged = True
                self._emit_build("debounce", 0.02, "Settings changed — waiting to stabilise")
            return None
        if now - self._sig_changed_at < _REBUILD_DEBOUNCE_S:
            return None
        self._debounce_logged = False  # reset for next debounce cycle

        with self._build_lock:
            # Re-check after acquiring the lock (another thread may have built it).
            if self._session is not None and self._sig == sig:
                self._pending_sig = None
                return self._session

            # Log what actually changed to help diagnose spurious rebuilds.
            _log_sig_diff(self._sig, sig, logger, "StreamSession")
            logger.info("StreamSession: rebuilding")
            # Reset torch.compile CUDA-graph state before rebuilding.
            # Changing t-indices changes the UNet batch size; without a reset,
            # the old graph recording leaks into the new session and raises
            # "beginAllocateToPool: already recording to mempool_id".
            try:
                import torch._dynamo as _dynamo
                _dynamo.reset()
            except Exception:
                pass
            model_label = (settings.get("model_path") or "model")
            model_label = (model_label.replace("\\", "/").rsplit("/", 1)[-1] or model_label)[:40]
            self._emit_build("loading_pipe", 0.10, f"Loading {model_label}")
            pipe = self._ensure_pipe(settings)
            if pipe is None:
                self._emit_build("error", 1.0, "Pipeline load failed")
                return None
            try:
                w, h = _clamp_dims(
                    int(settings.get("width", 512)),
                    int(settings.get("height", 512)),
                )
                t_idx = resolve_t_indices(settings)
                device = settings.get("device") or default_device()
                self._emit_build("building", 0.50, f"Building session {w}×{h}")
                cn_model = _load_cn_for_streaming(settings, device, pipe)
                session = StreamSession(
                    pipe=pipe,
                    device=device,
                    prompt=settings.get("prompt", ""),
                    negative_prompt=settings.get("negative_prompt", ""),
                    t_indices=t_idx,
                    frame_buffer_size=int(settings.get("stream_frame_buffer_size", 1)),
                    cfg_type=settings.get("stream_cfg_type", "self"),
                    cfg_scale=float(settings.get("cfg", 1.0)),
                    width=w,
                    height=h,
                    use_tiny_vae=_resolve_use_tiny_vae(settings),
                    prompt_b=settings.get("prompt_b", ""),
                    rcfg_delta=float(os.getenv("RTD_STREAM_RCFG_DELTA", "1.0")),
                    controlnet=cn_model,
                    seed=_stream_seed(settings),
                )
                # Always warm up at least len(t_idx) passes to fill the latent buffer.
                # Zero-initialised latent_buffer → first frames are garbage without this.
                # RTD_STREAM_WARMUP=0 skips only the *extra* quality passes.
                min_warmup = len(t_idx)
                extra_warmup = (
                    len(t_idx) * 2
                    if os.getenv("RTD_STREAM_WARMUP", "1").strip() not in ("0", "false", "no")
                    else 0
                )
                warmup_total = min_warmup + extra_warmup
                env_compile = os.getenv("RTD_STREAM_COMPILE", "1").strip() not in ("0", "false", "no")
                triton_compile = _TRITON_OK and bool(settings.get("stream_triton_compile", True)) and env_compile
                if triton_compile:
                    self._emit_build("compiling", 0.65,
                        "Compiling Triton kernels — first run takes 30-120 s (subsequent runs are instant)")
                else:
                    self._emit_build("warming", 0.80, f"Warming up ({warmup_total} passes)")
                warmup_image = self.last_output if isinstance(self.last_output, Image.Image) else None
                session.warmup(warmup_total, image=warmup_image)
                self._session = session
                self._sig = sig
                self._pending_sig = None
                self._generation += 1
                self._emit_build("ready", 1.0, f"Session ready at {w}×{h}")
                logger.info(
                    "StreamSession ready at %dx%d (t=%s, gen=%d)",
                    w, h, t_idx, self._generation,
                )
                return session
            except Exception as exc:
                logger.exception("StreamSession build failed")
                self._emit_build("error", 1.0, f"Build failed: {exc.__class__.__name__}: {exc}")
                self._session = None
                self._sig = None
                # Drop the cached pipe too: a build failure often means the pipe
                # is malformed (e.g. SD 1.5 loaded into the SDXL pipeline class
                # with the wrong number of text encoders). Without this, retries
                # would reuse the broken pipe and hit the same error forever.
                self._pipe = None
                self._pipe_sig = None
                # Reset debounce timer so the next failure waits another full window
                # before retrying, instead of hammering the GPU on every inference tick.
                self._sig_changed_at = time.monotonic()
                return None

    def _ensure_pipe(self, settings: dict[str, Any]) -> Any | None:
        model_path = normalize_diffusers_model_id(str(settings.get("model_path") or env_model_path()), logger)
        device = settings.get("device") or default_device()
        loras = tuple(sorted(settings.get("lora_paths") or []))
        triton_compile = bool(settings.get("stream_triton_compile", True))
        session_directory = str(settings.get("session_directory") or "").strip()
        psig = repr((model_path, device, loras, triton_compile, session_directory))
        if self._pipe is not None and self._pipe_sig == psig:
            return self._pipe
        pipe = _load_pipe(model_path, device, list(loras), triton_compile=triton_compile, session_directory=session_directory)
        if pipe is None:
            return None
        self._pipe = pipe
        self._pipe_sig = psig
        return pipe

    def reset(self) -> None:
        self._session = None
        self._sig = None

    def _apply_live_seed(self, settings: dict[str, Any], session: StreamSession) -> None:
        try:
            session._noise_drift = max(0.0, min(0.25, float(settings.get("stream_noise_drift", session._noise_drift))))
        except Exception:
            pass
        seed = _stream_seed(settings)
        reset_noise = getattr(session, "reset_noise", None)
        if callable(reset_noise):
            reset_noise(seed)


def _session_sig(s: dict[str, Any]) -> str:
    cn_model_id = _extract_cn_model_id(s)
    cfg_type = s.get("stream_cfg_type", "self")
    # Only the *structure* of CFG matters for the session build (doubled batch
    # for full-CFG). The actual scale is a runtime scalar updated live.
    do_full_cfg = cfg_type == "full" and float(s.get("cfg", 1.0)) > 1.0
    env_compile = os.getenv("RTD_STREAM_COMPILE", "1").strip() not in ("0", "false", "no")
    triton_compile = _TRITON_OK and bool(s.get("stream_triton_compile", True)) and env_compile
    return repr((
        normalize_diffusers_model_id(str(s.get("model_path") or _env_model_path()), logger),
        s.get("device") or default_device(),
        s.get("negative_prompt", ""),
        tuple(resolve_t_indices(s)),
        int(s.get("stream_frame_buffer_size", 1)),
        cfg_type,
        do_full_cfg,
        _clamp_dims(int(s.get("width", 512)), int(s.get("height", 512))),
        tuple(sorted(s.get("lora_paths") or [])),
        s.get("prompt_b", ""),
        cn_model_id,
        triton_compile,
        str(s.get("session_directory") or "").strip(),
        _stream_vae_mode(s),
    ))


def _stream_vae_mode(settings: dict[str, Any]) -> str:
    mode = str(settings.get("stream_vae_mode") or "auto").strip().lower()
    if mode in {"tiny", "full"}:
        return mode
    return "auto"


def _resolve_use_tiny_vae(settings: dict[str, Any]) -> bool:
    mode = _stream_vae_mode(settings)
    if mode == "tiny":
        return True
    if mode == "full":
        return False
    return os.getenv("RTD_STREAM_TINY_VAE", "1").strip().lower() not in ("0", "false", "no")


def _stream_seed(settings: dict[str, Any]) -> int:
    raw_seed = settings.get("seed")
    try:
        base_seed = int(raw_seed) if raw_seed is not None else 42
    except (TypeError, ValueError):
        base_seed = 42
    mode = str(settings.get("seed_mode") or "fixed").lower()
    try:
        serial = int(settings.get("seed_rotation_serial") or 0)
    except (TypeError, ValueError):
        serial = 0
    if mode == "increment":
        return (base_seed + serial) % (2**32)
    if mode == "decrement":
        return (base_seed - serial) % (2**32)
    if mode == "random":
        payload = repr((base_seed, serial, settings.get("scene_id", ""), settings.get("prompt", ""))).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return base_seed % (2**32)


def _extract_cn_model_id(settings: dict[str, Any]) -> str:
    """Return a stable string identifying the ControlNet model (or '' if none)."""
    for cond in (settings.get("layer_conditions") or []):
        model = cond.get("controlnet_model", "")
        if model:
            path = cond.get("controlnet_model_path", "")
            return path or model  # use explicit path if given, else short name
    return ""


def _load_cn_for_streaming(settings: dict[str, Any], device: str, pipe: Any) -> Any | None:
    """Load the first active ControlNet model from layer_conditions, or None."""
    from .. import controlnet as _cn_mod

    for cond in (settings.get("layer_conditions") or []):
        short = cond.get("controlnet_model", "")
        if not short:
            continue
        explicit_path = cond.get("controlnet_model_path", "")
        is_sdxl = hasattr(pipe, "text_encoder_2")
        try:
            dtype = next(pipe.unet.parameters()).dtype
            model_id = _cn_mod.resolve_model_id(short, explicit_path or None, is_sdxl)
            logger.info("StreamSession: loading ControlNet %s for streaming", model_id)
            cn = _cn_mod.load_controlnet(model_id, device, dtype)
            logger.info("StreamSession: ControlNet ready (%s)", short)
            return cn
        except Exception as exc:
            logger.error("StreamSession: ControlNet load failed (%s): %s", short, exc)
    return None


def _clamp_dims(w: int, h: int) -> tuple[int, int]:
    """Preserve requested dimensions unless explicit stream resize envs are set."""
    return clamp_dims_64(w, h)


def _default_device() -> str:
    return default_device()


def _env_model_path() -> str:
    return env_model_path()


def _load_pipe(model_path: str | None, device: str, lora_paths: list[str], *, triton_compile: bool = True, session_directory: str = "") -> Any | None:
    """Load a Diffusers pipeline suitable for StreamDiffusion."""
    import torch

    if not model_path:
        try:
            from .assets import default_model_path as dmp
            model_path = dmp()
        except Exception:
            pass
    if not model_path:
        logger.error("StreamSession: no model configured (set RTD_MODEL_PATH)")
        return None

    model_path = normalize_diffusers_model_id(model_path, logger)
    is_cu = _is_cuda(device)
    dtype = torch.float16 if is_cu else torch.float32
    path = Path(model_path)

    # navigate up from component files to the diffusers dir
    if path.is_file():
        for p in path.parents:
            if (p / "model_index.json").is_file():
                logger.info("StreamSession: resolved component → diffusers dir %s", p)
                model_path, path = str(p), p
                break

    logger.info("StreamSession: loading pipe %s on %s", model_path, device)
    t0 = time.perf_counter()

    try:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting

        if path.is_file():
            pipe = _load_single_file(path, dtype)
        else:
            # prefer inpaint pipeline (9-ch UNet handles inpainting natively)
            # but fall back to img2img which works fine with compositing
            try:
                pipe = load_pretrained_pipe(AutoPipelineForInpainting, model_path, dtype, logger=logger)
            except Exception:
                pipe = load_pretrained_pipe(AutoPipelineForImage2Image, model_path, dtype, logger=logger)

        pipe = pipe.to(device)
        # from_single_file can place sub-models on unexpected devices.  Explicitly
        # ensure every text encoder lands on the target device.
        for _attr in ("text_encoder", "text_encoder_2"):
            _te = getattr(pipe, _attr, None)
            if _te is not None:
                try:
                    _te.to(device)
                except Exception:
                    pass
        pipe.set_progress_bar_config(disable=True)

        if is_cu:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # channels_last for better CUDA kernel utilization
        try:
            pipe.unet.to(memory_format=torch.channels_last)
        except Exception:
            pass

        # xformers memory-efficient attention — big speedup on SDXL
        xformers_ok = False
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                xformers_ok = True
                logger.info("StreamSession: xformers memory-efficient attention enabled")
            except Exception as e:
                logger.info("StreamSession: xformers not available (%s)", e)

        _apply_loras(pipe, lora_paths)

        # Guard: older diffusers doesn't skip None tokenizers in maybe_convert_prompt.
        _patch_pipe_for_none_tokenizers(pipe)

        # Auto-inject LCM LoRA for SDXL models that aren't already fast
        _maybe_inject_lcm(pipe, model_path, lora_paths)

        # torch.compile — requires triton. ON by default (RTD_STREAM_COMPILE=0 to disable).
        # The first session pays a one-time compile cost (~30-60s); subsequent runs reuse the
        # cached graph. With 48GB VRAM and the realtime use case, the speedup pays back instantly.
        # On Triton-less systems (Windows) we skip silently rather than crash on first frame.
        env_compile = os.getenv("RTD_STREAM_COMPILE", "1").strip() not in ("0", "false", "no")
        if _TRITON_OK and triton_compile and env_compile:
            _compile_pipe(pipe, model_path=model_path or "", device=device, session_directory=session_directory)

        elapsed = time.perf_counter() - t0
        logger.info("StreamSession: pipe ready in %.1fs (xformers=%s)", elapsed, xformers_ok)
        return pipe

    except Exception:
        logger.exception("StreamSession: pipe load failed for %s", model_path)
        gc.collect()
        try:
            if is_cu:
                torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def _compile_pipe(pipe: Any, *, model_path: str = "", device: str = "", session_directory: str = "") -> None:
    """Apply torch.compile to UNet (and optionally VAE) for triton-accelerated inference."""
    import torch
    # "reduce-overhead" enables CUDA graphs which are tied to the thread that
    # recorded them (stored in TLS). asyncio.to_thread dispatches to arbitrary
    # pool threads, so CUDA graphs break immediately after the first warmup.
    # "default" still applies Triton kernel fusion (big win) without CUDA graphs.
    compile_mode = os.getenv("RTD_STREAM_COMPILE_MODE", "default")
    manifest_path = configure_torch_compile_cache(
        session_directory,
        model_id=model_path,
        device=device,
        backend="streamdiffusion_unet",
    ) if session_directory else None
    t0 = time.perf_counter()
    try:
        pipe.unet = torch.compile(pipe.unet, mode=compile_mode, fullgraph=False)
        logger.info("StreamSession: torch.compile applied to UNet (mode=%s, cache=%s)", compile_mode, manifest_path.parent if manifest_path else "default")
    except Exception as e:
        logger.warning("StreamSession: torch.compile failed: %s", e)
        return
    # Compile VAE decoder by default (RTD_STREAM_COMPILE_VAE=0 to disable).
    # VAE decode is ~25% of frame time at 512px; compiling it gives a free ~20% speedup.
    if os.getenv("RTD_STREAM_COMPILE_VAE", "1").strip() not in ("0", "false", "no"):
        try:
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "decoder"):
                pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode=compile_mode, fullgraph=False)
                logger.info("StreamSession: torch.compile applied to VAE decoder")
        except Exception:
            pass
    logger.info("StreamSession: torch.compile setup in %.1fs", time.perf_counter() - t0)


def _detect_sd_architecture(path: Path) -> str:
    """Peek at a ``.safetensors`` (or ``.ckpt``) checkpoint to detect the
    Stable Diffusion architecture without loading the full model.

    Returns ``"sdxl"``, ``"sd15"``, or ``"unknown"``.

    SDXL is distinguished by either:
      • a second CLIP text encoder under ``conditioner.embedders.1.*`` or
        ``cond_stage_model.1.*`` or ``text_encoder_2.*``;
      • a UNet cross-attention dimension of 2048 (vs 768 for SD 1.5).

    The check is purely metadata-level for ``.safetensors`` (no tensors copied);
    other formats fall through to the "unknown" branch and the caller may try
    both pipeline classes.
    """
    suffix = path.suffix.lower()
    if suffix not in (".safetensors", ".sft"):
        return "unknown"
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            # SDXL markers — second CLIP encoder in any of the known prefixes.
            if any(
                ".embedders.1." in k
                or "cond_stage_model.1." in k
                or k.startswith("text_encoder_2.")
                or k.startswith("conditioner.embedders.1.")
                for k in keys
            ):
                return "sdxl"
            # Fallback — inspect cross-attention key dimension. SDXL = 2048,
            # SD 1.5 = 768. The first match wins; we only need the input dim
            # (last axis of the weight matrix).
            for k in keys:
                if "attn2.to_k.weight" not in k:
                    continue
                shape = f.get_slice(k).get_shape()
                if not shape:
                    continue
                inner = int(shape[-1])
                if inner >= 1600:
                    return "sdxl"
                if inner <= 1024:
                    return "sd15"
    except Exception as exc:
        logger.debug("Architecture detection failed for %s: %s", path, exc)
    return "unknown"


def _detect_lora_cross_attention_dim(path: Path) -> int | None:
    """Peek at a LoRA's .safetensors and return its cross-attention input dim.

    Returns:
      • 2048 → SDXL LoRA
      • 768  → SD 1.5 LoRA
      • None → undetectable (non-safetensors, unknown format, or no attn2 keys)

    LoRA weight tensors come in several naming conventions:
      • diffusers: ``down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight``
      • Kohya:     ``lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight``

    The cross-attention input dim is the LAST axis of the ``lora_A`` / ``lora_down``
    matrix for any ``attn2.to_k`` (or ``to_v``) layer. We grab the first such match.
    """
    if path.suffix.lower() not in (".safetensors", ".sft"):
        return None
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                if "attn2" not in k:
                    continue
                # Match either diffusers (attn2.to_k.lora_A) or Kohya
                # (attn2_to_k.lora_down) cross-attention input projections.
                if not any(
                    needle in k
                    for needle in (
                        "to_k.lora_A", "to_v.lora_A",
                        "to_k.lora_down", "to_v.lora_down",
                    )
                ):
                    continue
                shape = f.get_slice(k).get_shape()
                if len(shape) >= 2:
                    return int(shape[-1])
    except Exception as exc:
        logger.debug("LoRA arch detection failed for %s: %s", path, exc)
    return None


def _apply_loras(pipe: Any, lora_paths: list[str]) -> None:
    """Apply each LoRA to the pipe, skipping incompatibly-shaped LoRAs.

    SDXL LoRAs use cross_attention_dim=2048 and SD 1.5 LoRAs use 768; loading
    a mismatched LoRA against the pipe's UNet would cause diffusers to print
    40+ lines of size-mismatch noise from its state_dict loader. We pre-validate
    each LoRA by peeking at its safetensors metadata, skipping with one clean
    warning line when the architectures don't match.
    """
    unet_cfg = getattr(getattr(pipe, "unet", None), "config", None)
    unet_dim = int(getattr(unet_cfg, "cross_attention_dim", 0) or 0)
    for lp in lora_paths:
        lora_dim = _detect_lora_cross_attention_dim(Path(lp))
        if lora_dim is not None and unet_dim and lora_dim != unet_dim:
            lora_arch = "SDXL" if lora_dim >= 1600 else "SD 1.5"
            model_arch = "SDXL" if unet_dim >= 1600 else "SD 1.5"
            logger.warning(
                "StreamSession: skipping %s LoRA on %s model — cross-attention "
                "dims don't match (lora=%d, model=%d): %s",
                lora_arch, model_arch, lora_dim, unet_dim, Path(lp).name,
            )
            continue
        try:
            pipe.load_lora_weights(lp, adapter_name=Path(lp).stem)
            logger.info("StreamSession: LoRA %s loaded", lp)
        except Exception as exc:
            logger.warning("StreamSession: LoRA load failed: %s — %s", lp, exc)


def _load_single_file(path: Path, dtype: Any) -> Any:
    """Load a single-file checkpoint into the correct pipeline class.

    The architecture is detected upfront so we don't shove an SD 1.5 checkpoint
    into ``StableDiffusionXLImg2ImgPipeline`` (which would "load" 6/7 components
    silently and crash later in ``encode_prompt`` looking for a non-existent
    ``text_encoder_2`` pooled output).
    """
    name = path.name.lower()
    is_inpaint = "inpaint" in name
    arch = _detect_sd_architecture(path)
    logger.info("StreamSession: detected %s architecture for %s", arch, path.name)

    sdxl_cls = "StableDiffusionXLInpaintPipeline" if is_inpaint else "StableDiffusionXLImg2ImgPipeline"
    sd15_cls = "StableDiffusionInpaintPipeline" if is_inpaint else "StableDiffusionImg2ImgPipeline"

    if arch == "sdxl":
        attempts = (sdxl_cls, sd15_cls)
    elif arch == "sd15":
        attempts = (sd15_cls, sdxl_cls)
    else:
        # Unknown — try SDXL first then fall back (preserves old behaviour
        # for non-safetensors checkpoints).
        attempts = (sdxl_cls, sd15_cls)

    last_err: Exception | None = None
    for cls_name in attempts:
        try:
            import diffusers
            cls = getattr(diffusers, cls_name)
            return cls.from_single_file(str(path), torch_dtype=dtype)
        except Exception as exc:
            last_err = exc
            logger.debug("StreamSession: %s.from_single_file failed: %s", cls_name, exc)
            continue
    raise RuntimeError(
        f"Could not load {path} with any known pipeline class (last error: {last_err})"
    )


def _patch_pipe_for_none_tokenizers(pipe: Any) -> None:
    """Guard against None tokenizer slots in SDXL pipelines.

    Some single-file checkpoints (or models with only one text encoder) end up
    with tokenizer_2 / text_encoder_2 as None even though tokenizer / text_encoder
    are valid.  The SDXL encode_prompt loop builds:

        tokenizers    = [self.tokenizer, self.tokenizer_2]   # when tokenizer  is not None
        text_encoders = [self.text_encoder, self.text_encoder_2]  # when text_encoder is not None

    Both lists then have 2 elements — the second being None — and zip produces a
    second iteration with tokenizer=None.  Two crash sites follow in order:
        1. maybe_convert_prompt(prompt, None) → tokenizer.tokenize() → AttributeError
        2. tokenizer(prompt, ...)             → NoneType is not callable → AttributeError

    Fix: wrap encode_prompt on the instance so that when tokenizer_2 is None but
    tokenizer is present, we temporarily move the encoder-1 pair into the encoder-2
    slot (and null out encoder-1).  This forces the pipeline into its single-encoder
    path: tokenizers=[tokenizer_2], text_encoders=[text_encoder_2] — one iteration,
    no None in sight.
    """
    import types

    # SDXL pipes declare ``text_encoder_2`` (and ``tokenizer_2``) attributes,
    # even when they happen to be None on a malformed checkpoint. SD 1.5 pipes
    # don't declare these attributes at all and use only ``self.tokenizer``.
    # Applying the swap to SD 1.5 would set ``self.tokenizer = None`` and crash
    # the very first ``maybe_convert_prompt(prompt, None)`` call.
    if not (hasattr(pipe, 'tokenizer_2') and hasattr(pipe, 'text_encoder_2')):
        return  # not an SDXL-style pipe — no patch needed

    t1 = getattr(pipe, 'tokenizer',   None)
    t2 = getattr(pipe, 'tokenizer_2', None)

    if t1 is None or t2 is not None:
        # Either no first tokenizer (nothing to fix), or second tokenizer already
        # present (no mismatch).  Nothing to patch.
        return

    logger.debug(
        "StreamSession: tokenizer_2 is None — wrapping encode_prompt "
        "to use single-encoder path (tokenizer / text_encoder moved to _2 slots)"
    )

    original_encode = getattr(type(pipe), 'encode_prompt', None)
    if original_encode is None:
        return

    def _safe_encode_prompt(self, prompt: Any, device: Any = None,
                             num_images_per_prompt: int = 1,
                             do_classifier_free_guidance: bool = True,
                             negative_prompt: Any = None,
                             **kwargs: Any) -> Any:
        # Swap encoder-1 → encoder-2 slot so the pipeline uses the
        # "only tokenizer_2" code path and the loop produces no None tokenizers.
        saved_t1  = self.tokenizer
        saved_te1 = getattr(self, 'text_encoder', None)
        self.tokenizer    = None
        self.tokenizer_2  = saved_t1
        if hasattr(self, 'text_encoder'):
            self.text_encoder   = None
        if hasattr(self, 'text_encoder_2'):
            self.text_encoder_2 = saved_te1
        try:
            # CRITICAL: forward as kwargs. SDXL's encode_prompt signature inserts
            # ``prompt_2`` between ``prompt`` and ``device``, so positional
            # forwarding shifts ``device`` into ``prompt_2`` and ``num_images=1``
            # into ``device`` → ``torch.device(1) == cuda:1`` on multi-GPU hosts,
            # causing "tensors on different devices" at the embedding lookup.
            return original_encode(
                self,
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                **kwargs,
            )
        finally:
            self.tokenizer    = saved_t1
            self.tokenizer_2  = None
            if hasattr(self, 'text_encoder'):
                self.text_encoder   = saved_te1
            if hasattr(self, 'text_encoder_2'):
                self.text_encoder_2 = None

    pipe.encode_prompt = types.MethodType(_safe_encode_prompt, pipe)


def _maybe_inject_lcm(pipe: Any, model_path: str, lora_paths: list[str] | None = None) -> None:
    """Auto-inject LCM LoRA for SDXL models without built-in speed."""
    if os.getenv("RTD_STREAM_LCM_LORA", "1").strip() in ("0", "false", "no"):
        return
    nm = model_path.replace("\\", "/").lower()
    if any(m in nm for m in ("turbo", "lightning", "lcm", "hyper", "distill")):
        logger.info("StreamSession: LCM LoRA skip (model name implies fast): %s", Path(model_path).name)
        return
    if not hasattr(pipe, "text_encoder_2"):
        return  # not SDXL
    # Skip if the user already has an LCM/latent-consistency LoRA in their lora_paths.
    # Check the full path (directory + stem) so "lcm-lora-sdxl/pytorch_lora_weights.safetensors" is caught.
    if lora_paths:
        for lp in lora_paths:
            lp_lower = lp.replace("\\", "/").lower()
            if any(m in lp_lower for m in ("lcm", "latent_consistency", "latent-consistency")):
                # Also activate the user's LoRA adapter so it's actually used.
                adapter_name = Path(lp).stem
                try:
                    pipe.set_adapters([adapter_name], [1.0])
                except Exception:
                    pass
                logger.info("StreamSession: LCM LoRA skip (user LoRA path contains %r)", adapter_name)
                return
    try:
        is_xl = hasattr(pipe, "text_encoder_2")
        lora_id = "latent-consistency/lcm-lora-sdxl" if is_xl else "latent-consistency/lcm-lora-sdv1-5"
        logger.info("StreamSession: injecting LCM LoRA %s …", lora_id)
        pipe.load_lora_weights(lora_id, adapter_name="_lcm")
        pipe.set_adapters(["_lcm"], [1.0])
        logger.info("StreamSession: LCM LoRA injected (%s)", lora_id)
    except Exception as e:
        logger.info("StreamSession: LCM LoRA injection skipped (%s): %s", Path(model_path).name, e)
