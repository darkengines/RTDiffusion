import json
import os
import sys
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Literal

from PIL import Image, ImageOps

from .image_io import decode_data_url_rgba
from .schemas import MotionClip, MotionClipFrame, MotionClipRequest
from .video_pipeline import generate_native_video_loop, native_video_configured, native_video_missing_message

MotionKind = Literal["idle", "local", "transition"]

CACHE_ROOT = Path("outputs/motion-cache")
MANIFEST_NAME = "manifest.json"
VIDEO_NAME = "playback.mp4"


def list_motion_clips() -> list[MotionClip]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    clips: list[MotionClip] = []
    for manifest_path in CACHE_ROOT.glob(f"*/{MANIFEST_NAME}"):
        try:
            clips.append(_motion_clip_from_manifest(manifest_path))
        except Exception:
            continue
    return sorted(clips, key=lambda item: item.created_at, reverse=True)


def motion_clip_path(clip_id: str) -> Path:
    return CACHE_ROOT / clip_id


def motion_frame_path(clip_id: str, frame_index: int) -> Path:
    return motion_clip_path(clip_id) / f"frame-{frame_index:04d}.png"


def motion_video_path(clip_id: str) -> Path:
    return motion_clip_path(clip_id) / VIDEO_NAME


def load_motion_clip(clip_id: str) -> MotionClip | None:
    manifest_path = motion_clip_path(clip_id) / MANIFEST_NAME
    if not manifest_path.exists():
        return None
    return _motion_clip_from_manifest(manifest_path)


def _motion_clip_from_manifest(manifest_path: Path) -> MotionClip:
    clip = MotionClip.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    if not clip.video_url and motion_video_path(clip.id).exists():
        clip.video_url = f"/motion/clips/{clip.id}/video"
    return clip


ProgressCallback = Callable[..., None]


def _emit(progress: ProgressCallback | None, phase: str, value: float, message: str, preview_frames: list[MotionClipFrame] | None = None) -> None:
    if progress is not None:
        progress(phase, max(0.0, min(1.0, value)), message, preview_frames)


def generate_motion_clip(frame: MotionClipRequest, progress: ProgressCallback | None = None) -> MotionClip:
    clip_id = uuid.uuid4().hex
    clip_dir = motion_clip_path(clip_id)
    clip_dir.mkdir(parents=True, exist_ok=True)
    source = decode_data_url_rgba(frame.source_image).resize((frame.width, frame.height), Image.Resampling.LANCZOS)
    arrival = decode_data_url_rgba(frame.arrival_image).resize((frame.width, frame.height), Image.Resampling.LANCZOS) if frame.arrival_image else None
    total_frames = _generate_true_loop_frames(source, arrival, frame, clip_dir, progress)
    frames = [MotionClipFrame(index=index, url=f"/motion/clips/{clip_id}/frames/{index}", duration_ms=round(1000 / frame.fps)) for index in range(total_frames)]
    video_url = _encode_playback_video(clip_dir, clip_id, total_frames, frame.fps, frame.loop)
    clip = MotionClip(
        id=clip_id,
        name=frame.name or _default_name(frame),
        motion=frame.motion,
        region=frame.region,
        model=frame.model,
        prompt=frame.prompt,
        arrival_prompt=frame.arrival_prompt,
        negative_prompt=frame.negative_prompt,
        keyframes=["input", *(["arrival"] if arrival is not None else [])],
        fps=frame.fps,
        duration_seconds=frame.duration_seconds,
        loop=frame.loop,
        width=frame.width,
        height=frame.height,
        frame_count=total_frames,
        frames=frames,
        video_url=video_url,
        created_at=time.time(),
    )
    (clip_dir / MANIFEST_NAME).write_text(clip.model_dump_json(indent=2), encoding="utf-8")
    return clip


def _encode_playback_video(clip_dir: Path, clip_id: str, frame_count: int, fps: int, loop: str) -> str:
    output_path = clip_dir / VIDEO_NAME
    if output_path.exists():
        return f"/motion/clips/{clip_id}/video"
    if shutil.which("ffmpeg") is None or frame_count < 2:
        return ""
    sequence = list(range(frame_count))
    if loop == "pingpong" and frame_count > 2:
        sequence.extend(range(frame_count - 2, 0, -1))
    playback_dir = clip_dir / "playback-frames"
    temporary_output = clip_dir / "playback.tmp.mp4"
    if playback_dir.exists():
        shutil.rmtree(playback_dir, ignore_errors=True)
    playback_dir.mkdir(parents=True, exist_ok=True)
    try:
        for playback_index, frame_index in enumerate(sequence):
            shutil.copyfile(clip_dir / f"frame-{frame_index:04d}.png", playback_dir / f"frame-{playback_index:04d}.png")
        if temporary_output.exists():
            temporary_output.unlink()
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(playback_dir / "frame-%04d.png"),
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-level",
                "4.2",
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-movflags",
                "+frag_keyframe+empty_moov+default_base_moof",
                str(temporary_output),
            ],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            timeout=max(60, min(600, round(len(sequence) / max(1, fps) * 30) + 30)),
            check=False,
        )
        if completed.returncode != 0 or not temporary_output.exists():
            return ""
        temporary_output.replace(output_path)
        return f"/motion/clips/{clip_id}/video"
    except Exception:
        return ""
    finally:
        shutil.rmtree(playback_dir, ignore_errors=True)
        if temporary_output.exists():
            temporary_output.unlink(missing_ok=True)


def motion_adapter_configured(model: str) -> bool:
    if _is_causal_forcing_model(model):
        return _causal_forcing_configured(model)
    if _is_krea_realtime_model(model):
        return _krea_realtime_configured()
    if _is_fastvideo_model(model):
        return bool(_fastvideo_command_configured()) or native_video_configured("fastvideo") or native_video_configured("wan")
    if _motion_backend() == "command":
        return bool(os.getenv(_command_env_name(model), "") or os.getenv("RTD_MOTION_COMMAND", ""))
    return native_video_configured(model)


def missing_motion_adapter_message(model: str) -> str:
    if _is_causal_forcing_model(model):
        return _causal_forcing_missing_message(model)
    if _is_krea_realtime_model(model):
        return _krea_realtime_missing_message()
    if _is_fastvideo_model(model):
        return (
            f"{native_video_missing_message('fastvideo')} For Dasiwa/WAN GGUF or NVFP4 models, set "
            "RTD_WAN_GGUF_COMMAND or RTD_FASTVIDEO_GGUF_COMMAND to a Wan-Video/Dasiwa runtime command."
        )
    if _motion_backend() != "command":
        return native_video_missing_message(model)
    return (
        f"No true-loop generator configured for {model}. Set {_command_env_name(model)} or RTD_MOTION_COMMAND "
        "to a WAN/LTX/Chrono command that reads {input}, optionally reads {arrival}, and writes PNG frames to {output_dir}."
    )


def _generate_true_loop_frames(
    source: Image.Image,
    arrival: Image.Image | None,
    frame: MotionClipRequest,
    clip_dir: Path,
    progress: ProgressCallback | None = None,
) -> int:
    if _is_causal_forcing_model(frame.model):
        return _generate_causal_forcing_loop(source, frame, clip_dir, progress)
    if _is_krea_realtime_model(frame.model):
        return _generate_krea_realtime_video(source, frame, clip_dir, progress)
    if _is_fastvideo_model(frame.model):
        if _requires_wan_gguf_runtime(frame.model_path or ""):
            command = _wan_gguf_command()
            if not command:
                raise RuntimeError(
                    "Selected WAN model requires a GGUF/NVFP4 Wan-Video/Dasiwa runtime. Set RTD_WAN_GGUF_COMMAND "
                    "or RTD_FASTVIDEO_GGUF_COMMAND. Available placeholders include {input}, {output_dir}, "
                    "{prompt}, {negative_prompt}, {frames}, {width}, {height}, {wan_high}, and {wan_low}."
                )
            return _generate_external_loop_frames(source, arrival, frame, clip_dir, command)
        command = os.getenv("RTD_FASTVIDEO_COMMAND", "").strip()
        if command:
            return _generate_external_loop_frames(source, arrival, frame, clip_dir, command)
        return generate_native_video_loop(source, arrival, frame, clip_dir, progress)
    if _motion_backend() == "command":
        return _generate_external_loop_frames(source, arrival, frame, clip_dir)
    return generate_native_video_loop(source, arrival, frame, clip_dir, progress)


def _is_causal_forcing_model(model: str) -> bool:
    return model in {"causal-forcing", "causal-forcing-1step", "causal-forcing-2step"}


def _is_krea_realtime_model(model: str) -> bool:
    return model in {"krea-realtime-video", "krea-realtime-14b"}


def _is_fastvideo_model(model: str) -> bool:
    return model == "fastvideo"


def _fastvideo_command_configured() -> str:
    return os.getenv("RTD_FASTVIDEO_COMMAND", "").strip() or _wan_gguf_command()


def _wan_gguf_command() -> str:
    return os.getenv("RTD_WAN_GGUF_COMMAND", "").strip() or os.getenv("RTD_FASTVIDEO_GGUF_COMMAND", "").strip()


def _requires_wan_gguf_runtime(model_path: str) -> bool:
    normalized = model_path.replace("\\", "/").lower()
    return normalized.endswith(".gguf") or "dasiwa" in normalized or "nvfp4" in normalized


def _causal_forcing_configured(model: str) -> bool:
    repo = _causal_forcing_repo()
    return bool(repo and repo.exists() and _causal_forcing_config_path(model).exists() and _causal_forcing_checkpoint_path(model).exists())


def _causal_forcing_missing_message(model: str) -> str:
    return (
        "Causal-Forcing I2V is not configured. Clone https://github.com/thu-ml/Causal-Forcing, "
        "install its environment, download Wan2.1 plus the frame-wise Causal-Forcing checkpoint, then set "
        "RTD_CAUSAL_FORCING_REPO, RTD_CAUSAL_FORCING_PYTHON, RTD_CAUSAL_FORCING_CONFIG, and "
        f"RTD_CAUSAL_FORCING_CHECKPOINT. Current expected config={_causal_forcing_config_path(model)} "
        f"checkpoint={_causal_forcing_checkpoint_path(model)}"
    )


def _generate_causal_forcing_loop(
    source: Image.Image,
    frame: MotionClipRequest,
    clip_dir: Path,
    progress: ProgressCallback | None = None,
) -> int:
    clip_dir = clip_dir.resolve()
    repo = _causal_forcing_repo()
    if repo is None:
        raise RuntimeError(_causal_forcing_missing_message(frame.model))
    config_path = _causal_forcing_config_path(frame.model)
    checkpoint_path = _causal_forcing_checkpoint_path(frame.model)
    if not repo.exists() or not config_path.exists() or not checkpoint_path.exists():
        raise RuntimeError(_causal_forcing_missing_message(frame.model))
    target_ratio = os.getenv("RTD_CAUSAL_FORCING_TARGET_RATIO", "26-15").strip() or "26-15"
    data_dir = clip_dir / "causal-forcing-i2v"
    image_dir = data_dir / target_ratio
    output_dir = clip_dir / "causal-forcing-output"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = image_dir / "input.png"
    causal_source = _fit_image(source, _causal_forcing_input_size(), os.getenv("RTD_CAUSAL_FORCING_FIT", "crop"))
    causal_source.save(source_path, format="PNG")
    prompt = _safe_causal_forcing_caption(_combined_motion_prompt(frame))
    metadata = [
        {
            "file_name": source_path.name,
            "caption": prompt,
            "target_crop": {"target_bbox": [0, 0, causal_source.width, causal_source.height], "target_ratio": target_ratio},
            "type": "rtdiffusion-i2v",
            "origin_width": causal_source.width,
            "origin_height": causal_source.height,
        }
    ]
    (data_dir / f"target_crop_info_{target_ratio}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    frames = _causal_forcing_frame_count(frame)
    command = [
        _causal_forcing_python(),
        "inference.py",
        "--config_path",
        str(config_path),
        "--output_folder",
        str(output_dir),
        "--checkpoint_path",
        str(checkpoint_path),
        "--data_path",
        str(data_dir),
        "--num_output_frames",
        str(frames),
        "--i2v",
    ]
    if os.getenv("RTD_CAUSAL_FORCING_USE_EMA", "1") != "0":
        command.append("--use_ema")
    if os.getenv("RTD_CAUSAL_FORCING_REPORT_TIMING", "0") == "1":
        command.append("--report_timing")
    _emit(progress, "load", 0.05, f"Launching Causal-Forcing I2V {frame.model}")
    timeout = max(1, int(os.getenv("RTD_CAUSAL_FORCING_TIMEOUT", os.getenv("RTD_MOTION_TIMEOUT", "1800"))))
    completed = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "Causal-Forcing inference failed").strip()
        raise RuntimeError(f"Causal-Forcing I2V failed: {details[-3000:]}")
    output_video = _latest_file(output_dir, "*.mp4")
    if output_video is None:
        raise RuntimeError(f"Causal-Forcing did not write an MP4 to {output_dir}")
    shutil.copyfile(output_video, clip_dir / VIDEO_NAME)
    _emit(progress, "extract", 0.92, "Extracting Causal-Forcing video frames")
    count = _extract_video_frames(output_video, clip_dir, frame.width, frame.height, "Causal-Forcing")
    _emit(progress, "complete", 1.0, f"Generated {count} Causal-Forcing frame(s)")
    return count


def _krea_realtime_configured() -> bool:
    if os.getenv("RTD_KREA_REALTIME_COMMAND", "").strip():
        return True
    python = _krea_realtime_python()
    if Path(python).parent != Path(".") and not Path(python).exists():
        return False
    if Path(python).parent == Path(".") and shutil.which(python) is None:
        return False
    repo = _krea_realtime_repo()
    if repo and repo.exists():
        return _krea_realtime_snapshot_ready(repo)
    return os.getenv("RTD_KREA_REALTIME_ALLOW_DOWNLOAD", "0") == "1"


def _krea_realtime_missing_message() -> str:
    repo = _krea_realtime_repo()
    return (
        "Krea Realtime Video is not configured. Finish the krea/krea-realtime-video snapshot download, install a "
        "Diffusers main environment with ModularPipeline support, then set RTD_KREA_REALTIME_REPO and "
        "RTD_KREA_REALTIME_PYTHON. The local snapshot should contain modular_model_index.json and "
        f"krea-realtime-video-14b.safetensors. Current repo={repo or 'unset'} python={_krea_realtime_python()}"
    )


def _krea_realtime_repo() -> Path | None:
    value = os.getenv("RTD_KREA_REALTIME_REPO", "").strip()
    return Path(value) if value else None


def _krea_realtime_python() -> str:
    return os.getenv("RTD_KREA_REALTIME_PYTHON", sys.executable).strip() or sys.executable


def _krea_realtime_snapshot_ready(repo: Path) -> bool:
    required = ["modular_model_index.json", "modular_config.json", "krea-realtime-video-14b.safetensors"]
    return all((repo / name).exists() for name in required) and not any(repo.glob("*.crdownload"))


def _generate_krea_realtime_video(
    source: Image.Image,
    frame: MotionClipRequest,
    clip_dir: Path,
    progress: ProgressCallback | None = None,
) -> int:
    command_template = os.getenv("RTD_KREA_REALTIME_COMMAND", "").strip()
    if command_template:
        return _generate_external_loop_frames(source, None, frame, clip_dir, command_template)
    if not _krea_realtime_configured():
        raise RuntimeError(_krea_realtime_missing_message())
    clip_dir = clip_dir.resolve()
    source_path = clip_dir / "krea_input.png"
    output_path = clip_dir / "krea_raw.mp4"
    runner_path = clip_dir / "krea_realtime_runner.py"
    log_path = clip_dir / "krea_realtime.log"
    krea_width, krea_height = _krea_realtime_input_size()
    _fit_image(source, (krea_width, krea_height), os.getenv("RTD_KREA_REALTIME_INPUT_FIT", "crop")).save(source_path, format="PNG")
    target_frames = _krea_realtime_frame_count(frame)
    frames_per_block = max(1, int(os.getenv("RTD_KREA_REALTIME_FRAMES_PER_BLOCK", "3")))
    blocks = max(1, int(os.getenv("RTD_KREA_REALTIME_BLOCKS", str(target_frames))))
    prompt = _safe_causal_forcing_caption(_combined_motion_prompt(frame))
    repo = _krea_realtime_repo()
    repo_id = str(repo) if repo is not None and _krea_realtime_snapshot_ready(repo) else os.getenv("RTD_KREA_REALTIME_MODEL_ID", "krea/krea-realtime-video")
    local_files_only = bool(repo and _krea_realtime_snapshot_ready(repo)) and os.getenv("RTD_KREA_REALTIME_ALLOW_DOWNLOAD", "0") != "1"
    runner_path.write_text(_krea_realtime_runner_source(), encoding="utf-8")
    env = os.environ.copy()
    hf_home = Path(os.getenv("RTD_KREA_REALTIME_HF_HOME", str((repo / ".hf-cache") if repo is not None else (clip_dir / "hf-cache")))).resolve()
    env.update(
        {
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "RTD_KREA_REPO_ID": repo_id,
            "RTD_KREA_LOCAL_ONLY": "1" if local_files_only else "0",
            "RTD_KREA_INPUT": str(source_path),
            "RTD_KREA_OUTPUT": str(output_path),
            "RTD_KREA_LOG": str(log_path),
            "RTD_KREA_PROMPT": prompt,
            "RTD_KREA_NEGATIVE_PROMPT": frame.negative_prompt,
            "RTD_KREA_WIDTH": str(krea_width),
            "RTD_KREA_HEIGHT": str(krea_height),
            "RTD_KREA_FPS": str(frame.fps),
            "RTD_KREA_BLOCKS": str(blocks),
            "RTD_KREA_STEPS": os.getenv("RTD_KREA_REALTIME_STEPS", "1"),
            "RTD_KREA_STRENGTH": os.getenv("RTD_KREA_REALTIME_STRENGTH", "0.3"),
            "RTD_KREA_FRAME_SAMPLE_LEN": os.getenv("RTD_KREA_REALTIME_FRAME_SAMPLE_LEN", str(frames_per_block)),
            "RTD_KREA_SEED": os.getenv("RTD_KREA_REALTIME_SEED", "42"),
            "RTD_KREA_DTYPE": os.getenv("RTD_KREA_REALTIME_DTYPE", "bfloat16"),
            "RTD_KREA_DISABLE_TORCH_COMPILE": os.getenv("RTD_KREA_DISABLE_TORCH_COMPILE", "1"),
            "TORCHDYNAMO_DISABLE": os.getenv("TORCHDYNAMO_DISABLE", "1"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    _emit(progress, "load", 0.05, "Launching Krea Realtime Video")
    timeout = max(1, int(os.getenv("RTD_KREA_REALTIME_TIMEOUT", os.getenv("RTD_MOTION_TIMEOUT", "1800"))))
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        completed = subprocess.run([_krea_realtime_python(), str(runner_path)], cwd=str(clip_dir), env=env, text=True, stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if completed.returncode != 0 or not output_path.exists():
        details = log_path.read_text(encoding="utf-8", errors="replace").strip() if log_path.exists() else "Krea realtime inference failed"
        raise RuntimeError(f"Krea Realtime Video failed: {details[-6000:]}")
    _emit(progress, "extract", 0.92, "Extracting Krea video frames")
    count = _extract_video_frames(output_path, clip_dir, frame.width, frame.height, "Krea Realtime Video")
    _emit(progress, "complete", 1.0, f"Generated {count} Krea frame(s)")
    return count


def _krea_realtime_frame_count(frame: MotionClipRequest) -> int:
    requested = max(2, round(frame.duration_seconds * frame.fps))
    limit = max(2, int(os.getenv("RTD_KREA_REALTIME_MAX_FRAMES", "3")))
    return min(requested, limit)


def _krea_realtime_input_size() -> tuple[int, int]:
    value = os.getenv("RTD_KREA_REALTIME_INPUT_SIZE", "832x480").lower().replace("*", "x")
    try:
        width, height = [int(part.strip()) for part in value.split("x", 1)]
        return max(128, width), max(128, height)
    except Exception:
        return 832, 480


def _krea_realtime_runner_source() -> str:
    return r'''
import os
import json
import sys
from pathlib import Path

import ftfy
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from PIL import Image
from diffusers import ModularPipeline
from diffusers.models import AutoModel
from diffusers.modular_pipelines import PipelineState
from diffusers.utils import export_to_video


if os.environ.get("RTD_KREA_DISABLE_TORCH_COMPILE", "1") == "1":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass

    def _identity_compile(model=None, *args, **kwargs):
        if model is None:
            return lambda fn: fn
        return model

    torch.compile = _identity_compile


def log(message):
    print(message, flush=True)


def krea_dtype(device):
    value = os.environ.get("RTD_KREA_DTYPE", "float16" if device == "cuda" else "float32").strip().lower()
    if value in {"bf16", "bfloat16"}:
        if device != "cuda" or not torch.cuda.is_bf16_supported():
            log("bfloat16 requested but unsupported on this device; falling back to float16")
            return torch.float16 if device == "cuda" else torch.float32
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16 if device == "cuda" else torch.float32
    if value in {"fp32", "float32"}:
        return torch.float32
    raise RuntimeError(f"Unsupported RTD_KREA_DTYPE={value!r}")


def load_krea_transformer(repo_id, device, local_files_only, dtype):
    repo = Path(repo_id)
    log(f"device={device} cuda_available={torch.cuda.is_available()} dtype={dtype}")
    if torch.cuda.is_available():
        log(f"cuda_device={torch.cuda.get_device_name(torch.cuda.current_device())}")
    if repo.exists() and (repo / "krea-realtime-video-14b.safetensors").exists():
        sys.path.insert(0, str(repo))
        from transformer.causal_model import CausalWanModel

        with init_empty_weights():
            transformer = CausalWanModel.from_config(str(repo / "transformer"), torch_dtype=dtype)
        state_device = "cuda" if device == "cuda" else "cpu"
        log(f"loading transformer checkpoint to {state_device}")
        state = load_file(str(repo / "krea-realtime-video-14b.safetensors"), device=state_device)
        state = {key.removeprefix("model."): value for key, value in state.items()}
        log("assigning transformer state dict")
        missing, unexpected = transformer.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"Krea transformer checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        transformer.to(device=device, dtype=dtype)
        transformer.eval()
        log("transformer ready")
        return transformer
    return AutoModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=dtype,
        device_map="cuda" if device == "cuda" else None,
    )

repo_id = os.environ["RTD_KREA_REPO_ID"]
local_files_only = os.environ.get("RTD_KREA_LOCAL_ONLY", "0") == "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = krea_dtype(device)
log(f"loading modular pipeline from {repo_id}")
pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True, local_files_only=local_files_only)
log("loading text_encoder/tokenizer/scheduler/vae")
pipe.load_components(
    names=["text_encoder", "tokenizer", "scheduler", "vae"],
    trust_remote_code=True,
    device_map="cuda" if device == "cuda" else None,
    torch_dtype={"default": dtype, "vae": torch.float16 if device == "cuda" else torch.float32},
)
transformer = load_krea_transformer(repo_id, device, local_files_only, dtype)
pipe.update_components(transformer=transformer)
transformer = getattr(pipe, "transformer", None)
if transformer is None:
    raise RuntimeError("Krea transformer failed to load; pipeline.transformer is still None")
if transformer is not None:
    log("fusing transformer attention projections")
    for block in getattr(transformer, "blocks", []):
        if hasattr(block, "self_attn") and hasattr(block.self_attn, "fuse_projections"):
            block.self_attn.fuse_projections()

width = int(os.environ["RTD_KREA_WIDTH"])
height = int(os.environ["RTD_KREA_HEIGHT"])
source = Image.open(os.environ["RTD_KREA_INPUT"]).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
prompt = [os.environ.get("RTD_KREA_PROMPT", "a video with smooth natural motion")]
blocks = int(os.environ.get("RTD_KREA_BLOCKS", "9"))
steps = int(os.environ.get("RTD_KREA_STEPS", "6"))
strength = float(os.environ.get("RTD_KREA_STRENGTH", "0.3"))
frame_sample_len = int(os.environ.get("RTD_KREA_FRAME_SAMPLE_LEN", "12"))
frame_sample_len = max(1, min(frame_sample_len, int(getattr(pipe.config, "num_frames_per_block", frame_sample_len))))
fps = int(os.environ.get("RTD_KREA_FPS", "24"))
seed = int(os.environ.get("RTD_KREA_SEED", "42"))
generator_device = "cuda" if device == "cuda" else "cpu"
generator = torch.Generator(generator_device).manual_seed(seed)

frames = []
state = PipelineState()
for block_idx in range(blocks):
    log(f"running block {block_idx + 1}/{blocks}")
    frame_sample = [source.copy() for _ in range(frame_sample_len)]
    state = pipe(
        state,
        video_stream=frame_sample,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        num_blocks=blocks,
        strength=strength,
        block_idx=block_idx,
        generator=generator,
    )
    frames.extend(state.values["videos"][0])

log(f"exporting {len(frames)} frames")
export_to_video(frames, os.environ["RTD_KREA_OUTPUT"], fps=fps)
'''.lstrip()


def _causal_forcing_repo() -> Path | None:
    value = os.getenv("RTD_CAUSAL_FORCING_REPO", "").strip()
    return Path(value) if value else None


def _causal_forcing_python() -> str:
    return os.getenv("RTD_CAUSAL_FORCING_PYTHON", sys.executable).strip() or sys.executable


def _causal_forcing_config_path(model: str) -> Path:
    configured = os.getenv("RTD_CAUSAL_FORCING_CONFIG", "").strip()
    if configured:
        return Path(configured)
    repo = _causal_forcing_repo() or Path("")
    config_name = {
        "causal-forcing-1step": "causal_forcing_dmd_framewise_1step.yaml",
        "causal-forcing-2step": "causal_forcing_dmd_framewise_2step.yaml",
    }.get(model, "causal_forcing_dmd_framewise.yaml")
    return repo / "configs" / config_name


def _causal_forcing_checkpoint_path(model: str) -> Path:
    configured = os.getenv("RTD_CAUSAL_FORCING_CHECKPOINT", "").strip()
    if configured:
        return Path(configured)
    repo = _causal_forcing_repo() or Path("")
    if model == "causal-forcing-1step":
        return repo / "checkpoints" / "causal-forcing++" / "framewise-1step.pt"
    if model == "causal-forcing-2step":
        return repo / "checkpoints" / "causal-forcing++" / "framewise-2step.pt"
    return repo / "checkpoints" / "framewise" / "causal_forcing.pt"


def _causal_forcing_frame_count(frame: MotionClipRequest) -> int:
    requested = max(2, round(frame.duration_seconds * frame.fps))
    limit = max(2, int(os.getenv("RTD_CAUSAL_FORCING_MAX_FRAMES", "33")))
    latent_window = max(2, int(os.getenv("RTD_CAUSAL_FORCING_LATENT_WINDOW", "21")))
    return min(requested, limit, latent_window)


def _causal_forcing_input_size() -> tuple[int, int]:
    value = os.getenv("RTD_CAUSAL_FORCING_INPUT_SIZE", "832x480").lower().replace("*", "x")
    try:
        width, height = [int(part.strip()) for part in value.split("x", 1)]
        return max(128, width), max(128, height)
    except Exception:
        return 832, 480


def _combined_motion_prompt(frame: MotionClipRequest) -> str:
    prompt = frame.prompt.strip() or _default_name(frame)
    arrival_prompt = frame.arrival_prompt.strip()
    if arrival_prompt and arrival_prompt not in prompt:
        prompt = f"{prompt}, target motion and appearance: {arrival_prompt}"
    if frame.motion and frame.motion not in prompt:
        prompt = f"{prompt}, {frame.motion} motion"
    return prompt


def _safe_causal_forcing_caption(prompt: str) -> str:
    safe = "".join(character if character not in '<>:"/\\|?*\n\r\t' else " " for character in prompt)
    safe = " ".join(safe.split())
    return (safe or "rtdiffusion alive character")[:96]


def _fit_image(image: Image.Image, size: tuple[int, int], mode: str = "crop") -> Image.Image:
    image = image.convert("RGB")
    if mode.strip().lower() in {"contain", "letterbox", "pad"}:
        return ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color=(0, 0, 0), centering=(0.5, 0.5))
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _latest_file(root: Path, pattern: str) -> Path | None:
    files = [path for path in root.glob(pattern) if path.is_file()]
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def _extract_video_frames(video_path: Path, clip_dir: Path, width: int, height: int, label: str, rotate: int = 0) -> int:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"ffmpeg is required to extract {label} MP4 frames")
    extracted_dir = clip_dir / "video-frames"
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir, ignore_errors=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            str(extracted_dir / "frame-%04d.png"),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "ffmpeg frame extraction failed").strip()
        raise RuntimeError(f"Could not extract {label} frames: {details[-2000:]}")
    extracted = sorted(path for path in extracted_dir.glob("*.png") if path.is_file())
    if len(extracted) < 2:
        raise RuntimeError(f"{label} MP4 contained only {len(extracted)} extracted frame(s); increase the model block/frame count")
    for index, path in enumerate(extracted):
        image = Image.open(path).convert("RGB")
        if rotate == 90:
            image = image.transpose(Image.Transpose.ROTATE_90)
        elif rotate == 180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        elif rotate == 270:
            image = image.transpose(Image.Transpose.ROTATE_270)
        image = _fit_image(image, (width, height), os.getenv("RTD_MOTION_OUTPUT_FIT", "crop"))
        image.save(clip_dir / f"frame-{index:04d}.png", format="PNG", optimize=False)
    shutil.rmtree(extracted_dir, ignore_errors=True)
    return len(extracted)


def _motion_backend() -> str:
    return os.getenv("RTD_MOTION_BACKEND", "native").strip().lower()


def _generate_external_loop_frames(source: Image.Image, arrival: Image.Image | None, frame: MotionClipRequest, clip_dir: Path, command_template: str | None = None) -> int:
    command_template = command_template or os.getenv(_command_env_name(frame.model), "") or os.getenv("RTD_MOTION_COMMAND", "")
    if not command_template:
        raise RuntimeError(missing_motion_adapter_message(frame.model))
    input_path = clip_dir / "input.png"
    arrival_path = clip_dir / "arrival.png"
    request_path = clip_dir / "request.json"
    output_dir = clip_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    source.save(input_path, format="PNG")
    if arrival is not None:
        arrival.save(arrival_path, format="PNG")
    request_path.write_text(frame.model_dump_json(indent=2), encoding="utf-8")
    values = {
        "input": str(input_path),
        "arrival": str(arrival_path) if arrival is not None else "",
        "output_dir": str(output_dir),
        "request": str(request_path),
        "prompt": frame.prompt,
        "arrival_prompt": frame.arrival_prompt,
        "negative_prompt": frame.negative_prompt,
        "model": frame.model,
        "model_path": frame.model_path or "",
        "wan_model": frame.model_path or "",
        "wan_high": str(_wan_high_path(frame.model_path or "")),
        "wan_low": str(_wan_low_path(frame.model_path or "")),
        "motion": frame.motion,
        "region": frame.region,
        "fps": str(frame.fps),
        "duration": str(frame.duration_seconds),
        "frames": str(max(2, int(frame.duration_seconds * frame.fps))),
        "width": str(frame.width),
        "height": str(frame.height),
    }
    command = command_template.format_map(_SafeFormat(values))
    timeout = max(1, int(os.getenv("RTD_MOTION_TIMEOUT", "900")))
    completed = subprocess.run(command, cwd=Path.cwd(), shell=True, text=True, capture_output=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "motion command failed").strip()
        raise RuntimeError(f"{frame.model} loop generator failed: {details[-2000:]}")
    generated = sorted(path for path in output_dir.glob("*.png") if path.is_file())
    if len(generated) < 2:
        output_video = _latest_file(output_dir, "*.mp4") or _latest_file(clip_dir, "*.mp4")
        if output_video is not None:
            shutil.copyfile(output_video, clip_dir / VIDEO_NAME)
            return _extract_video_frames(output_video, clip_dir, frame.width, frame.height, frame.model)
        raise RuntimeError(f"{frame.model} loop generator did not write at least two PNG frames or an MP4 to {output_dir}")
    for index, path in enumerate(generated):
        target = clip_dir / f"frame-{index:04d}.png"
        if path.resolve() != target.resolve():
            shutil.copyfile(path, target)
    return len(generated)


def _wan_high_path(model_path: str) -> str:
    configured = os.getenv("RTD_WAN_GGUF_HIGH_MODEL", "").strip() or os.getenv("RTD_FASTVIDEO_GGUF_HIGH_MODEL", "").strip()
    if configured:
        return configured
    path = Path(model_path)
    if "high" in path.name.lower():
        return str(path)
    paired = _paired_wan_component_path(path, "low", "high")
    return str(paired or path)


def _wan_low_path(model_path: str) -> str:
    configured = os.getenv("RTD_WAN_GGUF_LOW_MODEL", "").strip() or os.getenv("RTD_FASTVIDEO_GGUF_LOW_MODEL", "").strip()
    if configured:
        return configured
    path = Path(model_path)
    if "low" in path.name.lower():
        return str(path)
    paired = _paired_wan_component_path(path, "high", "low")
    return str(paired or path)


def _paired_wan_component_path(path: Path, source_token: str, target_token: str) -> Path | None:
    name = path.name
    for source, target in (
        (source_token, target_token),
        (source_token.capitalize(), target_token.capitalize()),
        (source_token.upper(), target_token.upper()),
    ):
        if source in name:
            candidate = path.with_name(name.replace(source, target))
            if candidate.exists():
                return candidate
    return None


class _SafeFormat(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _command_env_name(model: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in model.upper()).strip("_")
    return f"RTD_MOTION_{normalized}_COMMAND"


def _default_name(frame: MotionClipRequest) -> str:
    region = frame.region if frame.region != "full" else "scene"
    return f"{frame.motion} {region}"
