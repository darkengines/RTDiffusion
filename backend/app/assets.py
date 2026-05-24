import os
from pathlib import Path

from .config import load_local_env

load_local_env()

ROOT = Path(__file__).resolve().parents[2]


def _hf_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.getenv(env_name, "").strip()
        if value:
            roots.append(Path(value).expanduser())
    for env_name in ("HF_HOME", "RTD_NATIVE_VIDEO_HF_CACHE"):
        value = os.getenv(env_name, "").strip()
        if value:
            root = Path(value).expanduser()
            roots.extend([root, root / "hub"])
    roots.extend([ROOT / "outputs" / "hf-cache", ROOT / "outputs" / "hf-cache" / "hub", Path.home() / ".cache" / "huggingface" / "hub"])
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _path_list(env_name: str, defaults: list[Path]) -> list[Path]:
    configured = [Path(value).expanduser() for value in os.getenv(env_name, "").split(os.pathsep) if value]
    return configured or defaults


MODEL_DIRS = _path_list("RTD_MODEL_DIRS", [ROOT / "models", ROOT / "checkpoints", ROOT / "diffusers"]) + _hf_roots()
LORA_DIRS = _path_list("RTD_LORA_DIRS", [ROOT / "loras", ROOT / "models" / "loras"]) + _hf_roots()
VIDEO_MODEL_DIRS = _path_list("RTD_VIDEO_MODEL_DIRS", MODEL_DIRS) + _hf_roots()

MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt"}
VIDEO_MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt"}
LORA_SUFFIXES = {".safetensors", ".pt"}
DEFAULT_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def list_assets() -> dict[str, list[dict[str, str | int | bool]]]:
    models = _model_assets()
    video_models = _video_model_assets()
    # Discover LoRAs in the configured/default LORA_DIRS *and* alongside every
    # discovered model — this is the ComfyUI convention (a single tree with
    # sibling ``loras/`` and ``checkpoints/`` directories). Without this, a user
    # whose models are at ``D:\comfyui\comfy\models\checkpoints\*.safetensors``
    # would see "No LoRAs available" even when ``loras/`` exists right next to
    # the checkpoints — only because that path isn't in ``RTD_LORA_DIRS``.
    lora_dirs = list(LORA_DIRS) + _lora_dirs_from_models(models)
    lora_candidates = _scan(_dedupe_paths(lora_dirs), LORA_SUFFIXES)
    # Anything we already classified as a model must not also appear as a LoRA.
    model_paths = {_resolve_str(item.get("path")) for item in models}
    loras = [
        _describe(path, preferred=False)
        for path in lora_candidates
        if _resolve_str(str(path)) not in model_paths
    ]
    return {
        "models": sorted(models, key=lambda item: (not bool(item["preferred"]), str(item["name"]).lower())),
        "video_models": sorted(video_models, key=lambda item: (not bool(item["preferred"]), str(item["name"]).lower())),
        "loras": sorted(_dedupe_assets(loras), key=lambda item: str(item["name"]).lower()),
    }


def _lora_dirs_from_models(models: list[dict[str, str | int | bool]]) -> list[Path]:
    """Derive plausible LoRA directories from the discovered models.

    We look ONLY at sibling directories named ``loras`` / ``Lora`` / ``lora`` /
    ``LoRA`` next to the model file or its parent (i.e. the ComfyUI layout with
    ``checkpoints/`` and ``loras/`` siblings, or the Automatic1111 layout with
    ``models/Stable-diffusion/`` and ``models/Lora/`` siblings).

    We deliberately do NOT add the model's own parent directory — without a
    way to distinguish a model file from a LoRA file at the path level
    (both are ``.safetensors``), the same files would get classified as both.
    Users who keep LoRAs in the same folder as their models must add that
    folder to ``RTD_LORA_DIRS`` explicitly.
    """
    candidates: list[Path] = []
    for item in models:
        raw = str(item.get("path") or "")
        if not raw:
            continue
        model_path = Path(raw).expanduser()
        if not model_path.exists() or not model_path.is_file():
            continue
        parent = model_path.parent
        ancestors: list[Path] = [parent]
        if parent.parent != parent:
            ancestors.append(parent.parent)
        for ancestor in ancestors:
            for name in ("loras", "Lora", "lora", "LoRA"):
                sibling = ancestor / name
                if sibling.exists() and sibling.is_dir():
                    candidates.append(sibling)
    return candidates


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = _resolve_str(str(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _resolve_str(value: object | None) -> str:
    """Best-effort canonical string for a path. Falls back to the raw string
    when the path is not resolvable (e.g. an HF model ID)."""
    if not value:
        return ""
    try:
        return str(Path(str(value)).expanduser().resolve())
    except (OSError, RuntimeError):
        return str(value)


def default_model_path() -> str:
    configured_path = os.getenv("RTD_MODEL_PATH", "").strip()
    if configured_path:
        return str(Path(configured_path).expanduser())
    configured_id = os.getenv("RTD_MODEL_ID", "").strip()
    if configured_id:
        return configured_id
    return DEFAULT_MODEL_ID


def _model_assets() -> list[dict[str, str | int | bool]]:
    items: list[dict[str, str | int | bool]] = []
    configured_path = os.getenv("RTD_MODEL_PATH", "").strip()
    configured_id = os.getenv("RTD_MODEL_ID", "").strip()
    if configured_path:
        items.append(_describe_configured_path(configured_path, preferred=True))
    if configured_id:
        items.append(_describe_model_id(configured_id, preferred=not configured_path))
    for path in _scan(MODEL_DIRS, MODEL_SUFFIXES):
        items.append(_describe(path, preferred=_is_preferred_model(path)))
    items.append(_describe_model_id(DEFAULT_MODEL_ID, preferred=not configured_path and not configured_id))
    return _dedupe_assets(items)


def _video_model_assets() -> list[dict[str, str | int | bool]]:
    items: list[dict[str, str | int | bool]] = []
    for env_name in (
        "RTD_NATIVE_WAN_MODEL",
        "RTD_NATIVE_WAN_HIGH_MODEL",
        "RTD_NATIVE_WAN_LOW_MODEL",
        "RTD_NATIVE_FASTVIDEO_MODEL",
        "RTD_NATIVE_FASTVIDEO_HIGH_MODEL",
        "RTD_NATIVE_FASTVIDEO_LOW_MODEL",
        "RTD_FASTVIDEO_MODEL",
        "RTD_FASTVIDEO_HIGH_MODEL",
        "RTD_FASTVIDEO_LOW_MODEL",
    ):
        configured = os.getenv(env_name, "").strip()
        if configured:
            items.append(_describe_configured_path(configured, preferred=True))
    for path in _scan(VIDEO_MODEL_DIRS, VIDEO_MODEL_SUFFIXES):
        name = path.name.lower()
        if any(token in name for token in ("wan", "fastvideo", "ltx", "hunyuan", "cogvideo")) and not _unsupported_native_video_asset(path):
            items.append(_describe(path, preferred=False))
    return _dedupe_assets(items)


def _unsupported_native_video_asset(path: Path) -> bool:
    if _wan_gguf_runtime_configured():
        return False
    if "dasiwa" in path.name.lower():
        return True
    if path.suffix.lower() == ".gguf":
        return True
    if path.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
        return metadata.get("quantization.bits", "").upper() == "NVFP4"
    except Exception:
        return False


def _wan_gguf_runtime_configured() -> bool:
    return bool(os.getenv("RTD_WAN_GGUF_COMMAND", "").strip() or os.getenv("RTD_FASTVIDEO_GGUF_COMMAND", "").strip())


def _scan(roots: list[Path], suffixes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            model_dirs = [path.parent for path in root.rglob("model_index.json") if path.is_file()]
            paths.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes and not _inside_diffusers_model_dir(path, model_dirs))
            paths.extend(model_dirs)
    return paths


def _inside_diffusers_model_dir(path: Path, model_dirs: list[Path]) -> bool:
    return any(path == model_dir or model_dir in path.parents for model_dir in model_dirs)


def _describe(path: Path, *, preferred: bool) -> dict[str, str | int | bool]:
    return {
        "name": path.stem,
        "path": str(path),
        "size": _path_size(path),
        "preferred": preferred,
    }


def _describe_configured_path(path: str, *, preferred: bool) -> dict[str, str | int | bool]:
    model_path = Path(path).expanduser()
    return {
        "name": model_path.stem or model_path.name or path,
        "path": str(model_path),
        "size": _path_size(model_path) if model_path.exists() else 0,
        "preferred": preferred,
    }


def _describe_model_id(model_id: str, *, preferred: bool) -> dict[str, str | int | bool]:
    return {
        "name": model_id.rstrip("/").split("/")[-1] or model_id,
        "path": model_id,
        "size": 0,
        "preferred": preferred,
    }


def _dedupe_assets(items: list[dict[str, str | int | bool]]) -> list[dict[str, str | int | bool]]:
    deduped: dict[str, dict[str, str | int | bool]] = {}
    for item in items:
        key = str(item["path"])
        if key in deduped:
            deduped[key]["preferred"] = bool(deduped[key]["preferred"]) or bool(item["preferred"])
        else:
            deduped[key] = item
    return list(deduped.values())


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _is_preferred_model(path: Path) -> bool:
    return path == Path(os.getenv("RTD_MODEL_PATH", "")).expanduser()
