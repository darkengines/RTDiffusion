import os
from pathlib import Path

def _path_list(env_name: str) -> list[Path]:
    return [Path(value).expanduser() for value in os.getenv(env_name, "").split(os.pathsep) if value]


MODEL_DIRS = _path_list("RTD_MODEL_DIRS")
LORA_DIRS = _path_list("RTD_LORA_DIRS")

MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt"}
LORA_SUFFIXES = {".safetensors", ".pt"}


def list_assets() -> dict[str, list[dict[str, str | int | bool]]]:
    models = [_describe(path, preferred=_is_preferred_model(path)) for path in _scan(MODEL_DIRS, MODEL_SUFFIXES)]
    loras = [_describe(path, preferred=False) for path in _scan(LORA_DIRS, LORA_SUFFIXES)]
    return {
        "models": sorted(models, key=lambda item: (not bool(item["preferred"]), str(item["name"]).lower())),
        "loras": sorted(loras, key=lambda item: str(item["name"]).lower()),
    }


def default_model_path() -> str:
    preferred = [item for item in list_assets()["models"] if item["preferred"]]
    if preferred:
        return str(preferred[0]["path"])
    models = list_assets()["models"]
    if models:
        return str(models[0]["path"])
    return "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def _scan(roots: list[Path], suffixes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
            paths.extend(path.parent for path in root.rglob("model_index.json") if path.is_file())
    return paths


def _describe(path: Path, *, preferred: bool) -> dict[str, str | int | bool]:
    return {
        "name": path.stem,
        "path": str(path),
        "size": _path_size(path),
        "preferred": preferred,
    }


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _is_preferred_model(path: Path) -> bool:
    return path == Path(os.getenv("RTD_MODEL_PATH", "")).expanduser()
