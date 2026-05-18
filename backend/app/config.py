import os
from pathlib import Path


def load_local_env() -> None:
    root = Path(__file__).resolve().parents[2]
    explicit_env = os.getenv("RTD_ENV_FILE", "").strip()
    if explicit_env:
        env_files = (Path(explicit_env),)
    elif Path("/.dockerenv").exists():
        env_files = (root / ".env.docker", root / ".env")
    else:
        env_files = (root / ".env.local", root / ".env")
    for env_file in env_files:
        if env_file.exists():
            _load_env_file(env_file)


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value