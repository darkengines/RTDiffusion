"""LoRA discovery tests for the ComfyUI-style sibling-folder convention.

Run:  pytest backend/tests/test_lora_sibling_discovery.py -v

Why this exists
---------------
The default ``RTD_LORA_DIRS`` scans ``<project>/loras`` and ``<project>/models/loras``.
Users with a ComfyUI-style layout

    D:\\comfyui\\comfy\\models\\
        checkpoints\\dreamshaper_8.safetensors
        loras\\sdxl_latent_consistency.safetensors

would see "No LoRAs available" in the UI even with the LoRA file present —
because the checkpoint dir is in ``RTD_MODEL_DIRS`` but the LoRA dir isn't in
``RTD_LORA_DIRS``. The fix derives candidate LoRA dirs from every discovered
model's sibling/parent directories.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _make_safetensors(path: Path, *, contents: bytes = b"\x00\x00\x00\x00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def _isolated_assets(monkeypatch, model_dirs: list[Path], lora_dirs: list[Path]):
    """Reload ``app.assets`` and force its module-level scan dirs to point only
    at the supplied paths. Without this the user's real environment (HF cache,
    .env file, other RTD_* vars) leaks into the discovery results and the
    tmp_path fixtures get drowned by hundreds of unrelated LoRAs."""
    import sys
    # Clear EVERY env var that contributes to the global scan dirs. This way
    # the developer's real HF cache / model folders can't sneak in.
    for var in (
        "RTD_MODEL_DIRS", "RTD_LORA_DIRS", "RTD_VIDEO_MODEL_DIRS",
        "RTD_MODEL_PATH", "RTD_MODEL_ID",
        "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME",
        "RTD_NATIVE_VIDEO_HF_CACHE",
        "RTD_ENV_FILE",  # so load_local_env can't reload .env
    ):
        monkeypatch.delenv(var, raising=False)
    # Force load_local_env to a non-existent path so it can't read .env.local
    monkeypatch.setenv("RTD_ENV_FILE", "/__does_not_exist__")
    if "app.assets" in sys.modules:
        del sys.modules["app.assets"]
    from app import assets as _assets
    # Override the module's scan dirs directly so it cannot see anything else.
    monkeypatch.setattr(_assets, "MODEL_DIRS", model_dirs, raising=True)
    monkeypatch.setattr(_assets, "LORA_DIRS", lora_dirs, raising=True)
    monkeypatch.setattr(_assets, "VIDEO_MODEL_DIRS", model_dirs, raising=True)
    return _assets


class TestLoraDiscoverySiblings:
    def test_comfyui_layout_picks_up_loras(self, tmp_path, monkeypatch):
        """Models in ``checkpoints/`` reveal LoRAs in sibling ``loras/`` dir."""
        # Layout:
        #   <tmp>/comfy/models/checkpoints/dummy_model.safetensors
        #   <tmp>/comfy/models/loras/dummy_lora.safetensors
        models_dir = tmp_path / "comfy" / "models" / "checkpoints"
        loras_dir = tmp_path / "comfy" / "models" / "loras"
        _make_safetensors(models_dir / "dummy_model.safetensors")
        _make_safetensors(loras_dir / "dummy_lora.safetensors")

        assets = _isolated_assets(monkeypatch, [models_dir], [tmp_path / "nonexistent"])
        catalog = assets.list_assets()
        lora_paths = [str(item["path"]) for item in catalog["loras"]]
        assert any("dummy_lora.safetensors" in p for p in lora_paths), \
            f"sibling LoRA was not discovered; catalog={lora_paths}"

    def test_explicit_rtd_lora_dirs_used(self, tmp_path, monkeypatch):
        """Honours ``RTD_LORA_DIRS`` even when the dir isn't a model sibling."""
        far_lora_dir = tmp_path / "elsewhere" / "loras"
        _make_safetensors(far_lora_dir / "far_lora.safetensors")
        models_dir = tmp_path / "models"
        _make_safetensors(models_dir / "model.safetensors")

        assets = _isolated_assets(monkeypatch, [models_dir], [far_lora_dir])
        catalog = assets.list_assets()
        lora_names = {str(item["name"]) for item in catalog["loras"]}
        assert "far_lora" in lora_names

    def test_no_duplicate_when_lora_dir_in_both_configured_and_sibling(self, tmp_path, monkeypatch):
        loras_dir = tmp_path / "loras"
        _make_safetensors(loras_dir / "shared.safetensors")
        models_dir = tmp_path / "ckpt"
        _make_safetensors(models_dir / "m.safetensors")

        assets = _isolated_assets(monkeypatch, [models_dir], [loras_dir])
        catalog = assets.list_assets()
        shared_count = sum(1 for item in catalog["loras"] if "shared.safetensors" in str(item["path"]))
        assert shared_count == 1, f"LoRA discovered twice: {catalog['loras']}"

    def test_automatic1111_layout_also_works(self, tmp_path, monkeypatch):
        """The A1111 layout uses capitalized ``Lora`` next to ``models``."""
        models_dir = tmp_path / "models" / "Stable-diffusion"
        lora_dir = tmp_path / "models" / "Lora"
        _make_safetensors(models_dir / "ck.safetensors")
        _make_safetensors(lora_dir / "a1111.safetensors")

        assets = _isolated_assets(monkeypatch, [models_dir], [tmp_path / "nonexistent"])
        catalog = assets.list_assets()
        assert any("a1111.safetensors" in str(item["path"]) for item in catalog["loras"])

    def test_model_files_never_appear_as_loras(self, tmp_path, monkeypatch):
        """Regression: when MODEL_DIRS and LORA_DIRS overlap, files already
        classified as models never appear in the LoRA catalog. This is the
        ComfyUI scenario where ``checkpoints/`` is in MODEL_DIRS and
        ``loras/`` is its sibling — but the SAME .safetensors file shouldn't
        be reachable as both."""
        models_dir = tmp_path / "checkpoints"
        loras_dir = tmp_path / "loras"
        _make_safetensors(models_dir / "the_model.safetensors")
        _make_safetensors(loras_dir / "the_lora.safetensors")

        assets = _isolated_assets(monkeypatch, [models_dir], [tmp_path / "no"])
        catalog = assets.list_assets()
        model_names = {str(item["name"]) for item in catalog["models"]}
        lora_names = {str(item["name"]) for item in catalog["loras"]}
        # Models stay in models, LoRAs stay in loras — no cross-classification
        assert "the_model" in model_names
        assert "the_model" not in lora_names
        assert "the_lora" in lora_names
        assert "the_lora" not in model_names
