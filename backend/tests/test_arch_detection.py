"""Tests for SD 1.5 / SDXL architecture detection.

Run:  pytest backend/tests/test_arch_detection.py -v

Why this exists
---------------
``_load_single_file`` previously tried ``StableDiffusionXLImg2ImgPipeline``
first and fell back to SD 1.5. For an SD 1.5 checkpoint, the SDXL load would
"succeed" with 6/7 components (no ``text_encoder_2``), and the resulting
malformed pipe crashed inside diffusers' ``encode_prompt`` looking for the
pooled output of the missing encoder.

The fix peeks at the .safetensors metadata to pick the right class up front.
These tests verify the detector using tiny synthetic safetensors files that
expose the same keys real SD/SDXL checkpoints use to signal their topology.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Other tests install ``sys.modules`` stubs (bare ``types.ModuleType``) for
# heavy packages they don't need. We want the *real* torch / diffusers /
# safetensors here, so evict ONLY the stubbed entries — leaving real packages
# alone (deleting a real torch breaks the unreloadable C extension state).
_NAMESPACES = ("torch", "diffusers", "transformers", "huggingface_hub",
               "onnxruntime", "safetensors", "tokenizers", "accelerate")


def _is_stub(mod) -> bool:
    """A test stub is a bare ``types.ModuleType`` with no real file."""
    if mod is None:
        return False
    if not hasattr(mod, "__file__"):
        return True
    return getattr(mod, "__file__", None) is None


for _name in list(sys.modules):
    if _name in _NAMESPACES or any(_name.startswith(ns + ".") for ns in _NAMESPACES):
        if _is_stub(sys.modules[_name]):
            del sys.modules[_name]

# Skip the whole module if real torch / safetensors aren't installed —
# the detector is a best-effort fast path that gracefully falls back to
# "unknown" in their absence, so these tests are only meaningful with them.
safetensors = pytest.importorskip("safetensors")
_real_torch = pytest.importorskip("torch")
if not hasattr(_real_torch, "zeros"):
    pytest.skip("torch was stubbed by another test module", allow_module_level=True)


from app.stream.manager import (  # noqa: E402
    _apply_loras,
    _detect_lora_cross_attention_dim,
    _detect_sd_architecture,
)


# ── Synthetic .safetensors fixtures ──────────────────────────────────────────

def _write_safetensors(tmp_path: Path, name: str, tensors: dict) -> Path:
    from safetensors.torch import save_file
    path = tmp_path / name
    save_file(tensors, str(path))
    return path


# ── Tests ────────────────────────────────────────────────────────────────────

class TestSDXLDetection:
    def test_detects_sdxl_via_conditioner_embedders_1(self, tmp_path):
        # SDXL ComfyUI/Stability format
        tensors = {
            "conditioner.embedders.1.model.transformer.text_model.embeddings.token_embedding.weight":
                _real_torch.zeros(49408, 1280),
            "model.diffusion_model.input_blocks.0.0.weight": _real_torch.zeros(320, 4, 3, 3),
        }
        path = _write_safetensors(tmp_path, "sdxl_dummy.safetensors", tensors)
        assert _detect_sd_architecture(path) == "sdxl"

    def test_detects_sdxl_via_cond_stage_model_1(self, tmp_path):
        tensors = {"cond_stage_model.1.transformer.text_model.embeddings.token_embedding.weight":
                       _real_torch.zeros(49408, 1280)}
        path = _write_safetensors(tmp_path, "sdxl_alt.safetensors", tensors)
        assert _detect_sd_architecture(path) == "sdxl"

    def test_detects_sdxl_via_text_encoder_2_prefix(self, tmp_path):
        tensors = {"text_encoder_2.text_model.embeddings.position_embedding.weight":
                       _real_torch.zeros(77, 1280)}
        path = _write_safetensors(tmp_path, "sdxl_diffusers.safetensors", tensors)
        assert _detect_sd_architecture(path) == "sdxl"

    def test_detects_sdxl_via_cross_attention_dim_2048(self, tmp_path):
        # A UNet cross-attention to_k with input dim 2048 = SDXL
        tensors = {
            "model.diffusion_model.up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight":
                _real_torch.zeros(640, 2048),
        }
        path = _write_safetensors(tmp_path, "sdxl_unet_only.safetensors", tensors)
        assert _detect_sd_architecture(path) == "sdxl"


class TestSD15Detection:
    def test_detects_sd15_via_cross_attention_dim_768(self, tmp_path):
        tensors = {
            "model.diffusion_model.up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight":
                _real_torch.zeros(640, 768),
        }
        path = _write_safetensors(tmp_path, "sd15_dummy.safetensors", tensors)
        assert _detect_sd_architecture(path) == "sd15"

    def test_sd15_has_no_embedders_1(self, tmp_path):
        # SD 1.5 has cond_stage_model.transformer.* but NOT cond_stage_model.1.*
        tensors = {
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight":
                _real_torch.zeros(77, 768),
            "model.diffusion_model.up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight":
                _real_torch.zeros(640, 768),
        }
        path = _write_safetensors(tmp_path, "sd15_complete.safetensors", tensors)
        assert _detect_sd_architecture(path) == "sd15"


class TestUnknownAndEdgeCases:
    def test_unknown_for_non_safetensors_extension(self, tmp_path):
        # A .ckpt file (or anything not .safetensors) should return unknown
        bogus = tmp_path / "model.ckpt"
        bogus.write_bytes(b"fake ckpt")
        assert _detect_sd_architecture(bogus) == "unknown"

    def test_unknown_for_corrupt_safetensors(self, tmp_path):
        # A file with .safetensors extension but bad content
        path = tmp_path / "corrupt.safetensors"
        path.write_bytes(b"not actually safetensors")
        assert _detect_sd_architecture(path) == "unknown"

    def test_unknown_when_no_relevant_keys(self, tmp_path):
        # Safetensors with completely unrelated keys
        tensors = {"foo.bar.baz": _real_torch.zeros(1, 2)}
        path = _write_safetensors(tmp_path, "weird.safetensors", tensors)
        assert _detect_sd_architecture(path) == "unknown"

    def test_missing_file_unknown(self, tmp_path):
        # Detector must not crash on a missing file — it returns unknown.
        assert _detect_sd_architecture(tmp_path / "does_not_exist.safetensors") == "unknown"


class TestPipelineSelection:
    """Verify _load_single_file uses architecture detection to order attempts."""

    def test_detected_sd15_tries_sd15_first(self, tmp_path, monkeypatch):
        import diffusers as _diffusers

        # Use the real detector with a tiny SD1.5-shaped checkpoint
        tensors = {
            "model.diffusion_model.up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight":
                _real_torch.zeros(640, 768),
        }
        path = _write_safetensors(tmp_path, "sd15_pick.safetensors", tensors)

        # Patch every pipeline class so we can record which one was attempted
        # without actually loading a real model.
        call_order: list[str] = []

        class _StubCls:
            def __init__(self, name):
                self._name = name

            def from_single_file(self, *_a, **_kw):
                call_order.append(self._name)
                raise RuntimeError(f"stub:{self._name}")

        stub_xl = _StubCls("XL")
        stub_15 = _StubCls("SD15")
        monkeypatch.setattr(_diffusers, "StableDiffusionXLImg2ImgPipeline", stub_xl, raising=False)
        monkeypatch.setattr(_diffusers, "StableDiffusionImg2ImgPipeline", stub_15, raising=False)

        from app.stream.manager import _load_single_file
        with pytest.raises(RuntimeError):
            _load_single_file(path, dtype=_real_torch.float32)

        # SD15 must be attempted FIRST when arch detection said sd15
        assert call_order[0] == "SD15"

    def test_detected_sdxl_tries_xl_first(self, tmp_path, monkeypatch):
        import diffusers as _diffusers

        tensors = {"text_encoder_2.foo.weight": _real_torch.zeros(2, 2)}
        path = _write_safetensors(tmp_path, "sdxl_pick.safetensors", tensors)

        call_order: list[str] = []

        class _StubCls:
            def __init__(self, name): self._name = name
            def from_single_file(self, *_a, **_kw):
                call_order.append(self._name)
                raise RuntimeError(f"stub:{self._name}")

        monkeypatch.setattr(_diffusers, "StableDiffusionXLImg2ImgPipeline", _StubCls("XL"), raising=False)
        monkeypatch.setattr(_diffusers, "StableDiffusionImg2ImgPipeline", _StubCls("SD15"), raising=False)

        from app.stream.manager import _load_single_file
        with pytest.raises(RuntimeError):
            _load_single_file(path, dtype=_real_torch.float32)

        assert call_order[0] == "XL"


# ── LoRA architecture detection ──────────────────────────────────────────────

class TestLoRACrossAttentionDimDetection:
    """``_detect_lora_cross_attention_dim`` reads the cross-attention input dim
    from a LoRA's safetensors keys so we can refuse to apply an SDXL LoRA on an
    SD 1.5 pipe (and vice versa) without provoking a 40-line size-mismatch
    stack trace from diffusers' state_dict loader."""

    def test_sdxl_lora_diffusers_format_returns_2048(self, tmp_path):
        tensors = {
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight":
                _real_torch.zeros(64, 2048),
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.weight":
                _real_torch.zeros(640, 64),
        }
        path = _write_safetensors(tmp_path, "lora_sdxl.safetensors", tensors)
        assert _detect_lora_cross_attention_dim(path) == 2048

    def test_sd15_lora_diffusers_format_returns_768(self, tmp_path):
        tensors = {
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.weight":
                _real_torch.zeros(64, 768),
        }
        path = _write_safetensors(tmp_path, "lora_sd15.safetensors", tensors)
        assert _detect_lora_cross_attention_dim(path) == 768

    def test_kohya_format_lora_down_recognised(self, tmp_path):
        tensors = {
            "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight":
                _real_torch.zeros(64, 2048),
        }
        path = _write_safetensors(tmp_path, "lora_kohya_sdxl.safetensors", tensors)
        assert _detect_lora_cross_attention_dim(path) == 2048

    def test_no_attn2_keys_returns_none(self, tmp_path):
        # A "LoRA" that only contains text-encoder weights — undetectable
        # cross-attention dim. The caller should not gate on the result.
        tensors = {"lora_te1_text_model_encoder_layers_0.weight": _real_torch.zeros(768, 768)}
        path = _write_safetensors(tmp_path, "lora_te_only.safetensors", tensors)
        assert _detect_lora_cross_attention_dim(path) is None

    def test_non_safetensors_returns_none(self, tmp_path):
        bogus = tmp_path / "lora.ckpt"
        bogus.write_bytes(b"")
        assert _detect_lora_cross_attention_dim(bogus) is None

    def test_corrupt_safetensors_returns_none(self, tmp_path):
        bad = tmp_path / "lora.safetensors"
        bad.write_bytes(b"not safetensors")
        assert _detect_lora_cross_attention_dim(bad) is None


class TestApplyLorasGate:
    """``_apply_loras`` must refuse incompatible LoRAs before diffusers' loader
    is invoked, so the pipe never sees the noisy size-mismatch error.

    These tests use a fake pipe that records every ``load_lora_weights`` call.
    """

    class _FakePipe:
        def __init__(self, cross_attention_dim: int):
            self.unet = MagicMock()
            self.unet.config = MagicMock()
            self.unet.config.cross_attention_dim = cross_attention_dim
            self.lora_loads: list[str] = []

        def load_lora_weights(self, path, adapter_name=None, **_kwargs):
            self.lora_loads.append(str(path))

    def test_sdxl_lora_skipped_on_sd15_pipe(self, tmp_path):
        # SD 1.5 pipe (cross_attention_dim=768) + SDXL LoRA (2048-dim)
        pipe = self._FakePipe(cross_attention_dim=768)
        sdxl_lora = _write_safetensors(tmp_path, "sdxl.safetensors", {
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight":
                _real_torch.zeros(64, 2048),
        })
        _apply_loras(pipe, [str(sdxl_lora)])
        # Crucially: load_lora_weights was NEVER called for the incompatible LoRA
        assert pipe.lora_loads == []

    def test_sd15_lora_skipped_on_sdxl_pipe(self, tmp_path):
        pipe = self._FakePipe(cross_attention_dim=2048)
        sd15_lora = _write_safetensors(tmp_path, "sd15.safetensors", {
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight":
                _real_torch.zeros(64, 768),
        })
        _apply_loras(pipe, [str(sd15_lora)])
        assert pipe.lora_loads == []

    def test_matching_lora_is_loaded(self, tmp_path):
        # SD 1.5 pipe + SD 1.5 LoRA → load goes through
        pipe = self._FakePipe(cross_attention_dim=768)
        sd15_lora = _write_safetensors(tmp_path, "sd15_compat.safetensors", {
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight":
                _real_torch.zeros(64, 768),
        })
        _apply_loras(pipe, [str(sd15_lora)])
        assert len(pipe.lora_loads) == 1
        assert "sd15_compat" in pipe.lora_loads[0]

    def test_undetectable_lora_attempted(self, tmp_path):
        # If we can't detect the LoRA arch (e.g. text-encoder-only LoRA), we
        # let diffusers try — never silently skip on undetectable.
        pipe = self._FakePipe(cross_attention_dim=768)
        te_lora = _write_safetensors(tmp_path, "te_only.safetensors", {
            "lora_te1_text_model.layer.weight": _real_torch.zeros(768, 768),
        })
        _apply_loras(pipe, [str(te_lora)])
        assert len(pipe.lora_loads) == 1

    def test_mixed_batch_only_skips_mismatch(self, tmp_path):
        pipe = self._FakePipe(cross_attention_dim=768)
        good = _write_safetensors(tmp_path, "good.safetensors", {
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight":
                _real_torch.zeros(64, 768),
        })
        bad = _write_safetensors(tmp_path, "bad.safetensors", {
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight":
                _real_torch.zeros(64, 2048),
        })
        _apply_loras(pipe, [str(good), str(bad), str(good)])
        # First "good" and third "good" loaded; "bad" skipped silently
        assert len(pipe.lora_loads) == 2
        assert all("good" in p for p in pipe.lora_loads)

    def test_clean_warning_message_on_skip(self, tmp_path, caplog):
        pipe = self._FakePipe(cross_attention_dim=768)
        sdxl_lora = _write_safetensors(tmp_path, "lcm_sdxl.safetensors", {
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight":
                _real_torch.zeros(64, 2048),
        })
        import logging
        with caplog.at_level(logging.WARNING, logger="rtdiffusion.stream"):
            _apply_loras(pipe, [str(sdxl_lora)])
        # One concise warning, not a 40-line stack trace
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "SDXL" in msg and "SD 1.5" in msg
        assert "lcm_sdxl.safetensors" in msg
