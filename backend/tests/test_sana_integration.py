"""End-to-end SANA integration tests.

Run:  pytest backend/tests/test_sana_integration.py -v

Verifies the integration between:
  • scene_from_legacy (composition.py)
  • format_for_sana (sana_prompt.py)
  • SanaSprintSession.infer signature (per-frame prompt override)
  • _session_sig (no rebuild on prompt change)

These tests don't touch a real SANA pipeline; they verify the contract.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# Heavy deps stubs so importing sana_pipeline doesn't pull diffusers/torch.
def _stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Stubs only created if real modules are missing — pytest discovery may run
# these tests after other tests already imported the real torch/diffusers.
def _ensure_stub(name: str, attrs: dict | None = None):
    if name not in sys.modules:
        _stub(name, attrs)


_ensure_stub("torch")
_ensure_stub("diffusers")
_ensure_stub("transformers")

# torch.device is touched in SanaSprintSession.__init__; provide a no-op shim
# so we can instantiate sessions in tests without a real torch install.
import sys as _sys
if not hasattr(_sys.modules.get("torch"), "device"):
    _sys.modules["torch"].device = lambda x: x  # type: ignore[attr-defined]
if not hasattr(_sys.modules.get("torch"), "inference_mode"):
    class _NoOpCtx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    _sys.modules["torch"].inference_mode = lambda: _NoOpCtx()  # type: ignore[attr-defined]

from app.composition import scene_from_legacy
from app.sana_prompt import format_for_sana
from app.sana_pipeline import _session_sig


W, H = 64, 64


def _opaque_data_url(w: int = 64, h: int = 64) -> str:
    """Build a data-URL for an opaque white RGBA image — used as legacy layer mask."""
    import base64
    import io
    img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── _session_sig stability ───────────────────────────────────────────────────

class TestSanaSessionSigStability:
    def test_prompt_change_does_not_change_sig(self):
        s1 = {"prompt": "a cat", "cfg": 4.5, "sana_steps": 2}
        s2 = {"prompt": "a dog", "cfg": 4.5, "sana_steps": 2}
        assert _session_sig(s1) == _session_sig(s2)

    def test_negative_prompt_change_does_not_change_sig(self):
        s1 = {"prompt": "x", "negative_prompt": "blur", "cfg": 4.5}
        s2 = {"prompt": "x", "negative_prompt": "lowres", "cfg": 4.5}
        assert _session_sig(s1) == _session_sig(s2)

    def test_cfg_change_changes_sig(self):
        s1 = {"prompt": "x", "cfg": 4.5}
        s2 = {"prompt": "x", "cfg": 7.0}
        assert _session_sig(s1) != _session_sig(s2)

    def test_steps_change_changes_sig(self):
        s1 = {"sana_steps": 2}
        s2 = {"sana_steps": 4}
        assert _session_sig(s1) != _session_sig(s2)

    def test_dims_change_changes_sig(self):
        s1 = {"width": 512, "height": 512}
        s2 = {"width": 1024, "height": 1024}
        assert _session_sig(s1) != _session_sig(s2)


# ── Composition → SANA pipeline ──────────────────────────────────────────────

class TestComposeToSana:
    def test_simple_legacy_payload_yields_scene_prompt(self):
        scene = scene_from_legacy(
            [], base_prompt="a dragon", base_negative_prompt="",
            base_denoise=0.7, base_cfg=4.5, width=W, height=H,
        )
        out = format_for_sana(scene)
        assert "Scene:\na dragon" == out.positive

    def test_layered_legacy_payload_yields_sections(self):
        conds = [
            {"layer_id": "bg", "prompt": "forest"},
            {"layer_id": "fg", "prompt": "knight"},
        ]
        # Provide masks so the regions are well-defined.
        bg_mask = Image.new("L", (W, H), 255)
        fg_mask = Image.new("L", (W, H), 0)
        fg_arr = np.asarray(fg_mask).copy()
        fg_arr[20:44, 20:44] = 255
        fg_mask = Image.fromarray(fg_arr, "L")

        scene = scene_from_legacy(
            conds, base_prompt="oil painting", base_negative_prompt="",
            base_denoise=0.7, base_cfg=4.5, width=W, height=H,
            masks_by_layer={"bg": bg_mask, "fg": fg_mask},
        )
        out = format_for_sana(scene)
        assert "Scene:\noil painting" in out.positive
        assert "Background:\nforest" in out.positive
        # FG region has a centre locator
        assert "(centre) knight" in out.positive or "Foreground:\n(centre) knight" in out.positive

    def test_legacy_tagger_override_appears_in_section(self):
        conds = [{"layer_id": "L1", "prompt": "manual"}]
        scene = scene_from_legacy(
            conds, base_prompt="", base_negative_prompt="",
            base_denoise=0.7, base_cfg=4.5, width=W, height=H,
            masks_by_layer={"L1": Image.new("L", (W, H), 255)},
            prompt_overrides={"L1": "tagger: cat, dog"},
        )
        out = format_for_sana(scene)
        assert "tagger: cat, dog" in out.positive
        assert "manual" not in out.positive

    def test_warns_on_per_region_cfg(self):
        conds = [{"layer_id": "L1", "prompt": "x", "cfg": 8.0}]
        scene = scene_from_legacy(
            conds, base_prompt="", base_negative_prompt="",
            base_denoise=0.7, base_cfg=4.5, width=W, height=H,
            masks_by_layer={"L1": Image.new("L", (W, H), 255)},
        )
        out = format_for_sana(scene)
        assert any("CFG" in w for w in out.warnings)

    def test_warns_on_per_region_schedule(self):
        conds = [{
            "layer_id": "L1", "prompt": "x",
            "schedule_start": 0.2, "schedule_end": 0.8,
        }]
        scene = scene_from_legacy(
            conds, base_prompt="", base_negative_prompt="",
            base_denoise=0.7, base_cfg=4.5, width=W, height=H,
            masks_by_layer={"L1": Image.new("L", (W, H), 255)},
        )
        out = format_for_sana(scene)
        assert any("schedule" in w for w in out.warnings)


# ── SanaSprintSession.infer contract ─────────────────────────────────────────

class TestSanaInferContract:
    def _build_session(self) -> "SanaSprintSession":  # type: ignore[name-defined]
        from app.sana_pipeline import SanaSprintSession
        # Mock pipe; infer() exercises the wrapper path
        pipe = MagicMock()
        # Make `inspect.signature(pipe.__call__).parameters` return what
        # diffusers SanaSprintImg2ImgPipeline has (no `negative_prompt`).
        pipe.__call__ = lambda **kw: MagicMock(images=[Image.new("RGB", (8, 8), (1, 2, 3))])
        # Override sentinel for the `import torch` inside _infer_inner
        return SanaSprintSession(
            pipe=pipe,
            device="cpu",
            prompt="default-prompt",
            negative_prompt="default-neg",
            cfg_scale=4.5,
            num_steps=2,
            width=8, height=8,
        )

    def test_infer_signature_accepts_prompt_override(self):
        # Just checking the method signature exists and accepts the new kwargs
        from app.sana_pipeline import SanaSprintSession
        import inspect
        sig = inspect.signature(SanaSprintSession.infer)
        assert "prompt" in sig.parameters
        assert "negative_prompt" in sig.parameters

    def test_infer_prompt_override_defaults_to_session_default(self):
        # Smoke-check: omitted prompt uses session default. We don't run torch
        # inference here — we only verify the parameter falls through.
        from app.sana_pipeline import SanaSprintSession
        recorded: dict = {}
        original = SanaSprintSession._infer_inner

        def fake_inner(self, image, strength, prompt_override=None, negative_override=None):
            recorded["prompt_override"] = prompt_override
            recorded["negative_override"] = negative_override
            return Image.new("RGB", (8, 8))

        SanaSprintSession._infer_inner = fake_inner  # type: ignore[method-assign]
        try:
            s = self._build_session()
            s.infer(Image.new("RGB", (8, 8)))
            assert recorded["prompt_override"] is None
            assert recorded["negative_override"] is None

            s.infer(Image.new("RGB", (8, 8)), prompt="override!", negative_prompt="x")
            assert recorded["prompt_override"] == "override!"
            assert recorded["negative_override"] == "x"
        finally:
            SanaSprintSession._infer_inner = original  # type: ignore[method-assign]
