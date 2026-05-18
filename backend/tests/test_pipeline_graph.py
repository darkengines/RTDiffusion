"""
Unit tests for pipeline_graph.py.

Run with:  pytest backend/tests/test_pipeline_graph.py -v
"""

from __future__ import annotations

import base64
import io
import types
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Stub heavy optional deps so tests don't need a GPU / HuggingFace download
# ---------------------------------------------------------------------------

def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

# onnxruntime stub
_ort = _make_stub_module("onnxruntime")
_ort.InferenceSession = MagicMock  # type: ignore[attr-defined]

# huggingface_hub stub
_hf = _make_stub_module("huggingface_hub")
_hf.hf_hub_download = MagicMock(return_value="/tmp/fake")  # type: ignore[attr-defined]

# Now safe to import project modules
from app.pipeline_graph import (  # noqa: E402
    ConditioningResult,
    GraphResult,
    SessionGraph,
    _TaggerState,
    merge_layer_prompts,
    resolve_conditioning_mask,
)
from app.schemas import ControlNetPreprocessorParams  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank(size: tuple[int, int] = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, color=(128, 128, 128))

def _mask(size: tuple[int, int] = (64, 64)) -> Image.Image:
    m = Image.new("L", size, color=0)
    # white square in the centre
    for y in range(16, 48):
        for x in range(16, 48):
            m.putpixel((x, y), 255)
    return m


def _opaque_png_data_url(size: tuple[int, int] = (32, 32)) -> str:
    img = Image.new("RGBA", size, (255, 255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


# ---------------------------------------------------------------------------
# GraphResult
# ---------------------------------------------------------------------------

class TestGraphResult:
    def test_defaults_are_empty(self):
        r = GraphResult()
        assert r.prompt_overrides == {}
        assert r.cn_images == {}
        assert r.cn_params == {}


# ---------------------------------------------------------------------------
# SessionGraph — node compilation
# ---------------------------------------------------------------------------

class TestSessionGraphCompile:
    def test_empty_nodes(self):
        g = SessionGraph([])
        assert g._tagger_states == []
        assert g._cn_states == []

    def test_tagger_node_compiled(self):
        nodes = [{"id": "t1", "type": "tagger", "layer_id": "L1",
                  "tagger_threshold": 0.4, "tagger_refresh_frames": 3}]
        g = SessionGraph(nodes)
        assert len(g._tagger_states) == 1
        s = g._tagger_states[0]
        assert s.node_id == "t1"
        assert s.layer_id == "L1"
        assert s.threshold == pytest.approx(0.4)
        assert s.refresh_frames == 3

    def test_cn_node_compiled(self):
        nodes = [{"id": "c1", "type": "cn_from_layer", "layer_id": "L2",
                  "cn_model": "depth", "cn_scale": 0.45, "cn_start": 0.2, "cn_end": 0.8}]
        g = SessionGraph(nodes)
        assert len(g._cn_states) == 1
        s = g._cn_states[0]
        assert s.node_id == "c1"
        assert s.layer_id == "L2"
        assert s.cn_model == "depth"
        assert s.cn_scale == pytest.approx(0.45)
        assert s.cn_start == pytest.approx(0.2)
        assert s.cn_end == pytest.approx(0.8)

    def test_cn_preprocessor_params_forwarded(self):
        nodes = [{"id": "c2", "type": "cn_from_layer", "layer_id": "L3",
                  "cn_model": "canny",
                  "cn_preprocessor_params": {"canny_low": 80, "canny_high": 200}}]
        g = SessionGraph(nodes)
        p = g._cn_states[0].preprocessor_params
        assert p.canny_low == 80
        assert p.canny_high == 200

    def test_unknown_node_type_skipped(self):
        nodes = [{"id": "x1", "type": "mystery", "layer_id": "L4"}]
        g = SessionGraph(nodes)
        assert g._tagger_states == []
        assert g._cn_states == []

    def test_mixed_nodes(self):
        nodes = [
            {"id": "t1", "type": "tagger", "layer_id": "L1"},
            {"id": "c1", "type": "cn_from_layer", "layer_id": "L2"},
        ]
        g = SessionGraph(nodes)
        assert len(g._tagger_states) == 1
        assert len(g._cn_states) == 1

    def test_tagger_defaults(self):
        nodes = [{"id": "t2", "type": "tagger", "layer_id": "L5"}]
        g = SessionGraph(nodes)
        s = g._tagger_states[0]
        assert s.threshold == pytest.approx(0.35)
        assert s.refresh_frames == 5
        assert s.model_name == "wd-eva02-large-v3"


# ---------------------------------------------------------------------------
# SessionGraph.signature
# ---------------------------------------------------------------------------

class TestSessionGraphSignature:
    def test_same_nodes_same_sig(self):
        nodes = [{"id": "t1", "type": "tagger", "layer_id": "L1"}]
        assert SessionGraph.signature(nodes) == SessionGraph.signature(nodes)

    def test_different_nodes_different_sig(self):
        a = [{"id": "t1", "type": "tagger", "layer_id": "L1"}]
        b = [{"id": "t1", "type": "tagger", "layer_id": "L2"}]
        assert SessionGraph.signature(a) != SessionGraph.signature(b)

    def test_empty_sig_stable(self):
        assert SessionGraph.signature([]) == SessionGraph.signature([])


# ---------------------------------------------------------------------------
# _TaggerState — frame counter / caching
# ---------------------------------------------------------------------------

class TestTaggerState:
    def _state(self, refresh: int = 5) -> _TaggerState:
        return _TaggerState(
            node_id="t1",
            layer_id="L1",
            model_name="wd-eva02-large-v3",
            threshold=0.35,
            refresh_frames=refresh,
        )

    def test_runs_on_first_frame(self):
        state = self._state(refresh=5)
        fake_result = "cat, indoors"
        with patch("app.tagger.tag_image", return_value=fake_result) as mock_tag:
            result = state.maybe_run(_blank(), None)
        mock_tag.assert_called_once()
        assert result == fake_result

    def test_cached_between_runs(self):
        state = self._state(refresh=5)
        fake_result = "dog, outdoors"
        with patch("app.tagger.tag_image", return_value=fake_result):
            state.maybe_run(_blank(), None)  # frame 1 — runs
        with patch("app.tagger.tag_image", return_value="SHOULD NOT SEE") as mock_tag2:
            result2 = state.maybe_run(_blank(), None)  # frame 2 — cached
        mock_tag2.assert_not_called()
        assert result2 == fake_result

    def test_reruns_when_frame_changes_before_refresh(self):
        state = self._state(refresh=99)
        img_a = Image.new("RGB", (64, 64), (10, 10, 10))
        img_b = Image.new("RGB", (64, 64), (220, 220, 220))
        with patch("app.tagger.tag_image", side_effect=["dark", "bright"]) as mock_tag:
            r1 = state.maybe_run(img_a, None)
            r2 = state.maybe_run(img_b, None)
        assert r1 == "dark"
        assert r2 == "bright"
        assert mock_tag.call_count == 2

    def test_refreshes_at_interval(self):
        state = self._state(refresh=3)
        calls = []
        def fake_tag(img, **kw):
            calls.append(len(calls))
            return f"tag_{len(calls)}"

        with patch("app.tagger.tag_image", side_effect=fake_tag):
            r1 = state.maybe_run(_blank(), None)  # frame 1 → runs (count=1)
            r2 = state.maybe_run(_blank(), None)  # frame 2 → cached
            r3 = state.maybe_run(_blank(), None)  # frame 3 → cached (3 % 3 == 0 ≠ 1)
            r4 = state.maybe_run(_blank(), None)  # frame 4 → runs (4 % 3 == 1)

        assert r1 == "tag_1"
        assert r2 == "tag_1"
        assert r3 == "tag_1"
        assert r4 == "tag_2"

    def test_passes_mask_to_tagger(self):
        state = self._state(refresh=1)
        mask = _mask()
        with patch("app.tagger.tag_image", return_value="x") as mock_tag:
            state.maybe_run(_blank(), mask)
        _, kwargs = mock_tag.call_args
        assert kwargs.get("mask") is mask or mock_tag.call_args[0][1] is mask

    def test_empty_string_clears_on_refresh(self):
        """When refresh runs and no tags are found, stale tags should be cleared."""
        state = self._state(refresh=1)
        with patch("app.tagger.tag_image", return_value="cat"):
            state.maybe_run(_blank(), None)  # frame 1
        with patch("app.tagger.tag_image", return_value=""):
            state.maybe_run(_blank(), None)  # frame 2 — would run (refresh=1) but result is empty
        assert state._cached == ""


# ---------------------------------------------------------------------------
# SessionGraph.run — missing layer frames handled gracefully
# ---------------------------------------------------------------------------

class TestSessionGraphRun:
    def test_missing_layer_frame_skipped(self):
        nodes = [{"id": "t1", "type": "tagger", "layer_id": "MISSING"}]
        g = SessionGraph(nodes)
        result = g.run(layer_frames={})
        assert result.prompt_overrides == {}

    def test_tagger_result_in_prompt_overrides(self):
        nodes = [{"id": "t1", "type": "tagger", "layer_id": "L1",
                  "tagger_refresh_frames": 1}]
        g = SessionGraph(nodes)
        with patch("app.tagger.tag_image", return_value="anime, girl"):
            result = g.run(layer_frames={"L1": _blank()})
        assert result.prompt_overrides.get("L1") == "anime, girl"

    def test_cn_node_missing_layer_skipped(self):
        nodes = [{"id": "c1", "type": "cn_from_layer", "layer_id": "MISSING",
                  "cn_model": "canny"}]
        g = SessionGraph(nodes)
        result = g.run(layer_frames={})
        assert result.cn_images == {}

    def test_cn_preprocess_called(self):
        nodes = [{"id": "c1", "type": "cn_from_layer", "layer_id": "L2",
                  "cn_model": "canny", "cn_scale": 0.35, "cn_start": 0.1, "cn_end": 0.9}]
        g = SessionGraph(nodes)
        fake_img = _blank()
        with patch("app.controlnet.preprocess_image", return_value=fake_img) as mock_pre:
            result = g.run(layer_frames={"L2": _blank()})
        mock_pre.assert_called_once()
        assert result.cn_images.get("L2") is fake_img
        assert result.cn_params["L2"] == {"scale": 0.35, "start": 0.1, "end": 0.9}

    def test_cn_preprocess_exception_handled(self):
        nodes = [{"id": "c1", "type": "cn_from_layer", "layer_id": "L2",
                  "cn_model": "canny"}]
        g = SessionGraph(nodes)
        with patch("app.controlnet.preprocess_image", side_effect=RuntimeError("oops")):
            result = g.run(layer_frames={"L2": _blank()})
        assert result.cn_images == {}


# ---------------------------------------------------------------------------
# merge_layer_prompts
# ---------------------------------------------------------------------------

class TestMergeLayerPrompts:
    def test_no_layers_returns_base(self):
        assert merge_layer_prompts([], "base prompt", None) == "base prompt"

    def test_layer_prompt_appended(self):
        conds = [{"layer_id": "L1", "prompt": "red car"}]
        result = merge_layer_prompts(conds, "base", None)
        assert "base" in result
        assert "red car" in result

    def test_tagger_override_replaces_layer_prompt(self):
        conds = [{"layer_id": "L1", "prompt": "explicit prompt"}]
        gr = GraphResult(prompt_overrides={"L1": "tagged: cat"})
        result = merge_layer_prompts(conds, "base", gr)
        assert "tagged: cat" in result
        assert "explicit prompt" not in result

    def test_empty_layer_prompt_skipped(self):
        conds = [{"layer_id": "L1", "prompt": ""}]
        result = merge_layer_prompts(conds, "base", None)
        assert result == "base"

    def test_base_prompt_empty_still_works(self):
        conds = [{"layer_id": "L1", "prompt": "sunset"}]
        result = merge_layer_prompts(conds, "", None)
        assert "sunset" in result

    def test_multiple_layers_all_included(self):
        conds = [
            {"layer_id": "L1", "prompt": "fire"},
            {"layer_id": "L2", "prompt": "water"},
        ]
        result = merge_layer_prompts(conds, "base", None)
        assert "fire" in result
        assert "water" in result
        assert "base" in result

    def test_whitespace_only_prompt_skipped(self):
        conds = [{"layer_id": "L1", "prompt": "   "}]
        result = merge_layer_prompts(conds, "base", None)
        assert result == "base"

    def test_no_graph_result_uses_layer_prompts(self):
        conds = [{"layer_id": "L1", "prompt": "dragon"}]
        result = merge_layer_prompts(conds, "base", None)
        assert "dragon" in result


class TestResolveConditioningMask:
    def test_denoise_does_not_dim_output_mask(self):
        base = Image.new("L", (32, 32), 255)
        conds = [{
            "mode": "mask",
            "image": _opaque_png_data_url((32, 32)),
            "weight": 1.0,
            "denoise": 0.2,
            "schedule": "linear",
            "schedule_start": 0.0,
            "schedule_end": 1.0,
        }]
        resolved = resolve_conditioning_mask(base, conds, 32, 32, 0.2)
        arr = np.asarray(resolved.mask, dtype=np.uint8)
        assert int(arr.max()) == 255
        assert resolved.strength == pytest.approx(0.2)

    def test_override_painters_algorithm(self):
        """Layer A (bottom, denoise=0, override, full) then Layer B (top, denoise=1.0,
        override, left half) → only left half is in mask; strength = 1.0."""
        # Build a half-white PNG for Layer B (left 16 columns white, right 16 black)
        b_alpha = Image.new("L", (32, 32), 0)
        import PIL.ImageDraw as _draw
        _draw.Draw(b_alpha).rectangle([0, 0, 15, 31], fill=255)
        b_rgba = Image.new("RGBA", (32, 32), (128, 64, 32, 0))
        b_rgba.putalpha(b_alpha)
        import io as _io, base64 as _b64
        buf = _io.BytesIO()
        b_rgba.save(buf, "PNG")
        b_url = "data:image/png;base64," + _b64.b64encode(buf.getvalue()).decode()

        base = Image.new("L", (32, 32), 255)
        conds = [
            # Layer A: full coverage, denoise=0, override (bottom of stack)
            {
                "mode": "override",
                "image": _opaque_png_data_url((32, 32)),
                "weight": 1.0,
                "denoise": 0.0,
                "schedule": "linear",
                "schedule_start": 0.0,
                "schedule_end": 1.0,
            },
            # Layer B: left-half coverage, denoise=1.0, override (top of stack)
            {
                "mode": "override",
                "image": b_url,
                "weight": 1.0,
                "denoise": 0.999,
                "schedule": "linear",
                "schedule_start": 0.0,
                "schedule_end": 1.0,
            },
        ]
        resolved = resolve_conditioning_mask(base, conds, 32, 32, 0.5)
        arr = np.asarray(resolved.mask, dtype=np.uint8)
        # Left half: B's override of 1.0 → white in mask
        assert int(arr[:, :16].max()) == 255
        # Right half: A's override of 0 → black in mask (nothing to inpaint)
        assert int(arr[:, 16:].max()) == 0
        assert resolved.strength > 0.9

    def test_override_mode_skips_prompt_mix(self):
        """prompt_mix mode should not contribute to the spatial denoise map."""
        base = Image.new("L", (32, 32), 0)
        conds = [{
            "mode": "prompt_mix",
            "image": _opaque_png_data_url((32, 32)),
            "weight": 1.0,
            "denoise": 0.999,
        }]
        resolved = resolve_conditioning_mask(base, conds, 32, 32, 0.0)
        arr = np.asarray(resolved.mask, dtype=np.uint8)
        assert int(arr.max()) == 0  # no spatial effect
