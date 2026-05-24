"""Unit tests for SANA prompt formatter.

Run:  pytest backend/tests/test_sana_prompt.py -v

Tests verify:
  • Base prompt becomes "Scene:" section.
  • Single layer gets "Scene" label.
  • Two layers get "Background" / "Foreground".
  • Three layers get "Background" / "Midground" / "Foreground".
  • >3 layers get numbered midgrounds.
  • Custom layer.name overrides default label.
  • Partial-coverage masks add an English spatial locator.
  • Full-coverage masks add no locator.
  • Per-region negatives are accumulated.
  • Warnings flag unsupported features (per-region CFG, schedule).
  • Empty scenes produce empty positive.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.composition import Layer, PromptOp, Region, Scene
from app.sana_prompt import SanaPrompt, _layer_labels, _region_locator, format_for_sana


W, H = 64, 64


def _full_mask(width: int = W, height: int = H) -> Image.Image:
    return Image.new("L", (width, height), 255)


def _block_mask(x0: int, y0: int, x1: int, y1: int,
                width: int = W, height: int = H) -> Image.Image:
    m = Image.new("L", (width, height), 0)
    arr = np.asarray(m).copy()
    arr[y0:y1, x0:x1] = 255
    return Image.fromarray(arr, "L")


# ── Region locator ────────────────────────────────────────────────────────────

class TestRegionLocator:
    def test_none_mask_no_locator(self):
        assert _region_locator(None, W, H) == ""

    def test_full_coverage_no_locator(self):
        assert _region_locator(_full_mask(), W, H) == ""

    def test_upper_left_block(self):
        m = _block_mask(0, 0, 16, 16)
        assert _region_locator(m, W, H) == "upper-left"

    def test_lower_right_block(self):
        m = _block_mask(48, 48, 64, 64)
        assert _region_locator(m, W, H) == "lower-right"

    def test_centre_block(self):
        m = _block_mask(24, 24, 40, 40)
        assert _region_locator(m, W, H) == "centre"

    def test_left_band(self):
        m = _block_mask(0, 0, 16, 64)
        # centroid at (x≈8, y≈32) → horz=left, vert=centre
        assert _region_locator(m, W, H) == "left side"

    def test_top_band(self):
        m = _block_mask(0, 0, 64, 16)
        # centroid at (x≈32, y≈8) → horz=centre, vert=upper
        assert _region_locator(m, W, H) == "upper centre"


# ── Layer labels ──────────────────────────────────────────────────────────────

class TestLayerLabels:
    def test_one_layer_scene_label(self):
        layers = (Layer(layer_id="L1"),)
        assert _layer_labels(1, layers) == ["Scene"]

    def test_two_layers_bg_fg(self):
        layers = (Layer(layer_id="L1"), Layer(layer_id="L2"))
        assert _layer_labels(2, layers) == ["Background", "Foreground"]

    def test_three_layers_bg_mg_fg(self):
        layers = tuple(Layer(layer_id=f"L{i}") for i in range(3))
        assert _layer_labels(3, layers) == ["Background", "Midground", "Foreground"]

    def test_five_layers_numbered_midgrounds(self):
        layers = tuple(Layer(layer_id=f"L{i}") for i in range(5))
        assert _layer_labels(5, layers) == [
            "Background", "Midground 1", "Midground 2", "Midground 3", "Foreground",
        ]

    def test_explicit_layer_name_wins(self):
        layers = (
            Layer(layer_id="L1", name="Character"),
            Layer(layer_id="L2", name="Sky"),
        )
        assert _layer_labels(2, layers) == ["Character", "Sky"]

    def test_layer_name_equal_to_id_uses_default(self):
        # Layer.name defaults to layer_id in scene_from_legacy; ensure the
        # formatter still picks the role-based default in that case.
        layers = (
            Layer(layer_id="L1", name="L1"),
            Layer(layer_id="L2", name="L2"),
        )
        assert _layer_labels(2, layers) == ["Background", "Foreground"]


# ── format_for_sana ──────────────────────────────────────────────────────────

class TestFormatForSana:
    def test_base_prompt_only(self):
        scene = Scene(base_prompt="a cat", width=W, height=H)
        out = format_for_sana(scene)
        assert out.positive == "Scene:\na cat"
        assert out.negative == ""
        assert out.warnings == ()

    def test_empty_scene(self):
        scene = Scene(width=W, height=H)
        out = format_for_sana(scene)
        assert out.positive == ""
        assert out.negative == ""

    def test_single_layer_with_prompt(self):
        region = Region(mask=_full_mask(), prompt="dragon")
        scene = Scene(
            layers=(Layer(layer_id="L1", regions=(region,)),),
            base_prompt="dark fantasy",
            width=W, height=H,
        )
        out = format_for_sana(scene)
        assert "Scene:\ndark fantasy" in out.positive
        assert "Scene:\ndragon" in out.positive  # single layer → "Scene" label

    def test_two_layers_bg_fg_sections(self):
        bg = Region(mask=_full_mask(), prompt="forest landscape")
        fg = Region(mask=_block_mask(20, 20, 44, 44), prompt="knight")
        scene = Scene(
            layers=(
                Layer(layer_id="bg", regions=(bg,)),
                Layer(layer_id="fg", regions=(fg,)),
            ),
            base_prompt="oil painting",
            width=W, height=H,
        )
        out = format_for_sana(scene)
        # All three sections present, in order
        idx_scene = out.positive.find("Scene:")
        idx_bg = out.positive.find("Background:")
        idx_fg = out.positive.find("Foreground:")
        assert 0 <= idx_scene < idx_bg < idx_fg
        # Foreground region has centre locator
        assert "(centre)" in out.positive

    def test_three_layers_get_midground(self):
        scene = Scene(
            layers=(
                Layer(layer_id="a", regions=(Region(mask=_full_mask(), prompt="alpha-zzz"),)),
                Layer(layer_id="b", regions=(Region(mask=_full_mask(), prompt="beta-zzz"),)),
                Layer(layer_id="c", regions=(Region(mask=_full_mask(), prompt="gamma-zzz"),)),
            ),
            width=W, height=H,
        )
        out = format_for_sana(scene)
        assert "Background:" in out.positive
        assert "Midground:" in out.positive
        assert "Foreground:" in out.positive
        # Order: a=bg, b=mg, c=fg — use distinctive prompt tokens
        assert out.positive.index("alpha-zzz") < out.positive.index("beta-zzz") < out.positive.index("gamma-zzz")

    def test_layer_default_prompt_inherited(self):
        region = Region(mask=_full_mask())  # no prompt
        layer = Layer(layer_id="L", regions=(region,), default_prompt="forest")
        scene = Scene(layers=(layer,), width=W, height=H)
        out = format_for_sana(scene)
        assert "forest" in out.positive

    def test_two_regions_in_same_layer_concat(self):
        r1 = Region(mask=_block_mask(0, 0, 16, H), prompt="left thing")
        r2 = Region(mask=_block_mask(48, 0, W, H), prompt="right thing")
        layer = Layer(layer_id="L", regions=(r1, r2))
        scene = Scene(layers=(layer,), width=W, height=H)
        out = format_for_sana(scene)
        assert "left thing" in out.positive
        assert "right thing" in out.positive

    def test_empty_layer_skipped(self):
        empty = Layer(layer_id="empty", regions=(Region(mask=_full_mask()),))
        full = Layer(layer_id="x", regions=(Region(mask=_full_mask(), prompt="dragon"),))
        scene = Scene(layers=(empty, full), width=W, height=H)
        out = format_for_sana(scene)
        # Only one non-empty layer → "Scene" label
        assert "Scene:\ndragon" in out.positive
        # No "Background:" since empty layer was filtered out
        assert "Background:" not in out.positive

    def test_negatives_accumulated(self):
        region = Region(mask=_full_mask(), prompt="x", negative_prompt="blurry")
        scene = Scene(
            layers=(Layer(layer_id="L", regions=(region,)),),
            base_negative_prompt="lowres",
            width=W, height=H,
        )
        out = format_for_sana(scene)
        assert "lowres" in out.negative
        assert "blurry" in out.negative

    def test_warning_per_region_cfg(self):
        region = Region(mask=_full_mask(), prompt="x", cfg=7.0)
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        out = format_for_sana(scene)
        assert any("CFG" in w for w in out.warnings)

    def test_warning_per_region_schedule(self):
        region = Region(mask=_full_mask(), prompt="x", schedule=(0.2, 0.8))
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        out = format_for_sana(scene)
        assert any("schedule" in w for w in out.warnings)

    def test_warning_embed_blend_falls_back(self):
        region = Region(mask=_full_mask(), prompt="x", prompt_op=PromptOp.EMBED_BLEND)
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        out = format_for_sana(scene)
        assert any("embed_blend" in w for w in out.warnings)

    def test_no_warnings_for_supported_features(self):
        region = Region(mask=_full_mask(), prompt="x")
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        out = format_for_sana(scene)
        assert out.warnings == ()


# ── Realistic end-to-end scenarios ───────────────────────────────────────────

class TestRealisticScenarios:
    def test_avatar_in_environment(self):
        bg = Region(mask=_full_mask(), prompt="cyberpunk city, neon lights")
        avatar = Region(
            mask=_block_mask(16, 8, 48, 56),  # centre portrait area
            prompt="anime girl with pink hair",
        )
        scene = Scene(
            layers=(
                Layer(layer_id="env", name="Environment", regions=(bg,)),
                Layer(layer_id="char", name="Character", regions=(avatar,)),
            ),
            base_prompt="masterpiece, detailed",
            width=W, height=H,
        )
        out = format_for_sana(scene)
        # All sections present
        assert "Scene:" in out.positive
        # Custom layer names win
        assert "Environment:" in out.positive
        assert "Character:" in out.positive
        # Character has a spatial hint
        assert "(centre)" in out.positive or "centre" in out.positive
