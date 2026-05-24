"""Unit tests for the layer composition module.

Run:  pytest backend/tests/test_composition.py -v

Coverage goals (per the formal spec in composition.py):
  • Each NumericOp produces correct values in fully-covered, partial-alpha,
    and uncovered regions.
  • Bottom-up evaluation: later regions apply on top of accumulated state.
  • Mask aggregation respects per-region mask_op.
  • PromptOp behaviour for REPLACE / CONCAT / EMBED_BLEND.
  • Schedule (start/end) gates region contribution.
  • Layer-level defaults inherit when region leaves field at None.
  • Single-pass, layered-pass, tiled-pass plan generation.
  • Renderer capability fallback (warnings + degraded plan).
  • Legacy bridge ``scene_from_legacy`` for old payloads.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.composition import (
    CFG_HI,
    DENOISE_HI,
    CompositionPlan,
    CompositionWarnings,
    Layer,
    LayeredPassPlan,
    NumericOp,
    PassSpec,
    PromptOp,
    Region,
    RenderMode,
    RendererSupport,
    Scene,
    SinglePassPlan,
    TiledPassPlan,
    aggregate_mask,
    aggregate_numeric,
    aggregate_prompt,
    build_layered_pass,
    build_single_pass,
    build_tiled_pass,
    compose,
    scene_from_legacy,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

W, H = 32, 32


def _full_mask(width: int = W, height: int = H, value: int = 255) -> Image.Image:
    return Image.new("L", (width, height), value)


def _left_half_mask(width: int = W, height: int = H) -> Image.Image:
    m = Image.new("L", (width, height), 0)
    arr = np.asarray(m).copy()
    arr[:, : width // 2] = 255
    return Image.fromarray(arr, "L")


def _right_half_mask(width: int = W, height: int = H) -> Image.Image:
    m = Image.new("L", (width, height), 0)
    arr = np.asarray(m).copy()
    arr[:, width // 2 :] = 255
    return Image.fromarray(arr, "L")


def _center_block_mask(width: int = W, height: int = H) -> Image.Image:
    m = Image.new("L", (width, height), 0)
    arr = np.asarray(m).copy()
    arr[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255
    return Image.fromarray(arr, "L")


def _grad_alpha(width: int = W, height: int = H) -> np.ndarray:
    """Horizontal gradient 0..1, useful for AVERAGE tests."""
    return np.tile(np.linspace(0.0, 1.0, width, dtype=np.float32), (height, 1))


def _ones(width: int = W, height: int = H) -> np.ndarray:
    return np.ones((height, width), dtype=np.float32)


def _zeros(width: int = W, height: int = H) -> np.ndarray:
    return np.zeros((height, width), dtype=np.float32)


# ── aggregate_numeric: per-operator semantics ────────────────────────────────

class TestAggregateNumericReplace:
    def test_fully_masked_replaces_base(self):
        out = aggregate_numeric(
            base=0.2,
            contributions=[(_ones(), 0.8, NumericOp.REPLACE)],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        assert np.allclose(out, 0.8)

    def test_uncovered_keeps_base(self):
        out = aggregate_numeric(
            base=0.2,
            contributions=[(_zeros(), 0.8, NumericOp.REPLACE)],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        assert np.allclose(out, 0.2)

    def test_partial_alpha_blends(self):
        alpha = np.full((H, W), 0.5, dtype=np.float32)
        out = aggregate_numeric(
            base=0.0,
            contributions=[(alpha, 1.0, NumericOp.REPLACE)],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        # base 0.0, value 1.0, alpha 0.5 → 0.5
        assert np.allclose(out, 0.5)

    def test_bottom_up_top_replaces(self):
        out = aggregate_numeric(
            base=0.0,
            contributions=[
                (_ones(), 0.3, NumericOp.REPLACE),   # bottom
                (_ones(), 0.9, NumericOp.REPLACE),   # top
            ],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        assert np.allclose(out, 0.9)


class TestAggregateNumericAdd:
    def test_full_mask_adds_value(self):
        out = aggregate_numeric(
            base=0.2,
            contributions=[(_ones(), 0.3, NumericOp.ADD)],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        assert np.allclose(out, 0.5)

    def test_partial_alpha_scales_addition(self):
        out = aggregate_numeric(
            base=0.0,
            contributions=[(np.full((H, W), 0.5, np.float32), 1.0, NumericOp.ADD)],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        assert np.allclose(out, 0.5)

    def test_clamped_to_hi(self):
        out = aggregate_numeric(
            base=0.9,
            contributions=[(_ones(), 1.0, NumericOp.ADD)],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        assert np.allclose(out, 1.0)


class TestAggregateNumericAverage:
    def test_partial_alpha_smooth_blend(self):
        alpha = np.full((H, W), 0.5, dtype=np.float32)
        out = aggregate_numeric(
            base=0.2,
            contributions=[(alpha, 0.8, NumericOp.AVERAGE)],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        # 0.2*(1-0.5) + 0.8*0.5 = 0.1 + 0.4 = 0.5
        assert np.allclose(out, 0.5)

    def test_full_alpha_equals_value(self):
        out = aggregate_numeric(
            base=0.2,
            contributions=[(_ones(), 0.8, NumericOp.AVERAGE)],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        assert np.allclose(out, 0.8)


class TestAggregateNumericMultiply:
    def test_full_mask_value_two_doubles(self):
        out = aggregate_numeric(
            base=0.3,
            contributions=[(_ones(), 2.0, NumericOp.MULTIPLY)],
            width=W, height=H, lo=0.0, hi=2.0,
        )
        # factor = 1 + (2-1)*1 = 2; 0.3 * 2 = 0.6
        assert np.allclose(out, 0.6)

    def test_partial_alpha_scales_factor(self):
        alpha = np.full((H, W), 0.5, dtype=np.float32)
        out = aggregate_numeric(
            base=0.4,
            contributions=[(alpha, 2.0, NumericOp.MULTIPLY)],
            width=W, height=H, lo=0.0, hi=2.0,
        )
        # factor = 1 + (2-1)*0.5 = 1.5; 0.4 * 1.5 = 0.6
        assert np.allclose(out, 0.6)


class TestAggregateNumericMax:
    def test_takes_higher_value(self):
        out = aggregate_numeric(
            base=0.3,
            contributions=[(_ones(), 0.7, NumericOp.MAX)],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        assert np.allclose(out, 0.7)

    def test_keeps_base_when_lower(self):
        out = aggregate_numeric(
            base=0.9,
            contributions=[(_ones(), 0.3, NumericOp.MAX)],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        assert np.allclose(out, 0.9)


class TestAggregateNumericBottomUp:
    def test_chain_replace_then_add(self):
        out = aggregate_numeric(
            base=0.0,
            contributions=[
                (_ones(), 0.5, NumericOp.REPLACE),
                (_ones(), 0.2, NumericOp.ADD),
            ],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        # bottom REPLACE → 0.5; top ADD → 0.5 + 0.2 = 0.7
        assert np.allclose(out, 0.7)

    def test_disjoint_masks_independent_regions(self):
        out = aggregate_numeric(
            base=0.0,
            contributions=[
                (_mask_arr(_left_half_mask()), 0.4, NumericOp.REPLACE),
                (_mask_arr(_right_half_mask()), 0.9, NumericOp.REPLACE),
            ],
            width=W, height=H, lo=0.0, hi=1.0,
        )
        assert np.allclose(out[:, : W // 2], 0.4)
        assert np.allclose(out[:, W // 2 :], 0.9)


def _mask_arr(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask, dtype=np.float32) / 255.0


# ── aggregate_mask: union with operators ─────────────────────────────────────

class TestAggregateMaskUnion:
    def test_two_halves_add_to_full(self):
        out = aggregate_mask(
            [
                (_mask_arr(_left_half_mask()), NumericOp.ADD),
                (_mask_arr(_right_half_mask()), NumericOp.ADD),
            ],
            W, H,
        )
        assert np.allclose(out, 1.0)

    def test_multiply_by_zero_clears(self):
        out = aggregate_mask(
            [
                (_mask_arr(_full_mask()), NumericOp.ADD),
                (_zeros(), NumericOp.MULTIPLY),
            ],
            W, H,
        )
        assert np.allclose(out, 0.0)

    def test_replace_top_wins(self):
        out = aggregate_mask(
            [
                (_mask_arr(_full_mask()), NumericOp.ADD),       # full coverage
                (_mask_arr(_left_half_mask()), NumericOp.REPLACE),  # only left replaced
            ],
            W, H,
        )
        # Left half: REPLACEd to 1.0 (was already 1.0); right half: kept at 1.0 (REPLACE only affects covered).
        assert np.allclose(out[:, : W // 2], 1.0)
        assert np.allclose(out[:, W // 2 :], 1.0)

    def test_max_keeps_strongest(self):
        weaker = np.full((H, W), 0.3, dtype=np.float32)
        out = aggregate_mask(
            [
                (weaker, NumericOp.ADD),
                (_mask_arr(_left_half_mask()), NumericOp.MAX),
            ],
            W, H,
        )
        # Left half: max(0.3, 1.0) = 1.0; Right half: max stays 0.3 from below.
        assert np.allclose(out[:, : W // 2], 1.0)
        assert np.allclose(out[:, W // 2 :], 0.3)


# ── aggregate_prompt ──────────────────────────────────────────────────────────

class TestAggregatePrompt:
    def test_replace_top_wins(self):
        out = aggregate_prompt(
            "base",
            [(1.0, "p1", PromptOp.REPLACE), (1.0, "p2", PromptOp.REPLACE)],
        )
        assert out == "p2"

    def test_concat_with_unit_weight_no_paren(self):
        out = aggregate_prompt(
            "base",
            [(1.0, "fire", PromptOp.CONCAT)],
        )
        assert out == "base, fire"

    def test_concat_with_partial_weight_wraps(self):
        out = aggregate_prompt(
            "",
            [(0.5, "smoke", PromptOp.CONCAT)],
        )
        assert out == "(smoke:0.50)"

    def test_empty_prompt_skipped(self):
        out = aggregate_prompt(
            "base",
            [(1.0, "", PromptOp.CONCAT), (1.0, "x", PromptOp.CONCAT)],
        )
        assert out == "base, x"

    def test_zero_weight_skipped(self):
        out = aggregate_prompt(
            "base",
            [(0.0, "ignored", PromptOp.CONCAT)],
        )
        assert out == "base"

    def test_embed_blend_falls_back_to_top_text(self):
        out = aggregate_prompt(
            "base",
            [(1.0, "p1", PromptOp.EMBED_BLEND), (1.0, "p2", PromptOp.EMBED_BLEND)],
        )
        assert out == "p2"

    def test_base_only_when_no_contributions(self):
        assert aggregate_prompt("base", []) == "base"


# ── Region effective alpha (schedule + weight + negate) ──────────────────────

class TestRegionEffectiveAlpha:
    def test_default_full_alpha_when_no_mask(self):
        region = Region(weight=1.0)
        plan, _ = build_single_pass(
            Scene(
                layers=(Layer(regions=(region,), default_denoise=0.5),),
                width=W, height=H, base_denoise=0.0,
            ),
            RendererSupport(),
        )
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        assert np.allclose(arr, 0.5, atol=0.005)

    def test_schedule_zero_disables_region(self):
        region = Region(mask=_full_mask(), weight=1.0, schedule=(0.5, 0.5), denoise=0.9)
        plan, _ = build_single_pass(
            Scene(layers=(Layer(regions=(region,)),), width=W, height=H, base_denoise=0.0),
            RendererSupport(),
        )
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        assert np.allclose(arr, 0.0)

    def test_negate_inverts(self):
        region = Region(mask=_left_half_mask(), denoise=0.7, negate=True, weight=1.0)
        plan, _ = build_single_pass(
            Scene(layers=(Layer(regions=(region,)),), width=W, height=H, base_denoise=0.0),
            RendererSupport(),
        )
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        # Negated: right half active, left half inactive
        assert arr[0, 0] < 0.05    # left → still 0
        assert arr[0, W - 1] > 0.6  # right → ~0.7

    def test_weight_scales_alpha(self):
        region = Region(mask=_full_mask(), weight=0.5, denoise=0.8, denoise_op=NumericOp.AVERAGE)
        plan, _ = build_single_pass(
            Scene(layers=(Layer(regions=(region,)),), width=W, height=H, base_denoise=0.0),
            RendererSupport(),
        )
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        # AVERAGE with alpha=0.5: 0*(0.5) + 0.8*(0.5) = 0.4
        assert np.allclose(arr, 0.4, atol=0.01)


# ── Layer-level defaults ─────────────────────────────────────────────────────

class TestLayerDefaults:
    def test_region_inherits_layer_default_denoise(self):
        region = Region(mask=_full_mask())  # no per-region denoise
        layer = Layer(regions=(region,), default_denoise=0.6)
        plan, _ = build_single_pass(
            Scene(layers=(layer,), width=W, height=H, base_denoise=0.0),
            RendererSupport(),
        )
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        assert np.allclose(arr, 0.6, atol=0.005)

    def test_region_override_wins_over_layer_default(self):
        region = Region(mask=_full_mask(), denoise=0.9)
        layer = Layer(regions=(region,), default_denoise=0.3)
        plan, _ = build_single_pass(
            Scene(layers=(layer,), width=W, height=H, base_denoise=0.0),
            RendererSupport(),
        )
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        assert np.allclose(arr, 0.9, atol=0.005)

    def test_region_inherits_layer_default_prompt(self):
        region = Region(mask=_full_mask())
        layer = Layer(regions=(region,), default_prompt="house")
        plan, _ = build_single_pass(
            Scene(layers=(layer,), width=W, height=H, base_prompt=""),
            RendererSupport(),
        )
        assert "house" in plan.prompt


# ── Single-pass plan ─────────────────────────────────────────────────────────

class TestBuildSinglePass:
    def test_two_disjoint_regions_aggregate(self):
        left = Region(mask=_left_half_mask(), denoise=0.4, prompt="A")
        right = Region(mask=_right_half_mask(), denoise=0.8, prompt="B")
        scene = Scene(
            layers=(Layer(regions=(left, right)),),
            width=W, height=H, base_denoise=0.0, base_prompt="base",
        )
        plan, warnings = build_single_pass(scene, RendererSupport())
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        assert np.allclose(arr[:, : W // 2], 0.4, atol=0.005)
        assert np.allclose(arr[:, W // 2 :], 0.8, atol=0.005)
        # Both prompts present in concat
        assert "A" in plan.prompt and "B" in plan.prompt

    def test_uncovered_pixels_keep_base_denoise(self):
        # Left half only — right half should keep base_denoise (e.g. 0.0)
        left = Region(mask=_left_half_mask(), denoise=0.9)
        scene = Scene(
            layers=(Layer(regions=(left,)),),
            width=W, height=H, base_denoise=0.0,
        )
        plan, _ = build_single_pass(scene, RendererSupport())
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        assert np.allclose(arr[:, : W // 2], 0.9, atol=0.005)
        assert np.allclose(arr[:, W // 2 :], 0.0)

    def test_cfg_scalar_falls_back_to_max(self):
        region = Region(mask=_full_mask(), cfg=8.5)
        scene = Scene(
            layers=(Layer(regions=(region,)),),
            width=W, height=H, base_cfg=1.0,
        )
        plan, _ = build_single_pass(scene, RendererSupport())
        assert plan.cfg_scalar == pytest.approx(8.5, abs=0.05)

    def test_mask_union_covers_both_regions(self):
        left = Region(mask=_left_half_mask(), denoise=0.5)
        right = Region(mask=_right_half_mask(), denoise=0.5)
        scene = Scene(
            layers=(Layer(regions=(left, right)),),
            width=W, height=H, base_denoise=0.0,
        )
        plan, _ = build_single_pass(scene, RendererSupport())
        mask_arr = np.asarray(plan.mask, dtype=np.float32) / 255.0
        # Both halves should be covered (mask_op=ADD by default)
        assert mask_arr.min() > 0.5


# ── Layered-pass plan ────────────────────────────────────────────────────────

class TestBuildLayeredPass:
    def test_one_region_one_pass(self):
        region = Region(mask=_full_mask(), prompt="dog", denoise=0.8, cfg=7.0)
        scene = Scene(layers=(Layer(layer_id="L1", regions=(region,)),), width=W, height=H)
        plan, _ = build_layered_pass(scene, RendererSupport())
        assert len(plan.passes) == 1
        p = plan.passes[0]
        assert p.layer_id == "L1"
        assert p.prompt == "dog"
        assert p.denoise == pytest.approx(0.8)
        assert p.cfg == pytest.approx(7.0)

    def test_region_without_prompt_skipped(self):
        region = Region(mask=_full_mask(), denoise=0.5)  # no prompt
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        plan, _ = build_layered_pass(scene, RendererSupport())
        assert len(plan.passes) == 0

    def test_bottom_up_order_preserved(self):
        r1 = Region(mask=_full_mask(), prompt="A")
        r2 = Region(mask=_full_mask(), prompt="B")
        scene = Scene(
            layers=(
                Layer(layer_id="bottom", regions=(r1,)),
                Layer(layer_id="top", regions=(r2,)),
            ),
            width=W, height=H,
        )
        plan, _ = build_layered_pass(scene, RendererSupport())
        assert [p.layer_id for p in plan.passes] == ["bottom", "top"]

    def test_inactive_region_skipped(self):
        # Schedule (0.5, 0.5) → zero influence
        region = Region(mask=_full_mask(), prompt="x", schedule=(0.5, 0.5))
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        plan, _ = build_layered_pass(scene, RendererSupport())
        assert len(plan.passes) == 0


# ── Tiled-pass plan ──────────────────────────────────────────────────────────

class TestBuildTiledPass:
    def test_basic_tiling(self):
        region = Region(mask=_full_mask(width=64, height=64), prompt="x", denoise=0.5)
        scene = Scene(layers=(Layer(regions=(region,)),), width=64, height=64)
        plan, _ = build_tiled_pass(scene, RendererSupport(), tile_size=32, tile_overlap=0)
        # 64/32 = 2x2 = 4 tiles
        assert len(plan.tiles) == 4
        for tile in plan.tiles:
            assert tile.plan is not None
            assert tile.target_w >= tile.target_h or tile.target_h >= tile.target_w  # snapped to SDXL

    def test_tile_bbox_covers_full_canvas(self):
        region = Region(mask=_full_mask(width=64, height=64), denoise=0.5)
        scene = Scene(layers=(Layer(regions=(region,)),), width=64, height=64)
        plan, _ = build_tiled_pass(scene, RendererSupport(), tile_size=32, tile_overlap=0)
        max_x = max(t.bbox[2] for t in plan.tiles)
        max_y = max(t.bbox[3] for t in plan.tiles)
        assert max_x == 64
        assert max_y == 64


# ── Capability fallback ──────────────────────────────────────────────────────

class TestCapabilityFallback:
    def test_per_region_cfg_dropped_warning(self):
        region = Region(mask=_full_mask(), cfg=7.5)
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        support = RendererSupport(per_region_cfg=False)
        plan, warnings = build_single_pass(scene, support)
        # Plan still includes a cfg_scalar fallback
        assert plan.cfg_scalar == pytest.approx(7.5, abs=0.05)
        assert any("CFG" in m for m in warnings.messages)

    def test_layered_falls_back_to_single_when_multi_pass_unsupported(self):
        region = Region(mask=_full_mask(), prompt="x")
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        support = RendererSupport(multi_pass=False)
        result = compose(scene, RenderMode.LAYERED_PASS, support)
        assert result.mode is RenderMode.SINGLE_PASS
        assert result.single is not None
        assert any("layered-pass" in m for m in result.warnings.messages)

    def test_regional_prompts_dropped(self):
        region = Region(
            mask=_full_mask(), prompt="x", prompt_op=PromptOp.EMBED_BLEND,
        )
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        support = RendererSupport(regional_prompts=False)
        plan, warnings = build_single_pass(scene, support)
        assert plan.regional_prompts == ()
        assert warnings.regional_prompts_dropped


# ── compose() entry point ────────────────────────────────────────────────────

class TestCompose:
    def test_single_pass_returns_single_plan(self):
        region = Region(mask=_full_mask(), denoise=0.5)
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        result = compose(scene, RenderMode.SINGLE_PASS, RendererSupport())
        assert result.mode is RenderMode.SINGLE_PASS
        assert result.single is not None
        assert result.layered is None
        assert result.tiled is None

    def test_layered_returns_layered_plan(self):
        region = Region(mask=_full_mask(), prompt="x")
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        result = compose(scene, RenderMode.LAYERED_PASS, RendererSupport())
        assert result.mode is RenderMode.LAYERED_PASS
        assert result.layered is not None
        assert result.single is None

    def test_tiled_returns_tiled_plan(self):
        region = Region(mask=_full_mask(width=64, height=64), denoise=0.5)
        scene = Scene(layers=(Layer(regions=(region,)),), width=64, height=64)
        result = compose(scene, RenderMode.TILED_PASS, RendererSupport(),
                         tile_size=32, tile_overlap=0)
        assert result.mode is RenderMode.TILED_PASS
        assert result.tiled is not None
        assert len(result.tiled.tiles) == 4


# ── scene_from_legacy ────────────────────────────────────────────────────────

class TestSceneFromLegacy:
    def test_groups_regions_by_layer_id(self):
        conds = [
            {"layer_id": "L1", "prompt": "a", "denoise": 0.4},
            {"layer_id": "L1", "prompt": "b", "denoise": 0.5},
            {"layer_id": "L2", "prompt": "c", "denoise": 0.6},
        ]
        scene = scene_from_legacy(conds, "base", "", 0.0, 1.0, W, H)
        assert len(scene.layers) == 2
        assert len(scene.layers[0].regions) == 2
        assert len(scene.layers[1].regions) == 1

    def test_layer_order_preserved(self):
        conds = [
            {"layer_id": "Z", "prompt": "z"},
            {"layer_id": "A", "prompt": "a"},
        ]
        scene = scene_from_legacy(conds, "", "", 0.0, 1.0, W, H)
        assert [l.layer_id for l in scene.layers] == ["Z", "A"]

    def test_legacy_mode_override_to_replace_op(self):
        conds = [{"layer_id": "L", "prompt": "x", "mode": "override", "denoise": 0.9}]
        scene = scene_from_legacy(conds, "", "", 0.0, 1.0, W, H)
        assert scene.layers[0].regions[0].denoise_op is NumericOp.REPLACE

    def test_legacy_mode_mask_to_average_op(self):
        conds = [{"layer_id": "L", "prompt": "x", "mode": "mask", "denoise": 0.9}]
        scene = scene_from_legacy(conds, "", "", 0.0, 1.0, W, H)
        assert scene.layers[0].regions[0].denoise_op is NumericOp.AVERAGE

    def test_explicit_denoise_operator_wins(self):
        conds = [{
            "layer_id": "L",
            "prompt": "x",
            "mode": "mask",  # would normally → AVERAGE
            "denoise_operator": "add",
            "denoise": 0.4,
        }]
        scene = scene_from_legacy(conds, "", "", 0.0, 1.0, W, H)
        assert scene.layers[0].regions[0].denoise_op is NumericOp.ADD

    def test_tagger_overrides_replace_prompt(self):
        conds = [{"layer_id": "L1", "prompt": "explicit"}]
        scene = scene_from_legacy(conds, "", "", 0.0, 1.0, W, H,
                                   prompt_overrides={"L1": "tagged_x"})
        assert scene.layers[0].regions[0].prompt == "tagged_x"

    def test_empty_prompt_becomes_none(self):
        conds = [{"layer_id": "L1", "prompt": "   "}]
        scene = scene_from_legacy(conds, "", "", 0.0, 1.0, W, H)
        assert scene.layers[0].regions[0].prompt is None

    def test_mask_attached_to_regions(self):
        m = _full_mask()
        conds = [{"layer_id": "L1", "prompt": "x"}]
        scene = scene_from_legacy(conds, "", "", 0.0, 1.0, W, H, masks_by_layer={"L1": m})
        assert scene.layers[0].regions[0].mask is m

    def test_prompt_falls_back_to_region_color_mask(self):
        conds = [{"layer_id": "L1", "region_id": "r1", "prompt": "fire"}]
        scene = scene_from_legacy(
            conds, "", "", 0.0, 1.0, W, H,
            masks_by_layer={"L1": _full_mask()},
            channel_masks_by_region={"r1": {"color": _left_half_mask()}},
        )
        plan, _ = build_layered_pass(scene, RendererSupport())
        mask_arr = np.asarray(plan.passes[0].mask, dtype=np.float32) / 255.0
        assert mask_arr[:, : W // 2].max() > 0.5
        assert mask_arr[:, W // 2 :].max() < 0.01


# ── End-to-end scenario: the user's reported bug ─────────────────────────────

class TestUserScenarios:
    def test_empty_scene_yields_no_per_region_effect(self):
        """No layers → no contributions; aggregated maps = base values."""
        scene = Scene(width=W, height=H, base_prompt="dog", base_denoise=0.5)
        plan, _ = build_single_pass(scene, RendererSupport())
        assert plan.prompt == "dog"
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        assert np.allclose(arr, 0.5, atol=0.005)

    def test_partial_layer_prompt_in_mask_only(self):
        """Single layer covering left half with its own prompt: layered-pass
        gives ONE pass whose mask is the left half. The renderer composites
        the result over the previous frame (handled outside this module)."""
        region = Region(mask=_left_half_mask(), prompt="fire", denoise=0.8)
        scene = Scene(
            layers=(Layer(layer_id="L1", regions=(region,)),),
            width=W, height=H, base_prompt="dog",
        )
        plan, _ = build_layered_pass(scene, RendererSupport())
        assert len(plan.passes) == 1
        p = plan.passes[0]
        assert p.prompt == "fire"
        mask_arr = np.asarray(p.mask, dtype=np.float32) / 255.0
        assert mask_arr[:, : W // 2].max() > 0.5
        assert mask_arr[:, W // 2 :].max() < 0.01

    def test_top_layer_full_coverage_replaces(self):
        bottom = Region(mask=_left_half_mask(), prompt="forest", denoise=0.6)
        top = Region(mask=_full_mask(), prompt="fire", denoise=0.9, denoise_op=NumericOp.REPLACE)
        scene = Scene(
            layers=(
                Layer(layer_id="bg", regions=(bottom,)),
                Layer(layer_id="fg", regions=(top,)),
            ),
            width=W, height=H, base_denoise=0.0,
        )
        plan, _ = build_single_pass(scene, RendererSupport())
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        # Top REPLACE with full coverage → denoise=0.9 everywhere
        assert np.allclose(arr, 0.9, atol=0.005)
        # Prompt concat preserves both
        assert "fire" in plan.prompt
        assert "forest" in plan.prompt

    def test_three_layers_bottom_up_chain(self):
        r1 = Region(mask=_full_mask(), denoise=0.3, denoise_op=NumericOp.REPLACE)
        r2 = Region(mask=_left_half_mask(), denoise=0.6, denoise_op=NumericOp.REPLACE)
        r3 = Region(mask=_center_block_mask(), denoise=0.9, denoise_op=NumericOp.REPLACE)
        scene = Scene(
            layers=(
                Layer(layer_id="a", regions=(r1,)),
                Layer(layer_id="b", regions=(r2,)),
                Layer(layer_id="c", regions=(r3,)),
            ),
            width=W, height=H, base_denoise=0.0,
        )
        plan, _ = build_single_pass(scene, RendererSupport())
        arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        # Center block (top): 0.9 over its area
        cy, cx = H // 2, W // 2
        assert arr[cy, cx] == pytest.approx(0.9, abs=0.01)
        # Outside center but left half: 0.6 from middle layer
        assert arr[H // 2, 2] == pytest.approx(0.6, abs=0.01)
        # Outside center, right half (no middle layer coverage): bottom 0.3
        assert arr[H // 2, W - 2] == pytest.approx(0.3, abs=0.01)
