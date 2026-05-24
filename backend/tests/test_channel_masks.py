"""Per-channel mask tests for the composition module.

Run:  pytest backend/tests/test_channel_masks.py -v

Verifies the semantic introduced by ``docs/LAYER_SYSTEM.md`` §3: each region
exposes optional channel-specific masks (``denoise_mask``, ``prompt_mask``,
``cfg_mask``, ``color_mask``) and each parameter aggregation honours its own
channel mask, falling back to ``Region.mask`` when missing.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.composition import (
    CFG_HI,
    Layer,
    NumericOp,
    Region,
    RendererSupport,
    Scene,
    _region_channel_mask,
    _region_effective_alpha,
    build_layered_pass,
    build_single_pass,
    scene_from_legacy,
)


W, H = 32, 32


def _full(value: int = 255, w: int = W, h: int = H) -> Image.Image:
    return Image.new("L", (w, h), value)


def _left_half(value: int = 255) -> Image.Image:
    m = Image.new("L", (W, H), 0)
    arr = np.asarray(m).copy()
    arr[:, : W // 2] = value
    return Image.fromarray(arr, "L")


def _right_half(value: int = 255) -> Image.Image:
    m = Image.new("L", (W, H), 0)
    arr = np.asarray(m).copy()
    arr[:, W // 2 :] = value
    return Image.fromarray(arr, "L")


def _center_block(value: int = 255) -> Image.Image:
    m = Image.new("L", (W, H), 0)
    arr = np.asarray(m).copy()
    arr[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = value
    return Image.fromarray(arr, "L")


def _full_f32(value: float, w: int = W, h: int = H) -> Image.Image:
    return Image.fromarray(np.full((h, w), value, dtype=np.float32), "F")


# ── Channel mask resolution ──────────────────────────────────────────────────

class TestRegionChannelMaskResolution:
    def test_specific_channel_mask_wins_over_fallback(self):
        region = Region(
            mask=_full(255),                      # legacy fallback (everywhere)
            denoise_mask=_left_half(255),         # explicit denoise-only mask
        )
        assert _region_channel_mask(region, "denoise") is region.denoise_mask
        # color falls back to ``mask`` because no color_mask is set
        assert _region_channel_mask(region, "color") is region.mask

    def test_color_mask_alias_for_color_channel(self):
        region = Region(color_mask=_left_half(255))
        assert _region_channel_mask(region, "color") is region.color_mask
        assert _region_channel_mask(region, "color_mask") is region.color_mask

    def test_all_none_returns_none(self):
        region = Region()
        # No mask anywhere → None (uniform full-canvas application)
        assert _region_channel_mask(region, "denoise") is None
        assert _region_channel_mask(region, "prompt") is None


# ── Effective alpha per channel ──────────────────────────────────────────────

class TestPerChannelEffectiveAlpha:
    def test_denoise_alpha_uses_denoise_mask(self):
        region = Region(
            mask=_full(255),
            denoise_mask=_left_half(255),
        )
        alpha = _region_effective_alpha(region, W, H, "denoise")
        # Left half = 1.0, right half = 0.0
        assert np.allclose(alpha[:, : W // 2], 1.0)
        assert np.allclose(alpha[:, W // 2 :], 0.0)

    def test_color_alpha_falls_back_to_main_mask(self):
        region = Region(
            mask=_full(255),
            denoise_mask=_left_half(255),       # only denoise has a channel mask
        )
        alpha = _region_effective_alpha(region, W, H, "color")
        # color falls back to ``mask`` (full white) → 1.0 everywhere
        assert np.allclose(alpha, 1.0)

    def test_prompt_alpha_uses_prompt_mask(self):
        region = Region(
            mask=_full(255),
            prompt_mask=_right_half(128),       # 50% on the right half only
        )
        alpha = _region_effective_alpha(region, W, H, "prompt")
        assert np.allclose(alpha[:, : W // 2], 0.0)
        assert np.allclose(alpha[:, W // 2 :], 128 / 255, atol=0.01)


# ── Decoupled aggregation through single-pass ────────────────────────────────

class TestSinglePassChannelDecoupling:
    def test_denoise_and_prompt_can_target_different_halves(self):
        """Region covers the full canvas with denoise_mask=LEFT and
        prompt_mask=RIGHT. The aggregated denoise map only shows up on the
        LEFT, but the prompt weight comes from the RIGHT half."""
        region = Region(
            mask=_full(255),                      # legacy "presence"
            denoise=0.8,
            denoise_mask=_left_half(255),         # denoise applies on the LEFT
            prompt="fire",
            prompt_mask=_right_half(255),         # prompt applies on the RIGHT
        )
        scene = Scene(
            layers=(Layer(regions=(region,)),),
            width=W, height=H,
            base_denoise=0.0, base_prompt="base",
        )
        plan, _ = build_single_pass(scene, RendererSupport())

        denoise_arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        # Denoise non-zero only on LEFT
        assert denoise_arr[H // 2, 4] == pytest.approx(0.8, abs=0.01)
        assert denoise_arr[H // 2, W - 4] == pytest.approx(0.0, abs=0.01)
        # Prompt is concatenated (full weight because prompt_mask had max 1.0)
        assert "fire" in plan.prompt

    def test_cfg_uses_cfg_mask_not_main_mask(self):
        region = Region(
            mask=_full(255),
            cfg=8.0,
            cfg_mask=_left_half(255),
        )
        scene = Scene(
            layers=(Layer(regions=(region,)),),
            width=W, height=H,
            base_cfg=1.0,
        )
        plan, _ = build_single_pass(scene, RendererSupport())
        cfg_arr = np.asarray(plan.cfg_map, dtype=np.float32) / 255.0  # normalised
        # On the LEFT the explicit CFG mask is absolute full-scale CFG (30.0).
        # On the RIGHT it should be the base cfg (1.0/30 ≈ 0.033)
        left_val = float(cfg_arr[H // 2, 4])
        right_val = float(cfg_arr[H // 2, W - 4])
        assert left_val > right_val
        assert plan.cfg_scalar == pytest.approx(CFG_HI, abs=0.1)

    def test_cfg_mask_luma_is_absolute_cfg_value(self):
        region = Region(
            mask=_full(255),
            cfg=10.0,
            cfg_mask=_full(25),
        )
        scene = Scene(
            layers=(Layer(regions=(region,)),),
            width=W, height=H,
            base_cfg=10.0,
        )
        plan, _ = build_single_pass(scene, RendererSupport())
        cfg_arr = np.asarray(plan.cfg_map, dtype=np.float32) / 255.0 * CFG_HI

        expected = CFG_HI * 25.0 / 255.0
        assert float(cfg_arr[H // 2, W // 2]) == pytest.approx(expected, abs=0.08)
        assert plan.cfg_scalar == pytest.approx(expected, abs=0.1)

    def test_cfg_float_mask_preserves_absolute_sub_byte_precision(self):
        region = Region(
            mask=_full(255),
            cfg=10.0,
            cfg_mask=_full_f32(1.0375),
        )
        scene = Scene(
            layers=(Layer(regions=(region,)),),
            width=W,
            height=H,
            base_cfg=0.0,
        )
        plan, _ = build_single_pass(scene, RendererSupport())

        assert plan.cfg_scalar == pytest.approx(1.0375, abs=0.01)

    def test_missing_channel_falls_back_to_legacy_mask(self):
        """Old payload: only ``mask`` set. The per-channel masks should all
        resolve to ``mask`` and the aggregation behaves like before."""
        region = Region(
            mask=_left_half(255),
            denoise=0.9,
            prompt="dragon",
        )
        scene = Scene(
            layers=(Layer(regions=(region,)),),
            width=W, height=H, base_denoise=0.0,
        )
        plan, _ = build_single_pass(scene, RendererSupport())
        denoise_arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        assert denoise_arr[H // 2, 4] == pytest.approx(0.9, abs=0.01)
        assert denoise_arr[H // 2, W - 4] == pytest.approx(0.0, abs=0.01)
        assert "dragon" in plan.prompt


# ── Layered-pass uses prompt_mask ────────────────────────────────────────────

class TestLayeredPassPromptMask:
    def test_layered_pass_uses_prompt_mask_for_inpaint_region(self):
        """In layered-pass each region's mask in the PassSpec is the prompt
        channel mask — that's what drives the regional inpaint."""
        region = Region(
            mask=_full(255),
            prompt="fire",
            prompt_mask=_left_half(255),
        )
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        plan, _ = build_layered_pass(scene, RendererSupport())
        assert len(plan.passes) == 1
        mask_arr = np.asarray(plan.passes[0].mask, dtype=np.float32) / 255.0
        # The pass mask = the prompt_mask, not the main ``mask``
        assert np.allclose(mask_arr[:, : W // 2], 1.0)
        assert np.allclose(mask_arr[:, W // 2 :], 0.0)

    def test_layered_pass_keeps_denoise_mask_separate(self):
        region = Region(
            mask=_full(255),
            prompt="fire",
            prompt_mask=_full(255),
            denoise_mask=_left_half(255),
            denoise=0.8,
        )
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        plan, _ = build_layered_pass(scene, RendererSupport())
        assert len(plan.passes) == 1
        prompt_arr = np.asarray(plan.passes[0].mask, dtype=np.float32) / 255.0
        denoise_arr = np.asarray(plan.passes[0].denoise_mask, dtype=np.float32) / 255.0
        assert np.allclose(prompt_arr, 1.0)
        assert np.allclose(denoise_arr[:, : W // 2], 1.0)
        assert np.allclose(denoise_arr[:, W // 2 :], 0.0)

    def test_layered_pass_cfg_mask_sets_scalar_cfg(self):
        region = Region(
            mask=_full(255),
            prompt="fire",
            cfg=10.0,
            cfg_mask=_full(25),
        )
        scene = Scene(layers=(Layer(regions=(region,)),), width=W, height=H)
        plan, _ = build_layered_pass(scene, RendererSupport())
        assert len(plan.passes) == 1
        assert plan.passes[0].cfg == pytest.approx(25.0 / 255.0 * 30.0, abs=0.01)


# ── Signal stack disposition tests ──────────────────────────────────────────

class TestSignalStackDispositions:
    def test_single_pass_keeps_color_prompt_denoise_and_cfg_signals_decoupled(self):
        region = Region(
            mask=_full(255),
            color_mask=_left_half(255),
            prompt_mask=_right_half(255),
            denoise_mask=_center_block(255),
            cfg_mask=_full_f32(4.25),
            prompt="monster",
            denoise=0.75,
            cfg=9.0,
        )
        scene = Scene(
            layers=(Layer(layer_id="L1", regions=(region,)),),
            width=W,
            height=H,
            base_prompt="room",
            base_denoise=0.0,
            base_cfg=1.0,
        )

        plan, _ = build_single_pass(scene, RendererSupport())

        color_arr = np.asarray(plan.mask, dtype=np.float32) / 255.0
        denoise_arr = np.asarray(plan.denoise_map, dtype=np.float32) / 255.0
        assert np.allclose(color_arr[:, : W // 2], 1.0)
        assert np.allclose(color_arr[:, W // 2 :], 0.0)
        assert denoise_arr[H // 2, W // 2] == pytest.approx(0.75, abs=0.01)
        assert denoise_arr[1, 1] == pytest.approx(0.0, abs=0.01)
        assert plan.cfg_scalar == pytest.approx(4.25, abs=0.01)
        assert "room" in plan.prompt
        assert "monster" in plan.prompt

    def test_rgba_alpha_is_prompt_mask_fallback_for_layered_pass(self):
        region = Region(
            color_mask=_left_half(255),
            prompt="monster",
        )
        scene = Scene(layers=(Layer(layer_id="L1", regions=(region,)),), width=W, height=H)

        plan, _ = build_layered_pass(scene, RendererSupport())

        mask_arr = np.asarray(plan.passes[0].mask, dtype=np.float32) / 255.0
        assert np.allclose(mask_arr[:, : W // 2], 1.0)
        assert np.allclose(mask_arr[:, W // 2 :], 0.0)

    def test_explicit_prompt_mask_overrides_rgba_alpha_for_layered_pass(self):
        region = Region(
            color_mask=_left_half(255),
            prompt_mask=_right_half(255),
            prompt="monster",
        )
        scene = Scene(layers=(Layer(layer_id="L1", regions=(region,)),), width=W, height=H)

        plan, _ = build_layered_pass(scene, RendererSupport())

        mask_arr = np.asarray(plan.passes[0].mask, dtype=np.float32) / 255.0
        assert np.allclose(mask_arr[:, : W // 2], 0.0)
        assert np.allclose(mask_arr[:, W // 2 :], 1.0)

    def test_bottom_up_layers_keep_independent_prompt_pass_order(self):
        bottom = Region(region_id="bottom", prompt="jungle", prompt_mask=_left_half(255))
        top = Region(region_id="top", prompt="monster", prompt_mask=_right_half(255))
        scene = Scene(
            layers=(
                Layer(layer_id="bottom-layer", regions=(bottom,)),
                Layer(layer_id="top-layer", regions=(top,)),
            ),
            width=W,
            height=H,
        )

        plan, _ = build_layered_pass(scene, RendererSupport())

        assert [p.prompt for p in plan.passes] == ["jungle", "monster"]
        bottom_arr = np.asarray(plan.passes[0].mask, dtype=np.float32) / 255.0
        top_arr = np.asarray(plan.passes[1].mask, dtype=np.float32) / 255.0
        assert np.allclose(bottom_arr[:, : W // 2], 1.0)
        assert np.allclose(bottom_arr[:, W // 2 :], 0.0)
        assert np.allclose(top_arr[:, : W // 2], 0.0)
        assert np.allclose(top_arr[:, W // 2 :], 1.0)


# ── scene_from_legacy with channel_masks_by_region ───────────────────────────

class TestSceneFromLegacyWithChannelMasks:
    def test_channel_masks_attached_to_region(self):
        conds = [{"layer_id": "L1", "region_id": "R1",
                  "prompt": "x", "denoise": 0.5}]
        legacy_mask = _full(255)
        denoise_mask = _left_half(255)
        prompt_mask = _right_half(255)
        scene = scene_from_legacy(
            conds, "", "", 0.0, 1.0, W, H,
            masks_by_layer={"L1": legacy_mask},
            channel_masks_by_region={"R1": {
                "denoise": denoise_mask,
                "prompt": prompt_mask,
            }},
        )
        region = scene.layers[0].regions[0]
        # Channel-specific masks were attached
        assert region.denoise_mask is denoise_mask
        assert region.prompt_mask is prompt_mask
        # Legacy mask is preserved as the fallback
        assert region.mask is legacy_mask
        # Other channels fall back to the legacy mask via resolution
        assert _region_channel_mask(region, "cfg") is legacy_mask
        assert _region_channel_mask(region, "color") is legacy_mask

    def test_layer_qualified_channel_masks_do_not_collide(self):
        conds = [
            {"layer_id": "L1", "region_id": "shared", "prompt": "cat"},
            {"layer_id": "L2", "region_id": "shared", "prompt": "dog"},
        ]
        mask_a = _left_half(255)
        mask_b = _right_half(255)
        scene = scene_from_legacy(
            conds, "", "", 0.0, 1.0, W, H,
            masks_by_layer={"L1": _full(255), "L2": _full(255)},
            channel_masks_by_region={
                "L1:shared": {"color": mask_a},
                "L2:shared": {"color": mask_b},
            },
        )
        assert scene.layers[0].regions[0].color_mask is mask_a
        assert scene.layers[1].regions[0].color_mask is mask_b

    def test_no_channel_masks_means_full_legacy_compat(self):
        conds = [{"layer_id": "L1", "region_id": "R1", "prompt": "x"}]
        scene = scene_from_legacy(
            conds, "", "", 0.0, 1.0, W, H,
            masks_by_layer={"L1": _full(255)},
            channel_masks_by_region=None,
        )
        r = scene.layers[0].regions[0]
        assert r.denoise_mask is None
        assert r.prompt_mask is None
        assert r.cfg_mask is None
        # All channels fall back to ``mask``
        assert _region_channel_mask(r, "denoise") is r.mask
