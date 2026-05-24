"""Unit tests for the tile executor.

Run:  pytest backend/tests/test_tile_executor.py -v

Tile executor responsibilities verified here:
  • Single-tile pass-through: result == renderer output
  • Multi-tile stitching: each tile renders its colour into its bbox
  • Feather blending: seams between adjacent tiles are smooth (no hard edge)
  • Dirty-tile skipping: only changed tiles re-render between frames
  • Resize: when the renderer returns at a different size than the bbox,
    the result is scaled back to the bbox
  • Error handling: a tile that raises does not abort the whole frame
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.composition import (
    RenderMode,
    RendererSupport,
    Scene,
    Layer,
    Region,
    SinglePassPlan,
    TilePassSpec,
    TiledPassPlan,
    compose,
)
from app.tile_executor import TileExecutor, _feather_weights, _tile_signature


# ── Helpers ───────────────────────────────────────────────────────────────────

def _solid_plan(prompt: str = "x") -> SinglePassPlan:
    """Build a minimal SinglePassPlan suitable for executor tests."""
    return SinglePassPlan(
        prompt=prompt,
        negative_prompt="",
        mask=Image.new("L", (4, 4), 255),
        denoise_map=Image.new("L", (4, 4), 128),
        cfg_map=Image.new("L", (4, 4), 128),
        cfg_scalar=1.5,
    )


def _solid_renderer(colour: tuple[int, int, int]):
    """Return a renderer that ignores its plan and paints the whole tile colour."""
    def render(plan: SinglePassPlan, w: int, h: int) -> Image.Image:
        return Image.new("RGB", (w, h), colour)
    return render


def _per_tile_renderer(colour_by_prompt: dict[str, tuple[int, int, int]]):
    """Renderer that paints a different colour depending on the tile prompt."""
    def render(plan: SinglePassPlan, w: int, h: int) -> Image.Image:
        return Image.new("RGB", (w, h), colour_by_prompt.get(plan.prompt, (0, 0, 0)))
    return render


# ── Feather weights ───────────────────────────────────────────────────────────

class TestFeatherWeights:
    def test_zero_feather_uniform_ones(self):
        w = _feather_weights(10, 10, 0)
        assert w.shape == (10, 10)
        assert np.allclose(w, 1.0)

    def test_feather_edges_zero(self):
        w = _feather_weights(32, 32, 8)
        # corners should be near 0 with non-zero feather
        assert w[0, 0] < 0.01
        # centre should be 1
        assert w[16, 16] == pytest.approx(1.0, abs=1e-3)

    def test_feather_capped_to_half_size(self):
        # feather >= tile/2 → graceful, weights still valid
        w = _feather_weights(4, 4, 100)
        assert w.shape == (4, 4)
        assert (w >= 0.0).all() and (w <= 1.0).all()


# ── Single-tile execution ────────────────────────────────────────────────────

class TestSingleTile:
    def test_single_tile_passes_renderer_output_through(self):
        tile = TilePassSpec(
            bbox=(0, 0, 32, 32),
            target_w=32, target_h=32,
            plan=_solid_plan(),
        )
        plan = TiledPassPlan(tiles=(tile,))
        executor = TileExecutor(feather=0)
        result = executor.execute(plan, (32, 32), _solid_renderer((255, 0, 0)))
        arr = np.asarray(result.image)
        assert arr.shape == (32, 32, 3)
        assert np.allclose(arr[:, :, 0], 255)
        assert np.allclose(arr[:, :, 1], 0)
        assert np.allclose(arr[:, :, 2], 0)
        assert result.rendered_tiles == 1
        assert result.skipped_tiles == 0

    def test_renderer_size_mismatch_is_resized(self):
        tile = TilePassSpec(
            bbox=(0, 0, 32, 32),
            target_w=64, target_h=64,  # renderer paints 64×64
            plan=_solid_plan(),
        )
        plan = TiledPassPlan(tiles=(tile,))
        executor = TileExecutor(feather=0)
        # renderer paints at target_w/h (64x64)
        result = executor.execute(plan, (32, 32), _solid_renderer((100, 200, 50)))
        arr = np.asarray(result.image)
        assert arr.shape == (32, 32, 3)
        assert arr[16, 16, 1] == pytest.approx(200, abs=2)


# ── Multi-tile stitching ─────────────────────────────────────────────────────

class TestMultiTileStitching:
    def test_two_tiles_side_by_side_colours_preserved(self):
        left = TilePassSpec(
            bbox=(0, 0, 16, 32),
            target_w=16, target_h=32,
            plan=_solid_plan(prompt="L"),
        )
        right = TilePassSpec(
            bbox=(16, 0, 32, 32),
            target_w=16, target_h=32,
            plan=_solid_plan(prompt="R"),
        )
        plan = TiledPassPlan(tiles=(left, right))
        executor = TileExecutor(feather=0)
        renderer = _per_tile_renderer({"L": (255, 0, 0), "R": (0, 0, 255)})
        result = executor.execute(plan, (32, 32), renderer)
        arr = np.asarray(result.image)
        # Left half red, right half blue
        assert arr[16, 4, 0] > 200    # red dominates on left
        assert arr[16, 28, 2] > 200   # blue dominates on right
        assert result.rendered_tiles == 2

    def test_feather_smooths_overlapping_tiles(self):
        # Two overlapping tiles with different colours: in the overlap, pixels
        # should be a blend (not pure either colour).
        left = TilePassSpec(
            bbox=(0, 0, 20, 32),
            target_w=20, target_h=32,
            plan=_solid_plan(prompt="L"),
        )
        right = TilePassSpec(
            bbox=(12, 0, 32, 32),
            target_w=20, target_h=32,
            plan=_solid_plan(prompt="R"),
        )
        plan = TiledPassPlan(tiles=(left, right))
        executor = TileExecutor(feather=8)
        renderer = _per_tile_renderer({"L": (255, 0, 0), "R": (0, 0, 255)})
        result = executor.execute(plan, (32, 32), renderer)
        arr = np.asarray(result.image)
        # In the overlap region (x ∈ [12, 20)), both red and blue should appear
        overlap = arr[:, 12:20, :]
        assert overlap[..., 0].max() > 50  # some red
        assert overlap[..., 2].max() > 50  # some blue
        # And no single pixel is fully one colour (perfect seam)
        assert overlap[..., 0].min() < 250 or overlap[..., 2].min() < 250


# ── Dirty-tile skipping ──────────────────────────────────────────────────────

class TestDirtyTileSkipping:
    def test_unchanged_tile_skipped_second_call(self):
        tile = TilePassSpec(
            bbox=(0, 0, 32, 32),
            target_w=32, target_h=32,
            plan=_solid_plan(prompt="static"),
        )
        plan = TiledPassPlan(tiles=(tile,))
        executor = TileExecutor(feather=0)

        # First frame: tile renders
        call_count = [0]
        def counting_renderer(p, w, h):
            call_count[0] += 1
            return Image.new("RGB", (w, h), (100, 100, 100))

        r1 = executor.execute(plan, (32, 32), counting_renderer)
        assert r1.rendered_tiles == 1
        assert call_count[0] == 1

        # Second frame with same plan: tile should be skipped (sig matches)
        r2 = executor.execute(plan, (32, 32), counting_renderer)
        assert r2.skipped_tiles == 1
        assert r2.rendered_tiles == 0
        assert call_count[0] == 1  # renderer not called again

    def test_changed_prompt_invalidates_cache(self):
        tile_v1 = TilePassSpec(
            bbox=(0, 0, 32, 32),
            target_w=32, target_h=32,
            plan=_solid_plan(prompt="v1"),
        )
        tile_v2 = TilePassSpec(
            bbox=(0, 0, 32, 32),
            target_w=32, target_h=32,
            plan=_solid_plan(prompt="v2"),
        )
        executor = TileExecutor(feather=0)
        executor.execute(TiledPassPlan(tiles=(tile_v1,)), (32, 32),
                         _solid_renderer((10, 10, 10)))
        r2 = executor.execute(TiledPassPlan(tiles=(tile_v2,)), (32, 32),
                              _solid_renderer((250, 250, 250)))
        # Prompt changed → re-render
        assert r2.rendered_tiles == 1

    def test_explicit_dirty_set_overrides_auto_detection(self):
        tile = TilePassSpec(
            bbox=(0, 0, 32, 32),
            target_w=32, target_h=32,
            plan=_solid_plan(prompt="x"),
        )
        plan = TiledPassPlan(tiles=(tile,))
        executor = TileExecutor(feather=0)
        executor.execute(plan, (32, 32), _solid_renderer((50, 50, 50)))
        # Even with identical plan, explicit dirty={bbox} forces re-render
        r2 = executor.execute(plan, (32, 32), _solid_renderer((200, 200, 200)),
                              dirty={(0, 0, 32, 32)})
        assert r2.rendered_tiles == 1
        arr = np.asarray(r2.image)
        # Second renderer's output dominates
        assert arr[16, 16, 0] == pytest.approx(200, abs=3)


# ── Tile signature stability ─────────────────────────────────────────────────

class TestTileSignature:
    def test_identical_tiles_same_sig(self):
        a = TilePassSpec(bbox=(0, 0, 32, 32), target_w=32, target_h=32, plan=_solid_plan("x"))
        b = TilePassSpec(bbox=(0, 0, 32, 32), target_w=32, target_h=32, plan=_solid_plan("x"))
        assert _tile_signature(a) == _tile_signature(b)

    def test_different_prompt_different_sig(self):
        a = TilePassSpec(bbox=(0, 0, 32, 32), target_w=32, target_h=32, plan=_solid_plan("a"))
        b = TilePassSpec(bbox=(0, 0, 32, 32), target_w=32, target_h=32, plan=_solid_plan("b"))
        assert _tile_signature(a) != _tile_signature(b)

    def test_different_cfg_different_sig(self):
        p_a = _solid_plan("x")
        p_b = SinglePassPlan(
            prompt="x", negative_prompt="", mask=p_a.mask,
            denoise_map=p_a.denoise_map, cfg_map=p_a.cfg_map, cfg_scalar=5.0,
        )
        a = TilePassSpec(bbox=(0, 0, 32, 32), target_w=32, target_h=32, plan=p_a)
        b = TilePassSpec(bbox=(0, 0, 32, 32), target_w=32, target_h=32, plan=p_b)
        assert _tile_signature(a) != _tile_signature(b)


# ── Error handling ───────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_renderer_exception_is_captured(self):
        tile = TilePassSpec(
            bbox=(0, 0, 32, 32),
            target_w=32, target_h=32,
            plan=_solid_plan(),
        )
        plan = TiledPassPlan(tiles=(tile,))
        executor = TileExecutor(feather=0)

        def boom(p, w, h):
            raise RuntimeError("oops")

        result = executor.execute(plan, (32, 32), boom)
        assert len(result.errors) == 1
        assert result.errors[0][0] == (0, 0, 32, 32)
        assert "oops" in result.errors[0][1]
        assert result.rendered_tiles == 0

    def test_one_failing_tile_does_not_break_others(self):
        good = TilePassSpec(
            bbox=(0, 0, 16, 16),
            target_w=16, target_h=16,
            plan=_solid_plan(prompt="ok"),
        )
        bad = TilePassSpec(
            bbox=(16, 0, 32, 16),
            target_w=16, target_h=16,
            plan=_solid_plan(prompt="bad"),
        )
        plan = TiledPassPlan(tiles=(good, bad))
        executor = TileExecutor(feather=0)

        def selective_renderer(p, w, h):
            if p.prompt == "bad":
                raise RuntimeError("nope")
            return Image.new("RGB", (w, h), (50, 100, 150))

        result = executor.execute(plan, (32, 16), selective_renderer)
        assert result.rendered_tiles == 1
        assert len(result.errors) == 1
        arr = np.asarray(result.image)
        # Left half (good tile) painted
        assert arr[8, 8, 1] == pytest.approx(100, abs=2)


# ── End-to-end with compose() ────────────────────────────────────────────────

class TestEndToEndWithCompose:
    def test_tiled_compose_then_execute(self):
        # Build a real composition plan and run it through the executor.
        region = Region(
            mask=Image.new("L", (64, 64), 255),
            prompt="background",
            denoise=0.6,
        )
        scene = Scene(layers=(Layer(regions=(region,)),), width=64, height=64)
        plan = compose(scene, RenderMode.TILED_PASS, RendererSupport(),
                       tile_size=32, tile_overlap=0)
        assert plan.tiled is not None
        assert len(plan.tiled.tiles) == 4

        executor = TileExecutor(feather=4)
        result = executor.execute(plan.tiled, (64, 64), _solid_renderer((80, 80, 80)))
        assert result.rendered_tiles == 4
        arr = np.asarray(result.image)
        # All pixels approximately the solid colour
        assert np.allclose(arr.mean(axis=(0, 1)), 80, atol=5)
