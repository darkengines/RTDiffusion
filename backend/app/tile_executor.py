"""Tile executor for ``TiledPassPlan``.

Executes a tiled composition plan against any renderer callback and stitches
the tile outputs back into a single canvas image. The executor handles:

  • Resolution snapping per tile (each tile is rendered at its SDXL-optimal
    ``(target_w, target_h)`` then resized back to the bbox size).
  • Linear feather blending on overlap regions so seams disappear.
  • Dirty-tile tracking: skip tiles whose plan is identical to the previous
    frame (caller passes a stable hash via ``TileExecutor.execute(dirty=...)``).
  • Graceful failure: a tile that fails to render keeps the previous frame's
    pixels in that area, with the error reported in ``TileExecResult.errors``.

The renderer is a pure callback ``(plan, target_w, target_h) -> PIL.Image``.
No GPU code lives in this module — it's pure orchestration so it can be unit
tested without any model dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

from .composition import SinglePassPlan, TilePassSpec, TiledPassPlan

logger = logging.getLogger("rtdiffusion.tile")


TileRenderer = Callable[[SinglePassPlan, int, int], Image.Image]
"""Renderer callback. Receives the per-tile plan + the snapped target size
(SDXL-optimal). Returns an RGB PIL image at ``(target_w, target_h)``.
The executor then resizes it back to the actual tile bbox."""


@dataclass
class TileExecResult:
    """Result of executing a tiled plan."""

    image: Image.Image                                 # composed RGB canvas
    rendered_tiles: int = 0                            # tiles actually inferred
    skipped_tiles: int = 0                             # tiles skipped via dirty mask
    errors: tuple[tuple[tuple[int, int, int, int], str], ...] = ()
    """Per-tile (bbox, message) for tiles whose renderer raised."""


# ── Weight mask for overlap blending ──────────────────────────────────────────

def _feather_weights(tile_w: int, tile_h: int, feather: int) -> np.ndarray:
    """A ``(H, W)`` float array in [0,1], 1 in the centre, fading to 0 at edges.

    ``feather`` controls the gradient width (pixels). When ``feather == 0`` the
    mask is uniformly 1 and tiles do hard-edge replace.
    """
    if feather <= 0:
        return np.ones((tile_h, tile_w), dtype=np.float32)
    feather = min(feather, tile_w // 2, tile_h // 2)
    x = np.minimum(
        np.minimum(np.arange(tile_w), tile_w - 1 - np.arange(tile_w)),
        feather,
    ) / max(feather, 1)
    y = np.minimum(
        np.minimum(np.arange(tile_h), tile_h - 1 - np.arange(tile_h)),
        feather,
    ) / max(feather, 1)
    return np.clip(np.outer(y, x), 0.0, 1.0).astype(np.float32)


# ── Tile executor ─────────────────────────────────────────────────────────────

class TileExecutor:
    """Stateful executor that supports dirty-tile caching between frames."""

    def __init__(self, feather: int = 16) -> None:
        self.feather = max(0, int(feather))
        # Cache the last canvas so dirty-tile skipping has somewhere to read
        # the previous pixels from.
        self._last_canvas: Image.Image | None = None
        # Cache per-tile plan signatures so the caller can skip re-render
        # automatically without having to provide an explicit ``dirty`` set.
        self._last_tile_sig: dict[tuple[int, int, int, int], str] = {}

    def reset(self) -> None:
        self._last_canvas = None
        self._last_tile_sig = {}

    def execute(
        self,
        plan: TiledPassPlan,
        canvas_size: tuple[int, int],
        renderer: TileRenderer,
        *,
        dirty: set[tuple[int, int, int, int]] | None = None,
        base_image: Image.Image | None = None,
    ) -> TileExecResult:
        """Run the tiled plan.

        ``canvas_size`` is ``(width, height)`` of the final canvas (each tile's
        bbox is expressed in those coords).

        ``dirty`` optionally restricts the set of tile bboxes that get re-rendered;
        omitted tiles reuse the previous frame's pixels (or ``base_image``).

        ``base_image`` is the fallback canvas used to seed pixels that no tile
        covers. Defaults to a black canvas when ``self._last_canvas`` is None.
        """
        w, h = canvas_size
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid canvas size {(w, h)}")

        # Seed the working canvas from the previous frame (cache), the supplied
        # base, or black if both are missing.
        if self._last_canvas is not None and self._last_canvas.size == (w, h):
            canvas_arr = np.asarray(self._last_canvas.convert("RGB"), dtype=np.float32).copy()
        elif base_image is not None:
            base = base_image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
            canvas_arr = np.asarray(base, dtype=np.float32).copy()
        else:
            canvas_arr = np.zeros((h, w, 3), dtype=np.float32)

        weight_arr = np.zeros((h, w), dtype=np.float32)
        rendered = 0
        skipped = 0
        errors: list[tuple[tuple[int, int, int, int], str]] = []

        new_sigs: dict[tuple[int, int, int, int], str] = {}

        for tile in plan.tiles:
            sig = _tile_signature(tile)
            new_sigs[tile.bbox] = sig

            tile_is_dirty = True
            if dirty is not None:
                tile_is_dirty = tile.bbox in dirty
            elif self._last_tile_sig.get(tile.bbox) == sig:
                tile_is_dirty = False

            x0, y0, x1, y1 = tile.bbox
            tile_w = x1 - x0
            tile_h = y1 - y0
            if tile_w <= 0 or tile_h <= 0:
                continue

            if not tile_is_dirty:
                # Tile unchanged: skip the inference call entirely. The cached
                # canvas already has the previous frame's pixels; just record a
                # full-weight contribution so the fallback fill below doesn't
                # zero this region.
                weight_arr[y0:y1, x0:x1] = np.maximum(
                    weight_arr[y0:y1, x0:x1],
                    _feather_weights(tile_w, tile_h, self.feather),
                )
                skipped += 1
                continue

            try:
                rendered_img = renderer(tile.plan, tile.target_w, tile.target_h)
            except Exception as exc:  # noqa: BLE001 — orchestrator must keep going
                logger.warning("Tile renderer failed at %s: %s", tile.bbox, exc)
                errors.append((tile.bbox, f"{type(exc).__name__}: {exc}"))
                continue
            rendered += 1

            if rendered_img.size != (tile_w, tile_h):
                rendered_img = rendered_img.convert("RGB").resize(
                    (tile_w, tile_h), Image.Resampling.LANCZOS,
                )
            tile_arr = np.asarray(rendered_img.convert("RGB"), dtype=np.float32)
            tile_weight = _feather_weights(tile_w, tile_h, self.feather)

            # Feather only modulates blending with *overlapping* neighbours.
            # Pixels with no prior contribution take the tile pixel at full
            # strength so corners of feathered tiles aren't black when no
            # neighbour overlaps them.
            region = canvas_arr[y0:y1, x0:x1]
            region_w = weight_arr[y0:y1, x0:x1]
            first_contrib = region_w <= 0.0
            tile_weight_eff = np.where(first_contrib, 1.0, tile_weight)
            new_w = region_w + tile_weight_eff
            safe_w = np.where(new_w > 0, new_w, 1.0)[..., None]
            blended = (
                region * region_w[..., None]
                + tile_arr * tile_weight_eff[..., None]
            ) / safe_w
            canvas_arr[y0:y1, x0:x1] = blended
            weight_arr[y0:y1, x0:x1] = new_w

        result_img = Image.fromarray(np.clip(canvas_arr, 0.0, 255.0).astype(np.uint8), "RGB")
        self._last_canvas = result_img
        self._last_tile_sig = new_sigs

        return TileExecResult(
            image=result_img,
            rendered_tiles=rendered,
            skipped_tiles=skipped,
            errors=tuple(errors),
        )


# ── Tile signature for dirty-tile auto-detection ──────────────────────────────

def _tile_signature(tile: TilePassSpec) -> str:
    """Compute a stable signature for a tile's plan.

    Two tiles with identical input (same prompt, same masks, same params)
    produce the same signature → executor can skip the renderer call.
    """
    p = tile.plan
    parts = [
        p.prompt,
        p.negative_prompt,
        f"{p.cfg_scalar:.4f}",
        _img_signature(p.mask),
        _img_signature(p.denoise_map),
        _img_signature(p.cfg_map),
    ]
    for prompt_text, mask_img in p.regional_prompts:
        parts.append(prompt_text)
        parts.append(_img_signature(mask_img))
    return "|".join(parts)


def _img_signature(img: Image.Image) -> str:
    """Cheap fingerprint of a PIL image. 16-pixel digest is enough to detect
    visual changes while staying O(1) for tile-level dirty checking."""
    small = img.convert("L").resize((4, 4), Image.Resampling.NEAREST)
    return small.tobytes().hex()


__all__ = ["TileExecutor", "TileExecResult", "TileRenderer"]
