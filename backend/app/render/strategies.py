"""Three rendering strategies for the layered SDXL engine.

All strategies share the same call signature:
    result: Image = strategy.render(layers, canvas, settings, session, frame_index)

Strategies
──────────
  SinglePassStrategy  — one SceneComposer call + one session.step() (fastest,
                        lowest memory; all layers share the same denoise pass)

  PerLayerStrategy    — N independent diffusion passes, back-to-front; each
                        layer's bbox is fitted to the nearest SDXL resolution,
                        the crop is inpainted in isolation and stitched back.
                        Supports per-layer LoRA swaps and per-layer CFG/sampler
                        overrides.  Cache skips layers whose content hasn't
                        changed.

  TiledStrategy       — the canvas is covered by a grid of SDXL-resolution
                        tiles (with configurable overlap).  Only tiles that
                        intersect at least one dirty layer are re-rendered.
                        Tiles are blended with cosine feathering at boundaries.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from ..compose.composer import SceneComposer
from ..compose.layer import Layer
from ..inference.caps import LayerRole
from ..inference.session import (
    CondInput,
    FrameRequest,
    InferenceSession,
    PromptBundle,
    RegionalPrompt,
    SamplerSpec,
)
from .resolution import (
    SDXLCrop,
    cosine_tile_weight,
    extract_crop,
    fit_region_to_sdxl,
    optimal_sdxl_resolution,
    paste_crop,
)
from .types import (
    ConditioningProvider,
    PostProcessor,
    RenderLayer,
    SceneSettings,
    SDXLMicroCond,
    SourceProvider,
)

logger = logging.getLogger("rtdiffusion.render")


# ── Layer → compose.Layer adapter ────────────────────────────────────────────

def _to_compose_layer(rl: RenderLayer) -> Layer:
    """Adapt a RenderLayer to the compose.Layer expected by SceneComposer."""
    layer = Layer(
        layer_id=rl.layer_id,
        role=_denoise_mode_to_role(rl.denoise_mode),
        color=rl.color,
        cond_image=rl.hidden_source,
        denoise_strength=rl.denoise,
        mask_color=rl.mask_color,
        mask_cond=rl.mask_cond,
        mask_denoise=rl.mask_denoise,
        schedule=rl.schedule,
        weight=rl.weight,
        cn=rl.cn,
        prompt_fragment=rl.prompt or None,
        negative_fragment=rl.negative_prompt or None,
    )
    # Preserve the denoise mode for composer's _layer_denoise_mode()
    layer._legacy_mode = rl.denoise_mode  # type: ignore[attr-defined]
    return layer


def _denoise_mode_to_role(mode: str) -> LayerRole:
    if mode == "prompt_mix":
        return "cond_only"
    return "color"


def _make_prompt_bundle(settings: SceneSettings, layers: list[RenderLayer]) -> PromptBundle:
    positive_parts: list[str] = []
    negative_parts: list[str] = []
    regional: list[RegionalPrompt] = []
    for rl in layers:
        if rl.prompt.strip():
            positive_parts.append(rl.prompt.strip())
        if rl.negative_prompt.strip():
            negative_parts.append(rl.negative_prompt.strip())
        if rl.prompt.strip() and rl.color is not None:
            mask = rl.color.getchannel("A") if rl.color.mode == "RGBA" else None
            regional.append(RegionalPrompt(
                layer_id=rl.layer_id,
                prompt=rl.prompt.strip(),
                negative_prompt=rl.negative_prompt.strip(),
                mask=mask,
                weight=rl.weight,
                schedule=rl.schedule,
            ))
    return PromptBundle(
        positive=", ".join(positive_parts) if positive_parts else "",
        negative=", ".join(negative_parts) if negative_parts else "",
        regional=regional,
    )


def _sampler_spec(settings: SceneSettings, layer: RenderLayer | None = None) -> SamplerSpec:
    return SamplerSpec(
        steps=layer.steps if (layer and layer.steps is not None) else settings.steps,
        cfg=(layer.cfg.scale if layer else settings.cfg),
        sampler=layer.sampler if (layer and layer.sampler) else settings.sampler,
        scheduler=layer.scheduler if (layer and layer.scheduler) else settings.scheduler,
        seed=layer.seed if (layer and layer.seed is not None) else settings.seed,
        seed_mode=settings.seed_mode,
        seed_variation=settings.seed_variation,
    )


def _micro_cond_hints(micro: SDXLMicroCond) -> dict[str, Any]:
    return {
        "sdxl_orig_size": micro.orig_size,
        "sdxl_target_size": micro.target_size,
        "sdxl_crop_coords": micro.crop_coords,
    }


def _apply_source_providers(
    layers: list[RenderLayer],
    providers: list[SourceProvider],
    frame_index: int,
) -> list[RenderLayer]:
    if not providers:
        return layers
    updated: list[RenderLayer] = []
    for rl in layers:
        img = None
        for p in providers:
            img = p.get_image(rl.layer_id, frame_index)
            if img is not None:
                break
        if img is not None:
            from dataclasses import replace
            rl = replace(rl, color=img)
        updated.append(rl)
    return updated


def _apply_conditioning_providers(
    layers: list[RenderLayer],
    cond_inputs: list[CondInput],
    providers: list[ConditioningProvider],
    frame_index: int,
) -> list[CondInput]:
    extra: list[CondInput] = []
    for rl in layers:
        for p in providers:
            extra.extend(p.get_conditioning(rl, frame_index))
    return cond_inputs + extra


def _apply_post_processors(
    result: Image.Image,
    layers: list[RenderLayer],
    processors: list[PostProcessor],
) -> Image.Image:
    for p in processors:
        result = p.process(result, layers)
    return result


# ── SinglePassStrategy ────────────────────────────────────────────────────────

class SinglePassStrategy:
    """One composed SceneComposer call + one session.step() call.

    All layers are merged into a single `FrameRequest` and denoised together.
    This is the fastest strategy and the right default for real-time work.

    Providers and post-processors are optional plug-in points:
        source_providers        — override layer images (e.g., video frames)
        conditioning_providers  — inject extra CondInputs (e.g., runtime tagger)
        post_processors         — chained image post-processing after inference
    """

    def __init__(
        self,
        source_providers: list[SourceProvider] | None = None,
        conditioning_providers: list[ConditioningProvider] | None = None,
        post_processors: list[PostProcessor] | None = None,
    ) -> None:
        self._composer = SceneComposer()
        self._sources = source_providers or []
        self._cond_providers = conditioning_providers or []
        self._post = post_processors or []

    def render(
        self,
        layers: list[RenderLayer],
        canvas: Image.Image,
        settings: SceneSettings,
        session: InferenceSession,
        frame_index: int = 0,
        session_id: str = "",
    ) -> Image.Image:
        layers = _apply_source_providers(layers, self._sources, frame_index)

        compose_layers = [_to_compose_layer(rl) for rl in layers]
        base_prompt = _make_prompt_bundle(settings, layers)
        scene = self._composer.compose(
            compose_layers,
            canvas,
            base_prompt,
            base_strength=settings.cfg,  # scene-level fallback
        )

        cond = _apply_conditioning_providers(layers, scene.cond_inputs, self._cond_providers, frame_index)

        hints: dict[str, Any] = _micro_cond_hints(settings.micro_cond)
        if settings.loras:
            hints["loras"] = [(l.path, l.weight) for l in settings.loras]

        request = FrameRequest(
            color=scene.color,
            denoise_map=scene.denoise_map,
            width=settings.width,
            height=settings.height,
            prompt=scene.prompt,
            sampler=_sampler_spec(settings),
            cond=cond,
            session_id=session_id,
            backend_hints=hints,
        )

        result_frame = session.step(request)
        result = result_frame.image
        return _apply_post_processors(result, layers, self._post)


# ── PerLayerStrategy ──────────────────────────────────────────────────────────

def _layer_content_hash(rl: RenderLayer) -> str:
    """Cheap fingerprint of a layer's visual content for cache invalidation."""
    parts: list[bytes] = [rl.layer_id.encode()]
    for img in (rl.color, rl.hidden_source, rl.mask_denoise):
        if img is not None:
            thumb = img.convert("L").resize((16, 16), Image.Resampling.BILINEAR).tobytes()
            parts.append(thumb)
    parts.append(repr((rl.denoise, rl.denoise_mode, rl.prompt, rl.weight, rl.cfg.scale)).encode())
    return hashlib.blake2b(b"".join(parts), digest_size=8).hexdigest()


class PerLayerStrategy:
    """N independent diffusion passes, one per visible layer, back-to-front.

    Each layer's bbox is extracted from the canvas, fitted to the nearest SDXL
    resolution, inpainted, and stitched back.  Layers whose content hash hasn't
    changed since the last call are skipped (their cached render is re-used).

    Per-layer isolation enables:
      - Different LoRA adapters per layer (pass in `backend_hints["loras"]`)
      - Different CFG / steps / sampler per layer
      - Independent inpainting without cross-layer bleed in the conditioning

    Constraints
    ───────────
    - `session` must support LoRA hot-swapping in `backend_hints["loras"]` for
      per-layer LoRAs to take effect.  Backends that ignore the hint still work
      but all layers share the session's loaded LoRAs.
    - Per-layer CFG spatial maps are passed as `backend_hints["cfg_map"]`;
      backends that don't support them fall back to the scalar CFG.
    """

    def __init__(
        self,
        bbox_padding: int = 64,
        source_providers: list[SourceProvider] | None = None,
        conditioning_providers: list[ConditioningProvider] | None = None,
        post_processors: list[PostProcessor] | None = None,
    ) -> None:
        self._bbox_padding = bbox_padding
        self._sources = source_providers or []
        self._cond_providers = conditioning_providers or []
        self._post = post_processors or []
        self._cache: dict[str, tuple[str, Image.Image]] = {}  # layer_id → (hash, image)

    def render(
        self,
        layers: list[RenderLayer],
        canvas: Image.Image,
        settings: SceneSettings,
        session: InferenceSession,
        frame_index: int = 0,
        session_id: str = "",
    ) -> Image.Image:
        layers = _apply_source_providers(layers, self._sources, frame_index)
        current = canvas.convert("RGB")

        for rl in layers:
            if rl.denoise_mode == "prompt_mix" or rl.denoise <= 0:
                continue

            content_hash = _layer_content_hash(rl)
            cached = self._cache.get(rl.layer_id)
            if cached is not None and cached[0] == content_hash:
                # Re-use cached render — composite into running canvas
                cached_img = cached[1]
                crop = self._get_crop(current, rl, settings)
                if crop is not None:
                    current = self._composite_cached(current, cached_img, rl, crop)
                continue

            crop = self._get_crop(current, rl, settings)
            if crop is None:
                continue

            result_img = self._render_layer(current, rl, crop, settings, session, session_id)
            if result_img is None:
                continue

            self._cache[rl.layer_id] = (content_hash, result_img)
            current = paste_crop(current, result_img, crop)

        result = _apply_post_processors(current, layers, self._post)
        return result

    def _get_crop(
        self, canvas: Image.Image, rl: RenderLayer, settings: SceneSettings
    ) -> SDXLCrop | None:
        box = rl.bbox(settings.width, settings.height)
        if box is None:
            return None
        x, y, w, h = box
        if w < 8 or h < 8:
            return None
        return fit_region_to_sdxl(canvas, x, y, w, h, padding=self._bbox_padding)

    def _render_layer(
        self,
        canvas: Image.Image,
        rl: RenderLayer,
        crop: SDXLCrop,
        settings: SceneSettings,
        session: InferenceSession,
        session_id: str,
    ) -> Image.Image | None:
        color_crop = extract_crop(canvas, crop)

        # Denoise map: resize layer mask to crop resolution
        if rl.mask_denoise is not None:
            mask_src = rl.mask_denoise
        elif rl.color is not None and rl.color.mode == "RGBA":
            mask_src = rl.color.getchannel("A")
        else:
            mask_src = None

        if mask_src is not None:
            mask_crop = mask_src.convert("L").crop(
                (crop.src_x, crop.src_y, crop.src_x + crop.src_w, crop.src_y + crop.src_h)
            ).resize((crop.sdxl_w, crop.sdxl_h), Image.Resampling.NEAREST)
            denoise_val = max(0.0, min(0.999, rl.denoise))
            mask_arr = (np.asarray(mask_crop, dtype=np.float32) / 255.0 * denoise_val * 255.0).clip(0, 255).round().astype(np.uint8)
            denoise_map = Image.fromarray(mask_arr, "L")
        else:
            blank = np.full((crop.sdxl_h, crop.sdxl_w), int(rl.denoise * 254), dtype=np.uint8)
            denoise_map = Image.fromarray(blank, "L")

        # ControlNet: extract cond image from crop region
        cond_inputs: list[CondInput] = []
        for spec in rl.cn:
            src = rl.hidden_source or rl.color
            if src is None:
                continue
            cn_img = extract_crop(src.convert("RGB"), crop)
            cond_mask = None
            if rl.mask_cond is not None:
                cond_mask = rl.mask_cond.convert("L").crop(
                    (crop.src_x, crop.src_y, crop.src_x + crop.src_w, crop.src_y + crop.src_h)
                ).resize((crop.sdxl_w, crop.sdxl_h), Image.Resampling.NEAREST)
            cond_inputs.append(CondInput(spec=spec, image=cn_img, mask=cond_mask))

        for p in self._cond_providers:
            cond_inputs.extend(p.get_conditioning(rl, 0))

        micro = rl.micro_cond or SDXLMicroCond(
            orig_size=(crop.orig_w, crop.orig_h),
            target_size=(crop.sdxl_w, crop.sdxl_h),
            crop_coords=(crop.crop_x, crop.crop_y),
        )
        hints: dict[str, Any] = _micro_cond_hints(micro)
        if rl.cfg.spatial_map is not None:
            cfg_map_crop = extract_crop(rl.cfg.spatial_map.convert("L"), crop)
            hints["cfg_map"] = cfg_map_crop
        if rl.loras:
            hints["loras"] = [(l.path, l.weight) for l in rl.loras]

        prompt = PromptBundle(positive=rl.prompt, negative=rl.negative_prompt)

        request = FrameRequest(
            color=color_crop,
            denoise_map=denoise_map,
            width=crop.sdxl_w,
            height=crop.sdxl_h,
            prompt=prompt,
            sampler=_sampler_spec(settings, rl),
            cond=cond_inputs,
            session_id=session_id,
            backend_hints=hints,
        )

        try:
            result_frame = session.step(request)
            return result_frame.image
        except Exception:
            logger.exception("PerLayerStrategy: inference failed for layer %s", rl.layer_id)
            return None

    def _composite_cached(
        self,
        canvas: Image.Image,
        cached: Image.Image,
        rl: RenderLayer,
        crop: SDXLCrop,
    ) -> Image.Image:
        return paste_crop(canvas, cached, crop)

    def invalidate(self, layer_id: str | None = None) -> None:
        """Invalidate the render cache (all layers, or one by id)."""
        if layer_id is None:
            self._cache.clear()
        else:
            self._cache.pop(layer_id, None)


# ── TiledStrategy ─────────────────────────────────────────────────────────────

@dataclass
class _Tile:
    """One SDXL-resolution tile in the grid."""
    col: int
    row: int
    x: int          # top-left in canvas pixels
    y: int
    w: int          # tile width (may be < tile_w at right/bottom edge)
    h: int
    tile_w: int     # canonical SDXL tile resolution
    tile_h: int
    dirty: bool = True
    cached: Image.Image | None = None
    # layer ids whose bbox overlaps this tile
    overlapping_layers: set[str] = field(default_factory=set)


@dataclass
class TileGrid:
    """Dirty-bit tile cache covering a canvas.

    Built once per canvas size change; invalidate per-tile when overlapping
    layers change.
    """
    canvas_w: int
    canvas_h: int
    tile_w: int
    tile_h: int
    overlap: int
    tiles: list[_Tile] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
        self.tiles = []
        stride_x = self.tile_w - self.overlap
        stride_y = self.tile_h - self.overlap
        col = 0
        x = 0
        while x < self.canvas_w:
            row = 0
            y = 0
            while y < self.canvas_h:
                w = min(self.tile_w, self.canvas_w - x)
                h = min(self.tile_h, self.canvas_h - y)
                self.tiles.append(_Tile(
                    col=col, row=row, x=x, y=y, w=w, h=h,
                    tile_w=self.tile_w, tile_h=self.tile_h,
                    dirty=True,
                ))
                row += 1
                y += stride_y
            col += 1
            x += stride_x

    def mark_dirty_by_layers(self, changed_layer_ids: set[str]) -> None:
        """Mark tiles that overlap any of the given layer ids as dirty."""
        for tile in self.tiles:
            if tile.overlapping_layers & changed_layer_ids:
                tile.dirty = True

    def mark_all_dirty(self) -> None:
        for tile in self.tiles:
            tile.dirty = True

    def update_layer_coverage(self, layers: list[RenderLayer], width: int, height: int) -> None:
        """Recompute which layer bboxes overlap each tile."""
        for tile in self.tiles:
            tile.overlapping_layers.clear()
        for rl in layers:
            box = rl.bbox(width, height)
            if box is None:
                continue
            lx, ly, lw, lh = box
            for tile in self.tiles:
                if _rects_overlap(tile.x, tile.y, tile.w, tile.h, lx, ly, lw, lh):
                    tile.overlapping_layers.add(rl.layer_id)

    def dirty_tiles(self) -> list[_Tile]:
        return [t for t in self.tiles if t.dirty]

    def composite(self, canvas: Image.Image) -> Image.Image:
        """Blend all tiles (cached renders) back into a canvas via cosine feathering."""
        if not any(t.cached is not None for t in self.tiles):
            return canvas

        canvas_w, canvas_h = canvas.size
        accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for tile in self.tiles:
            img = tile.cached
            if img is None:
                continue
            img_resized = img.resize((tile.w, tile.h), Image.Resampling.LANCZOS).convert("RGB")
            weight_map = cosine_tile_weight(tile.w, tile.h)

            arr = np.asarray(img_resized, dtype=np.float32)
            w_arr = np.asarray(weight_map, dtype=np.float32) / 255.0

            x0, y0 = tile.x, tile.y
            x1, y1 = x0 + tile.w, y0 + tile.h

            accum[y0:y1, x0:x1] += arr * w_arr[:, :, np.newaxis]
            weight_sum[y0:y1, x0:x1] += w_arr

        # Normalize; fall back to canvas where weight is zero
        canvas_arr = np.asarray(canvas.convert("RGB"), dtype=np.float32)
        safe_w = np.where(weight_sum > 1e-6, weight_sum, 1.0)[:, :, np.newaxis]
        blended = np.where(
            weight_sum[:, :, np.newaxis] > 1e-6,
            accum / safe_w,
            canvas_arr,
        )
        return Image.fromarray(blended.clip(0, 255).round().astype(np.uint8), "RGB")


def _rects_overlap(ax: int, ay: int, aw: int, ah: int, bx: int, by: int, bw: int, bh: int) -> bool:
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


class TiledStrategy:
    """SDXL-tile grid with dirty-bit tracking and cosine feathering.

    The canvas is divided into a grid of tiles at the nearest SDXL resolution.
    Only tiles intersecting at least one dirty layer are re-rendered each call.
    Tile borders are blended with a cosine weight map to hide seams.

    Parameters
    ──────────
    tile_size_hint   — (width, height) hint; the nearest SDXL resolution is
                       selected automatically.
    overlap          — overlap in pixels between adjacent tiles for seamless
                       blending (default 128).
    source_providers — see SinglePassStrategy.
    """

    def __init__(
        self,
        tile_size_hint: tuple[int, int] = (1024, 1024),
        overlap: int = 128,
        source_providers: list[SourceProvider] | None = None,
        conditioning_providers: list[ConditioningProvider] | None = None,
        post_processors: list[PostProcessor] | None = None,
    ) -> None:
        tile_w, tile_h = optimal_sdxl_resolution(*tile_size_hint)
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._overlap = overlap
        self._sources = source_providers or []
        self._cond_providers = conditioning_providers or []
        self._post = post_processors or []
        self._grid: TileGrid | None = None
        self._prev_canvas_size: tuple[int, int] = (0, 0)
        self._composer = SceneComposer()

    def render(
        self,
        layers: list[RenderLayer],
        canvas: Image.Image,
        settings: SceneSettings,
        session: InferenceSession,
        frame_index: int = 0,
        session_id: str = "",
        changed_layer_ids: set[str] | None = None,
    ) -> Image.Image:
        layers = _apply_source_providers(layers, self._sources, frame_index)

        canvas_size = (settings.width, settings.height)
        canvas_r = canvas.resize(canvas_size, Image.Resampling.LANCZOS)

        # Rebuild grid if canvas size changed
        if self._grid is None or self._prev_canvas_size != canvas_size:
            self._grid = TileGrid(
                canvas_w=settings.width,
                canvas_h=settings.height,
                tile_w=self._tile_w,
                tile_h=self._tile_h,
                overlap=self._overlap,
            )
            self._prev_canvas_size = canvas_size

        grid = self._grid
        grid.update_layer_coverage(layers, settings.width, settings.height)

        if changed_layer_ids is not None:
            grid.mark_dirty_by_layers(changed_layer_ids)
        # On first render all tiles are already dirty

        dirty = grid.dirty_tiles()
        logger.debug("TiledStrategy: %d/%d tiles dirty", len(dirty), len(grid.tiles))

        for tile in dirty:
            rendered = self._render_tile(tile, layers, canvas_r, settings, session, session_id, frame_index)
            tile.cached = rendered
            tile.dirty = False

        result = grid.composite(canvas_r)
        return _apply_post_processors(result, layers, self._post)

    def _render_tile(
        self,
        tile: _Tile,
        layers: list[RenderLayer],
        canvas: Image.Image,
        settings: SceneSettings,
        session: InferenceSession,
        session_id: str,
        frame_index: int,
    ) -> Image.Image:
        # Crop canvas to tile, resize to SDXL resolution
        tile_crop_img = canvas.crop(
            (tile.x, tile.y, tile.x + tile.w, tile.y + tile.h)
        ).resize((tile.tile_w, tile.tile_h), Image.Resampling.LANCZOS).convert("RGB")

        # Compose layers clipped to tile bounds
        tile_settings = SceneSettings(
            width=tile.tile_w,
            height=tile.tile_h,
            steps=settings.steps,
            cfg=settings.cfg,
            sampler=settings.sampler,
            scheduler=settings.scheduler,
            seed=settings.seed,
            seed_mode=settings.seed_mode,
            loras=settings.loras,
            micro_cond=SDXLMicroCond(
                orig_size=(settings.width, settings.height),
                target_size=(tile.tile_w, tile.tile_h),
                crop_coords=(tile.x, tile.y),
            ),
        )

        compose_layers = [_to_compose_layer(self._clip_layer_to_tile(rl, tile, settings)) for rl in layers]
        base_prompt = _make_prompt_bundle(tile_settings, layers)
        scene = self._composer.compose(
            compose_layers,
            tile_crop_img,
            base_prompt,
            base_strength=settings.cfg,
        )

        cond = _apply_conditioning_providers(layers, scene.cond_inputs, self._cond_providers, frame_index)

        hints: dict[str, Any] = _micro_cond_hints(tile_settings.micro_cond)
        if settings.loras:
            hints["loras"] = [(l.path, l.weight) for l in settings.loras]

        request = FrameRequest(
            color=scene.color,
            denoise_map=scene.denoise_map,
            width=tile.tile_w,
            height=tile.tile_h,
            prompt=scene.prompt,
            sampler=_sampler_spec(tile_settings),
            cond=cond,
            session_id=session_id,
            backend_hints=hints,
        )

        try:
            result_frame = session.step(request)
            return result_frame.image.convert("RGB")
        except Exception:
            logger.exception("TiledStrategy: tile (%d,%d) inference failed", tile.col, tile.row)
            return tile_crop_img

    def _clip_layer_to_tile(self, rl: RenderLayer, tile: _Tile, settings: SceneSettings) -> RenderLayer:
        """Resize layer masks so they align with the tile crop (not full canvas)."""
        from dataclasses import replace

        def _crop_mask(img: Image.Image | None) -> Image.Image | None:
            if img is None:
                return None
            # Crop mask to tile region then resize to tile_w × tile_h
            src_w, src_h = settings.width, settings.height
            m = img.convert("L").resize((src_w, src_h), Image.Resampling.NEAREST)
            cropped = m.crop((tile.x, tile.y, tile.x + tile.w, tile.y + tile.h))
            return cropped.resize((tile.tile_w, tile.tile_h), Image.Resampling.NEAREST)

        def _crop_img(img: Image.Image | None) -> Image.Image | None:
            if img is None:
                return None
            src_w, src_h = settings.width, settings.height
            c = img.resize((src_w, src_h), Image.Resampling.LANCZOS)
            cropped = c.crop((tile.x, tile.y, tile.x + tile.w, tile.y + tile.h))
            return cropped.resize((tile.tile_w, tile.tile_h), Image.Resampling.LANCZOS)

        return replace(
            rl,
            color=_crop_img(rl.color),
            hidden_source=_crop_img(rl.hidden_source),
            mask_color=_crop_mask(rl.mask_color),
            mask_cond=_crop_mask(rl.mask_cond),
            mask_denoise=_crop_mask(rl.mask_denoise),
        )

    def invalidate_all(self) -> None:
        if self._grid is not None:
            self._grid.mark_all_dirty()
