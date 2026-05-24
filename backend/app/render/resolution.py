"""SDXL optimal resolution fitting.

All resolutions are ~1 MP (1024×1024 = 1 048 576 px).  The sampler is
trained on these aspect ratios; using an arbitrary size degrades quality.

Usage:
    res_w, res_h = optimal_sdxl_resolution(layer_bbox_w, layer_bbox_h)
    crop = fit_region_to_sdxl(canvas, bbox_x, bbox_y, bbox_w, bbox_h)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

# Training-set aspect ratios for SDXL (all ≈ 1 048 576 px)
SDXL_RESOLUTIONS: list[tuple[int, int]] = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
]


def optimal_sdxl_resolution(w: int, h: int) -> tuple[int, int]:
    """Return the SDXL resolution whose aspect ratio is closest to w:h."""
    if h == 0:
        return (1024, 1024)
    aspect = w / h
    return min(SDXL_RESOLUTIONS, key=lambda r: abs(r[0] / r[1] - aspect))


@dataclass
class SDXLCrop:
    """Describes how a canvas region is mapped to an SDXL-resolution tile.

    Workflow:
        1. `src_rect`  — (x, y, w, h) in canvas pixels (original bbox, possibly padded)
        2. Crop + resize canvas src_rect → (sdxl_w × sdxl_h) for the diffusion pass.
        3. After inference, resize result → (src_w × src_h).
        4. Composite result into canvas at (src_x, src_y).

    `micro_cond` records the SDXL aesthetic micro-conditioning tensors
    (orig_size, target_size, crop_coords) to pass in `backend_hints`.
    """

    src_x: int
    src_y: int
    src_w: int
    src_h: int
    sdxl_w: int
    sdxl_h: int
    # SDXL micro-cond values derived from this crop
    orig_w: int
    orig_h: int
    crop_x: int
    crop_y: int


def fit_region_to_sdxl(
    canvas: Image.Image,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
    padding: int = 0,
) -> SDXLCrop:
    """Compute the SDXL crop descriptor for a layer bbox on a canvas.

    The bbox is optionally padded outward (useful for giving diffusion context
    around the edited region).  The padded rect is clamped to canvas bounds,
    then the closest SDXL resolution is selected and a `SDXLCrop` is returned.

    The actual image extraction + resize is left to the caller so this function
    stays pure (no PIL I/O).
    """
    canvas_w, canvas_h = canvas.size

    # Expand bbox by padding, clamp to canvas
    x0 = max(0, bbox_x - padding)
    y0 = max(0, bbox_y - padding)
    x1 = min(canvas_w, bbox_x + bbox_w + padding)
    y1 = min(canvas_h, bbox_y + bbox_h + padding)
    src_w = max(1, x1 - x0)
    src_h = max(1, y1 - y0)

    sdxl_w, sdxl_h = optimal_sdxl_resolution(src_w, src_h)

    return SDXLCrop(
        src_x=x0, src_y=y0,
        src_w=src_w, src_h=src_h,
        sdxl_w=sdxl_w, sdxl_h=sdxl_h,
        orig_w=canvas_w, orig_h=canvas_h,
        crop_x=x0, crop_y=y0,
    )


def extract_crop(canvas: Image.Image, crop: SDXLCrop) -> Image.Image:
    """Crop canvas to `src_rect` and resize to SDXL resolution."""
    region = canvas.crop((crop.src_x, crop.src_y, crop.src_x + crop.src_w, crop.src_y + crop.src_h))
    return region.resize((crop.sdxl_w, crop.sdxl_h), Image.Resampling.LANCZOS)


def paste_crop(canvas: Image.Image, result: Image.Image, crop: SDXLCrop) -> Image.Image:
    """Resize diffusion result back to src_rect dimensions and paste into canvas."""
    canvas = canvas.copy()
    resized = result.resize((crop.src_w, crop.src_h), Image.Resampling.LANCZOS)
    canvas.paste(resized.convert("RGB"), (crop.src_x, crop.src_y))
    return canvas


def cosine_tile_weight(tile_w: int, tile_h: int) -> Image.Image:
    """Generate a cosine weight map for feathered tile blending.

    Center = 1.0, edges = 0.0.  Values are cos²(π/2 · d) where d is the
    normalized distance from center (0=center, 1=edge).
    Returns an L-mode image with values 0..255.
    """
    import numpy as np

    xs = np.linspace(-1.0, 1.0, tile_w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, tile_h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    # Cosine weight along each axis, product = 2-D tent
    wx = np.cos(xg * math.pi / 2.0) ** 2
    wy = np.cos(yg * math.pi / 2.0) ** 2
    w = (wx * wy * 255.0).clip(0, 255).round().astype(np.uint8)
    return Image.fromarray(w, "L")
