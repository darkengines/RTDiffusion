"""Canonical decode/encode helpers for transport boundaries.

Single rule everywhere: **white = act**. White pixels mean "regenerate this
pixel" / "apply this conditioning" / "paint this color here". Black means
"leave alone". Every transport layer is expected to normalize through these
helpers so downstream code never has to think about polarity.

The legacy `image_io.mask_to_luma` and `image_io.alpha_to_empty_mask` already
produce white-act output; both are kept as thin wrappers that delegate here
and will be removed in a follow-up cleanup.
"""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image


def decode_mask(
    source: str | Image.Image | None,
    size: tuple[int, int],
    *,
    polarity: str = "white-act",
    threshold: int = 12,
) -> Image.Image:
    """Decode any mask payload into a canonical L-mode image of the given size.

    Inputs:
        source: a data URL, a raw PIL Image, or None.
        size:   target (width, height); always resized via NEAREST.
        polarity: "white-act" (default; no change), "black-act" (input is
                  inverted polarity, will be flipped to white-act).
        threshold: values strictly below this are clamped to 0 (noise floor).

    None / empty payload → all-white mask (full canvas is "act"). This is the
    canonical "no mask provided" semantic and replaces every ad-hoc fallback.
    """
    width, height = size

    if source is None or (isinstance(source, str) and not source):
        return Image.new("L", (width, height), 255)

    if isinstance(source, Image.Image):
        img = source
    else:
        _, _, payload = source.partition(",")
        raw = base64.b64decode(payload or source)
        img = Image.open(BytesIO(raw))

    luma = img.convert("L").resize((width, height), Image.Resampling.NEAREST)
    if polarity == "black-act":
        luma = luma.point(lambda v: 255 - v)
    if threshold > 0:
        luma = luma.point(lambda v: v if v > threshold else 0)
    return luma


def decode_empty_alpha_as_mask(
    source: str | Image.Image,
    size: tuple[int, int],
    *,
    transparent_threshold: int = 250,
) -> Image.Image:
    """Build a white-act mask from an RGBA source's *transparent* pixels.

    Where the RGBA input has alpha < `transparent_threshold` (i.e., the user
    hasn't painted that area), the mask is white (regenerate). Where the
    source is opaque, the mask is black (preserve). This is the canonical
    "fill empty canvas with new content" mask, replacing
    `image_io.alpha_to_empty_mask`.
    """
    width, height = size
    if isinstance(source, Image.Image):
        rgba = source.convert("RGBA")
    else:
        _, _, payload = source.partition(",")
        raw = base64.b64decode(payload or source)
        rgba = Image.open(BytesIO(raw)).convert("RGBA")
    if rgba.size != (width, height):
        rgba = rgba.resize((width, height), Image.Resampling.LANCZOS)
    alpha = rgba.getchannel("A")
    return alpha.point(lambda v: 255 if v < transparent_threshold else 0)


def encode_jpeg_data_url(image: Image.Image, *, quality: int = 90) -> str:
    """JPEG data URL encoder shared by all transports."""
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def encode_png_data_url(image: Image.Image) -> str:
    """PNG data URL encoder for paths that need lossless output."""
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
