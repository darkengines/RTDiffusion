"""Parity + invariant tests for the canonical mask decoders."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from backend.app.compose.io import (
    decode_empty_alpha_as_mask,
    decode_mask,
    encode_jpeg_data_url,
    encode_png_data_url,
)
from backend.app.image_io import alpha_to_empty_mask, mask_to_luma


SIZE = (32, 32)


def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_decode_mask_none_is_all_white() -> None:
    out = decode_mask(None, SIZE)
    arr = np.asarray(out)
    assert out.mode == "L"
    assert out.size == SIZE
    assert int(arr.min()) == 255


def test_decode_mask_empty_string_is_all_white() -> None:
    out = decode_mask("", SIZE)
    arr = np.asarray(out)
    assert int(arr.min()) == 255


def test_decode_mask_threshold_clamps_noise() -> None:
    src = Image.new("L", SIZE, 6)
    out = decode_mask(src, SIZE)
    assert int(np.asarray(out).max()) == 0  # 6 < default threshold 12 → clamped


def test_decode_mask_polarity_flip() -> None:
    src = Image.new("L", SIZE, 200)
    flipped = decode_mask(src, SIZE, polarity="black-act")
    assert int(np.asarray(flipped).max()) == 55


def test_decode_mask_legacy_parity() -> None:
    src = Image.new("L", SIZE)
    px = src.load()
    for y in range(SIZE[1]):
        for x in range(SIZE[0]):
            px[x, y] = (x * 8) % 256
    legacy = mask_to_luma(src)
    new = decode_mask(src, SIZE)
    assert np.array_equal(np.asarray(legacy), np.asarray(new))


def test_decode_empty_alpha_legacy_parity() -> None:
    rgba = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    px = rgba.load()
    for y in range(SIZE[1]):
        for x in range(SIZE[0]):
            px[x, y] = (10, 20, 30, x * 8 % 256)
    legacy = alpha_to_empty_mask(rgba)
    new = decode_empty_alpha_as_mask(rgba, SIZE)
    assert np.array_equal(np.asarray(legacy), np.asarray(new))


def test_decode_mask_resizes() -> None:
    src = Image.new("L", (8, 8), 200)
    out = decode_mask(src, (64, 64))
    assert out.size == (64, 64)
    assert int(np.asarray(out).min()) == 200


def test_encode_jpeg_round_trip() -> None:
    img = Image.new("RGB", (16, 16), (40, 80, 160))
    url = encode_jpeg_data_url(img, quality=90)
    assert url.startswith("data:image/jpeg;base64,")


def test_encode_png_round_trip() -> None:
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    url = encode_png_data_url(img)
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    decoded = Image.open(io.BytesIO(raw)).convert("RGB")
    assert np.array_equal(np.asarray(img), np.asarray(decoded))
