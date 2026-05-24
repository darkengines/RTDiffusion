"""Unit tests for the StreamDiffusion quality auto-picker.

`auto_t_indices` maps a 0..1 quality slider to a sorted list of t-indices for
the LCM scheduler. The mapping must be monotonic in count and span the full
scheduler range so the first index is the noisiest (most creative) and the
last is the cleanest (closest to input).
"""

from __future__ import annotations

import pytest
from PIL import Image

from app.stream.helpers import auto_t_indices, resolve_t_indices
from app.stream.manager import _stream_seed
from app.stream.session import _apply_denoise_output_blend


def test_quality_zero_gives_two_steps() -> None:
    idx = auto_t_indices(0.0)
    assert len(idx) == 2
    assert idx[0] == 0


def test_quality_one_gives_eight_steps() -> None:
    idx = auto_t_indices(1.0)
    assert len(idx) == 8
    assert idx[0] == 0
    assert idx[-1] == 49


def test_quality_monotonic_in_count() -> None:
    counts = [len(auto_t_indices(q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert counts == sorted(counts)


@pytest.mark.parametrize("q", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_indices_sorted_and_spanning(q: float) -> None:
    idx = auto_t_indices(q)
    assert idx == sorted(idx)
    assert idx[0] == 0
    assert idx[-1] == 49


def test_clamps_out_of_range() -> None:
    assert auto_t_indices(-0.5) == auto_t_indices(0.0)
    assert auto_t_indices(1.5) == auto_t_indices(1.0)


def test_resolve_prefers_quality_over_raw() -> None:
    settings = {
        "stream_quality": 0.5,
        "stream_timestep_indices": [99, 99, 99],  # should be ignored
    }
    out = resolve_t_indices(settings)
    assert out == auto_t_indices(0.5)


def test_resolve_falls_back_to_raw_indices() -> None:
    settings = {"stream_timestep_indices": [5, 10, 20]}
    assert resolve_t_indices(settings) == [5, 10, 20]


def test_resolve_default_when_empty() -> None:
    assert resolve_t_indices({}) == [0, 16, 32, 45]


def test_stream_seed_increment_uses_rotation_serial() -> None:
    settings = {"seed": 100, "seed_mode": "increment", "seed_rotation_serial": 3}
    assert _stream_seed(settings) == 103


def test_stream_seed_random_changes_with_rotation_serial() -> None:
    base = {"seed": 100, "seed_mode": "random", "prompt": "same"}
    first = _stream_seed({**base, "seed_rotation_serial": 1})
    second = _stream_seed({**base, "seed_rotation_serial": 2})
    assert first != second


def test_denoise_output_blend_uses_denoise_map_as_strength() -> None:
    generated = Image.new("RGB", (4, 4), (20, 180, 130))
    original = Image.new("RGB", (4, 4), (128, 128, 128))
    denoise_map = Image.new("L", (4, 4), 64)

    blended = _apply_denoise_output_blend(generated, original, denoise_map, 4, 4)

    pixel = blended.getpixel((0, 0))
    assert pixel[0] == pytest.approx(101, abs=2)
    assert pixel[1] == pytest.approx(141, abs=2)
    assert pixel[2] == pytest.approx(128, abs=2)
