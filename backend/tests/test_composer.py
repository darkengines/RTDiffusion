"""Parity tests for the new SceneComposer.

The composer must produce byte-equal denoise maps to the legacy
`resolve_conditioning_mask` function for any input that the legacy
adapter (`layer_conditions_to_layers`) can express. These tests pin that
guarantee so the Step 4–6 cutover is safe.

Run with:  pytest backend/tests/test_composer.py -v
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from backend.app.compose import SceneComposer, layer_conditions_to_layers
from backend.app.inference import PromptBundle
from backend.app.pipeline_graph import resolve_conditioning_mask
from backend.app.schemas import LayerCondition


# ── fixtures ──────────────────────────────────────────────────────────────


def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _solid_rgba(w: int, h: int, alpha: int = 255, fill=(128, 128, 128)) -> Image.Image:
    rgba = Image.new("RGBA", (w, h), (*fill, alpha))
    return rgba


def _rect_alpha_rgba(w: int, h: int, rect: tuple[int, int, int, int]) -> Image.Image:
    """RGBA image with alpha=255 inside `rect`, 0 outside."""
    rgba = Image.new("RGBA", (w, h), (200, 100, 50, 0))
    alpha = Image.new("L", (w, h), 0)
    x0, y0, x1, y1 = rect
    box_alpha = Image.new("L", (x1 - x0, y1 - y0), 255)
    alpha.paste(box_alpha, (x0, y0))
    rgba.putalpha(alpha)
    return rgba


def _make_condition(
    *,
    rgba: Image.Image,
    mode: str = "mask",
    denoise: float | None = None,
    weight: float = 1.0,
    schedule_start: float = 0.0,
    schedule_end: float = 1.0,
    prompt: str = "",
) -> LayerCondition:
    return LayerCondition(
        layer_id="L1",
        image=_png_data_url(rgba),
        mode=mode,
        weight=weight,
        denoise=denoise,
        schedule_start=schedule_start,
        schedule_end=schedule_end,
        prompt=prompt,
    )


# ── denoise channel parity ────────────────────────────────────────────────


WIDTH, HEIGHT = 128, 128


@pytest.mark.parametrize(
    "mode,denoise,weight",
    [
        ("mask", 0.7, 1.0),
        ("mask", 0.3, 0.6),
        ("override", 0.0, 1.0),
        ("override", 0.95, 1.0),
        ("add", 0.4, 1.0),
        ("multiply", 0.5, 1.0),
        ("replace", 0.8, 1.0),
    ],
)
def test_denoise_parity_single_layer(mode: str, denoise: float, weight: float) -> None:
    rgba = _rect_alpha_rgba(WIDTH, HEIGHT, (32, 32, 96, 96))
    conditions = [_make_condition(rgba=rgba, mode=mode, denoise=denoise, weight=weight)]

    base_mask = Image.new("L", (WIDTH, HEIGHT), 255)
    base_strength = 0.5

    legacy = resolve_conditioning_mask(base_mask, conditions, WIDTH, HEIGHT, base_strength)

    composer = SceneComposer()
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    scene = composer.compose(
        layer_conditions_to_layers(conditions),
        canvas,
        PromptBundle(positive=""),
        base_strength=base_strength,
        canvas_mask=base_mask,
    )

    legacy_arr = np.asarray(legacy.denoise_map, dtype=np.uint8)
    new_arr = np.asarray(scene.denoise_map, dtype=np.uint8)
    # max abs diff bound is tight; pure parity expected.
    max_diff = int(np.abs(legacy_arr.astype(int) - new_arr.astype(int)).max())
    assert max_diff == 0, f"denoise map drift: max_diff={max_diff} for {mode=} {denoise=} {weight=}"


def test_denoise_parity_painters_algorithm() -> None:
    """Two layers, later one overrides earlier in its covered region."""
    rgba_bottom = _rect_alpha_rgba(WIDTH, HEIGHT, (16, 16, 80, 80))
    rgba_top = _rect_alpha_rgba(WIDTH, HEIGHT, (48, 48, 112, 112))
    conditions = [
        _make_condition(rgba=rgba_bottom, mode="override", denoise=0.2),
        _make_condition(rgba=rgba_top, mode="override", denoise=0.9),
    ]

    base_mask = Image.new("L", (WIDTH, HEIGHT), 255)
    legacy = resolve_conditioning_mask(base_mask, conditions, WIDTH, HEIGHT, 0.4)
    composer = SceneComposer()
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    scene = composer.compose(
        layer_conditions_to_layers(conditions),
        canvas,
        PromptBundle(),
        base_strength=0.4,
        canvas_mask=base_mask,
    )

    legacy_arr = np.asarray(legacy.denoise_map, dtype=np.uint8)
    new_arr = np.asarray(scene.denoise_map, dtype=np.uint8)
    assert int(np.abs(legacy_arr.astype(int) - new_arr.astype(int)).max()) == 0


def test_denoise_skips_inactive_schedule() -> None:
    rgba = _rect_alpha_rgba(WIDTH, HEIGHT, (32, 32, 96, 96))
    conditions = [_make_condition(rgba=rgba, mode="mask", denoise=0.9, schedule_start=0.7, schedule_end=0.7)]
    base_mask = Image.new("L", (WIDTH, HEIGHT), 255)
    legacy = resolve_conditioning_mask(base_mask, conditions, WIDTH, HEIGHT, 0.3)
    composer = SceneComposer()
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    scene = composer.compose(
        layer_conditions_to_layers(conditions),
        canvas,
        PromptBundle(),
        base_strength=0.3,
        canvas_mask=base_mask,
    )
    assert np.array_equal(np.asarray(scene.denoise_map), np.asarray(legacy.denoise_map))


def test_prompt_mix_mode_ignored_by_denoise() -> None:
    rgba = _rect_alpha_rgba(WIDTH, HEIGHT, (32, 32, 96, 96))
    conditions = [_make_condition(rgba=rgba, mode="prompt_mix", prompt="dragon")]
    base_mask = Image.new("L", (WIDTH, HEIGHT), 255)
    legacy = resolve_conditioning_mask(base_mask, conditions, WIDTH, HEIGHT, 0.42)
    composer = SceneComposer()
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    scene = composer.compose(
        layer_conditions_to_layers(conditions),
        canvas,
        PromptBundle(),
        base_strength=0.42,
        canvas_mask=base_mask,
    )
    assert np.array_equal(np.asarray(scene.denoise_map), np.asarray(legacy.denoise_map))


# ── prompt channel ────────────────────────────────────────────────────────


def test_prompt_merges_fragments() -> None:
    rgba = _rect_alpha_rgba(WIDTH, HEIGHT, (32, 32, 96, 96))
    conditions = [
        _make_condition(rgba=rgba, prompt="red dragon"),
        _make_condition(rgba=rgba, prompt="snow"),
    ]
    composer = SceneComposer()
    scene = composer.compose(
        layer_conditions_to_layers(conditions),
        Image.new("RGB", (WIDTH, HEIGHT)),
        PromptBundle(positive="mountain"),
    )
    assert scene.prompt.positive == "mountain, red dragon, snow"


def test_prompt_no_fragments_returns_base() -> None:
    composer = SceneComposer()
    scene = composer.compose([], Image.new("RGB", (WIDTH, HEIGHT)), PromptBundle(positive="base"))
    assert scene.prompt.positive == "base"


# ── canonical mask polarity ───────────────────────────────────────────────


def test_canvas_mask_defaults_to_white_act_everywhere() -> None:
    """No canvas_mask → composer treats whole canvas as 'regenerate'."""
    composer = SceneComposer()
    scene = composer.compose([], Image.new("RGB", (WIDTH, HEIGHT)), PromptBundle(), base_strength=0.6)
    arr = np.asarray(scene.denoise_map, dtype=np.uint8)
    assert arr.min() == arr.max()
    assert abs(int(arr.mean()) - int(round(0.6 * 255))) <= 1
