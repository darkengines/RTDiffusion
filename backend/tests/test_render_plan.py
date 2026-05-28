"""Unit tests for app.render_plan -- the v2 layer wire format decoder.

Run:  pytest backend/tests/test_render_plan.py -v
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.render_plan import CFG_HI, ControlNet, Prompt, RenderPlan, plan_from_wire


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _data_url(img: Image.Image) -> str:
    return "data:image/png;base64," + _png_b64(img)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_empty_payload_uses_base_defaults() -> None:
    plan = plan_from_wire({"width": 16, "height": 8, "base_cfg": 4.0, "base_denoise": 0.6})
    assert isinstance(plan, RenderPlan)
    assert plan.rgba.shape == (8, 16, 4)
    assert plan.rgba.dtype == np.uint8
    assert np.all(plan.rgba == 0)
    assert plan.cfg_map.shape == (8, 16)
    np.testing.assert_allclose(plan.cfg_map, 4.0)
    assert plan.denoise_map.shape == (8, 16)
    np.testing.assert_allclose(plan.denoise_map, 0.6)
    assert plan.prompts == ()
    assert plan.controlnet == ()
    assert plan.width == 16 and plan.height == 8


def test_rgba_b64_round_trip_preserves_alpha() -> None:
    src = np.zeros((4, 4, 4), dtype=np.uint8)
    src[1, 1] = [200, 100, 50, 255]
    src[2, 2] = [10, 20, 30, 128]
    img = Image.fromarray(src, mode="RGBA")
    plan = plan_from_wire({"width": 4, "height": 4, "rgba_b64": _png_b64(img)})
    assert np.array_equal(plan.rgba, src)


def test_cfg_map_16bit_decodes_to_cfg_units() -> None:
    arr16 = np.array([[0, 32768], [49152, 65535]], dtype=np.uint16)
    img = Image.fromarray(arr16, mode="I;16")
    plan = plan_from_wire({"width": 2, "height": 2, "cfg_map_b64": _png_b64(img)})
    expected = arr16.astype(np.float32) / 65535.0 * CFG_HI
    np.testing.assert_allclose(plan.cfg_map, expected, atol=1e-3)


def test_denoise_map_8bit_decodes_to_unit_interval() -> None:
    arr = np.array([[0, 127], [128, 255]], dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    plan = plan_from_wire({"width": 2, "height": 2, "denoise_map_b64": _png_b64(img)})
    expected = arr.astype(np.float32) / 255.0
    np.testing.assert_allclose(plan.denoise_map, expected, atol=1e-4)


def test_cfg_map_8bit_decodes_via_scale() -> None:
    arr = np.array([[0, 255]], dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    plan = plan_from_wire({"width": 2, "height": 1, "cfg_map_b64": _png_b64(img)})
    np.testing.assert_allclose(plan.cfg_map, np.array([[0.0, CFG_HI]], dtype=np.float32), atol=1e-3)


def test_prompts_decoded_with_optional_masks() -> None:
    mask_arr = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    mask_img = Image.fromarray(mask_arr, mode="L")
    payload = {
        "width": 2, "height": 2,
        "prompts": [
            {"text": "a cat", "negative": "blurry", "mask_b64": _png_b64(mask_img)},
            {"text": "a dog"},
            {"text": ""},  # dropped
            {"text": "  "},  # not dropped: whitespace counts as text
        ],
    }
    plan = plan_from_wire(payload)
    assert len(plan.prompts) == 3
    p0, p1, p2 = plan.prompts
    assert p0.text == "a cat" and p0.negative == "blurry"
    assert p0.mask is not None
    np.testing.assert_allclose(p0.mask, mask_arr.astype(np.float32) / 255.0, atol=1e-4)
    assert p1.text == "a dog" and p1.negative == "" and p1.mask is None
    assert p2.text == "  " and p2.mask is None


def test_controlnet_decoded() -> None:
    payload = {
        "width": 4, "height": 4,
        "controlnet": [
            {"layer_id": "L1", "model": "canny", "scale": 0.8, "start": 0.1, "end": 0.7,
             "preprocessor_params": {"canny_low": 50, "canny_high": 180}},
            {"layer_id": "L2", "model": "depth"},  # all defaults
        ],
    }
    plan = plan_from_wire(payload)
    assert len(plan.controlnet) == 2
    cn0, cn1 = plan.controlnet
    assert cn0.layer_id == "L1" and cn0.model == "canny"
    assert cn0.scale == pytest.approx(0.8)
    assert cn0.start == pytest.approx(0.1)
    assert cn0.end == pytest.approx(0.7)
    assert cn0.preprocessor_params == {"canny_low": 50, "canny_high": 180}
    assert cn1.layer_id == "L2" and cn1.model == "depth"
    assert cn1.scale == 1.0 and cn1.start == 0.0 and cn1.end == 1.0


def test_blob_resolver_used_when_ref_present() -> None:
    img = Image.fromarray(np.full((2, 2), 200, dtype=np.uint8), mode="L")
    raw = _png_bytes(img)
    seen: list[str] = []

    def resolver(key: str) -> bytes | None:
        seen.append(key)
        return raw if key == "blob123" else None

    payload = {"width": 2, "height": 2, "denoise_map_ref": "blob123"}
    plan = plan_from_wire(payload, blob_resolver=resolver)
    assert seen == ["blob123"]
    np.testing.assert_allclose(plan.denoise_map, np.full((2, 2), 200 / 255.0), atol=1e-4)


def test_inline_b64_wins_over_ref() -> None:
    inline = Image.fromarray(np.full((2, 2), 100, dtype=np.uint8), mode="L")
    payload = {
        "width": 2, "height": 2,
        "denoise_map_b64": _png_b64(inline),
        "denoise_map_ref": "should-not-be-resolved",
    }
    called: list[str] = []

    def resolver(key: str) -> bytes | None:  # pragma: no cover -- must not run
        called.append(key)
        return b""

    plan = plan_from_wire(payload, blob_resolver=resolver)
    assert called == []
    np.testing.assert_allclose(plan.denoise_map, np.full((2, 2), 100 / 255.0), atol=1e-4)


def test_data_url_prefix_is_stripped() -> None:
    src = np.full((2, 2, 4), 0, dtype=np.uint8)
    src[..., 3] = 255
    img = Image.fromarray(src, mode="RGBA")
    plan = plan_from_wire({"width": 2, "height": 2, "rgba_b64": _data_url(img)})
    assert int(plan.rgba[0, 0, 3]) == 255


def test_mismatched_size_resized_bilinear() -> None:
    img = Image.fromarray(np.full((8, 8, 4), 200, dtype=np.uint8), mode="RGBA")
    plan = plan_from_wire({"width": 4, "height": 4, "rgba_b64": _png_b64(img)})
    assert plan.rgba.shape == (4, 4, 4)
    assert int(plan.rgba[2, 2, 0]) == 200


def test_blob_resolver_miss_falls_back_to_default() -> None:
    def resolver(_: str) -> bytes | None:
        return None

    plan = plan_from_wire(
        {"width": 4, "height": 4, "base_denoise": 0.42, "denoise_map_ref": "missing"},
        blob_resolver=resolver,
    )
    np.testing.assert_allclose(plan.denoise_map, 0.42)


def test_base_prompts_carried_through() -> None:
    plan = plan_from_wire({
        "width": 2, "height": 2,
        "base_prompt": "global scene",
        "base_negative_prompt": "no text",
    })
    assert plan.base_prompt == "global scene"
    assert plan.base_negative_prompt == "no text"
