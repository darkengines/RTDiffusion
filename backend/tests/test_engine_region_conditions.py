from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from app.engine import DiffusionEngine, EngineConfig
from app.engine.regional_attention import RegionalAttentionRegion, RegionalAttentionSpec, regional_attention_context
from app.schemas import InpaintFrame, LayerCondition


def _png_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _condition(layer_id: str, z_index: int, prompt: str, alpha: Image.Image) -> LayerCondition:
    rgba = Image.new("RGBA", alpha.size, (128, 128, 128, 0))
    rgba.putalpha(alpha)
    return LayerCondition(
        layer_id=layer_id,
        z_index=z_index,
        region_id=layer_id,
        name=prompt,
        prompt=prompt,
        image=_png_url(rgba),
        mode="mask",
        denoise=0.999,
        weight=1,
        schedule="linear",
        schedule_start=0,
        schedule_end=1,
    )


def _frame(conditions: list[LayerCondition]) -> InpaintFrame:
    size = (256, 256)
    return InpaintFrame(
        prompt="cat, car",
        image=_png_url(Image.new("RGBA", size, (128, 128, 128, 255))),
        mask=_png_url(Image.new("L", size, 255)),
        width=size[0],
        height=size[1],
        strength=0.999,
        steps=1,
        cfg=1,
        layer_conditions=conditions,
    )


def _frame_with_prompt(prompt: str, conditions: list[LayerCondition]) -> InpaintFrame:
    return _frame(conditions).model_copy(update={"prompt": prompt})


def test_regional_mask_passes_are_z_ordered_and_prompt_isolated(monkeypatch) -> None:
    engine = DiffusionEngine(EngineConfig(device="cpu"))
    full = Image.new("L", (256, 256), 255)
    bottom = Image.new("L", (256, 256), 0)
    bottom.paste(255, (0, 128, 256, 256))
    calls: list[str] = []

    def fake_inpaint(image: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image:
        calls.append(frame.prompt)
        color = (20, 180, 130) if frame.prompt == "cat" else (190, 40, 80)
        return Image.new("RGB", image.size, color)

    monkeypatch.setattr(engine, "_diffusers_inpaint", fake_inpaint)
    engine.pipe = object()
    frame = _frame([
        _condition("top", 1, "car", bottom),
        _condition("bottom", 0, "cat", full),
    ])

    output = engine._apply_layer_region_conditions(Image.new("RGB", (256, 256), (128, 128, 128)), frame)

    assert calls == ["cat", "car"]
    assert output.getpixel((64, 64)) == (20, 180, 130)
    assert output.getpixel((64, 192)) == (190, 40, 80)


def test_mask_layer_runs_when_prompt_matches_scene_prompt(monkeypatch) -> None:
    engine = DiffusionEngine(EngineConfig(device="cpu"))
    full = Image.new("L", (256, 256), 255)
    calls: list[str] = []

    def fake_inpaint(image: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image:
        calls.append(frame.prompt)
        return Image.new("RGB", image.size, (20, 180, 130))

    monkeypatch.setattr(engine, "_diffusers_inpaint", fake_inpaint)
    engine.pipe = object()
    frame = _frame_with_prompt("cat", [_condition("bottom", 0, "cat", full)])

    output = engine._apply_layer_region_conditions(Image.new("RGB", (256, 256), (128, 128, 128)), frame)

    assert calls == ["cat"]
    assert output.getpixel((64, 64)) == (20, 180, 130)


def test_cfg_mask_luma_scales_regional_pass_cfg(monkeypatch) -> None:
    engine = DiffusionEngine(EngineConfig(device="cpu"))
    full = Image.new("L", (256, 256), 255)
    low_cfg_mask = Image.new("L", (256, 256), 25)
    cfg_values: list[float] = []

    def fake_inpaint(image: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image:
        cfg_values.append(frame.cfg)
        return Image.new("RGB", image.size, (20, 180, 130))

    monkeypatch.setattr(engine, "_diffusers_inpaint", fake_inpaint)
    engine.pipe = object()
    condition = _condition("bottom", 0, "cat", full).model_copy(update={
        "cfg": 10.0,
        "cfg_mask": _png_url(low_cfg_mask),
    })
    frame = _frame([condition]).model_copy(update={"cfg": 10.0})

    engine._apply_layer_region_conditions(Image.new("RGB", (256, 256), (128, 128, 128)), frame)

    assert cfg_values == pytest.approx([30.0 * 25.0 / 255.0], abs=0.05)


def test_denoise_override_scales_regional_composite(monkeypatch) -> None:
    engine = DiffusionEngine(EngineConfig(device="cpu"))
    full = Image.new("L", (256, 256), 255)
    masks: list[Image.Image] = []

    def fake_inpaint(image: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image:
        masks.append(mask.copy())
        return Image.new("RGB", image.size, (20, 180, 130))

    monkeypatch.setattr(engine, "_diffusers_inpaint", fake_inpaint)
    engine.pipe = object()
    condition = _condition("bottom", 0, "cat", full).model_copy(update={"denoise": 0.25})
    frame = _frame([condition])

    output = engine._apply_layer_region_conditions(Image.new("RGB", (256, 256), (128, 128, 128)), frame)

    assert masks and masks[0].getpixel((64, 64)) >= 250
    pixel = output.getpixel((64, 64))
    assert pixel[0] == pytest.approx(101, abs=2)
    assert pixel[1] == pytest.approx(141, abs=2)
    assert pixel[2] == pytest.approx(128, abs=2)


def test_generate_preserves_relative_denoise_in_inpaint_mask(monkeypatch) -> None:
    engine = DiffusionEngine(EngineConfig(device="cpu"))
    left = Image.new("L", (256, 256), 0)
    left.paste(255, (0, 0, 128, 256))
    captured: list[tuple[Image.Image, InpaintFrame]] = []

    def fake_inpaint(image: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image:
        captured.append((mask.copy(), frame))
        return image.convert("RGB")

    monkeypatch.setattr(engine, "_ensure_runtime", lambda _frame: None)
    monkeypatch.setattr(engine, "_apply_sampler", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "_diffusers_inpaint", fake_inpaint)
    engine.pipe = object()
    condition = _condition("bottom", 0, "cat", left).model_copy(update={"denoise": 0.25})
    frame = _frame([condition]).model_copy(update={"strength": 0.999})

    engine.generate(frame)

    assert captured
    mask, inpaint_frame = captured[0]
    assert inpaint_frame.strength == pytest.approx(0.999)
    assert mask.getpixel((64, 64)) == pytest.approx(64, abs=1)
    assert mask.getpixel((192, 64)) == 255


def test_generate_debug_includes_visible_aggregated_denoise_mask(monkeypatch) -> None:
    engine = DiffusionEngine(EngineConfig(device="cpu"))
    left = Image.new("L", (256, 256), 0)
    left.paste(255, (0, 0, 128, 256))

    monkeypatch.setattr(engine, "_ensure_runtime", lambda _frame: None)
    monkeypatch.setattr(engine, "_apply_sampler", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "_diffusers_inpaint", lambda image, _mask, _frame: image.convert("RGB"))
    engine.pipe = object()
    condition = _condition("bottom", 0, "cat", left).model_copy(update={"denoise": 0.25})
    frame = _frame([condition]).model_copy(update={"strength": 0.999, "debug_streams": True})

    result = engine.generate(frame)

    assert result.debug_channels["input/mask/aggregated/denoise"].startswith("data:image/")
    assert result.debug_channels["condition_denoise"] == result.debug_channels["input/mask/aggregated/denoise"]


def test_single_render_strategy_runs_regional_prompts_after_global_pass(monkeypatch) -> None:
    engine = DiffusionEngine(EngineConfig(device="cpu"))
    full = Image.new("L", (256, 256), 255)
    calls: list[str] = []

    monkeypatch.setattr(engine, "_ensure_runtime", lambda _frame: None)
    monkeypatch.setattr(engine, "_apply_sampler", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "pipe", object())

    def fake_inpaint(image: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image:
        calls.append(frame.prompt)
        return Image.new("RGB", image.size, (20, 180, 130))

    monkeypatch.setattr(engine, "_diffusers_inpaint", fake_inpaint)
    frame = _frame([_condition("bottom", 0, "regional cat", full)]).model_copy(update={"render_strategy": "single"})

    engine.generate(frame)

    assert len(calls) == 2
    assert calls[0].startswith("cat, car")
    assert "regional cat" not in calls[0]
    assert calls[1] == "regional cat"


def test_prompt_mix_conditions_still_extend_global_prompt() -> None:
    engine = DiffusionEngine(EngineConfig(device="cpu"))
    full = Image.new("L", (256, 256), 255)
    condition = _condition("mix", 0, "regional cat", full).model_copy(update={"mode": "prompt_mix"})
    frame = _frame([condition])

    mixed = engine._frame_with_prompt_mix_conditions(frame)

    assert mixed.prompt.startswith("cat, car")
    assert "regional cat" in mixed.prompt


def test_mask_conditions_do_not_become_global_regional_attention_specs() -> None:
    import torch

    class FakePipe:
        _execution_device = "cpu"

        def encode_prompt(self, prompt: str, **_kwargs):
            value = 3.0 if prompt == "regional cat" else 1.0
            return torch.full((1, 4, 8), value), torch.zeros((1, 4, 8)), None, None

    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.config = EngineConfig(device="cpu")
    engine.pipe = FakePipe()
    full = Image.new("L", (256, 256), 255)
    prompt_mix = _condition("mix", 0, "global mood", full).model_copy(update={"mode": "prompt_mix"})
    frame = _frame([_condition("mask", 1, "regional cat", full), prompt_mix])

    spec = engine._regional_attention_spec(frame, "", guidance_scale=2.0)

    assert spec is None


def test_regional_attention_context_restores_processors() -> None:
    import torch

    class FakeUnet:
        def __init__(self):
            self.attn_processors = {
                "block.attn1.processor": object(),
                "block.attn2.processor": object(),
            }

        def set_attn_processor(self, processors):
            self.attn_processors = processors

    class FakePipe:
        def __init__(self):
            self.unet = FakeUnet()

    pipe = FakePipe()
    original = dict(pipe.unet.attn_processors)
    spec = RegionalAttentionSpec(
        regions=(RegionalAttentionRegion(torch.zeros((1, 4, 8)), None, Image.new("L", (256, 256), 255)),),
        width=256,
        height=256,
    )

    with regional_attention_context(pipe, spec):
        assert pipe.unet.attn_processors != original
        assert pipe.unet.attn_processors["block.attn1.processor"] is original["block.attn1.processor"]
        assert pipe.unet.attn_processors["block.attn2.processor"] is not original["block.attn2.processor"]

    assert pipe.unet.attn_processors == original


def test_regional_attention_context_keeps_compiled_unet_active() -> None:
    import torch

    class FakeUnet:
        def __init__(self):
            self.attn_processors = {"block.attn2.processor": object()}

        def set_attn_processor(self, processors):
            self.attn_processors = processors

    class FakeCompiledUnet:
        def __init__(self, original):
            self._orig_mod = original

    class FakePipe:
        def __init__(self):
            self.original = FakeUnet()
            self.unet = FakeCompiledUnet(self.original)

    pipe = FakePipe()
    compiled = pipe.unet
    original_processor = pipe.original.attn_processors["block.attn2.processor"]
    spec = RegionalAttentionSpec(
        regions=(RegionalAttentionRegion(torch.zeros((1, 4, 8)), None, Image.new("L", (256, 256), 255)),),
        width=256,
        height=256,
    )

    with regional_attention_context(pipe, spec):
        assert pipe.unet is compiled
        assert pipe.original.attn_processors["block.attn2.processor"] is not original_processor

    assert pipe.unet is compiled
    assert pipe.original.attn_processors["block.attn2.processor"] is original_processor


def test_per_layer_render_strategy_can_run_regional_diffusion(monkeypatch) -> None:
    engine = DiffusionEngine(EngineConfig(device="cpu"))
    full = Image.new("L", (256, 256), 255)
    calls: list[str] = []

    monkeypatch.setattr(engine, "_ensure_runtime", lambda _frame: None)
    monkeypatch.setattr(engine, "_apply_sampler", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "pipe", object())

    def fake_inpaint(image: Image.Image, mask: Image.Image, frame: InpaintFrame) -> Image.Image:
        calls.append(frame.prompt)
        return Image.new("RGB", image.size, (20, 180, 130))

    monkeypatch.setattr(engine, "_diffusers_inpaint", fake_inpaint)
    frame = _frame([_condition("bottom", 0, "regional cat", full)]).model_copy(update={"render_strategy": "per_layer"})

    engine.generate(frame)

    assert len(calls) == 1


def test_soft_alpha_compositing_does_not_wipe_lower_layers() -> None:
    engine = DiffusionEngine(EngineConfig(device="cpu"))
    size = (256, 256)
    original = Image.new("RGB", size, (128, 128, 128))
    generated = Image.new("RGB", size, (20, 180, 130))
    generated.paste((190, 40, 80), (0, 128, 256, 256))
    full = Image.new("L", size, 255)
    top = Image.new("L", size, 0)
    top.paste(255, (0, 136, 256, 256))
    top.paste(128, (0, 128, 256, 136))
    frame = _frame([
        _condition("bottom", 0, "room", full),
        _condition("top", 1, "bed", top),
    ])

    output = engine._apply_layer_soft_alpha_compositing(generated, original, frame)

    assert output.getpixel((64, 64)) == (20, 180, 130)
    assert output.getpixel((64, 192)) == (190, 40, 80)


def test_realtime_frame_accepts_full_denoise_strength() -> None:
    size = (256, 256)
    alpha = Image.new("L", size, 255)
    frame = InpaintFrame(
        prompt="cat",
        image=_png_url(Image.new("RGBA", size, (128, 128, 128, 255))),
        mask=_png_url(Image.new("L", size, 255)),
        width=size[0],
        height=size[1],
        strength=1.0,
        layer_conditions=[LayerCondition(
            layer_id="layer",
            z_index=0,
            region_id="layer",
            prompt="cat",
            image=_png_url(Image.new("RGBA", size, (128, 128, 128, 255))),
            mode="mask",
            denoise=1.0,
        )],
    )

    assert frame.strength == 1.0
    assert frame.layer_conditions[0].denoise == 1.0


def test_lcm_scheduler_clamps_inpaint_steps_for_full_strength(monkeypatch) -> None:
    class FakeSchedulerConfig:
        original_inference_steps = 50

    class LCMScheduler:
        config = FakeSchedulerConfig()

    class FakePipe:
        scheduler = LCMScheduler()

    class FakeResult:
        images = [Image.new("RGB", (256, 256), (20, 180, 130))]

    engine = DiffusionEngine(EngineConfig(device="cpu"))
    engine.pipe = FakePipe()
    engine.pipeline_kind = "inpaint"
    captured: list[int] = []

    monkeypatch.setattr(engine, "_realtime_prompt_kwargs", lambda *_args, **_kwargs: {"prompt": "cat", "negative_prompt": None})
    monkeypatch.setattr(engine, "_realtime_generator", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "_refresh_flowmatch_scheduler", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "_reset_realtime_scheduler_state", lambda *_args, **_kwargs: None)

    def fake_call(kwargs, _frame):
        captured.append(kwargs["num_inference_steps"])
        return FakeResult()

    monkeypatch.setattr(engine, "_call_realtime_pipe", fake_call)
    frame = _frame([]).model_copy(update={"steps": 60, "strength": 1.0})

    output = engine._diffusers_inpaint(
        Image.new("RGB", (256, 256), (128, 128, 128)),
        Image.new("L", (256, 256), 255),
        frame,
    )

    assert output.getpixel((0, 0)) == (20, 180, 130)
    assert captured == [50]
