"""SceneComposer — single source of truth for layer → sampler inputs.

This composer mirrors the semantics of `pipeline_graph.resolve_conditioning_mask`
(denoise channel) and `merge_layer_prompts` / `_frame_with_prompt_mix_conditions`
(prompt channel) so refactor Step 4–6 can swap call sites without behavior
drift. Step 2 ships parity tests against the legacy implementation.

Canonical conventions
─────────────────────
- Mask polarity everywhere: white = act on this pixel.
- Default per-channel mask = layer.color alpha. If `mask_<channel>` is set,
  it overrides; otherwise alpha is used.
- Painter's algorithm: layers process in list order; later layers win in the
  pixels they cover (for `override` mode). Other modes blend.
- Three channels are composed independently. They do not interact except
  via shared masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageChops

from ..inference.session import CondInput, ControlNetSpec, PromptBundle, RegionalPrompt
from .layer import Layer


@dataclass
class ComposedScene:
    color: Image.Image  # RGB, ready to send to the sampler
    denoise_map: Image.Image  # L-mode, white=regenerate
    cond_inputs: list[CondInput] = field(default_factory=list)
    prompt: PromptBundle = field(default_factory=PromptBundle)


# Mode → denoise composition operation. Matches `resolve_conditioning_mask`
# semantics exactly so a legacy `LayerCondition.mode` maps 1:1.
DenoiseMode = str  # "override" | "mask" | "add" | "multiply" | "prompt_mix" | "lock"


class SceneComposer:
    """Stateless composer; safe to call from any thread."""

    def compose(
        self,
        layers: list[Layer],
        canvas: Image.Image,
        base_prompt: PromptBundle,
        base_strength: float = 1.0,
        canvas_mask: Image.Image | None = None,
    ) -> ComposedScene:
        width, height = canvas.size

        color = self._compose_color(layers, canvas, width, height)
        denoise_map = self._compose_denoise(layers, canvas_mask, width, height, base_strength)
        cond_inputs = self._compose_cond(layers, color, width, height)
        prompt = self._compose_prompt(layers, base_prompt)

        return ComposedScene(color=color, denoise_map=denoise_map, cond_inputs=cond_inputs, prompt=prompt)

    # ── color channel ───────────────────────────────────────────────────────

    @staticmethod
    def _compose_color(
        layers: list[Layer],
        canvas: Image.Image,
        width: int,
        height: int,
    ) -> Image.Image:
        result = canvas.convert("RGB")
        for layer in layers:
            if layer.role not in ("color", "guidance"):
                continue
            if layer.color is None:
                continue
            schedule_w = _schedule_influence(*_schedule_tuple(layer))
            if schedule_w <= 0:
                continue
            color = layer.color.resize((width, height), Image.Resampling.LANCZOS)
            if color.mode != "RGBA":
                color = color.convert("RGBA")
            mask = _resolve_channel_mask(layer.mask_color, color, width, height, layer.weight * schedule_w)
            if mask is None:
                continue
            # Step 2 supports "normal" alpha-over compositing. Other blend modes
            # are reserved in caps but lower priority; ship those in Step 2.1.
            result = Image.composite(color.convert("RGB"), result, mask)
        return result

    # ── denoise channel ────────────────────────────────────────────────────

    @staticmethod
    def _compose_denoise(
        layers: list[Layer],
        canvas_mask: Image.Image | None,
        width: int,
        height: int,
        base_strength: float,
    ) -> Image.Image:
        base_strength = _clamp01(base_strength, upper=0.999)
        if canvas_mask is not None:
            base_arr = np.asarray(
                canvas_mask.convert("L").resize((width, height), Image.Resampling.NEAREST),
                dtype=np.float32,
            ) / 255.0
        else:
            base_arr = np.ones((height, width), dtype=np.float32)
        denoise_arr = base_arr * base_strength

        for layer in layers:
            if layer.role == "cond_only":
                continue
            schedule_w = _schedule_influence(*_schedule_tuple(layer))
            if schedule_w <= 0:
                continue
            mask_img = _resolve_channel_mask(layer.mask_denoise, layer.color, width, height, 1.0)
            if mask_img is None:
                continue
            alpha = np.asarray(mask_img, dtype=np.float32) / 255.0
            influence = np.clip(alpha * max(0.0, layer.weight) * schedule_w, 0.0, 1.0)
            if float(influence.max(initial=0.0)) <= 0:
                continue
            target = _clamp01(layer.denoise_strength, upper=0.999)
            covered = influence > 0.01
            mode = _layer_denoise_mode(layer)
            if mode in ("override", "replace", "lock"):
                denoise_arr = np.where(covered, target, denoise_arr)
            elif mode == "add":
                denoise_arr = np.clip(denoise_arr + np.where(covered, influence * target, 0.0), 0.0, 0.999)
            elif mode == "multiply":
                denoise_arr = np.where(covered, denoise_arr * target, denoise_arr)
            else:  # "mask" / soft blend
                denoise_arr = np.where(
                    covered,
                    denoise_arr * (1.0 - influence) + target * influence,
                    denoise_arr,
                )

        denoise_arr = np.clip(denoise_arr, 0.0, 0.999)
        return Image.fromarray((denoise_arr * 255.0).round().astype(np.uint8), "L")

    # ── conditioning channel ────────────────────────────────────────────────

    @staticmethod
    def _compose_cond(
        layers: list[Layer],
        composed_color: Image.Image,
        width: int,
        height: int,
    ) -> list[CondInput]:
        """One CondInput per (layer, cn_spec).

        Merging CNs that target the same model across multiple layers is a
        later optimization; Step 2 emits per-layer entries so the downstream
        sampler sees the same fan-out the legacy code does.
        """
        out: list[CondInput] = []
        for layer in layers:
            if not layer.cn:
                continue
            schedule_w = _schedule_influence(*_schedule_tuple(layer))
            if schedule_w <= 0:
                continue
            if layer.cond_image is not None:
                cn_img = layer.cond_image.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
            elif layer.color is not None:
                cn_img = layer.color.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
            else:
                cn_img = composed_color
            mask = _resolve_channel_mask(layer.mask_cond, layer.color, width, height, 1.0)
            for spec in layer.cn:
                scaled = ControlNetSpec(
                    model_id=spec.model_id,
                    model_path=spec.model_path,
                    scale=max(0.0, spec.scale * layer.weight * schedule_w),
                    start=spec.start,
                    end=spec.end,
                )
                out.append(CondInput(spec=scaled, image=cn_img, mask=mask))
        return out

    # ── prompt channel ──────────────────────────────────────────────────────

    @staticmethod
    def _compose_prompt(layers: list[Layer], base_prompt: PromptBundle) -> PromptBundle:
        positive_parts: list[str] = []
        negative_parts: list[str] = []
        regional: list[RegionalPrompt] = list(base_prompt.regional)
        if base_prompt.positive.strip():
            positive_parts.append(base_prompt.positive.strip())
        if base_prompt.negative.strip():
            negative_parts.append(base_prompt.negative.strip())
        for layer in layers:
            if layer.prompt_fragment and layer.prompt_fragment.strip():
                fragment = layer.prompt_fragment.strip()
                positive_parts.append(fragment)
                regional.append(
                    RegionalPrompt(
                        layer_id=layer.layer_id,
                        prompt=fragment,
                        negative_prompt=(layer.negative_fragment or "").strip(),
                        mask=_layer_alpha_image(layer),
                        weight=layer.weight,
                        schedule=_schedule_tuple(layer),
                    )
                )
            if layer.negative_fragment and layer.negative_fragment.strip():
                negative_parts.append(layer.negative_fragment.strip())
        return PromptBundle(
            positive=", ".join(positive_parts) if positive_parts else base_prompt.positive,
            negative=", ".join(negative_parts) if negative_parts else base_prompt.negative,
            regional=regional,
            positive_b=base_prompt.positive_b,
            lerp_b=base_prompt.lerp_b,
        )


# ── helpers ────────────────────────────────────────────────────────────────


def _schedule_tuple(layer: Layer) -> tuple[str, float, float]:
    return ("linear", layer.schedule[0], layer.schedule[1])


def _layer_denoise_mode(layer: Layer) -> str:
    """Map a Layer's role + denoise_strength to a legacy mode string.

    The Layer dataclass intentionally does not expose `mode` directly; the
    intent is encoded by role + masks. This translation preserves parity
    with `resolve_conditioning_mask` for legacy-shaped inputs.
    """
    # `role` carries intent. For Step 2 we route everything through "mask"
    # (soft blend) which matches the legacy default. The legacy_adapter sets
    # an attribute when migrating an explicit `mode` field so the operator
    # can be preserved without polluting the public Layer API.
    return getattr(layer, "_legacy_mode", "mask")


def _resolve_channel_mask(
    explicit: Image.Image | None,
    fallback_color: Image.Image | None,
    width: int,
    height: int,
    weight: float,
) -> Image.Image | None:
    """Resolve the active mask for a channel.

    Order: explicit `mask_<channel>` → fallback `alpha(color)` → None.
    """
    if explicit is not None:
        mask = explicit.convert("L").resize((width, height), Image.Resampling.NEAREST)
    elif fallback_color is not None:
        rgba = fallback_color.resize((width, height), Image.Resampling.LANCZOS)
        if rgba.mode != "RGBA":
            rgba = rgba.convert("RGBA")
        mask = rgba.getchannel("A")
    else:
        return None
    if not mask.getbbox():
        return None
    if abs(weight - 1.0) <= 1e-6:
        return mask
    scaled_weight = max(0.0, min(1.0, weight))
    return mask.point(lambda v: int(v * scaled_weight))


def _layer_alpha_image(layer: Layer) -> Image.Image | None:
    if layer.color is None:
        return None
    rgba = layer.color if layer.color.mode == "RGBA" else layer.color.convert("RGBA")
    return rgba.getchannel("A")


def _schedule_influence(schedule: str, start: float, end: float) -> float:
    duration = max(0.0, min(1.0, end) - max(0.0, start))
    if duration <= 0:
        return 0.0
    if schedule == "ease_in":
        return duration * 0.82
    if schedule == "ease_out":
        return min(1.0, duration * 1.08)
    if schedule == "ease_in_out":
        return duration * 0.94
    return duration


def _clamp01(value: float, upper: float = 1.0) -> float:
    return max(0.0, min(upper, value))


# Unused helper kept to silence linter; ImageChops imported for future blend modes.
_ = ImageChops
