from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Iterator, cast

from PIL import Image

logger = logging.getLogger("rtdiffusion.engine")


@dataclass(frozen=True)
class RegionalAttentionRegion:
    prompt_embeds: Any
    negative_prompt_embeds: Any | None
    mask: Image.Image
    weight: float = 1.0


@dataclass(frozen=True)
class RegionalAttentionSpec:
    regions: tuple[RegionalAttentionRegion, ...]
    width: int
    height: int


@contextlib.contextmanager
def regional_attention_context(pipe: Any, spec: RegionalAttentionSpec | None) -> Iterator[None]:
    if spec is None or not spec.regions:
        yield
        return
    pipe_unet = getattr(pipe, "unet", None)
    eager_unet = getattr(pipe_unet, "_orig_mod", None)
    unet = eager_unet if eager_unet is not None else pipe_unet
    if unet is None or not hasattr(unet, "attn_processors") or not hasattr(unet, "set_attn_processor"):
        logger.warning("Regional attention unavailable: UNet attention processor API not found")
        yield
        return
    try:
        processors = dict(unet.attn_processors)
        wrapped = {
            name: MaskedRegionalAttnProcessor(proc, spec) if _is_cross_attention_processor(name) else proc
            for name, proc in processors.items()
        }
        unet.set_attn_processor(wrapped)
    except Exception:
        logger.warning("Regional attention unavailable: failed to install attention processors", exc_info=True)
        yield
        return
    try:
        yield
    finally:
        try:
            unet.set_attn_processor(processors)
        except Exception:
            logger.warning("Regional attention: failed to restore original attention processors", exc_info=True)


class MaskedRegionalAttnProcessor:
    def __init__(self, base_processor: Any, spec: RegionalAttentionSpec) -> None:
        self.base_processor = base_processor
        self.spec = spec
        self._mask_cache: dict[tuple[int, Any, Any, int], Any] = {}

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        temb: Any | None = None,
        image_rotary_emb: Any | None = None,
    ) -> Any:
        if encoder_hidden_states is None or not self.spec.regions:
            return self._call_base(attn, hidden_states, encoder_hidden_states, attention_mask, temb, image_rotary_emb)
        try:
            base = self._attention(attn, hidden_states, encoder_hidden_states, attention_mask, temb, image_rotary_emb)
            query_length = self._query_length(hidden_states)
            weighted_regions: list[tuple[Any, Any]] = []
            alpha_sum: Any | None = None
            for region in self.spec.regions:
                region_states = self._match_encoder_batch(region, encoder_hidden_states)
                if region_states is None:
                    continue
                region_out = self._attention(attn, hidden_states, region_states, attention_mask, temb, image_rotary_emb)
                alpha = self._mask_tensor(region, query_length, base)
                weighted_regions.append((region_out, alpha))
                alpha_sum = alpha if alpha_sum is None else alpha_sum + alpha
            if not weighted_regions or alpha_sum is None:
                return base
            current = base * (1.0 - alpha_sum.clamp(0.0, 1.0))
            normalizer = alpha_sum.clamp_min(1.0)
            for region_out, alpha in weighted_regions:
                current = current + region_out * (alpha / normalizer)
            return current
        except Exception:
            logger.debug("Regional attention processor fell back to base attention", exc_info=True)
            return self._call_base(attn, hidden_states, encoder_hidden_states, attention_mask, temb, image_rotary_emb)

    def _call_base(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any | None,
        attention_mask: Any | None,
        temb: Any | None,
        image_rotary_emb: Any | None,
    ) -> Any:
        if image_rotary_emb is not None:
            try:
                return self.base_processor(
                    attn,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
            except TypeError:
                pass
        return self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
        )

    @staticmethod
    def _query_length(hidden_states: Any) -> int:
        if getattr(hidden_states, "ndim", 0) == 4:
            return int(hidden_states.shape[-2] * hidden_states.shape[-1])
        return int(hidden_states.shape[1])

    def _match_encoder_batch(self, region: RegionalAttentionRegion, encoder_hidden_states: Any) -> Any | None:
        import torch

        positive = region.prompt_embeds.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        negative = region.negative_prompt_embeds
        if negative is not None:
            negative = negative.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        target_batch = int(encoder_hidden_states.shape[0])
        if negative is not None and target_batch == int(positive.shape[0]) + int(negative.shape[0]):
            return torch.cat([negative, positive], dim=0)
        if target_batch == int(positive.shape[0]):
            return positive
        if negative is not None and target_batch % 2 == 0:
            half = target_batch // 2
            neg = negative.repeat(max(1, half // int(negative.shape[0])), 1, 1)[:half]
            pos = positive.repeat(max(1, half // int(positive.shape[0])), 1, 1)[:half]
            return torch.cat([neg, pos], dim=0)
        if target_batch % int(positive.shape[0]) == 0:
            return positive.repeat(target_batch // int(positive.shape[0]), 1, 1)
        return None

    def _mask_tensor(self, region: RegionalAttentionRegion, query_length: int, like: Any) -> Any:
        import numpy as np
        import torch

        key = (id(region.mask), like.device, like.dtype, query_length)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        grid_w, grid_h = _grid_for_sequence(query_length, self.spec.width, self.spec.height)
        mask_img = region.mask.convert("L").resize((grid_w, grid_h), Image.Resampling.BILINEAR)
        arr = np.asarray(mask_img, dtype=np.float32).reshape(-1) / 255.0
        if arr.shape[0] != query_length:
            arr = np.resize(arr, query_length)
        arr = np.clip(arr * max(0.0, min(1.0, float(region.weight))), 0.0, 1.0)
        mask = torch.from_numpy(arr).to(device=like.device, dtype=like.dtype).view(1, query_length, 1)
        if getattr(like, "ndim", 0) == 4:
            mask = mask.view(1, grid_h, grid_w, 1).permute(0, 3, 1, 2)
        if int(like.shape[0]) > 1:
            mask = mask.repeat(int(like.shape[0]), *([1] * (mask.ndim - 1)))
        self._mask_cache[key] = mask
        return mask

    def _attention(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any | None,
        attention_mask: Any | None,
        temb: Any | None,
        image_rotary_emb: Any | None,
    ) -> Any:
        import torch.nn.functional as F

        try:
            from diffusers.models.embeddings import apply_rotary_emb
        except Exception:
            apply_rotary_emb = None

        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        channel = height = width = 0
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        if attention_mask is not None:
            prepared_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            if prepared_mask is not None:
                attention_mask = prepared_mask.view(batch_size, attn.heads, -1, prepared_mask.shape[-1])

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query: Any = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key: Any = attn.to_k(encoder_hidden_states)
        value: Any = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None and apply_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not getattr(attn, "is_cross_attention", False):
                key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = cast(Any, F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False))
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def _grid_for_sequence(sequence_length: int, width: int, height: int) -> tuple[int, int]:
    if sequence_length <= 0:
        return 1, 1
    aspect = max(1e-6, float(width) / max(1.0, float(height)))
    guess_w = max(1, int(round((sequence_length * aspect) ** 0.5)))
    candidates: list[tuple[int, int]] = []
    for w in range(max(1, guess_w - 8), guess_w + 9):
        if sequence_length % w == 0:
            candidates.append((w, sequence_length // w))
    if not candidates:
        return sequence_length, 1
    return min(candidates, key=lambda item: abs((item[0] / max(1, item[1])) - aspect))


def _is_cross_attention_processor(name: str) -> bool:
    parts = name.split(".")
    return "attn2" in parts or name.endswith("attn2.processor")


__all__ = ["RegionalAttentionRegion", "RegionalAttentionSpec", "regional_attention_context"]