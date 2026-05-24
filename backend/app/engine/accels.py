"""Hardware acceleration wrappers: ONNX and TensorRT UNet runners, debug channel builder."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


class _UNetONNXWrapper:
    """Flattens dict inputs so the UNet callable is wrappable for ONNX export.

    Not a nn.Module itself — use make_unet_onnx_module(torch, unet, has_sdxl_cond) to get an
    exportable nn.Module (defined at call-time when torch is already imported).
    """

    def __init__(self, unet: Any, has_sdxl_cond: bool) -> None:
        self._unet = unet
        self._has_sdxl_cond = has_sdxl_cond

    def __call__(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        kwargs: dict[str, Any] = {"encoder_hidden_states": encoder_hidden_states, "return_dict": False}
        if self._has_sdxl_cond:
            kwargs["added_cond_kwargs"] = {"text_embeds": text_embeds, "time_ids": time_ids}
        return (self._unet(sample, timestep, **kwargs)[0],)


class _TRTUNetRunner:
    """Drop-in replacement for a PyTorch UNet — runs inference via a TensorRT engine.

    The engine is pre-built for a fixed batch size and latent resolution.
    Output is written to a pre-allocated buffer that is reused every frame.
    """

    def __init__(self, engine_bytes: bytes, out_shape: tuple, has_sdxl_cond: bool) -> None:
        try:
            import tensorrt as trt  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("tensorrt package required for RTD_STREAM_TRT=1") from exc
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError("TRT engine deserialization failed")
        self._ctx = self._engine.create_execution_context()
        self._out_shape = out_shape
        self._has_sdxl_cond = has_sdxl_cond
        self._out_buf: Any = None

    def __call__(
        self,
        sample,
        timestep,
        encoder_hidden_states: Any = None,
        added_cond_kwargs: Any = None,
        return_dict: bool = False,
        **_: Any,
    ):
        import torch

        if self._out_buf is None or self._out_buf.shape != self._out_shape:
            self._out_buf = torch.empty(self._out_shape, device=sample.device, dtype=sample.dtype)

        tensors: dict[str, Any] = {
            "sample": sample.contiguous(),
            "timestep": timestep.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
        }
        if self._has_sdxl_cond and added_cond_kwargs:
            tensors["text_embeds"] = added_cond_kwargs["text_embeds"].contiguous()
            tensors["time_ids"] = added_cond_kwargs["time_ids"].contiguous()

        for name, t in tensors.items():
            self._ctx.set_tensor_address(name, t.data_ptr())
        self._ctx.set_tensor_address("noise_pred", self._out_buf.data_ptr())

        stream_handle = torch.cuda.current_stream().cuda_stream
        self._ctx.execute_async_v3(stream_handle=stream_handle)
        return (self._out_buf,)


def _collect_engine_debug(
    image: "Image.Image",
    mask: "Image.Image",
    output: "Image.Image",
    frame: Any,
    denoise_map: "Image.Image | None" = None,
) -> dict[str, str]:
    """Build debug thumbnails for the overlay — canvas, mask, denoise map, per-layer, CN preprocessed."""
    import base64
    import io as _io

    from PIL import Image as _PIL

    def _thumb(img: "Image.Image", max_side: int = 320) -> str:
        t = img.convert("RGB").copy()
        t.thumbnail((max_side, max_side), _PIL.BILINEAR)
        buf = _io.BytesIO()
        t.save(buf, format="JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _thumb_l(img: "Image.Image", max_side: int = 320) -> str:
        import numpy as np
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        r = np.clip(arr * 2 - 1, 0, 1)
        g = np.clip(1 - np.abs(arr * 2 - 1), 0, 1)
        b = np.clip(1 - arr * 2, 0, 1)
        rgb = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)
        t = _PIL.fromarray(rgb, "RGB")
        t.thumbnail((max_side, max_side), _PIL.BILINEAR)
        buf = _io.BytesIO()
        t.save(buf, format="JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    channels: dict[str, str] = {
        "canvas": _thumb(image),
        "mask": _thumb_l(mask),
        "input/frame/canvas": _thumb(image),
        "input/mask/aggregated": _thumb_l(mask),
        "condition_mask": _thumb_l(mask),
        "output": _thumb(output),
    }

    if denoise_map is not None:
        denoise_preview = _thumb_l(denoise_map)
        channels["denoise_map"] = denoise_preview
        channels["input/mask/aggregated/denoise"] = denoise_preview
        channels["condition_denoise"] = denoise_preview
    elif frame.layer_conditions:
        try:
            from ..pipeline_graph import resolve_conditioning_mask
            resolved = resolve_conditioning_mask(
                mask, list(frame.layer_conditions), frame.width, frame.height, frame.strength
            )
            denoise_preview = _thumb_l(resolved.denoise_map)
            channels["denoise_map"] = denoise_preview
            channels["input/mask/aggregated/denoise"] = denoise_preview
            channels["condition_denoise"] = denoise_preview
        except Exception:
            pass

    from ..image_io import decode_data_url_rgba
    for i, cond in enumerate(frame.layer_conditions or []):
        label = cond.name or cond.layer_id or f"layer_{i}"
        try:
            rgba = decode_data_url_rgba(cond.image).resize((frame.width, frame.height), _PIL.LANCZOS)
            channels[f"input/layer/{label}/alpha"] = _thumb_l(rgba.getchannel("A"))
            channels[f"input/layer/{label}/rgb"] = _thumb(rgba.convert("RGB"))
        except Exception:
            pass

    from .. import controlnet as _cn
    for cond in (frame.layer_conditions or []):
        if not cond.controlnet_model:
            continue
        try:
            from ..image_io import decode_data_url
            from ..schemas import ControlNetPreprocessorParams
            src = decode_data_url(cond.controlnet_image or cond.image).convert("RGB")
            params = cond.controlnet_preprocessor_params or ControlNetPreprocessorParams()
            preprocessed = (
                _cn.preprocess_image(src.resize((frame.width, frame.height)), cond.controlnet_model, params)
                if cond.controlnet_preprocessor
                else src.resize((frame.width, frame.height))
            )
            label = cond.name or cond.layer_id or cond.controlnet_model
            channels[f"cn/{label}/{cond.controlnet_model}"] = _thumb(preprocessed)
        except Exception:
            pass

    return channels
