from __future__ import annotations

from functools import cached_property

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file


def _zero_module(module: torch.nn.Module) -> torch.nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class _TransparentUNet1024(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block

        block_out_channels = (32, 32, 64, 128, 256, 512, 512)
        down_block_types = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        )
        up_block_types = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )
        layers_per_block = 2
        attention_head_dim = 8
        norm_groups = 4
        norm_eps = 1e-5

        self.conv_in = torch.nn.Conv2d(3, block_out_channels[0], kernel_size=3, padding=1)
        self.latent_conv_in = _zero_module(torch.nn.Conv2d(4, block_out_channels[2], kernel_size=1))
        self.down_blocks = torch.nn.ModuleList()
        self.up_blocks = torch.nn.ModuleList()

        output_channel = block_out_channels[0]
        for index, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[index]
            self.down_blocks.append(
                get_down_block(
                    down_block_type,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=None,
                    add_downsample=index != len(block_out_channels) - 1,
                    resnet_eps=norm_eps,
                    resnet_act_fn="silu",
                    resnet_groups=norm_groups,
                    attention_head_dim=attention_head_dim,
                    downsample_padding=1,
                    resnet_time_scale_shift="default",
                    downsample_type="conv",
                    dropout=0.0,
                )
            )

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=0.0,
            resnet_eps=norm_eps,
            resnet_act_fn="silu",
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=attention_head_dim,
            resnet_groups=norm_groups,
            attn_groups=None,
            add_attention=True,
        )

        reversed_channels = list(reversed(block_out_channels))
        output_channel = reversed_channels[0]
        for index, up_block_type in enumerate(up_block_types):
            previous_output_channel = output_channel
            output_channel = reversed_channels[index]
            input_channel = reversed_channels[min(index + 1, len(block_out_channels) - 1)]
            self.up_blocks.append(
                get_up_block(
                    up_block_type,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=previous_output_channel,
                    temb_channels=None,
                    add_upsample=index != len(block_out_channels) - 1,
                    resnet_eps=norm_eps,
                    resnet_act_fn="silu",
                    resnet_groups=norm_groups,
                    attention_head_dim=attention_head_dim,
                    resnet_time_scale_shift="default",
                    upsample_type="conv",
                    dropout=0.0,
                )
            )

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_groups, eps=norm_eps)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(block_out_channels[0], 4, kernel_size=3, padding=1)

    def forward(self, pixel: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        sample_latent = self.latent_conv_in(latent)
        sample = self.conv_in(pixel)
        residuals = (sample,)
        for index, down_block in enumerate(self.down_blocks):
            if index == 3:
                sample = sample + sample_latent
            sample, block_residuals = down_block(hidden_states=sample, temb=None)
            residuals += block_residuals
        sample = self.mid_block(sample, None)
        for up_block in self.up_blocks:
            block_residuals = residuals[-len(up_block.resnets) :]
            residuals = residuals[: -len(up_block.resnets)]
            sample = up_block(sample, block_residuals, None)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        return self.conv_out(sample)


class LayerDiffuseDecoder:
    def __init__(self, device: str, dtype: torch.dtype) -> None:
        self.device = torch.device(device)
        self.dtype = dtype

    @cached_property
    def model(self) -> _TransparentUNet1024:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="LayerDiffusion/layerdiffusion-v1",
            filename="vae_transparent_decoder.safetensors",
        )
        model = _TransparentUNet1024()
        model.load_state_dict(load_file(model_path), strict=True)
        model.to(self.device, dtype=self.dtype)
        model.eval()
        return model

    @torch.inference_mode()
    def decode_rgba(self, pixel: torch.Tensor, latent: torch.Tensor, *, augmented: bool = True) -> Image.Image:
        return self.decode_rgba_batch(pixel, latent, augmented=augmented)[0]

    @torch.inference_mode()
    def decode_rgba_batch(self, pixel: torch.Tensor, latent: torch.Tensor, *, augmented: bool = True) -> list[Image.Image]:
        model = self.model
        model_dtype = next(model.parameters()).dtype
        pixel = pixel.to(self.device, dtype=model_dtype)
        latent = latent.to(self.device, dtype=model_dtype)
        decoded = self._estimate_augmented(pixel, latent) if augmented else model(pixel, latent)
        images: list[Image.Image] = []
        for item in decoded.clamp(0, 1).detach().float().cpu():
            rgba = (item.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            alpha = rgba[..., :1]
            rgb = rgba[..., 1:]
            images.append(Image.fromarray(np.concatenate([rgb, alpha], axis=-1), mode="RGBA"))
        return images

    def _estimate_augmented(self, pixel: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for flip in (False, True):
            for rotation in range(4):
                feed_pixel = pixel.clone()
                feed_latent = latent.clone()
                if flip:
                    feed_pixel = torch.flip(feed_pixel, dims=(3,))
                    feed_latent = torch.flip(feed_latent, dims=(3,))
                feed_pixel = torch.rot90(feed_pixel, k=rotation, dims=(2, 3))
                feed_latent = torch.rot90(feed_latent, k=rotation, dims=(2, 3))
                output = self.model(feed_pixel, feed_latent).clamp(0, 1)
                output = torch.rot90(output, k=-rotation, dims=(2, 3))
                if flip:
                    output = torch.flip(output, dims=(3,))
                outputs.append(output)
        return torch.median(torch.stack(outputs, dim=0), dim=0).values