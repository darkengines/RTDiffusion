"""Layered SDXL rendering engine.

Public surface:
    RenderLayer        — full per-layer conditioning parameter set
    SceneSettings      — scene-level sampler / SDXL params
    CFGCond            — scalar + optional spatial CFG map
    LoRASpec           — LoRA path + weight
    SDXLMicroCond      — orig_size / target_size / crop_coords
    SourceProvider     — protocol for custom image sources
    ConditioningProvider — protocol for custom conditioning injectors
    PostProcessor      — protocol for chained post-processing
    SinglePassStrategy — one composed diffusion call (fastest)
    PerLayerStrategy   — N back-to-front passes, bbox fitted to SDXL resolution
    TiledStrategy      — SDXL-tile grid with dirty-bit tracking + cosine feathering
    optimal_sdxl_resolution — nearest SDXL aspect-ratio resolution for a bbox
"""

from .resolution import SDXL_RESOLUTIONS, optimal_sdxl_resolution, fit_region_to_sdxl, SDXLCrop
from .types import (
    CFGCond,
    LoRASpec,
    SDXLMicroCond,
    RenderLayer,
    SceneSettings,
    SourceProvider,
    ConditioningProvider,
    PostProcessor,
)
from .strategies import SinglePassStrategy, PerLayerStrategy, TiledStrategy, TileGrid

__all__ = [
    "SDXL_RESOLUTIONS",
    "optimal_sdxl_resolution",
    "fit_region_to_sdxl",
    "SDXLCrop",
    "CFGCond",
    "LoRASpec",
    "SDXLMicroCond",
    "RenderLayer",
    "SceneSettings",
    "SourceProvider",
    "ConditioningProvider",
    "PostProcessor",
    "SinglePassStrategy",
    "PerLayerStrategy",
    "TiledStrategy",
    "TileGrid",
]
