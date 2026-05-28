"""Resolution helpers (only).

The full layered SDXL rendering engine (SinglePassStrategy / PerLayerStrategy
/ TiledStrategy / RenderLayer / SceneSettings / CFGCond) has been removed.
Engines now consume :class:`app.render_plan.RenderPlan` directly.
"""

from .resolution import SDXL_RESOLUTIONS, optimal_sdxl_resolution, fit_region_to_sdxl, SDXLCrop

__all__ = [
    "SDXL_RESOLUTIONS",
    "optimal_sdxl_resolution",
    "fit_region_to_sdxl",
    "SDXLCrop",
]
