"""v2 layer composition (back-compat import surface).

The legacy Region/Layer/Scene model with operator aggregation has been
removed (see plan ``optimized-growing-marble``). The frontend now aggregates
the layer stack into three flat buffers (rgba, cfg_map, denoise_map) plus a
flat prompt list; the backend decodes via :func:`plan_from_wire` into a
:class:`RenderPlan`. New code should import from :mod:`app.render_plan`.
"""

from __future__ import annotations

from .render_plan import (
    CFG_HI,
    BlobResolver,
    ControlNet,
    Prompt,
    RenderPlan,
    plan_from_wire,
)

__all__ = [
    "CFG_HI",
    "BlobResolver",
    "ControlNet",
    "Prompt",
    "RenderPlan",
    "plan_from_wire",
]
