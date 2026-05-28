"""Transport-boundary helpers.

The legacy scene composer (SceneComposer / layer_conditions_to_layers /
Layer / AutoTagSpec) has been removed; the v2 wire format aggregates on
the frontend, see :mod:`app.render_plan`. Only :mod:`compose.io` (mask
polarity / data-URL helpers) survives because transports still need it.
"""
