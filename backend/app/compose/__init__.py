"""Scene composition.

The composer turns the editor's layer list into the inputs a sampler wants:
one color image, one denoise map, a CN-input per ControlNet id, and a
prompt bundle. Every transport path runs through this single composer.
"""

from .composer import ComposedScene, SceneComposer
from .layer import AutoTagSpec, Layer
from .legacy import layer_condition_to_layer, layer_conditions_to_layers

__all__ = [
    "AutoTagSpec",
    "ComposedScene",
    "Layer",
    "SceneComposer",
    "layer_condition_to_layer",
    "layer_conditions_to_layers",
]
