"""Preprocessors run before composition.

Today these live in `pipeline_graph.SessionGraph` (tagger, cn_from_layer)
and feed back into per-frame state. In Step 6 they move here as plain
functions that mutate a `Layer` in place (fill `prompt_fragment` from
tagger, fill `cond_image` from CN preprocessor).

Skeleton only in Step 1.
"""

from __future__ import annotations

from .layer import Layer


def run_auto_tag(layer: Layer) -> None:
    raise NotImplementedError("auto-tag preprocessor lands in refactor Step 6")


def run_cn_preprocessor(layer: Layer) -> None:
    raise NotImplementedError("CN preprocessor lands in refactor Step 6")
