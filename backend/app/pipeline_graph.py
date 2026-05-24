"""
Per-session processing graph for streaming conditioning.

The graph is compiled once from a list of PipelineNode dicts (sent by the
client in settings) and executed each frame from the inference thread pool.

Node types
──────────
  tagger       — runs WD image tagger on a layer image, produces a prompt string
  cn_from_layer — preprocesses a layer image for ControlNet conditioning

Graph execution is single-threaded per session (protected by the caller's
_infer_lock), so no additional locking is needed inside run().

Example node list (as Python dicts / JSON):
  [
    {"id": "n1", "type": "tagger", "layer_id": "abc123",
     "tagger_threshold": 0.35, "tagger_refresh_frames": 5},
    {"id": "n2", "type": "cn_from_layer", "layer_id": "abc123",
     "cn_model": "canny", "cn_scale": 0.8}
  ]
"""

from __future__ import annotations

import logging
import base64
import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from .schemas import ControlNetPreprocessorParams

logger = logging.getLogger("rtdiffusion.graph")


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class GraphResult:
    """Output of one graph execution pass."""
    prompt_overrides: dict[str, str] = field(default_factory=dict)
    """layer_id → prompt string produced by tagger nodes."""
    cn_images: dict[str, Image.Image] = field(default_factory=dict)
    """layer_id → preprocessed control image from cn_from_layer nodes."""
    cn_params: dict[str, dict[str, float]] = field(default_factory=dict)
    """layer_id → ControlNet scale/start/end produced by cn_from_layer nodes."""


@dataclass
class ConditioningResult:
    """Resolved spatial conditioning shared by SDXL, Z-Image and streaming paths."""
    mask: Image.Image
    strength: float
    denoise_map: Image.Image


# ── Internal node state ────────────────────────────────────────────────────────

@dataclass
class _TaggerState:
    node_id: str
    layer_id: str
    model_name: str
    threshold: float
    refresh_frames: int
    _counter: int = 0
    _cached: str = ""
    _last_probe: str = ""

    def maybe_run(self, image: Image.Image, mask: Image.Image | None) -> str:
        """Run tagger on refresh intervals and whenever the input content changes."""
        self._counter += 1
        probe = self._frame_probe(image, mask)
        should_refresh = (self._counter - 1) % self.refresh_frames == 0
        if should_refresh or probe != self._last_probe:
            from . import tagger as _tagger
            result = _tagger.tag_image(
                image,
                mask=mask,
                threshold=self.threshold,
                model_name=self.model_name,
            )
            self._last_probe = probe
            if result.strip():
                self._cached = result.strip()
                logger.debug("Tagger[%s]: %s…", self.layer_id, self._cached[:60])
            elif should_refresh:
                # Clear stale tags when refresh executes but the frame no longer
                # contains confident detections.
                self._cached = ""
        return self._cached

    @staticmethod
    def _frame_probe(image: Image.Image, mask: Image.Image | None) -> str:
        img = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR).tobytes()
        if mask is not None:
            m = mask.convert("L").resize((32, 32), Image.Resampling.BILINEAR).tobytes()
        else:
            m = b""
        return hashlib.blake2b(img + m, digest_size=12).hexdigest()


@dataclass
class _CnFromLayerState:
    node_id: str
    layer_id: str
    cn_model: str
    cn_scale: float
    cn_start: float
    cn_end: float
    preprocessor_params: ControlNetPreprocessorParams


# ── SessionGraph ───────────────────────────────────────────────────────────────

class SessionGraph:
    """
    Compiled pipeline graph for one InpaintSession.

    All methods are called from the inference thread pool (not the event loop).
    Instance is rebuilt whenever pipeline_nodes in settings change (detected via
    graph_sig comparison in InpaintSession).
    """

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self._tagger_states: list[_TaggerState] = []
        self._cn_states: list[_CnFromLayerState] = []
        self._compile(nodes)

    # ── public ──────────────────────────────────────────────────────────────

    def run(
        self,
        layer_frames: dict[str, Image.Image],
        layer_masks: dict[str, Image.Image | None] | None = None,
        input_image: Image.Image | None = None,
        primary_input_layers: set[str] | None = None,
        condition_images: dict[str, Image.Image] | None = None,
    ) -> GraphResult:
        """
        Execute all nodes for the current frame.

        layer_frames:       layer_id → PIL Image decoded from per-frame POST body
        layer_masks:        layer_id → L-mode PIL Image (alpha from condition RGBA) for tagger crop
        input_image:        composed input scene (fallback tagger image)
        primary_input_layers: layer_ids whose frame is actual scene content (video)
        condition_images:   layer_id → RGB content of the exported condition RGBA image;
                            used as the tagger image for non-video layers so the tagger
                            sees the actual layer content rather than the full scene composite
        """
        result = GraphResult()
        masks = layer_masks or {}
        primary_layers = primary_input_layers or set()
        cond_imgs = condition_images or {}

        for state in self._tagger_states:
            # Priority: condition RGB image (layer's own visual content) > video frame > scene
            if state.layer_id in cond_imgs:
                img = cond_imgs[state.layer_id]
            elif state.layer_id in primary_layers:
                img = layer_frames.get(state.layer_id)
            else:
                img = layer_frames.get(state.layer_id) or input_image
            if img is None:
                continue
            mask = masks.get(state.layer_id)
            prompt = state.maybe_run(img, mask)
            if prompt:
                result.prompt_overrides[state.layer_id] = prompt

        for state in self._cn_states:
            img = layer_frames.get(state.layer_id)
            if img is None:
                continue
            try:
                from . import controlnet as _cn
                cn_img = _cn.preprocess_image(img, state.cn_model, state.preprocessor_params)
                result.cn_images[state.layer_id] = cn_img
                result.cn_params[state.layer_id] = {
                    "scale": state.cn_scale,
                    "start": state.cn_start,
                    "end": state.cn_end,
                }
            except Exception as exc:
                logger.warning("Graph CN preprocess [%s] failed: %s", state.node_id, exc)

        return result

    @staticmethod
    def signature(nodes: list[dict[str, Any]]) -> str:
        """Stable string used to detect graph changes without rebuilding unnecessarily."""
        return repr(tuple(sorted(repr(sorted(n.items())) for n in nodes)))

    # ── private ─────────────────────────────────────────────────────────────

    def _compile(self, nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            ntype = node.get("type", "")
            nid   = node.get("id", "?")
            lid   = node.get("layer_id", "")

            if ntype == "tagger":
                self._tagger_states.append(_TaggerState(
                    node_id=nid,
                    layer_id=lid,
                    model_name=node.get("tagger_model", "wd-eva02-large-v3"),
                    threshold=float(node.get("tagger_threshold", 0.35)),
                    refresh_frames=max(1, int(node.get("tagger_refresh_frames", 5))),
                ))
                logger.info("Graph: compiled tagger node %s → layer %s", nid, lid)

            elif ntype == "cn_from_layer":
                params_raw = node.get("cn_preprocessor_params") or {}
                params = ControlNetPreprocessorParams(**params_raw) if isinstance(params_raw, dict) else ControlNetPreprocessorParams()
                self._cn_states.append(_CnFromLayerState(
                    node_id=nid,
                    layer_id=lid,
                    cn_model=node.get("cn_model", "canny"),
                    cn_scale=max(0.0, min(2.0, float(node.get("cn_scale", 1.0)))),
                    cn_start=max(0.0, min(1.0, float(node.get("cn_start", 0.0)))),
                    cn_end=max(0.0, min(1.0, float(node.get("cn_end", 1.0)))),
                    preprocessor_params=params,
                ))
                logger.info("Graph: compiled cn_from_layer node %s → layer %s", nid, lid)

            else:
                logger.warning("Graph: unknown node type %r (id=%s), skipping", ntype, nid)


# ── Prompt merge utility ───────────────────────────────────────────────────────

def merge_layer_prompts(
    layer_conditions: list[dict[str, Any]],
    base_prompt: str,
    graph_result: GraphResult | None,
) -> str:
    """
    Merge layer condition prompts (and tagger overrides) with the base prompt.

    Layer prompts are weighted by coverage and appended to the base prompt.
    Tagger overrides (from graph_result) replace the layer's explicit prompt.
    Returns the merged prompt string.
    """
    if not layer_conditions:
        return base_prompt

    parts = [base_prompt.strip()] if base_prompt.strip() else []
    overrides = graph_result.prompt_overrides if graph_result else {}

    for cond in layer_conditions:
        layer_id = cond.get("layer_id", "")
        # Tagger override takes priority over explicit prompt
        prompt = overrides.get(layer_id) or cond.get("prompt", "")
        if not prompt.strip():
            continue
        parts.append(prompt.strip())

    return ", ".join(parts) if parts else base_prompt


def resolve_conditioning_mask(
    base_mask: Image.Image,
    layer_conditions: list[Any],
    width: int,
    height: int,
    base_strength: float,
) -> ConditioningResult:
    """
    Aggregate layer conditions into one effective inpaint mask and global strength
    using painter's-algorithm compositing (layers processed in order; later layers
    override earlier ones in their covered region).

    Mode semantics per layer condition:
      override / replace  — hard-set denoise to target value in covered region
      mask                — blend denoise toward target weighted by influence
      add                 — add influence × target to current denoise
      multiply            — scale current denoise by target in covered region
      prompt_mix          — affects prompt only; skipped here
    """
    base_strength = _clamp01(float(base_strength), upper=0.999)
    base_arr = np.asarray(
        base_mask.convert("L").resize((width, height), Image.Resampling.NEAREST),
        dtype=np.float32,
    ) / 255.0
    # denoise_arr: starts as scene mask × base_strength (no-layer fallback)
    denoise_arr = base_arr * base_strength

    for cond in layer_conditions:
        mode = str(_cond_get(cond, "mode", "mask") or "mask").lower().strip()
        if mode == "prompt_mix":
            continue  # purely affects prompt; no spatial conditioning
        schedule = _schedule_influence(
            str(_cond_get(cond, "schedule", "linear") or "linear"),
            float(_cond_get(cond, "schedule_start", 0.0) or 0.0),
            float(_cond_get(cond, "schedule_end", 1.0) or 1.0),
        )
        if schedule <= 0:
            continue
        alpha = _condition_alpha(cond, width, height)
        if alpha is None:
            continue
        weight = max(0.0, float(_cond_get(cond, "weight", 1.0) or 0.0))
        influence = np.clip(alpha * weight * schedule, 0.0, 1.0)
        if float(influence.max(initial=0.0)) <= 0:
            continue
        denoise = _cond_get(cond, "denoise", None)
        target = base_strength if denoise is None else _clamp01(float(denoise), upper=0.999)
        covered = influence > 0.01

        if mode in ("override", "replace"):
            # Painter's algorithm: hard overwrite denoise in covered region.
            # Later layers win — so Layer B (top) overrides Layer A (bottom) in
            # B's masked area even if A set that region to 0.
            denoise_arr = np.where(covered, target, denoise_arr)
        elif mode == "add":
            denoise_arr = np.clip(
                denoise_arr + np.where(covered, influence * target, 0.0), 0.0, 0.999
            )
        elif mode == "multiply":
            denoise_arr = np.where(covered, denoise_arr * target, denoise_arr)
        else:  # "mask" — soft blend toward target weighted by influence
            denoise_arr = np.where(
                covered,
                denoise_arr * (1.0 - influence) + target * influence,
                denoise_arr,
            )

    denoise_arr = np.clip(denoise_arr, 0.0, 0.999)
    # Diffusers has one scalar ``strength`` for the whole inpaint call. Keep the
    # highest requested denoise as that scalar, then encode lower per-pixel
    # denoise values back into the mask intensity so RGBA denoise overrides
    # preserve those pixels during diffusion instead of becoming full-white mask.
    effective_strength = max(float(denoise_arr.max(initial=0.0)), 0.001)
    effective_mask = np.where(denoise_arr > 0.001, denoise_arr / effective_strength, 0.0)
    mask_img = Image.fromarray((np.clip(effective_mask, 0.0, 1.0) * 255.0).round().astype(np.uint8), "L")
    denoise_img = Image.fromarray((denoise_arr * 255.0).round().astype(np.uint8), "L")
    return ConditioningResult(mask=mask_img, strength=min(0.999, effective_strength), denoise_map=denoise_img)


def _cond_get(cond: Any, key: str, default: Any = None) -> Any:
    if isinstance(cond, dict):
        return cond.get(key, default)
    return getattr(cond, key, default)


def _condition_alpha(cond: Any, width: int, height: int) -> np.ndarray | None:
    image_url = _cond_get(cond, "denoise_mask", "") or _cond_get(cond, "image", "")
    if not image_url:
        return None
    try:
        _, sep, payload = str(image_url).partition(",")
        raw = base64.b64decode(payload if sep else str(image_url))
        img = Image.open(io.BytesIO(raw)).resize((width, height), Image.Resampling.LANCZOS)
        if _cond_get(cond, "denoise_mask", ""):
            return np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        return np.asarray(img.convert("RGBA").getchannel("A"), dtype=np.float32) / 255.0
    except Exception as exc:
        logger.warning("Conditioning mask decode failed: %s", exc)
        return None


def _valid_operator(value: str) -> str:
    value = value.lower().strip().replace("-", "_")
    if value in {"replace", "add", "average", "multiply", "max"}:
        return value
    return "add"


def _combine(current: np.ndarray, incoming: np.ndarray, influence: np.ndarray, operator: str) -> np.ndarray:
    active = influence > 1e-6
    if operator == "replace":
        return np.where(active, current * (1.0 - influence) + incoming, current)
    if operator == "average":
        return np.where(active, (current + incoming) * 0.5, current)
    if operator == "multiply":
        return np.where(active, current * np.clip(incoming, 0.0, 1.0), current)
    if operator == "max":
        return np.maximum(current, incoming)
    return np.clip(current + incoming, 0.0, 1.0)


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
