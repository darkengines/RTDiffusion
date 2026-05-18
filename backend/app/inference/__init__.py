"""Inference backend abstractions.

Every realtime / one-shot inference path implements `InferenceSession` and
declares a `RendererCaps`. The transport layer (WS, WebRTC, HTTP) talks to
these via a single seam so transport never reaches into engine internals.
"""

from .caps import BlendMode, Channel, LayerRole, RendererCaps
from .diffusers_session import DiffusersSession, inpaint_frame_to_request
from .stream_session import StreamInferenceSession
from .session import (
    CondInput,
    ControlNetSpec,
    FrameRequest,
    FrameResult,
    InferenceSession,
    PromptBundle,
    RegionalPrompt,
    SamplerSpec,
    WarmupHint,
)
from .registry import BACKENDS, get_backend_caps, list_backends, register_backend

__all__ = [
    "BACKENDS",
    "BlendMode",
    "Channel",
    "CondInput",
    "ControlNetSpec",
    "DiffusersSession",
    "FrameRequest",
    "FrameResult",
    "InferenceSession",
    "LayerRole",
    "PromptBundle",
    "RegionalPrompt",
    "RendererCaps",
    "SamplerSpec",
    "StreamInferenceSession",
    "WarmupHint",
    "get_backend_caps",
    "inpaint_frame_to_request",
    "list_backends",
    "register_backend",
]
