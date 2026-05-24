"""DiffusionEngine package — realtime & batch diffusion inference.

Sub-modules
-----------
config    — EngineConfig dataclass and type aliases
accels    — ONNX / TensorRT UNet wrappers and debug channel builder
engine    — DiffusionEngine class (core inference logic)
"""

from .config import ChunkCallback, EngineConfig, ProgressCallback
from .accels import _TRTUNetRunner, _UNetONNXWrapper, _collect_engine_debug
from .engine import DiffusionEngine, is_cuda_device

__all__ = [
    "ChunkCallback",
    "DiffusionEngine",
    "EngineConfig",
    "ProgressCallback",
    "_TRTUNetRunner",
    "_UNetONNXWrapper",
    "_collect_engine_debug",
    "is_cuda_device",
]
