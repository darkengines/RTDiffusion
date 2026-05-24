"""StreamDiffusion pipeline package.

Sub-modules
-----------
helpers  — module-level inference helpers (VAE encode, prompt encode, _predict_x0, etc.)
session  — StreamSession class (one continuous streaming session)
manager  — SessionManager class (caches sessions, handles debounced rebuilds)
"""

from .helpers import auto_t_indices, resolve_t_indices
from .session import StreamSession
from .manager import SessionManager

__all__ = [
    "StreamSession",
    "SessionManager",
    "auto_t_indices",
    "resolve_t_indices",
]
