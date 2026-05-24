# Backward-compatibility shim — all implementations moved to app/stream/
from .stream import SessionManager, StreamSession, auto_t_indices, resolve_t_indices

__all__ = ["SessionManager", "StreamSession", "auto_t_indices", "resolve_t_indices"]
