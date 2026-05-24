"""RTC streaming package.

Sub-modules
-----------
session  — InpaintSession class (per-connection inference loop)
router   — FastAPI APIRouter with /api/rtc/* endpoints
"""

from .router import router
from .session import _shared_session_manager

__all__ = ["router", "_shared_session_manager"]
