"""Shared test setup.

Force `RTD_MOCK=1` before any test module imports `backend.app.engine`, so the
DiffusionEngine constructor short-circuits its pipeline load. Tests that need
real GPU paths must override this explicitly.
"""

from __future__ import annotations

import os

os.environ.setdefault("RTD_MOCK", "1")
