"""Shared test setup.

Force `RTD_MOCK=1` before any test module imports `backend.app.engine`, so the
DiffusionEngine constructor short-circuits its pipeline load. Tests that need
real GPU paths must override this explicitly.

Also: ``torch`` is a complex package with C extension modules (``torch._C``)
that can only be loaded ONCE per process. Several test modules install
``sys.modules`` stubs for ``torch`` / ``diffusers`` / ``transformers`` to keep
their startup light. After such a stub is installed, a later ``import torch``
returns the stub — and even if we restore the real module, the partially
initialised state of ``torch._C`` is not fully recoverable.

We solve this by reordering collection: test files that need the *real* torch
run first. By the time stub-using tests execute, the real torch has finished
loading; the stubs cleanly replace the entries without breaking later imports.
"""

from __future__ import annotations

import os

os.environ.setdefault("RTD_MOCK", "1")


# Eagerly trigger lazy diffusers submodule loads we know test_arch_detection
# will need. The lazy loader caches the resolved class on the diffusers module
# so subsequent access bypasses the (sometimes torch-fragile) lazy import.
try:  # pragma: no cover — environment-dependent
    import torch  # noqa: F401
    import torch.distributed  # noqa: F401
    import diffusers as _diffusers  # noqa: F401
    # Force-resolve the lazy attributes; these would fail later if torch._C
    # state is corrupted by an intermediate test module's import sequence.
    _ = _diffusers.StableDiffusionXLImg2ImgPipeline  # noqa: F841
    _ = _diffusers.StableDiffusionImg2ImgPipeline  # noqa: F841
except Exception:  # noqa: BLE001
    pass


# Test files that touch the real torch / diffusers / safetensors stack — they
# must run before any test module that installs sys.modules stubs.
_NEEDS_REAL_TORCH = {
    "test_arch_detection.py",
    "test_e2e_rtc.py",
    "test_blob_channels.py",
}


def pytest_collection_modifyitems(config, items):  # noqa: D401 — pytest hook
    """Reorder collection so real-torch tests run before stubbed ones.

    Without this, stubbed tests can break the torch C extension state for any
    later import. Reordering ensures the real-torch tests run before any test
    file installs ``sys.modules`` stubs.
    """
    front: list = []
    back: list = []
    for item in items:
        # ``item.location[0]`` is the test file path relative to rootdir
        fname = item.location[0].replace("\\", "/").rsplit("/", 1)[-1]
        if fname in _NEEDS_REAL_TORCH:
            front.append(item)
        else:
            back.append(item)
    items[:] = front + back
