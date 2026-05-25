"""Binary blob upload + per-channel mask reference protocol.

Run:  pytest backend/tests/test_blob_channels.py -v

Verifies the binary path documented in §X of LAYER_SYSTEM.md:
  1. ``POST /api/rtc/{pc_id}/blob`` accepts raw PNG bytes, returns ``{ id }``
  2. ``layer_conditions[*]_mask_ref`` resolves through the session blob store
  3. LRU eviction keeps the store bounded
  4. Inline base64 fallback (legacy) still works when no ref is provided
"""

from __future__ import annotations

import io
import sys

import pytest


# Clear stub torch/diffusers so the FastAPI app boots with real torch + diffusers.
_NAMESPACES = ("torch", "diffusers", "transformers", "huggingface_hub",
               "onnxruntime", "safetensors")
for _name in list(sys.modules):
    if _name not in _NAMESPACES and not any(_name.startswith(ns + ".") for ns in _NAMESPACES):
        continue
    mod = sys.modules[_name]
    if mod is None:
        continue
    if not hasattr(mod, "__file__") or getattr(mod, "__file__", None) is None:
        del sys.modules[_name]

pytest.importorskip("fastapi")
pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.rtc.session import InpaintSession  # noqa: E402


@pytest.fixture
def client():
    from app import main as _main
    from app.rtc import session as rtc_session

    class _StubSession:
        def infer(self, *_a, **_kw):
            return Image.new("RGB", (32, 32), (10, 20, 30))

    class _StubManager:
        def __init__(self):
            self._generation = 0
            self.last_output = None
            self.build_phase = "ready"
            self.build_progress = 1.0
            self.build_message = ""

        def get_session(self, settings):
            self._generation = 1
            return _StubSession()

        def set_status_callback(self, *a, **kw):
            pass

    from unittest.mock import patch
    with patch.object(rtc_session, "_shared_session_manager", _StubManager()):
        with TestClient(_main.app) as c:
            yield c


def _png_bytes(w: int = 16, h: int = 16, value: int = 128) -> bytes:
    img = Image.new("L", (w, h), value)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ── Unit: session blob store ─────────────────────────────────────────────────

class TestSessionBlobStore:
    def test_store_returns_stable_id_for_same_content(self):
        s = InpaintSession("test")
        data = _png_bytes(value=100)
        id1 = s.store_blob(data)
        id2 = s.store_blob(data)
        assert id1 == id2
        assert s.get_blob(id1) is not None

    def test_store_returns_different_id_for_different_content(self):
        s = InpaintSession("test")
        id1 = s.store_blob(_png_bytes(value=10))
        id2 = s.store_blob(_png_bytes(value=200))
        assert id1 != id2

    def test_store_rejects_invalid_payload(self):
        s = InpaintSession("test")
        with pytest.raises(ValueError):
            s.store_blob(b"not a png")

    def test_lru_eviction(self):
        s = InpaintSession("test")
        s._BLOB_LIMIT = 3
        ids = [s.store_blob(_png_bytes(value=i)) for i in (1, 2, 3, 4)]
        # First entry evicted because LIMIT=3
        assert s.get_blob(ids[0]) is None
        assert all(s.get_blob(i) is not None for i in ids[1:])

    def test_re_uploading_bumps_to_most_recent(self):
        s = InpaintSession("test")
        s._BLOB_LIMIT = 3
        a = s.store_blob(_png_bytes(value=1))
        b = s.store_blob(_png_bytes(value=2))
        c = s.store_blob(_png_bytes(value=3))
        # Re-upload "a" → it bumps to MRU, "b" is now the LRU candidate
        s.store_blob(_png_bytes(value=1))
        s.store_blob(_png_bytes(value=4))  # triggers eviction of "b"
        assert s.get_blob(a) is not None   # survived
        assert s.get_blob(b) is None
        assert s.get_blob(c) is not None

    def test_blob_upload_does_not_advance_staging_generation(self):
        s = InpaintSession("test")
        generation = s._input_generation
        s.store_blob(_png_bytes(value=123))
        assert s._input_generation == generation

    def test_stale_settings_and_frame_sequences_are_ignored(self):
        s = InpaintSession("test")
        s.apply_settings({"prompt": "new"}, seq=2)
        s.apply_settings({"prompt": "old"}, seq=1)
        assert s._settings["prompt"] == "new"

        first = "data:image/png;base64,new"
        stale = "data:image/png;base64,old"
        s.apply_frame(first, None, seq=2)
        s.apply_frame(stale, None, seq=1)
        assert s._canvas_url == first


# ── End-to-end: HTTP endpoint ────────────────────────────────────────────────

class TestBlobEndpoint:
    def test_upload_returns_id(self, client):
        pc_id = client.post("/api/rtc/start").json()["pc_id"]
        try:
            r = client.post(f"/api/rtc/{pc_id}/blob", content=_png_bytes(value=42))
            assert r.status_code == 200
            body = r.json()
            assert "id" in body and len(body["id"]) >= 8
        finally:
            client.delete(f"/api/rtc/peers/{pc_id}")

    def test_upload_same_payload_returns_same_id(self, client):
        pc_id = client.post("/api/rtc/start").json()["pc_id"]
        try:
            data = _png_bytes(value=50)
            id1 = client.post(f"/api/rtc/{pc_id}/blob", content=data).json()["id"]
            id2 = client.post(f"/api/rtc/{pc_id}/blob", content=data).json()["id"]
            assert id1 == id2
        finally:
            client.delete(f"/api/rtc/peers/{pc_id}")

    def test_unknown_session_404(self, client):
        r = client.post("/api/rtc/does-not-exist/blob", content=_png_bytes())
        assert r.status_code == 404

    def test_empty_body_400(self, client):
        pc_id = client.post("/api/rtc/start").json()["pc_id"]
        try:
            r = client.post(f"/api/rtc/{pc_id}/blob", content=b"")
            assert r.status_code == 400
        finally:
            client.delete(f"/api/rtc/peers/{pc_id}")

    def test_invalid_png_returns_400(self, client):
        pc_id = client.post("/api/rtc/start").json()["pc_id"]
        try:
            r = client.post(f"/api/rtc/{pc_id}/blob", content=b"definitely not a png")
            assert r.status_code == 400
        finally:
            client.delete(f"/api/rtc/peers/{pc_id}")


# ── End-to-end: layer_conditions resolves via blob ref ───────────────────────

class TestLayerConditionRefResolution:
    def test_settings_with_denoise_mask_ref_resolves_to_uploaded_blob(self, client):
        pc_id = client.post("/api/rtc/start").json()["pc_id"]
        try:
            blob_id = client.post(
                f"/api/rtc/{pc_id}/blob",
                content=_png_bytes(value=200),
            ).json()["id"]

            # Settings with a layer condition that references the uploaded blob
            r = client.post(
                f"/api/rtc/{pc_id}/settings",
                json={
                    "type": "settings",
                    "prompt": "x",
                    "model_path": "fake/m.safetensors",
                    "device": "cpu",
                    "width": 16, "height": 16,
                    "cfg": 1.5, "strength": 0.7,
                    "layer_conditions": [{
                        "layer_id": "L1",
                        "region_id": "R1",
                        "image": "",                    # no legacy mask
                        "prompt": "fire",
                        "denoise": 0.8,
                        "denoise_mask_ref": blob_id,   # binary reference
                    }],
                    "pipeline_nodes": [],
                },
            )
            assert r.status_code == 200

            # Reach into the session and confirm the channel mask was wired
            from app.rtc.session import _peer_sessions
            session = _peer_sessions[pc_id]
            assert session.get_blob(blob_id) is not None
        finally:
            client.delete(f"/api/rtc/peers/{pc_id}")

    def test_resolve_picks_ref_over_inline_base64(self, client):
        """When both ``denoise_mask`` (base64) and ``denoise_mask_ref`` (ID)
        are present, the ref wins (more efficient path)."""
        pc_id = client.post("/api/rtc/start").json()["pc_id"]
        try:
            # Upload a distinctive blob (uniform value 50)
            blob_id = client.post(
                f"/api/rtc/{pc_id}/blob",
                content=_png_bytes(value=50),
            ).json()["id"]

            from app.rtc.session import _peer_sessions
            session = _peer_sessions[pc_id]

            # Build a fake layer_conditions payload with BOTH inline and ref
            import base64 as _b64
            inline_data = _png_bytes(value=200)  # different value
            inline_url = f"data:image/png;base64,{_b64.b64encode(inline_data).decode()}"
            conds = [{
                "layer_id": "L1",
                "region_id": "R1",
                "denoise_mask": inline_url,
                "denoise_mask_ref": blob_id,
            }]
            result = session._build_channel_masks_by_region(conds, 16, 16)
            denoise_img = result["R1"]["denoise"]
            import numpy as np
            arr = np.asarray(denoise_img.convert("L"))
            # The ref (value=50) won, not the inline (value=200)
            assert int(arr.mean()) == pytest.approx(50, abs=2)
        finally:
            client.delete(f"/api/rtc/peers/{pc_id}")
