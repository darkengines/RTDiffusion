"""
Unit tests for tagger.py — category filtering, sigmoid detection, preprocessing.

Run with:  pytest backend/tests/test_tagger.py -v

These tests exercise the real tagger logic (preprocessing, category filtering,
sigmoid detection) WITHOUT requiring a GPU or downloading models. A fake ONNX
session is injected via monkeypatching.
"""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Stub onnxruntime and huggingface_hub before importing tagger
# ---------------------------------------------------------------------------

def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]

_stub_module("onnxruntime")
_stub_module("huggingface_hub")

import app.tagger as tagger  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb(w: int = 64, h: int = 64, color=(128, 100, 50)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _make_tag_csv(rows: list[dict]) -> str:
    """Write a selected_tags.csv to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    writer = csv.DictWriter(f, fieldnames=["tag_id", "name", "category", "count"])
    writer.writeheader()
    writer.writerows(rows)
    f.close()
    return f.name


def _make_fake_session(output: np.ndarray):
    """Build a minimal ONNX session mock that returns `output`."""
    sess = MagicMock()
    sess.get_inputs.return_value = [MagicMock(name="input")]
    sess.run.return_value = [output]
    return sess


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_output_shape(self):
        img = _rgb(100, 100)
        out = tagger._preprocess(img)
        assert out.shape == (1, 448, 448, 3), f"expected (1,448,448,3) got {out.shape}"

    def test_output_dtype_float32(self):
        out = tagger._preprocess(_rgb())
        assert out.dtype == np.float32

    def test_output_range(self):
        out = tagger._preprocess(_rgb())
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_bgr_channel_order(self):
        """For a pure-red image, after BGR conversion the first channel (B) should be 0."""
        red_img = Image.new("RGB", (64, 64), (255, 0, 0))
        out = tagger._preprocess(red_img)  # shape (1, 448, 448, 3)
        # WD models use BGR: red (R=255,G=0,B=0) → BGR=(0,0,255)
        # After /255: (0.0, 0.0, 1.0) in B,G,R position
        assert out[0, 0, 0, 0] < 0.01, "Blue channel should be 0 for a pure-red input"
        assert out[0, 0, 0, 2] > 0.99, "Red channel (at BGR index 2) should be 1.0"

    def test_non_rgb_input_converted(self):
        rgba = _rgb().convert("RGBA")
        out = tagger._preprocess(rgba)
        assert out.shape == (1, 448, 448, 3)


# ---------------------------------------------------------------------------
# _crop_masked_area
# ---------------------------------------------------------------------------

class TestCropMaskedArea:
    def test_no_mask_returns_original(self):
        img = _rgb(100, 80)
        result = tagger._crop_masked_area(img, None)
        assert result is img

    def test_empty_mask_returns_original(self):
        img = _rgb(100, 80)
        mask = Image.new("L", (100, 80), 0)  # all black
        result = tagger._crop_masked_area(img, mask)
        assert result is img

    def test_full_mask_returns_full_image(self):
        img = _rgb(100, 80)
        mask = Image.new("L", (100, 80), 255)  # all white
        result = tagger._crop_masked_area(img, mask)
        assert result.size == img.size

    def test_partial_mask_crops_to_bbox(self):
        img = Image.new("RGB", (100, 100), (50, 50, 50))
        mask = Image.new("L", (100, 100), 0)
        # white square 25..74 x 25..74
        for y in range(25, 75):
            for x in range(25, 75):
                mask.putpixel((x, y), 255)
        result = tagger._crop_masked_area(img, mask)
        assert result.size == (50, 50)


# ---------------------------------------------------------------------------
# Sigmoid detection heuristic
# ---------------------------------------------------------------------------

class TestSigmoidDetection:
    """
    The tagger auto-detects whether the ONNX model outputs logits or
    probabilities and skips sigmoid if values are already in [0,1].
    """

    def _run_with_raw_scores(self, scores: np.ndarray) -> list[str]:
        """
        Inject a fake tagger cache entry with controlled output and run tag_image.
        Returns the comma-split list of returned tag strings.
        """
        tag_names = ["a_tag", "b_tag", "c_tag"]
        tag_categories = [0, 0, 0]  # all general — no filtering

        fake_csv = _make_tag_csv([
            {"tag_id": i, "name": n, "category": c, "count": 100}
            for i, (n, c) in enumerate(zip(tag_names, tag_categories))
        ])
        fake_session = _make_fake_session(scores.reshape(1, -1))

        # Patch the cache directly so no HF download happens
        old_cache = dict(tagger._tagger_cache)
        tagger._tagger_cache["_test_model"] = (fake_session, tag_names, tag_categories)
        try:
            result = tagger.tag_image(
                _rgb(),
                model_name="_test_model",
                threshold=0.5,
            )
        finally:
            tagger._tagger_cache.clear()
            tagger._tagger_cache.update(old_cache)
        return [t.strip() for t in result.split(",") if t.strip()]

    def test_probabilities_already_in_01_not_double_sigmoided(self):
        """If model outputs [0.9, 0.1, 0.8] (probs), both high ones should appear."""
        tags = self._run_with_raw_scores(np.array([0.9, 0.1, 0.8], dtype=np.float32))
        assert "a tag" in tags   # 0.9 ≥ 0.5 → should appear
        assert "c tag" in tags   # 0.8 ≥ 0.5 → should appear
        assert "b tag" not in tags  # 0.1 < 0.5 → should NOT appear

    def test_logits_get_sigmoided(self):
        """If model outputs raw logits like [5.0, -5.0, 3.0], sigmoid is applied."""
        # sigmoid(5.0) ≈ 0.993, sigmoid(-5.0) ≈ 0.007, sigmoid(3.0) ≈ 0.953
        tags = self._run_with_raw_scores(np.array([5.0, -5.0, 3.0], dtype=np.float32))
        assert "a tag" in tags   # sigmoid(5) >> 0.5
        assert "c tag" in tags   # sigmoid(3) >> 0.5
        assert "b tag" not in tags  # sigmoid(-5) << 0.5

    def test_zero_logit_gives_near_half(self):
        """sigmoid(0) = 0.5 — with threshold=0.5 it should NOT appear."""
        tags = self._run_with_raw_scores(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        # 0.0 is in range [-0.01, 1.01], so treated as probability (not logit),
        # and 0.0 < 0.5 threshold → empty
        assert tags == []


# ---------------------------------------------------------------------------
# Category filtering
# ---------------------------------------------------------------------------

class TestCategoryFiltering:
    """
    Tags with category 9 (rating), 5 (copyright), and 6 (artist) must
    be excluded from the tagger output, even if their score is very high.
    """

    def _build_cache_and_run(self, tag_rows, scores, threshold=0.3) -> str:
        tag_names = [r["name"] for r in tag_rows]
        tag_categories = [int(r["category"]) for r in tag_rows]
        scores_arr = np.array(scores, dtype=np.float32)
        fake_session = _make_fake_session(scores_arr.reshape(1, -1))

        old = dict(tagger._tagger_cache)
        tagger._tagger_cache["_cattest"] = (fake_session, tag_names, tag_categories)
        try:
            return tagger.tag_image(_rgb(), model_name="_cattest", threshold=threshold)
        finally:
            tagger._tagger_cache.clear()
            tagger._tagger_cache.update(old)

    def test_rating_tags_excluded(self):
        rows = [
            {"tag_id": 0, "name": "rating:safe",       "category": 9, "count": 1000},
            {"tag_id": 1, "name": "rating:explicit",   "category": 9, "count": 1000},
            {"tag_id": 2, "name": "cat",               "category": 0, "count": 500},
        ]
        # scores already in [0,1] range: all high
        result = self._build_cache_and_run(rows, [0.99, 0.95, 0.85], threshold=0.3)
        tags = [t.strip() for t in result.split(",") if t.strip()]
        assert "rating:safe" not in tags
        assert "rating:explicit" not in tags
        assert "cat" in tags

    def test_copyright_tags_excluded(self):
        rows = [
            {"tag_id": 0, "name": "kantai_collection", "category": 5, "count": 100},
            {"tag_id": 1, "name": "blue_sky",          "category": 0, "count": 100},
        ]
        result = self._build_cache_and_run(rows, [0.99, 0.80])
        tags = [t.strip() for t in result.split(",") if t.strip()]
        assert "kantai collection" not in tags
        assert "blue sky" in tags

    def test_artist_tags_excluded(self):
        rows = [
            {"tag_id": 0, "name": "some_artist",  "category": 6, "count": 100},
            {"tag_id": 1, "name": "green_grass",  "category": 0, "count": 100},
        ]
        result = self._build_cache_and_run(rows, [0.99, 0.70])
        tags = [t.strip() for t in result.split(",") if t.strip()]
        assert "some artist" not in tags
        assert "green grass" in tags

    def test_character_tags_included(self):
        """Category 4 (character names) should be included."""
        rows = [
            {"tag_id": 0, "name": "hatsune_miku", "category": 4, "count": 100},
            {"tag_id": 1, "name": "outdoor",      "category": 0, "count": 100},
        ]
        result = self._build_cache_and_run(rows, [0.95, 0.85])
        tags = [t.strip() for t in result.split(",") if t.strip()]
        assert "hatsune miku" in tags
        assert "outdoor" in tags

    def test_only_low_scoring_general_tags_excluded_by_threshold(self):
        rows = [
            {"tag_id": 0, "name": "sky",  "category": 0, "count": 100},
            {"tag_id": 1, "name": "tree", "category": 0, "count": 100},
        ]
        result = self._build_cache_and_run(rows, [0.8, 0.1], threshold=0.5)
        tags = [t.strip() for t in result.split(",") if t.strip()]
        assert "sky" in tags
        assert "tree" not in tags

    def test_underscore_replaced_with_space(self):
        rows = [{"tag_id": 0, "name": "blue_sky", "category": 0, "count": 100}]
        result = self._build_cache_and_run(rows, [0.9])
        assert "blue sky" in result
        assert "blue_sky" not in result

    def test_tags_sorted_by_confidence_descending(self):
        rows = [
            {"tag_id": 0, "name": "low",  "category": 0, "count": 100},
            {"tag_id": 1, "name": "high", "category": 0, "count": 100},
            {"tag_id": 2, "name": "mid",  "category": 0, "count": 100},
        ]
        result = self._build_cache_and_run(rows, [0.5, 0.9, 0.7], threshold=0.3)
        parts = [t.strip() for t in result.split(",")]
        assert parts[0] == "high"   # 0.9 first
        assert parts[1] == "mid"    # 0.7 second
        assert parts[2] == "low"    # 0.5 third

    def test_max_tags_limit_respected(self):
        rows = [{"tag_id": i, "name": f"tag_{i}", "category": 0, "count": 100}
                for i in range(50)]
        scores = np.ones(50, dtype=np.float32) * 0.9
        old = dict(tagger._tagger_cache)
        tag_names = [r["name"] for r in rows]
        tag_cats = [0] * 50
        fake_session = _make_fake_session(scores.reshape(1, -1))
        tagger._tagger_cache["_maxtest"] = (fake_session, tag_names, tag_cats)
        try:
            result = tagger.tag_image(_rgb(), model_name="_maxtest", threshold=0.5, max_tags=10)
        finally:
            tagger._tagger_cache.clear()
            tagger._tagger_cache.update(old)
        parts = [t for t in result.split(",") if t.strip()]
        assert len(parts) == 10

    def test_all_rating_returns_empty_string(self):
        """If every tag is a rating tag, result should be empty (not crash)."""
        rows = [
            {"tag_id": 0, "name": "rating:safe",     "category": 9, "count": 100},
            {"tag_id": 1, "name": "rating:general",  "category": 9, "count": 100},
        ]
        result = self._build_cache_and_run(rows, [0.99, 0.99])
        assert result == ""
