"""Regression test for the multi-GPU device mismatch bug.

Run:  pytest backend/tests/test_safe_encode_prompt.py -v

The bug
-------
``_patch_pipe_for_none_tokenizers`` installs a wrapper that forwards
``encode_prompt`` to the original class method. The wrapper used POSITIONAL
arguments matching the SD1.5 signature ``(self, prompt, device, num_images,
do_cfg, negative_prompt)``. But the same patch is applied to SDXL pipelines
whose signature is ``(self, prompt, prompt_2, device, num_images, ...)``.
The positional shift made:

  • our ``device`` end up in the SDXL ``prompt_2`` slot (silently ignored)
  • our ``num_images_per_prompt=1`` end up in the SDXL ``device`` slot
  • ``torch.device(1)`` == ``cuda:1``

On a multi-GPU host this manifested as a tensor-device mismatch the moment
the SDXL pipeline did ``text_input_ids.to(device)`` — the index landed on
cuda:1 while the encoder weights stayed on cuda:0.

This test reproduces the failure on the OLD wrapper and verifies the fix
(keyword-argument forwarding) routes every argument to its intended slot.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock



# Stub torch/diffusers so importing manager doesn't drag in CUDA.
def _ensure_stub(name: str):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)


_ensure_stub("torch")
_ensure_stub("diffusers")
_ensure_stub("transformers")
# torch.compile / torch.cuda / torch.backends shims used at module import time
_t = sys.modules["torch"]
if not hasattr(_t, "compile"):
    _t.compile = lambda *a, **kw: a[0] if a else None
if not hasattr(_t, "backends"):
    _t.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
                                         cudnn=types.SimpleNamespace(allow_tf32=False))


# ── Helpers ────────────────────────────────────────────────────────────────

class _FakeSDXLEncodePrompt:
    """Mimics the SDXL ``encode_prompt`` positional signature, recording calls."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(
        self,
        instance,
        prompt,
        prompt_2=None,
        device=None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt=None,
        negative_prompt_2=None,
        **kwargs,
    ):
        # Record what each parameter actually received — the wrapper test
        # asserts on these values to confirm positional shift never occurs.
        self.calls.append({
            "prompt": prompt,
            "prompt_2": prompt_2,
            "device": device,
            "num_images_per_prompt": num_images_per_prompt,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "negative_prompt": negative_prompt,
            "negative_prompt_2": negative_prompt_2,
            "kwargs": kwargs,
        })
        return ("EMBEDS", "NEG", "POOLED", "NEG_POOLED")


class _FakeSDXLPipe:
    """Minimal stand-in for an SDXL pipeline missing tokenizer_2 (triggers patch)."""

    encode_prompt = _FakeSDXLEncodePrompt()

    def __init__(self):
        # tokenizer present, tokenizer_2 missing → patch will fire
        self.tokenizer = MagicMock()
        self.tokenizer_2 = None
        self.text_encoder = MagicMock()
        self.text_encoder_2 = None
        # Reset call log per instance via class-level mock
        type(self).encode_prompt = _FakeSDXLEncodePrompt()


# ── Test the patch ─────────────────────────────────────────────────────────

class TestPatchedEncodePromptNoShift:
    """Verify the wrapper forwards by keyword, not by position."""

    def _patch(self, pipe):
        from app.stream.manager import _patch_pipe_for_none_tokenizers
        _patch_pipe_for_none_tokenizers(pipe)

    def test_device_lands_in_device_slot_not_prompt_2(self):
        pipe = _FakeSDXLPipe()
        original_call_recorder = type(pipe).encode_prompt
        self._patch(pipe)

        # Simulate the call pattern used by _encode_prompt in helpers.py
        result = pipe.encode_prompt(
            prompt="a cat",
            device="cuda:0",   # a *torch.device-like* object — string here for clarity
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
        )
        assert result == ("EMBEDS", "NEG", "POOLED", "NEG_POOLED")
        recorded = original_call_recorder.calls[0]
        assert recorded["prompt"] == "a cat"
        # CRITICAL: device must land in the device slot, not prompt_2
        assert recorded["device"] == "cuda:0"
        assert recorded["prompt_2"] is None
        # num_images_per_prompt must NOT shift into device
        assert recorded["num_images_per_prompt"] == 1
        assert recorded["device"] != 1  # would be cuda:1 if positional shift happened

    def test_num_images_default_does_not_pollute_device(self):
        pipe = _FakeSDXLPipe()
        original_call_recorder = type(pipe).encode_prompt
        self._patch(pipe)
        # Caller omits num_images_per_prompt → wrapper uses its default 1
        pipe.encode_prompt(prompt="x", device="cuda:0")
        recorded = original_call_recorder.calls[0]
        assert recorded["device"] == "cuda:0"
        assert recorded["num_images_per_prompt"] == 1
        # Sanity: prompt_2 stays at its default
        assert recorded["prompt_2"] is None

    def test_kwargs_pass_through(self):
        pipe = _FakeSDXLPipe()
        original_call_recorder = type(pipe).encode_prompt
        self._patch(pipe)
        pipe.encode_prompt(
            prompt="x",
            device="cuda:0",
            num_images_per_prompt=2,
            do_classifier_free_guidance=True,
            negative_prompt="blurry",
            negative_prompt_2="extra",   # SDXL-only kwarg goes via **kwargs
        )
        recorded = original_call_recorder.calls[0]
        assert recorded["num_images_per_prompt"] == 2
        assert recorded["negative_prompt"] == "blurry"
        # SDXL-only kwarg routes to its named slot (or **kwargs on older diffusers)
        assert recorded["negative_prompt_2"] == "extra"


class TestPatchSkippedWhenNotNeeded:
    """Patch must NOT fire when tokenizer_2 is already present."""

    def test_well_formed_sdxl_pipe_skipped(self):
        pipe = MagicMock()
        pipe.tokenizer = MagicMock()
        pipe.tokenizer_2 = MagicMock()  # both present → no patch
        pipe.text_encoder_2 = MagicMock()
        original = pipe.encode_prompt

        from app.stream.manager import _patch_pipe_for_none_tokenizers
        _patch_pipe_for_none_tokenizers(pipe)
        # encode_prompt was NOT replaced (MagicMock attribute identity check)
        assert pipe.encode_prompt is original

    def test_no_tokenizer_at_all_skipped(self):
        pipe = MagicMock()
        pipe.tokenizer = None
        pipe.tokenizer_2 = None
        pipe.text_encoder_2 = None
        original = pipe.encode_prompt
        from app.stream.manager import _patch_pipe_for_none_tokenizers
        _patch_pipe_for_none_tokenizers(pipe)
        assert pipe.encode_prompt is original


class TestPatchNotAppliedToSD15:
    """Patch must NOT be applied to SD 1.5 pipes.

    SD 1.5 pipelines (e.g. ``StableDiffusionImg2ImgPipeline``) declare neither
    ``tokenizer_2`` nor ``text_encoder_2``. The patch's swap (``self.tokenizer =
    None``) would otherwise corrupt the pipe's single-encoder code path and
    cause an ``AttributeError: 'NoneType' object has no attribute 'tokenize'``
    inside ``maybe_convert_prompt``.
    """

    class _SD15Pipe:
        """Realistic SD 1.5 stand-in — has ``tokenizer`` but no ``_2`` attrs."""

        def __init__(self):
            self.tokenizer = MagicMock()
            self.text_encoder = MagicMock()
            # NOTE: no tokenizer_2 or text_encoder_2 attributes at all

        def encode_prompt(self, *args, **kwargs):
            return ("EMBEDS", None, None, None)

    def test_sd15_pipe_not_patched(self):
        pipe = self._SD15Pipe()

        from app.stream.manager import _patch_pipe_for_none_tokenizers
        _patch_pipe_for_none_tokenizers(pipe)

        # No swap should have happened on the SD 1.5 pipe — encode_prompt
        # remains the original bound method, tokenizer remains intact.
        assert pipe.tokenizer is not None
        assert not hasattr(pipe, "tokenizer_2")
        # Calling encode_prompt does NOT crash (tokenizer is intact)
        result = pipe.encode_prompt(prompt="x", device="cuda:0")
        assert result == ("EMBEDS", None, None, None)

    def test_sd15_tokenizer_not_nulled(self):
        """Regression: patch was setting tokenizer=None on SD 1.5 pipes, crashing
        ``maybe_convert_prompt(prompt, self.tokenizer)`` on the next encode."""
        pipe = self._SD15Pipe()
        tokenizer_before = pipe.tokenizer

        from app.stream.manager import _patch_pipe_for_none_tokenizers
        _patch_pipe_for_none_tokenizers(pipe)

        assert pipe.tokenizer is tokenizer_before


class TestEncodePromptOnRealisticSD15Pipe:
    """End-to-end-ish test using a pipe whose ``encode_prompt`` mirrors the
    diffusers SD 1.5 signature. Catches the previous crash where the patch
    nulled ``self.tokenizer`` and the diffusers code then did
    ``self.maybe_convert_prompt(prompt, self.tokenizer)``."""

    class _RealisticSD15Pipe:
        def __init__(self):
            # Tokenizer behaves like a callable + has .tokenize()
            self.tokenizer = MagicMock()
            self.tokenizer.tokenize = MagicMock(return_value=["tok"])
            self.text_encoder = MagicMock()

        def maybe_convert_prompt(self, prompt, tokenizer):
            # Mirror the real call: would crash if tokenizer is None
            tokenizer.tokenize(prompt)
            return prompt

        def encode_prompt(self, prompt, device=None, **_kwargs):
            # SD 1.5 signature: maybe_convert_prompt is called first
            self.maybe_convert_prompt(prompt, self.tokenizer)
            return ("EMBEDS", None, None, None)

    def test_full_call_does_not_crash_on_sd15(self):
        pipe = self._RealisticSD15Pipe()
        from app.stream.manager import _patch_pipe_for_none_tokenizers

        # Apply patch — must be a no-op for SD 1.5
        _patch_pipe_for_none_tokenizers(pipe)

        # The real diffusers code does: maybe_convert_prompt(prompt, self.tokenizer)
        # If the patch had wrongly set self.tokenizer=None, .tokenize() would
        # crash with AttributeError. The fix lets this succeed.
        result = pipe.encode_prompt(prompt="a dog", device="cuda:0")
        assert result == ("EMBEDS", None, None, None)
        # Verify tokenize was actually called (proves we went through the real path)
        pipe.tokenizer.tokenize.assert_called_with("a dog")
