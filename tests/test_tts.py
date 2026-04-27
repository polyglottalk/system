"""
tests/test_tts.py — Verify MMS-TTS engine initialisation and synthesis.

Tests synthesise a short Hindi phrase and check that a valid WAV file
(at model.config.sampling_rate) is produced.  No reference audio is
necessary — MMS-TTS (facebook/mms-tts-hin) is a fixed-voice model.

Run:
    python -m pytest tests/test_tts.py -v

Integration tests (marked with @pytest.mark.integration) require a locally
cached HuggingFace model and are skipped in offline / CI environments unless
the POLYGLOT_TALK_RUN_INTEGRATION environment variable is set to "1".
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polyglot_talk import config  # noqa: F401

import pytest

_MODEL_ID = config.MMS_TTS_MODEL_MAP[config.TARGET_LANG]

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_SAMPLE_TEXT = "नमस्ते, यह एक परीक्षण है।"  # "Hello, this is a test."

# ---------------------------------------------------------------------------
# Helper: skip integration tests unless explicitly requested
# ---------------------------------------------------------------------------

_RUN_INTEGRATION = os.environ.get("POLYGLOT_TALK_RUN_INTEGRATION", "0") == "1"
integration = pytest.mark.skipif(
    not _RUN_INTEGRATION,
    reason=(
        "Skipped: requires a locally cached HuggingFace model. "
        "Set POLYGLOT_TALK_RUN_INTEGRATION=1 to run."
    ),
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@integration
def test_mms_tts_model_loads() -> None:
    """VitsModel.from_pretrained should load without error."""
    from transformers import VitsModel  # noqa: PLC0415

    model = VitsModel.from_pretrained(_MODEL_ID, local_files_only=True)
    assert model is not None, "VitsModel.from_pretrained returned None"
    print(f"✓ test_mms_tts_model_loads passed (model={_MODEL_ID})")


@integration
def test_mms_tts_synthesises_wav(tmp_path: Path) -> None:
    """MMS-TTS should produce a non-empty float32 WAV at model.config.sampling_rate."""
    import torch  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    from transformers import VitsModel, VitsTokenizer  # noqa: PLC0415

    tokenizer = VitsTokenizer.from_pretrained(_MODEL_ID, local_files_only=True)
    model = VitsModel.from_pretrained(_MODEL_ID, local_files_only=True)
    model.eval()

    inputs = tokenizer(_SAMPLE_TEXT, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)

    waveform = output.waveform[0].squeeze().cpu().numpy()
    sr = model.config.sampling_rate

    out_path = tmp_path / "test_synthesis.wav"
    sf.write(str(out_path), waveform.astype(np.float32), samplerate=sr)

    assert out_path.exists(), "Output WAV file was not created"
    data, file_sr = sf.read(str(out_path))
    assert file_sr == sr, f"Expected {sr} Hz, got {file_sr}"
    assert len(data) > 0, "Output WAV is empty"
    print(f"✓ test_mms_tts_synthesises_wav passed  (duration={len(data)/sr:.2f}s, sr={sr} Hz)")


@integration
def test_tts_engine_class(tmp_path: Path) -> None:
    """TTSEngine.run() must synthesise one segment and exit on sentinel."""
    import queue as _queue  # noqa: PLC0415
    from polyglot_talk.tts_engine import TTSEngine  # noqa: PLC0415
    from polyglot_talk.models import TranslatedSegment  # noqa: PLC0415

    results: dict[str, object] = {}

    tts_q: _queue.Queue = _queue.Queue()
    stop_event = threading.Event()

    engine = TTSEngine(
        tts_queue=tts_q,
        stop_event=stop_event,
        output_dir=str(tmp_path),
    )

    def _run() -> None:
        try:
            engine.run()
            results["ok"] = True
        except Exception as exc:  # pylint: disable=broad-except
            results["error"] = str(exc)

    t = threading.Thread(target=_run, name="TTSEngineTest", daemon=True)
    t.start()

    # Synchronise on the explicit readiness signal instead of an arbitrary sleep
    _MODEL_LOAD_TIMEOUT = 120  # seconds — generous for first-run GPU transfer
    engine._model_ready.wait(timeout=_MODEL_LOAD_TIMEOUT)

    if engine._startup_failed.is_set():
        pytest.fail("TTSEngine model loading failed — see log for details")

    seg = TranslatedSegment(chunk_id=42, text=_SAMPLE_TEXT)
    tts_q.put(seg)
    tts_q.put(None)  # sentinel

    # Wait for synthesis to complete
    t.join(timeout=60)
    assert not t.is_alive(), "TTSEngine thread did not exit within 60 s"

    if "error" in results:
        pytest.fail(f"TTSEngine.run() raised: {results['error']}")
    assert results.get("ok"), "TTSEngine did not signal success"

    out_wav = tmp_path / "chunk_0042.wav"
    assert out_wav.exists(), f"Expected output WAV not found: {out_wav}"
    print(f"✓ test_tts_engine_class passed  (output={out_wav})")


def test_model_map_completeness() -> None:
    """MMS_TTS_MODEL_MAP must have ≥6 entries; every key must have an MT backend."""
    assert len(config.MMS_TTS_MODEL_MAP) >= 6
    all_mt_langs = set(config.ARGOS_LANG_MAP) | set(config.MARIANMT_MODEL_MAP) | set(config.NLLB_LANG_MAP)
    for lang_code in config.MMS_TTS_MODEL_MAP:
        assert lang_code in all_mt_langs, (
            f"MMS_TTS_MODEL_MAP key {lang_code!r} missing from all MT backend maps"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
