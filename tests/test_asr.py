"""
tests/test_asr.py — WER-based ASR accuracy evaluation on LibriSpeech dev-clean.

Uses the locally installed LibriSpeech dev-clean dataset to evaluate
faster-whisper transcription accuracy (Word Error Rate) and per-sample latency.

Dataset layout expected at:
    data/dev-clean/LibriSpeech/dev-clean/<speaker>/<chapter>/<utterance>.flac
    data/dev-clean/LibriSpeech/dev-clean/<speaker>/<chapter>/<spk>-<ch>.trans.txt

Transcript format (LibriSpeech standard):
    <utterance_id> UPPERCASE REFERENCE TEXT WITHOUT PUNCTUATION

Thresholds (issue #2):
    faster-whisper base.en  → WER < 10 % on LibriSpeech dev-clean

Results are saved to benchmarks/asr_wer_results.csv for the PCL paper.

Run:
    python -m pytest tests/test_asr.py -v
    python -m pytest tests/test_asr.py -v -s   # shows per-sample output
    python tests/test_asr.py                   # standalone
"""

from __future__ import annotations

import csv
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polyglot_talk import config  # noqa: F401 — must set os.environ before faster_whisper import

import numpy as np
import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
LIBRISPEECH_ROOT = _REPO_ROOT / "data" / "dev-clean" / "LibriSpeech" / "dev-clean"
BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"
RESULTS_CSV = BENCHMARKS_DIR / "asr_wer_results.csv"

# ── Evaluation parameters ─────────────────────────────────────────────────────
WER_THRESHOLD = 0.10   # 10 % — issue #2 requirement for base.en on dev-clean
MAX_SAMPLES = 100      # cap to keep the test fast on CPU


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for WER comparison.

    LibriSpeech references are already uppercase without punctuation;
    Whisper output is mixed-case with punctuation — normalize both sides.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text.strip()


def _iter_samples(root: Path, max_n: int) -> Iterator[tuple[Path, str]]:
    """Yield (flac_path, reference_text) for up to *max_n* utterances.

    Traverses all *.trans.txt files under *root* in sorted order so results
    are deterministic across runs.
    """
    count = 0
    for trans_file in sorted(root.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                utterance_id, *words = line.split()
                reference = " ".join(words)
                flac_path = chapter_dir / f"{utterance_id}.flac"
                if flac_path.exists():
                    yield flac_path, reference
                    count += 1
                    if count >= max_n:
                        return


def _load_flac_as_float32(path: Path) -> np.ndarray:
    """Load a FLAC file as a float32 mono array at config.SAMPLE_RATE Hz."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)  # type: ignore[misc]
    if audio.ndim > 1:
        audio = audio.mean(axis=1)          # stereo → mono
        
    if sr != config.SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(config.SAMPLE_RATE, sr)
        audio = resample_poly(audio, config.SAMPLE_RATE // g, sr // g)

    return audio.astype(np.float32)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def asr_model():
    """Load WhisperModel once for the entire test module."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.ASR_MODEL_SIZE,
        device=config.ASR_DEVICE,
        compute_type=config.ASR_COMPUTE_TYPE,
    )


@pytest.fixture(scope="module")
def librispeech_available():
    """Skip any test that needs the dataset or optional deps if absent."""
    if not LIBRISPEECH_ROOT.exists():
        pytest.skip(
            f"LibriSpeech dev-clean not found at {LIBRISPEECH_ROOT}. "
            "Download and extract the dataset under data/dev-clean/ at the repo root."
        )
    missing = []
    for pkg in ("soundfile", "jiwer"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        pytest.skip(f"Missing dependencies: {', '.join(missing)}. Run: uv pip install {' '.join(missing)}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_transcribe_silence(asr_model) -> None:
    """Silent audio must not crash; any string output is acceptable."""
    silence = np.zeros(config.BLOCK_SIZE, dtype=np.float32)
    segments_gen, _ = asr_model.transcribe(
        silence,
        beam_size=config.ASR_BEAM_SIZE,
        language=config.ASR_LANGUAGE,
        vad_filter=False,
    )
    text = "".join(seg.text for seg in segments_gen).strip()
    assert isinstance(text, str)
    print(f"\nSilence transcript: {text!r}")
    print("✓ test_transcribe_silence passed")


def test_asr_engine_integration() -> None:
    """ASREngine._transcribe() returns a str on silent input without error."""
    from polyglot_talk.asr_engine import ASREngine

    engine = ASREngine(
        audio_queue=queue.Queue(),
        text_queue=queue.Queue(),
        stop_event=threading.Event(),
    )
    
    result = engine._transcribe(np.zeros(config.BLOCK_SIZE, dtype=np.float32))
    assert isinstance(result, str)
    print(f"\nASREngine._transcribe(silence) = {result!r}")
    print("✓ test_asr_engine_integration passed")


def test_wer_librispeech_devclean(asr_model, librispeech_available) -> None:
    """Evaluate faster-whisper on LibriSpeech dev-clean and assert WER < 10 %.

    Runs up to MAX_SAMPLES utterances, computes per-sample WER and latency,
    saves results to benchmarks/asr_wer_results.csv, then asserts that the
    overall (corpus-level) WER is below WER_THRESHOLD.

    This directly addresses GitHub issue #2.
    """
    import jiwer

    samples = list(_iter_samples(LIBRISPEECH_ROOT, MAX_SAMPLES))
    assert samples, f"No FLAC samples found under {LIBRISPEECH_ROOT}"

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    all_refs: list[str] = []
    all_hyps: list[str] = []

    print(f"\nEvaluating {len(samples)} utterances from LibriSpeech dev-clean…")
    print(f"  Model : {config.ASR_MODEL_SIZE}  ({config.ASR_COMPUTE_TYPE}, {config.ASR_DEVICE})")
    print(f"  WER threshold : {WER_THRESHOLD * 100:.0f}%")
    print()

    for i, (flac_path, reference) in enumerate(samples, 1):
        audio = _load_flac_as_float32(flac_path)

        t0 = time.perf_counter()
        segments_gen, _ = asr_model.transcribe(
            audio,
            beam_size=config.ASR_BEAM_SIZE,
            language=config.ASR_LANGUAGE,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        hypothesis = "".join(seg.text for seg in segments_gen).strip()
        latency = time.perf_counter() - t0

        ref_norm = _normalize(reference)
        hyp_norm = _normalize(hypothesis)

        # Per-sample WER (guard against empty reference)
        sample_wer = jiwer.wer(ref_norm, hyp_norm) if ref_norm else 0.0

        all_refs.append(ref_norm)
        all_hyps.append(hyp_norm)
        rows.append({
            "utterance_id": flac_path.stem,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(sample_wer, 4),
            "latency_s": round(latency, 4),
        })

        print(
            f"  [{i:>3}/{len(samples)}] WER={sample_wer*100:5.1f}%  "
            f"lat={latency:.2f}s  {flac_path.stem}"
        )

    # ── Corpus-level WER (counts errors across all words, not avg of per-sample) ──
    overall_wer = jiwer.wer(all_refs, all_hyps)
    avg_latency = sum(r["latency_s"] for r in rows) / len(rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["utterance_id", "reference", "hypothesis", "wer", "latency_s"]
        )
        writer.writeheader()
        writer.writerows(rows)
        # Summary footer — readable by both humans and pandas
        f.write(
            f"\n# summary: model={config.ASR_MODEL_SIZE}, n={len(rows)}, "
            f"overall_wer={overall_wer:.4f}, avg_latency_s={avg_latency:.4f}, "
            f"threshold={WER_THRESHOLD}\n"
        )

    print()
    print("=" * 60)
    print(f"  Model         : {config.ASR_MODEL_SIZE}")
    print(f"  Samples       : {len(rows)}")
    print(f"  Overall WER   : {overall_wer * 100:.2f}%  (threshold: {WER_THRESHOLD * 100:.0f}%)")
    print(f"  Avg latency   : {avg_latency:.3f}s per utterance")
    print(f"  Results saved : {RESULTS_CSV}")
    print("=" * 60)

    assert overall_wer < WER_THRESHOLD, (
        f"WER {overall_wer * 100:.2f}% exceeds the {WER_THRESHOLD * 100:.0f}% threshold "
        f"for model '{config.ASR_MODEL_SIZE}' on LibriSpeech dev-clean "
        f"({len(rows)} samples). See {RESULTS_CSV} for details."
    )
    print("✓ test_wer_librispeech_devclean passed")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
