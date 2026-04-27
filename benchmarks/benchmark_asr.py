"""
benchmark_asr.py — ASR model comparison benchmark.

Compares faster-whisper model sizes (tiny.en, base.en, small.en) on
synthesized test clips, measuring Word Error Rate (WER) and latency.

Usage
-----
    python benchmarks/benchmark_asr.py

Output
------
    results/asr_results.csv
"""

from __future__ import annotations

import os
import random
import sys
import time

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from polyglot_talk import config  # noqa: E402  — must be first project import

import numpy as np  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

# benchmarks/ is already the cwd-equivalent when running as a package
sys.path.insert(0, os.path.dirname(__file__))
import system_meta  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
LIBRISPEECH_DIR = os.path.join(PROJECT_ROOT, "data", "dev-clean", "LibriSpeech", "dev-clean")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "asr")
# Output CSV paths are computed per-machine inside run_benchmark().

MODEL_SIZES = ["tiny.en", "base.en", "small.en"]

# Maximum number of clips sampled per model run.
# A fresh random seed is drawn for each model so results are statistically
# independent; the seed is stored in the CSV metadata for reproducibility.
MAX_CLIPS: int = 500


# ── WER calculation ─────────────────────────────────────────────────────────

import re as _re

def _normalize(text: str) -> list[str]:
    """Lowercase and strip all punctuation, returning a word token list.

    Standard ASR WER normalization (NIST sclite convention): remove every
    character that is not a letter, digit, apostrophe-within-word, or
    whitespace so that "captors." and "captors" are treated identically.
    Contractions such as "don't" are kept intact.
    """
    # Lower-case first
    text = text.lower()
    # Remove sentence-boundary and other punctuation, but keep intra-word
    # apostrophes (e.g. "don't" → "don't", not "dont")
    text = _re.sub(r"[^\w\s']", "", text)          # strip all non-word except '
    text = _re.sub(r"(?<!\w)'|'(?!\w)", "", text)  # strip leading/trailing '
    return text.split()


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER using Levenshtein edit distance on normalized word sequences.

    Both reference and hypothesis are normalized (lowercased, punctuation
    stripped) before comparison so that capitalisation and trailing periods
    emitted by Whisper do not inflate the error rate.
    """
    ref_words = _normalize(reference)
    hyp_words = _normalize(hypothesis)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Standard DP edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_librispeech() -> list[tuple[str, str]]:
    """Return list of (flac_path, reference_text) from LibriSpeech dev-clean.

    Walks LIBRISPEECH_DIR, reads every ``*.trans.txt`` file, and maps each
    ``UTTID TEXT`` line to its corresponding .flac file on disk.
    Reference text is lowercased from the original ALL-CAPS transcripts.
    """
    import pathlib

    entries: list[tuple[str, str]] = []
    for trans_file in sorted(pathlib.Path(LIBRISPEECH_DIR).rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uttid, text = line.split(" ", 1)
                flac_path = str(chapter_dir / f"{uttid}.flac")
                if os.path.exists(flac_path):
                    entries.append((flac_path, text.lower()))
    return entries


def _load_audio(audio_path: str) -> np.ndarray:
    """Load a FLAC or WAV file as float32 numpy array at 16 kHz mono."""
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)

    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16 kHz if needed (LibriSpeech is already 16 kHz)
    if sr != 16000:
        from scipy.signal import resample
        num_samples = int(len(data) * 16000 / sr)
        data = resample(data, num_samples).astype(np.float32) # type: ignore

    return data


# ── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark() -> None:
    machine_dir = os.path.join(RESULTS_DIR, system_meta.machine_slug())
    os.makedirs(machine_dir, exist_ok=True)
    results_csv = os.path.join(machine_dir, "asr_results.csv")

    all_entries = _load_librispeech()
    print(f"Discovered {len(all_entries)} clips in LibriSpeech dev-clean.")
    print(f"Will randomly sample {MAX_CLIPS} clips per model.\n")

    if not all_entries:
        print(f"ERROR: No FLAC files found under {LIBRISPEECH_DIR}")
        sys.exit(1)

    # Collect system / config snapshot once (before any model load)
    meta = system_meta.collect()
    meta["dataset"] = "LibriSpeech dev-clean"
    meta["dataset_total_clips"] = str(len(all_entries))
    meta["max_clips_per_model"] = str(MAX_CLIPS)

    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_size in MODEL_SIZES:
        print(f"{'=' * 60}")
        print(f"  Model: {model_size}")
        print(f"{'=' * 60}")

        # ── Independent random sample for this model ──────────────────────
        seed = random.randrange(2 ** 32)
        rng = random.Random(seed)
        sample = rng.sample(all_entries, min(MAX_CLIPS, len(all_entries)))
        print(f"  Random seed: {seed}  |  Clips sampled: {len(sample)}")

        # Load audio only for the sampled clips
        print(f"  Loading {len(sample)} audio files...")
        audio_data: dict[str, np.ndarray] = {}
        for flac_path, _text in sample:
            audio_data[flac_path] = _load_audio(flac_path)

        # Load model
        t0 = time.perf_counter()
        model = WhisperModel(
            model_size,
            device=config.ASR_DEVICE,
            compute_type=config.ASR_COMPUTE_TYPE,
        )
        load_time = time.perf_counter() - t0
        print(f"  Model loaded in {load_time:.1f}s\n")

        wers = []
        latencies = []

        for flac_path, ground_truth in sample:
            if flac_path not in audio_data:
                continue

            audio = audio_data[flac_path]
            uttid = os.path.splitext(os.path.basename(flac_path))[0]

            # Transcribe
            t0 = time.perf_counter()
            segments_gen, _info = model.transcribe(
                audio,
                beam_size=config.ASR_BEAM_SIZE,
                language=config.ASR_LANGUAGE,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            hypothesis = "".join(seg.text for seg in segments_gen).strip()
            latency = time.perf_counter() - t0

            wer = _word_error_rate(ground_truth, hypothesis)
            wers.append(wer)
            latencies.append(latency)

            all_rows.append({
                "model": model_size,
                "clip": uttid,
                "ground_truth": ground_truth,
                "hypothesis": hypothesis,
                "wer": f"{wer:.4f}",
                "latency_s": f"{latency:.4f}",
            })

            print(f"  {uttid}: WER={wer:.2%}  latency={latency:.3f}s")
            print(f"    REF: {ground_truth}")
            print(f"    HYP: {hypothesis}")

        # Model summary
        avg_wer = np.mean(wers) if wers else 0
        avg_lat = np.mean(latencies) if latencies else 0
        std_lat = np.std(latencies) if latencies else 0

        summary_rows.append({
            "model": model_size,
            "num_clips": len(wers),
            "random_seed": seed,
            "avg_wer": f"{avg_wer:.4f}",
            "avg_latency_s": f"{avg_lat:.4f}",
            "std_latency_s": f"{std_lat:.4f}",
        })

        print(f"\n  ── Summary for {model_size} ──")
        print(f"  Average WER:     {avg_wer:.2%}")
        print(f"  Average latency: {avg_lat:.3f}s \u00b1 {std_lat:.3f}s")
        print()

        # Release model memory and audio cache
        del model
        del audio_data

    # ── Write CSV (metadata in .meta.json sidecar) ──────────────────────────────────────
    system_meta.write_csv(
        results_csv,
        fieldnames=["model", "clip", "ground_truth", "hypothesis", "wer", "latency_s"],
        rows=all_rows,
        meta=meta,
    )
    print(f"✓ Detailed results saved to {results_csv}")
    print(f"  (metadata sidecar: {results_csv.replace('.csv', '.meta.json')})")

    # Summary CSV
    summary_csv = os.path.join(machine_dir, "asr_summary.csv")
    system_meta.write_csv(
        summary_csv,
        fieldnames=["model", "num_clips", "random_seed", "avg_wer", "avg_latency_s", "std_latency_s"],
        rows=summary_rows,
        meta=meta,
    )
    print(f"✓ Summary saved to {summary_csv}")

    # Print paper-ready table
    print(f"\n{'=' * 60}")
    print("  ASR Results Table (for Paper Section 5)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<12} {'Avg WER':>10} {'Avg Latency':>14} {'Std Latency':>14}")
    print(f"  {'─' * 12} {'─' * 10} {'─' * 14} {'─' * 14}")
    for row in summary_rows:
        print(f"  {row['model']:<12} {float(row['avg_wer']):>9.2%} {row['avg_latency_s']:>13}s {row['std_latency_s']:>13}s")


if __name__ == "__main__":
    run_benchmark()
