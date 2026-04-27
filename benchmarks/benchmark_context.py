"""
benchmark_context.py — Context continuity validation benchmark.

Compares translation quality with and without the rolling context window
by running a 10-sentence scripted conversation through the Translator.

Metrics
-------
- Repetitions: consecutive outputs sharing >60% of their words
- Grammar breaks: outputs that are unusually short (<3 chars) or identical
  to the English input (failed translation)

Usage
-----
    python benchmarks/benchmark_context.py

Output
------
    results/context_results.csv
"""

# SCOPE NOTE: This benchmark measures translation-layer repetition only.
# Inputs are clean, non-overlapping sentences fed directly to the Translator.
# ASR-layer overlap deduplication (handled by ASREngine.deduplicate_overlap) is
# out of scope here and is tested separately via the live pipeline in main.py.

from __future__ import annotations

import collections
import difflib
import os
import sys
import time
from typing import Deque

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from polyglot_talk import config  # noqa: E402

import argostranslate.package  # noqa: E402
import argostranslate.translate  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import system_meta  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
LIBRISPEECH_DIR = os.path.join(PROJECT_ROOT, "data", "dev-clean", "LibriSpeech", "dev-clean")
# Number of consecutive sentences drawn from one chapter for context testing
CONTEXT_CLIP_COUNT: int = 10
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "context")
# Output CSV paths are computed per-machine inside run_benchmark().


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_conversation() -> list[str]:
    """Load CONTEXT_CLIP_COUNT consecutive sentences from the first LibriSpeech chapter.

    Sentences are drawn from the first ``*.trans.txt`` file found under
    LIBRISPEECH_DIR (sorted).  ALL-CAPS LibriSpeech text is title-cased for
    more natural English input to the translator.
    """
    import pathlib

    for trans_file in sorted(pathlib.Path(LIBRISPEECH_DIR).rglob("*.trans.txt")):
        sentences: list[str] = []
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                _uttid, text = line.split(" ", 1)
                sentences.append(text.title())
                if len(sentences) >= CONTEXT_CLIP_COUNT:
                    break
        if sentences:
            return sentences

    raise FileNotFoundError(
        f"No *.trans.txt files found under {LIBRISPEECH_DIR}"
    )


def _translate(text: str) -> str:
    """Single Argos translation en→hi."""
    return argostranslate.translate.translate(text, "en", "hi")


def _word_overlap(a: str, b: str) -> float:
    """Fraction of words in common between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def _is_repetition(prev_output: str, curr_output: str) -> bool:
    """Check if current output is too similar to previous (>60% word overlap)."""
    # Translation-layer repetition only. ASR overlap dedup is upstream in asr_engine.py.
    return _word_overlap(prev_output, curr_output) > 0.60


def _is_grammar_break(source: str, output: str) -> bool:
    """Check for grammar break indicators."""
    # Output too short
    if len(output.strip()) < 3:
        return True
    # Output identical to source (translation did nothing useful)
    if output.strip().lower() == source.strip().lower():
        return True
    # Output is just punctuation
    if not any(c.isalnum() for c in output):
        return True
    return False


# ── Context-aware translation (mirrors translator.py logic) ─────────────────

def _translate_with_context(
    new_text: str,
    context_source: Deque[str],
    context_translated: Deque[str],
) -> str:
    """Translate with rolling context prefix — same logic as Translator class."""
    prefix_source = " ".join(context_source).strip()
    prefix_translated = " ".join(context_translated).strip()

    if prefix_source:
        combined_input = f"{prefix_source} {new_text}"
    else:
        combined_input = new_text

    full_translation = _translate(combined_input)

    if prefix_translated:
        trimmed = _trim_prefix(full_translation, prefix_translated)
    else:
        trimmed = full_translation

    context_source.append(new_text)
    result = trimmed.strip() if trimmed.strip() else full_translation.strip()
    context_translated.append(result)
    return result


def _trim_prefix(full: str, prefix_tr: str) -> str:
    """Remove translated prefix — mirrors Translator._trim_prefix()."""
    if not prefix_tr:
        return full

    if full.startswith(prefix_tr):
        return full[len(prefix_tr):].strip()

    matcher = difflib.SequenceMatcher(None, full, prefix_tr, autojunk=False)
    blocks = matcher.get_matching_blocks()
    trimmed_end = 0
    for block in blocks:
        if block.a == trimmed_end and block.b == 0:
            trimmed_end = block.a + block.size
        else:
            break

    overlap_ratio = trimmed_end / max(len(prefix_tr), 1)
    if trimmed_end > 0 and overlap_ratio >= 0.30:
        return full[trimmed_end:].strip()

    return full


# ── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark() -> None:
    machine_dir = os.path.join(RESULTS_DIR, system_meta.machine_slug())
    os.makedirs(machine_dir, exist_ok=True)
    results_csv = os.path.join(machine_dir, "context_results.csv")

    sentences = _load_conversation()
    print(f"Loaded {len(sentences)} conversation sentences.\n")

    # Collect system / config snapshot
    meta = system_meta.collect()
    meta["dataset"] = "LibriSpeech dev-clean (first chapter)"
    meta["num_sentences"] = str(len(sentences))
    meta["context_maxlen"] = str(config.CONTEXT_MAXLEN)

    # Verify Argos model
    installed = argostranslate.package.get_installed_packages()
    if not any(p.from_code == "en" and p.to_code == "hi" for p in installed):
        print("ERROR: Argos en→hi package not installed. Run setup_models.py.")
        sys.exit(1)

    # ── Run WITH context ─────────────────────────────────────────────────────
    print(f"{'=' * 60}")
    print("  Condition: WITH context window")
    print(f"{'=' * 60}")

    context_source: Deque[str] = collections.deque(maxlen=config.CONTEXT_MAXLEN)
    context_translated: Deque[str] = collections.deque(maxlen=config.CONTEXT_MAXLEN)

    outputs_with: list[str] = []
    latencies_with: list[float] = []

    for idx, sentence in enumerate(sentences, 1):
        t0 = time.perf_counter()
        output = _translate_with_context(sentence, context_source, context_translated)
        latency = time.perf_counter() - t0
        outputs_with.append(output)
        latencies_with.append(latency)
        print(f"  [{idx:>2}] EN:  {sentence}")
        print(f"       HI:  {output}  ({latency:.3f}s)")

    # ── Run WITHOUT context ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Condition: WITHOUT context window")
    print(f"{'=' * 60}")

    outputs_without: list[str] = []
    latencies_without: list[float] = []

    for idx, sentence in enumerate(sentences, 1):
        t0 = time.perf_counter()
        output = _translate(sentence)
        latency = time.perf_counter() - t0
        outputs_without.append(output)
        latencies_without.append(latency)
        print(f"  [{idx:>2}] EN:  {sentence}")
        print(f"       HI:  {output}  ({latency:.3f}s)")

    # ── Count repetitions and grammar breaks ─────────────────────────────────
    reps_with = 0
    reps_without = 0
    breaks_with = 0
    breaks_without = 0

    for i in range(len(sentences)):
        # Grammar breaks
        if _is_grammar_break(sentences[i], outputs_with[i]):
            breaks_with += 1
        if _is_grammar_break(sentences[i], outputs_without[i]):
            breaks_without += 1

        # Repetitions (check against previous output)
        if i > 0:
            if _is_repetition(outputs_with[i - 1], outputs_with[i]):
                reps_with += 1
            if _is_repetition(outputs_without[i - 1], outputs_without[i]):
                reps_without += 1

    # ── Results ──────────────────────────────────────────────────────────────
    import numpy as np

    results = {
        "repetitions_with_context": reps_with,
        "repetitions_without_context": reps_without,
        "grammar_breaks_with_context": breaks_with,
        "grammar_breaks_without_context": breaks_without,
        "avg_latency_with_context": f"{np.mean(latencies_with):.4f}",
        "avg_latency_without_context": f"{np.mean(latencies_without):.4f}",
    }

    # Write per-sentence detail CSV
    detail_csv = os.path.join(machine_dir, "context_detail.csv")
    detail_rows = []
    for i in range(len(sentences)):
        is_rep_with = _is_repetition(outputs_with[i-1], outputs_with[i]) if i > 0 else False
        is_rep_without = _is_repetition(outputs_without[i-1], outputs_without[i]) if i > 0 else False
        detail_rows.append({
            "sentence_id": i + 1,
            "source": sentences[i],
            "output_with_context": outputs_with[i],
            "output_without_context": outputs_without[i],
            "latency_with_s": f"{latencies_with[i]:.4f}",
            "latency_without_s": f"{latencies_without[i]:.4f}",
            "translation_repetition_with": is_rep_with,
            "translation_repetition_without": is_rep_without,
            "is_grammar_break_with": _is_grammar_break(sentences[i], outputs_with[i]),
            "is_grammar_break_without": _is_grammar_break(sentences[i], outputs_without[i]),
        })
    system_meta.write_csv(
        detail_csv,
        fieldnames=[
            "sentence_id", "source", "output_with_context", "output_without_context",
            "latency_with_s", "latency_without_s",
            "translation_repetition_with", "translation_repetition_without",
            "is_grammar_break_with", "is_grammar_break_without",
        ],
        rows=detail_rows,
        meta=meta,
    )
    print(f"\n✓ Detail saved to {detail_csv}")
    print(f"  (metadata sidecar: {detail_csv.replace('.csv', '.meta.json')})")

    # Write summary CSV
    summary_rows = [
        {"metric": "repetitions",    "with_context": reps_with,   "without_context": reps_without},
        {"metric": "grammar_breaks", "with_context": breaks_with, "without_context": breaks_without},
        {"metric": "avg_latency_s",  "with_context": results["avg_latency_with_context"],
                                      "without_context": results["avg_latency_without_context"]},
    ]
    system_meta.write_csv(
        results_csv,
        fieldnames=["metric", "with_context", "without_context"],
        rows=summary_rows,
        meta=meta,
    )
    print(f"✓ Summary saved to {results_csv}")
    print(f"  (metadata sidecar: {results_csv.replace('.csv', '.meta.json')})")

    # Print paper-ready table
    print(f"\n{'=' * 60}")
    print("  Context Continuity Results (for Paper Section 5)")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'With Context':>15} {'Without Context':>18}")
    print(f"  {'─' * 25} {'─' * 15} {'─' * 18}")
    print(f"  {'Translation Repetitions':<25} {reps_with:>15} {reps_without:>18}")
    print(f"  {'Grammar Breaks':<25} {breaks_with:>15} {breaks_without:>18}")
    print(f"  {'Avg Latency (s)':<25} {results['avg_latency_with_context']:>15} {results['avg_latency_without_context']:>18}")


if __name__ == "__main__":
    run_benchmark()
