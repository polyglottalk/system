"""
benchmark_mt.py — Machine Translation comparison benchmark.

Compares Argos Translate vs MarianMT (Helsinki-NLP/opus-mt-en-hi)
on test sentences, measuring BLEU score and latency.

Usage
-----
    python benchmarks/benchmark_mt.py

Output
------
    results/mt_results.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from polyglot_talk import config  # noqa: E402

# benchmarks/ helpers
sys.path.insert(0, os.path.dirname(__file__))
import system_meta  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SENTENCES_FILE = os.path.join(PROJECT_ROOT, "data", "test_sentences", "sentences.txt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "mt")
# Output CSV paths are computed per-machine inside run_benchmark().


# ── BLEU calculation ────────────────────────────────────────────────────────

def _compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU score.
    
    Uses a simple 4-gram BLEU with brevity penalty.
    Falls back to sacrebleu if available, otherwise uses manual implementation.
    """
    try:
        import sacrebleu
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return bleu.score / 100.0  # normalize to 0-1
    except ImportError:
        pass

    # Manual BLEU implementation
    import math
    from collections import Counter

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))

    # n-gram precisions
    precisions = []
    for n in range(1, 5):  # 1-gram to 4-gram
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1)
        )
        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        clipped = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0))
                       for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    # Avoid log(0)
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / 4
    bleu = bp * math.exp(log_avg)
    return bleu


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_sentences() -> list[tuple[str, str]]:
    """Load (english, hindi_reference) pairs from sentences.txt."""
    pairs = []
    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


# ── Model runners ────────────────────────────────────────────────────────────

def _run_argos(sentences: list[tuple[str, str]]) -> list[dict]:
    """Benchmark Argos Translate en→hi."""
    import argostranslate.translate

    rows = []
    for idx, (en, hi_ref) in enumerate(sentences, 1):
        t0 = time.perf_counter()
        hypothesis = argostranslate.translate.translate(en, "en", "hi")
        latency = time.perf_counter() - t0

        bleu = _compute_bleu(hi_ref, hypothesis)
        rows.append({
            "model": "argos_translate",
            "sentence_id": idx,
            "source": en,
            "reference": hi_ref,
            "hypothesis": hypothesis,
            "bleu": f"{bleu:.4f}",
            "latency_s": f"{latency:.4f}",
        })
        print(f"  [{idx:>2}] BLEU={bleu:.3f}  latency={latency:.3f}s")
        print(f"       EN:  {en}")
        print(f"       REF: {hi_ref}")
        print(f"       HYP: {hypothesis}")

    return rows


def _run_marianmt(sentences: list[tuple[str, str]]) -> list[dict]:
    """Benchmark MarianMT (Helsinki-NLP/opus-mt-en-hi)."""
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        print("  ⚠ transformers not installed — skipping MarianMT benchmark.")
        print("    Install with: pip install transformers sentencepiece torch")
        return []

    model_name = "Helsinki-NLP/opus-mt-en-hi"
    print(f"  Loading MarianMT model: {model_name}...")
    t0 = time.perf_counter()
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s\n")

    rows = []
    for idx, (en, hi_ref) in enumerate(sentences, 1):
        t0 = time.perf_counter()
        inputs = tokenizer(en, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, num_beams=1, max_length=512)
        hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)
        latency = time.perf_counter() - t0

        bleu = _compute_bleu(hi_ref, hypothesis)
        rows.append({
            "model": "marianmt_opus-mt-en-hi",
            "sentence_id": idx,
            "source": en,
            "reference": hi_ref,
            "hypothesis": hypothesis,
            "bleu": f"{bleu:.4f}",
            "latency_s": f"{latency:.4f}",
        })
        print(f"  [{idx:>2}] BLEU={bleu:.3f}  latency={latency:.3f}s")
        print(f"       EN:  {en}")
        print(f"       REF: {hi_ref}")
        print(f"       HYP: {hypothesis}")

    return rows


# ── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark() -> None:
    machine_dir = os.path.join(RESULTS_DIR, system_meta.machine_slug())
    os.makedirs(machine_dir, exist_ok=True)
    results_csv = os.path.join(machine_dir, "mt_results.csv")

    sentences = _load_sentences()
    print(f"Loaded {len(sentences)} test sentences.\n")

    all_rows: list[dict] = []

    # --- Argos Translate ---
    print(f"{'=' * 60}")
    print("  Model: Argos Translate (en→hi)")
    print(f"{'=' * 60}")
    argos_rows = _run_argos(sentences)
    all_rows.extend(argos_rows)

    if argos_rows:
        bleus = [float(r["bleu"]) for r in argos_rows]
        lats = [float(r["latency_s"]) for r in argos_rows]
        print(f"\n  ── Summary: Argos Translate ──")
        print(f"  Average BLEU:    {sum(bleus)/len(bleus):.3f}")
        print(f"  Average latency: {sum(lats)/len(lats):.3f}s")
    print()

    # --- MarianMT ---
    print(f"{'=' * 60}")
    print("  Model: MarianMT (Helsinki-NLP/opus-mt-en-hi)")
    print(f"{'=' * 60}")
    marian_rows = _run_marianmt(sentences)
    all_rows.extend(marian_rows)

    if marian_rows:
        bleus = [float(r["bleu"]) for r in marian_rows]
        lats = [float(r["latency_s"]) for r in marian_rows]
        print(f"\n  ── Summary: MarianMT ──")
        print(f"  Average BLEU:    {sum(bleus)/len(bleus):.3f}")
        print(f"  Average latency: {sum(lats)/len(lats):.3f}s")
    print()

    # ── Write CSV ────────────────────────────────────────────────────────────
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "sentence_id", "source", "reference", "hypothesis",
            "bleu", "latency_s",
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✓ Detailed results saved to {results_csv}")

    # Summary CSV
    summary_csv = os.path.join(machine_dir, "mt_summary.csv")
    summary_rows = []
    for model_name, rows in [("argos_translate", argos_rows),
                              ("marianmt_opus-mt-en-hi", marian_rows)]:
        if not rows:
            continue
        bleus = [float(r["bleu"]) for r in rows]
        lats = [float(r["latency_s"]) for r in rows]
        import numpy as np
        summary_rows.append({
            "model": model_name,
            "avg_bleu": f"{np.mean(bleus):.4f}",
            "avg_latency_s": f"{np.mean(lats):.4f}",
            "std_latency_s": f"{np.std(lats):.4f}",
            "num_sentences": len(rows),
        })

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "avg_bleu", "avg_latency_s", "std_latency_s", "num_sentences",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"✓ Summary saved to {summary_csv}")

    # Print paper-ready table
    print(f"\n{'=' * 60}")
    print("  MT Results Table (for Paper Section 5)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<30} {'Avg BLEU':>10} {'Avg Latency':>14} {'Std Latency':>14}")
    print(f"  {'─' * 30} {'─' * 10} {'─' * 14} {'─' * 14}")
    for row in summary_rows:
        print(f"  {row['model']:<30} {float(row['avg_bleu']):>10.3f} {row['avg_latency_s']:>13}s {row['std_latency_s']:>13}s")


if __name__ == "__main__":
    run_benchmark()
