"""
tests/test_asr_dedup.py — Unit tests for ASREngine.deduplicate_overlap() and the
near-duplicate guard in ASREngine.run().

WhisperModel is patched out so no model is loaded during the test session.
"""

from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock, patch

import numpy as np

from polyglot_talk import config

# Patch WhisperModel before importing ASREngine so the model is never loaded.
with patch("faster_whisper.WhisperModel", MagicMock()):
    from polyglot_talk.asr_engine import ASREngine

from polyglot_talk.models import AudioChunk


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_engine() -> ASREngine:
    """Return an ASREngine with a mocked WhisperModel (no actual model loaded)."""
    with patch("faster_whisper.WhisperModel", MagicMock()):
        return ASREngine(
            audio_queue=queue.Queue(),
            text_queue=queue.Queue(),
            stop_event=threading.Event(),
        )


def _loud_audio(n: int = 1024) -> np.ndarray:
    """Return a float32 array whose RMS is well above the silence threshold."""
    return np.ones(n, dtype=np.float32)


# ── deduplicate_overlap tests ─────────────────────────────────────────────────

def test_basic_exact_overlap():
    """Removes the exact matching suffix-prefix from curr_words."""
    result = ASREngine._deduplicate_overlap(
        ["hello", "world", "how", "are"],
        ["how", "are", "you", "doing"],
    )
    assert result == ["you", "doing"]


def test_punctuation_mismatch_comma():
    """Matches 'fox,' to prev 'fox' after stripping trailing punctuation."""
    result = ASREngine._deduplicate_overlap(
        ["the", "quick", "brown", "fox"],
        ["fox,", "jumps", "over"],
    )
    assert result == ["jumps", "over"]


def test_capitalisation_mismatch():
    """Matches 'Brown Fox' to prev 'brown fox' case-insensitively."""
    result = ASREngine._deduplicate_overlap(
        ["brown", "fox"],
        ["Brown", "Fox", "jumps"],
    )
    assert result == ["jumps"]


def test_hyphenated_word_split():
    """Expands 'real-time' to tokens ['real', 'time'] before matching."""
    result = ASREngine._deduplicate_overlap(
        ["real", "time", "system"],
        ["real-time", "system", "works"],
    )
    assert result == ["works"]


def test_no_overlap():
    """Returns curr_words unchanged when there is no suffix-prefix match."""
    result = ASREngine._deduplicate_overlap(
        ["hello", "world"],
        ["something", "completely", "different"],
    )
    assert result == ["something", "completely", "different"]


def test_empty_prev_words():
    """Returns curr_words unchanged when prev_words is empty."""
    result = ASREngine._deduplicate_overlap(
        [],
        ["any", "words"],
    )
    assert result == ["any", "words"]


def test_empty_curr_words():
    """Returns an empty list when curr_words is empty."""
    result = ASREngine._deduplicate_overlap(
        ["hello", "world"],
        [],
    )
    assert result == []


def test_full_overlap_returns_empty():
    """Returns an empty list when all of curr_words is already in prev suffix."""
    result = ASREngine._deduplicate_overlap(
        ["a", "b", "c", "d"],
        ["c", "d"],
    )
    assert result == []


def test_apostrophe_preservation():
    """Preserves apostrophes so contractions like \"don't\" survive normalisation."""
    result = ASREngine._deduplicate_overlap(
        ["i", "don't", "know"],
        ["don't", "know", "either"],
    )
    assert result == ["either"]


# ── Near-duplicate guard test (run()) ────────────────────────────────────────

def test_near_duplicate_guard_skips_chunk():
    """run() skips a chunk whose words overlap > 85% with the previous chunk's text."""
    engine = _make_engine()

    # Build a prev text and a near-duplicate (6 out of 7 unique words match → 6/7 ≈ 85.7%)
    last_text = "alpha beta gamma delta epsilon sigma theta"
    near_dup_text = "Alpha Beta Gamma Delta Epsilon Sigma New"  # 6/7 > 0.85

    # Pre-seed last_text so the guard has a previous value to compare against.
    engine._last_text = last_text

    # Patch _transcribe to return the near-duplicate text.
    engine._transcribe = MagicMock(return_value=near_dup_text)

    chunk = AudioChunk(chunk_id=99, audio=_loud_audio())
    engine._audio_queue.put(chunk)
    engine._audio_queue.put(None)  # shutdown sentinel

    engine.run()

    # The text_queue should contain only the sentinel (None) — not the near-dup chunk.
    item = engine._text_queue.get_nowait()
    assert item is None, (
        f"Expected only the None sentinel in text_queue, got {item!r}"
    )
    assert engine._text_queue.empty(), "text_queue should be empty after consuming sentinel"


# ── Tail-replacement regression test ─────────────────────────────────────────

def test_tail_replacement_prevents_double_phrase():
    """Tail correction replaces the buffer tail when Jaccard overlap exceeds threshold.

    Scenario: buffer holds 'one two three'; chunk B transcribes to 'two three four'.
    Jaccard({"one","two","three"}, {"two","three","four"}) = 2/3 ≈ 0.67 > 0.60 threshold.
    Without tail correction the buffer would grow to 'one two three two three four'
    (audible double-phrase). With tail correction the buffer is replaced with
    'two three four', which is the only text in the flushed TextSegment.
    """
    engine = _make_engine()

    # Pre-seed the sentence buffer to simulate a prior chunk already accumulated.
    # _last_text_time stays at 0.0 so _maybe_flush_timeout() returns early
    # (the guard skips when last_text_time <= 0), leaving the buffer intact.
    engine._sentence_buf = ["one two three"]
    engine._sentence_chunk_id = 10

    # Chunk B: re-transcribes the overlapping tail ("two three") plus new content.
    chunk_b = AudioChunk(chunk_id=11, audio=_loud_audio())
    engine._transcribe = MagicMock(return_value="two three four")

    # Silent chunk triggers the silence-detection flush path.
    silent_chunk = AudioChunk(chunk_id=12, audio=np.zeros(1024, dtype=np.float32))

    engine._audio_queue.put(chunk_b)
    engine._audio_queue.put(silent_chunk)
    engine._audio_queue.put(None)  # shutdown sentinel

    # Force text-based dedup path so _transcribe() is used (not timestamp path).
    with patch.object(config, "ASR_USE_WORD_TIMESTAMPS", False):
        engine.run()

    segment = engine._text_queue.get_nowait()
    assert segment is not None, "Expected a TextSegment flushed by silence detection"
    text = segment.text
    assert "one two three two three" not in text, (
        f"Double-phrase detected — tail correction did not fire: {text!r}"
    )
    assert "two three four" in text, (
        f"Expected corrected tail 'two three four' in flushed segment: {text!r}"
    )
    sentinel = engine._text_queue.get_nowait()
    assert sentinel is None, "Expected shutdown sentinel after segment"
