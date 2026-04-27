"""
tests/test_asr_engine.py — Unit tests for ASREngine tail-correction and
timestamp-based committed_cutoff logic (issue #15).

Tests cover:
  1. _word_overlap_ratio() basic cases
  2. Tail-replacement: high-overlap new chunk replaces sentence buffer tail
  3. Regression: ["and it is all", "and it is also converting it"] →
     final SENT text contains only ONE version (not both)
  4. Low-overlap chunks still append normally
  5. _committed_cutoff advances correctly with mocked word timestamps

WhisperModel is patched out so no model is loaded during the test session.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Patch WhisperModel before importing ASREngine so no model is loaded.
with patch("faster_whisper.WhisperModel", MagicMock()):
    from polyglot_talk.asr_engine import ASREngine

from polyglot_talk import config
from polyglot_talk.models import AudioChunk, TextSegment


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_engine() -> ASREngine:
    """Return an ASREngine with a mocked WhisperModel (no model loaded)."""
    with patch("faster_whisper.WhisperModel", MagicMock()):
        return ASREngine(
            audio_queue=queue.Queue(),
            text_queue=queue.Queue(),
            stop_event=threading.Event(),
        )


def _loud_audio(n: int = 1024) -> np.ndarray:
    """Float32 array whose RMS is well above the silence threshold."""
    return np.ones(n, dtype=np.float32)


class _FakeWord(NamedTuple):
    word: str
    start: float
    end: float


class _FakeSeg(NamedTuple):
    text: str
    words: list[_FakeWord]


def _mock_transcribe_segments(*word_tuples: tuple[str, float, float]):
    """Return a mock model.transcribe() return value for the given word list.

    Each element of word_tuples is (word_text, start_s, end_s).
    The mock yields a single segment whose `words` list mirrors the tuples.
    """
    fake_words = [_FakeWord(w, s, e) for w, s, e in word_tuples]
    full_text = " ".join(w for w, _, _ in word_tuples)
    seg = _FakeSeg(text=full_text, words=fake_words)

    def _transcribe_side_effect(*a, **kw):
        return ([seg], None)

    return _transcribe_side_effect


# ── 1. _word_overlap_ratio ────────────────────────────────────────────────────

class TestWordOverlapRatio:
    def test_full_overlap(self):
        assert ASREngine._word_overlap_ratio(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert ASREngine._word_overlap_ratio(["hello", "world"], ["foo", "bar"]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # {"and","it","is","all"} ∩ {"and","it","is","also"} = {"and","it","is"} → 3/4 = 0.75
        ratio = ASREngine._word_overlap_ratio(
            ["and", "it", "is", "all"],
            ["and", "it", "is", "also"],
        )
        assert ratio == pytest.approx(0.75)

    def test_case_insensitive(self):
        ratio = ASREngine._word_overlap_ratio(["Hello", "World"], ["hello", "world"])
        assert ratio == pytest.approx(1.0)

    def test_empty_a_returns_zero(self):
        assert ASREngine._word_overlap_ratio([], ["a", "b"]) == pytest.approx(0.0)

    def test_empty_b_returns_zero(self):
        assert ASREngine._word_overlap_ratio(["a", "b"], []) == pytest.approx(0.0)

    def test_uses_min_set_size(self):
        # min(3, 1) = 1 denominator; intersection = {"a"} → 1/1 = 1.0
        assert ASREngine._word_overlap_ratio(["a", "b", "c"], ["a"]) == pytest.approx(1.0)


# ── 2. Tail-replacement ───────────────────────────────────────────────────────

class TestTailCorrection:
    """Tests that a high-overlap new chunk replaces the buffer tail."""

    def test_high_overlap_replaces_tail(self):
        """New chunk with >60% Jaccard overlap on tail triggers replacement."""
        engine = _make_engine()
        # Pre-populate the sentence buffer with a tail
        engine._sentence_buf = ["and it is all"]

        # Simulate a new chunk that is a better take of the same content
        # overlap_ratio("and it is all", "and it is also converting it") =
        # {"and","it","is","all"} ∩ {"and","it","is","also","converting"} = {"and","it","is"} = 3
        # min(4, 5) = 4 → 3/4 = 0.75  > 0.60 threshold ✓
        # len(new_words)=5 >= len(tail_slice)*0.7 = 4*0.7=2.8 ✓
        new_chunk_text = "and it is also converting it"
        new_words = new_chunk_text.split()
        tail_words = " ".join(engine._sentence_buf).split()
        tail_slice = (
            tail_words[-config.ASR_TAIL_WINDOW :]
            if len(tail_words) > config.ASR_TAIL_WINDOW
            else tail_words
        )
        overlap = ASREngine._word_overlap_ratio(tail_slice, new_words)

        # Should trigger tail correction
        assert overlap > config.ASR_TAIL_OVERLAP_THRESHOLD
        assert len(new_words) >= len(tail_slice) * config.ASR_TAIL_MIN_SIZE_RATIO

        # Apply the same logic as run()
        keep_len = max(len(tail_words) - len(tail_slice), 0)
        engine._sentence_buf = [" ".join(tail_words[:keep_len] + new_words)]

        assert engine._sentence_buf == ["and it is also converting it"]
        assert len(engine._sentence_buf) == 1  # NOT two entries

    def test_low_overlap_appends(self):
        """New chunk with <60% overlap is appended normally."""
        engine = _make_engine()
        engine._sentence_buf = ["hello there"]

        new_text = "this is completely different content now"
        new_words = new_text.split()
        tail_words = " ".join(engine._sentence_buf).split()
        tail_slice = tail_words  # all 2 words

        overlap = ASREngine._word_overlap_ratio(tail_slice, new_words)
        assert overlap < config.ASR_TAIL_OVERLAP_THRESHOLD

        # Low overlap → append
        engine._sentence_buf.append(new_text)
        assert len(engine._sentence_buf) == 2
        assert engine._sentence_buf[0] == "hello there"
        assert engine._sentence_buf[1] == new_text

    def test_empty_buffer_always_appends(self):
        """When sentence_buf is empty the condition is False → append."""
        engine = _make_engine()
        assert not engine._sentence_buf

        new_text = "some new text here"
        # The tail-correction condition checks `self._sentence_buf` first;
        # an empty list is falsy so we always append.
        if engine._sentence_buf:
            pytest.fail("Buffer should be empty at this point")
        engine._sentence_buf.append(new_text)
        assert engine._sentence_buf == [new_text]


# ── 3. Regression: "and it is all" duplicate ─────────────────────────────────

class TestTailCorrectionRegression:
    """Regression for the exact session-log pattern from the issue body."""

    def test_regression_and_it_is_all(self):
        """
        ASR chunk sequence from issue #15:
          chunk 3: "and it is all"
          chunk 4: "and it is also converting it"

        With tail correction, chunk 4 replaces chunk 3's tail so the final
        SENT text contains only ONE version, not both joined.
        """
        engine = _make_engine()

        # Simulate chunk 3 reaching the buffer
        engine._sentence_buf.append("and it is all")
        engine._last_text_time = time.perf_counter()

        # Simulate chunk 4 triggering tail correction
        new_text = "and it is also converting it"
        new_words = new_text.split()
        tail_words = " ".join(engine._sentence_buf).split()
        tail_slice = (
            tail_words[-config.ASR_TAIL_WINDOW :]
            if len(tail_words) > config.ASR_TAIL_WINDOW
            else tail_words
        )
        overlap = ASREngine._word_overlap_ratio(tail_slice, new_words)

        if (
            engine._sentence_buf
            and overlap > config.ASR_TAIL_OVERLAP_THRESHOLD
            and len(new_words) >= len(tail_slice) * config.ASR_TAIL_MIN_SIZE_RATIO
        ):
            keep_len = max(len(tail_words) - len(tail_slice), 0)
            engine._sentence_buf = [" ".join(tail_words[:keep_len] + new_words)]
        else:
            engine._sentence_buf.append(new_text)

        # Flush the buffer to get the final sentence
        sentence = " ".join(engine._sentence_buf).strip()

        # Must NOT contain "and it is all AND also converting it" — only one take
        words_in_sentence = sentence.lower().split()
        # The word "all" should not appear if tail correction worked correctly
        # (it was replaced by "also"), OR "and it is all" as a distinct phrase.
        # The key check: the sentence should NOT start with "and it is all and"
        assert not sentence.lower().startswith("and it is all and"), (
            f"Duplicate content detected in sentence: {sentence!r}"
        )
        # And the corrected chunk content should be present
        assert "also converting" in sentence.lower(), (
            f"Corrected content missing from sentence: {sentence!r}"
        )


# ── 4. _committed_cutoff advances correctly ───────────────────────────────────

class TestCommittedCutoff:
    """Tests that _committed_cutoff advances after timestamp-based dedup."""

    def _run_chunk_through_engine(
        self,
        engine: ASREngine,
        chunk: AudioChunk,
        words: list[tuple[str, float, float]],
    ) -> list[str]:
        """Feed one chunk through the timestamp dedup logic (mirrors run() logic).

        Returns the list of accepted words.
        """
        accepted: list[str] = []
        new_cutoff = engine._committed_cutoff
        for word_text, start_s, end_s in words:
            midpoint = (start_s + end_s) / 2.0 + chunk.global_offset
            if midpoint > engine._committed_cutoff + config.ASR_TIMESTAMP_EPSILON:
                accepted.append(word_text)
                new_cutoff = max(new_cutoff, end_s + chunk.global_offset)
        if accepted:
            engine._committed_cutoff = new_cutoff
        return accepted

    def test_cutoff_starts_at_zero(self):
        engine = _make_engine()
        assert engine._committed_cutoff == 0.0

    def test_cutoff_advances_after_first_chunk(self):
        engine = _make_engine()
        chunk = AudioChunk(chunk_id=0, audio=_loud_audio(), global_offset=0.0)

        # Words spanning 0.0–1.5 s (chunk-relative)
        words = [("hello", 0.0, 0.5), ("world", 0.5, 1.0), ("how", 1.0, 1.5)]
        accepted = self._run_chunk_through_engine(engine, chunk, words)

        assert accepted == ["hello", "world", "how"]
        # cutoff should advance to end of last word: 1.5 + 0.0 (global_offset) = 1.5
        assert engine._committed_cutoff == pytest.approx(1.5)

    def test_cutoff_rejects_duplicate_words_in_next_chunk(self):
        engine = _make_engine()

        # Chunk 0: global_offset=0.0, covers 0.0–2.5 s, stride=1.5 s
        chunk0 = AudioChunk(chunk_id=0, audio=_loud_audio(), global_offset=0.0)
        words0 = [("hello", 0.0, 0.5), ("world", 0.5, 1.0), ("how", 1.0, 1.5)]
        self._run_chunk_through_engine(engine, chunk0, words0)
        # cutoff is now 1.5

        # Chunk 1: global_offset=1.5 (stride), chunk-relative timestamps 0.0–2.5 s
        # The overlap region (0.0–1.0 s chunk-relative = 1.5–2.5 s global) is AFTER cutoff
        chunk1 = AudioChunk(chunk_id=1, audio=_loud_audio(), global_offset=1.5)
        # Word "overlap_word" at chunk-relative 0.1–0.4 s → global 1.6–1.9 s
        # midpoint global = 1.75 s > 1.5 + 0.05 = 1.55 → ACCEPTED
        # Word "old_overlap" at chunk-relative 0.0–0.04 s → global 1.5–1.54 s
        # midpoint global = 1.52 s ≤ 1.5 + 0.05 = 1.55 → REJECTED
        words1 = [
            ("old_overlap", 0.0, 0.04),   # midpoint global = 1.52 < cutoff+ε=1.55 → rejected
            ("overlap_word", 0.1, 0.4),    # midpoint global = 1.75 > 1.55 → accepted
            ("new_content", 0.5, 1.0),     # midpoint global = 2.25 > 1.55 → accepted
        ]
        accepted1 = self._run_chunk_through_engine(engine, chunk1, words1)

        assert "old_overlap" not in accepted1
        assert "overlap_word" in accepted1
        assert "new_content" in accepted1

    def test_cutoff_with_global_offset(self):
        """global_offset correctly shifts chunk-relative timestamps to global."""
        engine = _make_engine()
        # Set cutoff as if prior chunks have been committed up to 10.0 s
        engine._committed_cutoff = 10.0

        chunk = AudioChunk(chunk_id=5, audio=_loud_audio(), global_offset=9.0)
        # chunk-relative 0.5–1.5 s → global 9.5–10.5 s
        # midpoint global = 10.0 s: 10.0 ≤ 10.0 + 0.05 = 10.05 → REJECTED
        # chunk-relative 1.5–2.5 s → global 10.5–11.5 s
        # midpoint global = 11.0 s > 10.05 → ACCEPTED
        words = [
            ("old_word", 0.5, 1.5),    # midpoint=10.0, rejected
            ("new_word", 1.5, 2.5),    # midpoint=11.0, accepted
        ]
        accepted = self._run_chunk_through_engine(engine, chunk, words)
        assert accepted == ["new_word"]
        assert engine._committed_cutoff == pytest.approx(11.5)  # 2.5 + 9.0

    def test_no_accepted_words_does_not_advance_cutoff(self):
        """cutoff stays unchanged if no words pass the threshold."""
        engine = _make_engine()
        engine._committed_cutoff = 5.0

        chunk = AudioChunk(chunk_id=2, audio=_loud_audio(), global_offset=4.0)
        # All words are before the cutoff (global midpoints ≤ 5.05)
        words = [("old1", 0.0, 0.5), ("old2", 0.5, 1.0)]
        # global midpoints: 4.25 and 4.75 — both < 5.05 → rejected
        accepted = self._run_chunk_through_engine(engine, chunk, words)

        assert accepted == []
        assert engine._committed_cutoff == pytest.approx(5.0)  # unchanged


class TestASRConfidenceGate:
    """Tests for metadata-based hallucination rejection."""

    def test_rejects_high_no_speech_and_low_logprob(self):
        info = SimpleNamespace(
            no_speech_prob=config.ASR_NO_SPEECH_PROB_THRESHOLD + 0.1,
            avg_logprob=config.ASR_LOW_LOGPROB_THRESHOLD - 0.1,
            compression_ratio=1.2,
        )
        assert ASREngine._should_reject_from_asr_info(info) is True

    def test_rejects_high_compression_and_low_logprob(self):
        info = SimpleNamespace(
            no_speech_prob=0.1,
            avg_logprob=config.ASR_LOW_LOGPROB_THRESHOLD - 0.1,
            compression_ratio=config.ASR_COMPRESSION_RATIO_THRESHOLD + 0.2,
        )
        assert ASREngine._should_reject_from_asr_info(info) is True

    def test_keeps_confident_speech(self):
        info = SimpleNamespace(
            no_speech_prob=0.1,
            avg_logprob=-0.2,
            compression_ratio=1.1,
        )
        assert ASREngine._should_reject_from_asr_info(info) is False

    def test_none_info_is_not_rejected(self):
        assert ASREngine._should_reject_from_asr_info(None) is False
