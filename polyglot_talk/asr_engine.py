"""
asr_engine.py — Speech-to-text thread using faster-whisper.

Key constraints:
- WhisperModel is loaded ONCE in __init__ (not in run()).
- model.transcribe() returns a GENERATOR — it must be fully drained.
- Silent/hallucinated chunks are filtered before pushing to text_queue.

Overlap deduplication
---------------------
Because AudioCapture now emits overlapping chunks (see config.CHUNK_OVERLAP),
Whisper will re-transcribe the overlapping audio region.  The ASR engine
performs **suffix-prefix word matching** to deduplicate:

    prev_words = ["hello", "world", "how", "are"]
    curr_words = ["how", "are", "you", "doing"]
    → deduplicated = ["you", "doing"]

Research basis:
  • Whispy (Bevilacqua et al., 2024) — Levenshtein-distance deduplication
    over a shifting buffer of re-transcribed overlapping audio.
  • Whisper-Streaming (Machácek et al., 2023) — LocalAgreement-2 longest-
    common-prefix policy over consecutive overlapping transcriptions.

Sentence accumulation
---------------------
Whisper appends a period to virtually every chunk transcription, even when
speech continues into the next chunk.  Sending each fragment with an
artificial sentence-final period degrades translation quality and TTS
prosody.  Instead, fragments are buffered until a natural sentence boundary
is detected (pause / silence in next chunk, or accumulation timeout).
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import re

import numpy as np

from . import config
from .models import AudioChunk, TextSegment

# Import after config.py has set OMP_NUM_THREADS / CT2_INTER_THREADS
from faster_whisper import WhisperModel  # noqa: E402

logger = logging.getLogger(__name__)


def _get_broadcaster():
    """Lazy import so dashboard_server is only pulled in when --dashboard is used."""
    try:
        from dashboard_server import broadcaster  # noqa: PLC0415
        return broadcaster
    except ImportError:
        return None


class ASREngine:
    """Transcribes AudioChunk objects into TextSegment objects.

    Filters
    -------
    1. RMS silence filter — chunks whose RMS energy is below
       ``config.RMS_SILENCE_THRESHOLD`` are skipped.
    2. Duplicate filter — if the transcription is identical to the
       previous non-empty result, it is skipped (Whisper hallucination).

    Overlap deduplication
    ---------------------
    Since audio chunks now overlap, the engine keeps the previous
    transcription's word list and strips repeated words from the start
    of each new transcription (suffix-prefix matching).

    Sentence accumulation
    ---------------------
    Deduplicated fragments are buffered.  The buffer is flushed as a
    single TextSegment when any of these conditions is met:
      • The next audio chunk is silent (natural pause → sentence end).
      • No new text arrives for SENTENCE_BUFFER_TIMEOUT seconds.
      • The buffer exceeds SENTENCE_BUFFER_MAXWORDS words.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        text_queue: queue.Queue,
        stop_event: threading.Event,
        source_lang: str = config.SOURCE_LANG,
        model_size: str | None = None,
        compute_type: str = config.ASR_COMPUTE_TYPE,
        beam_size: int = config.ASR_BEAM_SIZE,
    ) -> None:
        self._audio_queue = audio_queue
        self._text_queue = text_queue
        self._stop_event = stop_event
        self._beam_size = beam_size
        self._source_lang = source_lang
        self._asr_language = config.ASR_TRANSCRIBE_LANG_MAP.get(source_lang, config.ASR_LANGUAGE)
        resolved_model_size = model_size or config.ASR_MODEL_MAP.get(source_lang, config.ASR_MODEL_SIZE)

        logger.info(
            "Loading ASR model (%s, %s) for source=%s (language=%s)…",
            resolved_model_size,
            compute_type,
            source_lang,
            self._asr_language,
        )
        t0 = time.perf_counter()
        self.model = WhisperModel(
            resolved_model_size,
            device=config.ASR_DEVICE,
            compute_type=compute_type,
        )
        logger.info("ASR model loaded in %.1fs", time.perf_counter() - t0)

        self._last_text: str = ""
        # Word list from the previous chunk's transcription — for text-based dedup
        self._prev_words: list[str] = []
        # Sentence accumulation buffer
        self._sentence_buf: list[str] = []
        self._sentence_chunk_id: int = 0
        self._sentence_capture_ts: float = 0.0
        self._last_text_time: float = 0.0  # perf_counter when last text appended
        # Medium-term: global audio-stream offset of the last committed word.
        # Updated each chunk when ASR_USE_WORD_TIMESTAMPS is True.
        self._committed_cutoff: float = 0.0

    # ── Thread target ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Consume AudioChunks, transcribe, deduplicate, buffer, push TextSegments."""
        while not self._stop_event.is_set():
            try:
                item = self._audio_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                # Check sentence buffer timeout while waiting
                self._maybe_flush_timeout()
                continue

            if item is None:  # shutdown sentinel
                # Flush any remaining sentence buffer
                self._flush_sentence_buffer()
                # Propagate sentinel downstream so Translator exits cleanly
                try:
                    self._text_queue.put_nowait(None)
                except queue.Full:
                    pass
                break

            assert isinstance(item, AudioChunk)

            # ── Silence filter ─────────────────────────────────────────────
            rms = float(np.sqrt(np.mean(item.audio ** 2)))
            if rms < config.RMS_SILENCE_THRESHOLD:
                logger.debug("Chunk #%d silent (rms=%.5f) — skipping.", item.chunk_id, rms)
                # Silence after speech → natural sentence boundary → flush buffer
                if self._sentence_buf:
                    logger.debug("Silence detected — flushing sentence buffer.")
                    self._flush_sentence_buffer()
                continue

            # ── Transcribe ─────────────────────────────────────────────────
            t0 = time.perf_counter()
            if config.ASR_USE_WORD_TIMESTAMPS:
                _words_with_ts, _asr_info = self._transcribe_with_timestamps_and_info(item.audio)
                text = " ".join(w for w, _, _ in _words_with_ts).strip()
            else:
                _words_with_ts = None
                text, _asr_info = self._transcribe_with_info(item.audio)
            elapsed = time.perf_counter() - t0

            if not text:
                logger.debug("Chunk #%d produced empty transcript.", item.chunk_id)
                continue

            # ── Hallucination blocklist filter ─────────────────────────────
            normalized = re.sub(r"[^\w\s]", "", text).strip().lower()
            if normalized in config.ASR_HALLUCINATION_BLOCKLIST:
                logger.debug(
                    "Chunk #%d blocked hallucination: %r", item.chunk_id, text
                )
                continue

            # ── Confidence gate (silence / non-speech hallucinations) ─────
            if self._should_reject_from_asr_info(_asr_info):
                logger.debug(
                    "Chunk #%d rejected by ASR confidence gate: %r",
                    item.chunk_id,
                    text,
                )
                continue

            # ── Duplicate / hallucination filter ───────────────────────────
            if text == self._last_text:
                logger.debug(
                    "Chunk #%d skipped — duplicate transcript: %r",
                    item.chunk_id,
                    text,
                )
                continue
            if config.ASR_USE_WORD_TIMESTAMPS:
                # ── Medium-term: timestamp-based dedup ────────────────────
                # Accept only words whose audio midpoint (chunk-relative
                # start/end + chunk global_offset) is strictly beyond the
                # last committed cutoff.  Replaces both suffix/prefix text
                # dedup and the near-duplicate guard for overlap regions.
                global_offset = item.global_offset
                accepted_words: list[str] = []
                new_cutoff = self._committed_cutoff
                for word_text, start_s, end_s in _words_with_ts:  # type: ignore[union-attr]
                    midpoint = (start_s + end_s) / 2.0 + global_offset
                    if midpoint > self._committed_cutoff + config.ASR_TIMESTAMP_EPSILON:
                        accepted_words.append(word_text)
                        new_cutoff = max(new_cutoff, end_s + global_offset)

                if not accepted_words:
                    logger.debug(
                        "Chunk #%d — all words before committed cutoff (%.3fs), skipping.",
                        item.chunk_id,
                        self._committed_cutoff,
                    )
                    continue

                self._committed_cutoff = new_cutoff
                deduped_text = " ".join(accepted_words)
            else:
                # ── Short-term: text-based overlap dedup ──────────────────
                curr_words = text.split()
                deduped_words = self._deduplicate_overlap(self._prev_words, curr_words)
                self._prev_words = curr_words  # keep full transcription for next round

                if not deduped_words:
                    logger.debug(
                        "Chunk #%d — all words duplicated from overlap, skipping.",
                        item.chunk_id,
                    )
                    continue

                deduped_text = " ".join(deduped_words)

                # ── Near-duplicate guard ──────────────────────────────────
                # Compares deduped_text (new words only) against the previous
                # chunk's full transcript.  Threshold raised to 0.92 so chunks
                # with a genuine tail correction are not discarded.
                if self._last_text:
                    _curr_set = set(w.lower() for w in deduped_text.split())
                    _prev_set = set(w.lower() for w in self._last_text.split())
                    _overlap_count = len(_curr_set & _prev_set)
                    _min_len = min(len(deduped_text.split()), len(self._last_text.split()))
                    if _min_len > 0 and _overlap_count / _min_len > config.ASR_NEAR_DUP_THRESHOLD:
                        logger.debug(
                            "Chunk %d near-duplicate of previous (%.0f%%), skipping.",
                            item.chunk_id,
                            _overlap_count / _min_len * 100,
                        )
                        continue

            self._last_text = text

            # ── Strip trailing period ──────────────────────────────────────
            deduped_text = self._normalize_punctuation(deduped_text)

            if not deduped_text:
                continue

            logger.debug(
                "Transcription done (%.3fs) chunk #%d: %r",
                elapsed,
                item.chunk_id,
                deduped_text,
            )
            # Live console output
            print(f"[ASR   #{item.chunk_id:>4d}] {deduped_text}", flush=True)
            _bc = _get_broadcaster()
            if _bc is not None:
                _bc.emit({"type": "asr_chunk", "chunk_id": item.chunk_id, "text": deduped_text})

            # ── Sentence buffer: timeout check + tail correction ───────────
            # Check timeout BEFORE adding new text: if the previous buffer
            # has been waiting longer than SENTENCE_BUFFER_TIMEOUT, flush it
            # first so the new content starts a fresh sentence.
            self._maybe_flush_timeout()
            if not self._sentence_buf:
                self._sentence_chunk_id = item.chunk_id
                self._sentence_capture_ts = item.timestamp

            # ── Short-term: tail-correction ────────────────────────────────
            # If the new chunk has high Jaccard overlap with the tail of the
            # buffer, replace the tail rather than appending.  Corrects
            # Whisper's garbled re-transcription of the overlapping region
            # that text-based dedup cannot reliably catch.
            new_words = deduped_text.split()
            tail_text = " ".join(self._sentence_buf)
            tail_words = tail_text.split()
            tail_slice = (
                tail_words[-config.ASR_TAIL_WINDOW :]
                if len(tail_words) > config.ASR_TAIL_WINDOW
                else tail_words
            )
            overlap_ratio = self._word_overlap_ratio(tail_slice, new_words)
            if (
                self._sentence_buf
                and overlap_ratio > config.ASR_TAIL_OVERLAP_THRESHOLD
                and len(new_words) >= len(tail_slice) * config.ASR_TAIL_MIN_SIZE_RATIO
            ):
                keep_len = max(len(tail_words) - len(tail_slice), 0)
                self._sentence_buf = [" ".join(tail_words[:keep_len] + new_words)]
                logger.debug(
                    "Tail correction: overlap=%.0f%%, replacing last %d words with new chunk.",
                    overlap_ratio * 100,
                    len(tail_slice),
                )
            else:
                self._sentence_buf.append(deduped_text)
            self._last_text_time = time.perf_counter()

            # Force-flush if buffer is too long
            buf_words = sum(len(s.split()) for s in self._sentence_buf)
            if buf_words >= config.SENTENCE_BUFFER_MAXWORDS:
                logger.debug("Sentence buffer maxwords reached — flushing.")
                self._flush_sentence_buffer()

        logger.info("ASREngine stopped.")

    # ── Overlap deduplication ───────────────────────────────────────────────

    @staticmethod
    def _word_overlap_ratio(a: list[str], b: list[str]) -> float:
        """Jaccard overlap between two word lists (case-insensitive).

        Returns the size of the intersection divided by the size of the
        *smaller* set, so a short new chunk can still register high overlap
        with a longer tail slice.

        Examples
        --------
        >>> ASREngine._word_overlap_ratio(["and", "it", "is", "all"], ["and", "it", "is", "also"])
        0.6
        """
        a_set = {w.lower() for w in a}
        b_set = {w.lower() for w in b}
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / min(len(a_set), len(b_set))

    @staticmethod
    def _normalize_token(w: str) -> str:
        """Normalise a single token for overlap comparison.

        Lowercases *w* and strips every character that is not a word
        character or an apostrophe, so contractions like ``don't`` survive
        while punctuation attached to a word (``fox,``, ``(Brown)``) is
        removed before comparison.

        Examples
        --------
        >>> ASREngine._normalize_token("Fox,")
        'fox'
        >>> ASREngine._normalize_token("don't")
        "don't"
        >>> ASREngine._normalize_token("(hello)")
        'hello'
        """
        return re.sub(r"[^\w']", "", w.lower())

    @staticmethod
    def _expand_to_tokens(
        words: list[str],
    ) -> tuple[list[str], list[int]]:
        """Tokenise a word list for overlap comparison.

        Splits hyphenated words (e.g. "real-time" → ["real", "time"]) and
        strips punctuation so that Whisper's inconsistent hyphenation across
        consecutive overlapping chunks does not defeat deduplication.

        Returns
        -------
        tokens : list[str]
            Normalised sub-word tokens.
        orig_indices : list[int]
            For each token, the index of the original word in *words* that
            produced it.  Used to map a token-level match back to the number
            of original words to skip.
        """
        tokens: list[str] = []
        orig_indices: list[int] = []
        for i, w in enumerate(words):
            w_clean = w.lower().strip(".,!?;:'\"")
            # Split on hyphen and en/em-dash so "real-time" → ["real", "time"]
            for part in re.split(r"[-\u2013\u2014]", w_clean):
                part = part.strip()
                if part:
                    tokens.append(part)
                    orig_indices.append(i)
        return tokens, orig_indices

    @staticmethod
    def _deduplicate_overlap(
        prev_words: list[str], curr_words: list[str]
    ) -> list[str]:
        """Remove overlapping words at the boundary between chunks.

        Expands hyphenated words into component tokens (via
        :meth:`_expand_to_tokens`) and then applies
        :meth:`_normalize_token` to every sub-token before comparison.
        This makes matching robust against Whisper's inconsistent
        punctuation and capitalisation across consecutive overlapping
        chunks (e.g. ``"fox"`` vs ``"fox,"``, ``"Brown"`` vs
        ``"brown"``, or mid-word punctuation that the simple
        ``strip()`` in ``_expand_to_tokens`` would miss).

        Finds the longest suffix of *prev* normalised tokens that equals
        a prefix of *curr* normalised tokens, then maps the match back to
        the original *curr_words* index so the returned list preserves the
        original Whisper capitalisation and punctuation.

        Example
        -------
        >>> ASREngine._deduplicate_overlap(
        ...     ["a", "real", "time"],
        ...     ["real-time,", "fully", "offline"],
        ... )
        ['fully', 'offline']
        """
        if not prev_words or not curr_words:
            return curr_words

        # Expand to sub-tokens (handles hyphen splitting) then normalise
        # each token for a punctuation- and case-insensitive comparison.
        prev_raw, _ = ASREngine._expand_to_tokens(prev_words)
        curr_raw, curr_orig = ASREngine._expand_to_tokens(curr_words)

        prev_norm = [ASREngine._normalize_token(t) for t in prev_raw]
        curr_norm = [ASREngine._normalize_token(t) for t in curr_raw]

        max_overlap = min(len(prev_norm), len(curr_norm))
        best = 0

        for k in range(1, max_overlap + 1):
            if prev_norm[-k:] == curr_norm[:k]:
                best = k

        if best == 0:
            return curr_words

        # Map the last matched token back to its original word index and
        # return everything after that word.
        last_matched_orig = curr_orig[best - 1]
        logger.debug(
            "Dedup: stripped %d token(s) (%d original word(s)): %s",
            best,
            last_matched_orig + 1,
            curr_words[: last_matched_orig + 1],
        )
        return curr_words[last_matched_orig + 1 :]

    # ── Punctuation normalisation ───────────────────────────────────────────

    @staticmethod
    def _normalize_punctuation(text: str) -> str:
        """Strip the trailing period that Whisper auto-appends to every chunk.

        Whisper systematically adds a period at the end of each transcription
        segment, even when the speaker is mid-sentence.  This causes the
        translator to treat each fragment as an independent sentence and the
        TTS to use sentence-final intonation on every chunk.

        Interrogative (?) and exclamatory (!) marks are preserved because they
        are genuine prosodic cues that Whisper only produces when actually
        warranted.
        """
        if not config.ASR_STRIP_TRAILING_PERIOD:
            return text
        # Strip trailing period(s) only — keep ? and !
        text = re.sub(r"\.+\s*$", "", text).strip()
        return text

    # ── Sentence buffer management ──────────────────────────────────────────

    def _flush_sentence_buffer(self) -> None:
        """Join buffered fragments into one TextSegment and push to text_queue."""
        if not self._sentence_buf:
            return

        sentence = " ".join(self._sentence_buf).strip()
        self._sentence_buf.clear()

        if not sentence:
            return

        # Re-add a proper sentence-ending period for the complete sentence
        if sentence and sentence[-1] not in ".!?":
            sentence += "."

        logger.debug("Flushing sentence buffer → %r", sentence)
        print(f"[SENT  #{self._sentence_chunk_id:>4d}] {sentence}", flush=True)
        _bc = _get_broadcaster()
        if _bc is not None:
            _bc.emit({"type": "sentence_flushed", "chunk_id": self._sentence_chunk_id, "text": sentence})

        segment = TextSegment(
            chunk_id=self._sentence_chunk_id,
            text=sentence,
            timestamp=time.perf_counter(),
            capture_timestamp=self._sentence_capture_ts,
        )
        self._put(segment)

    def _maybe_flush_timeout(self) -> None:
        """Flush the sentence buffer if enough time elapsed since last text."""
        if not self._sentence_buf:
            return
        if self._last_text_time <= 0:
            return
        elapsed = time.perf_counter() - self._last_text_time
        if elapsed >= config.SENTENCE_BUFFER_TIMEOUT:
            logger.debug("Sentence buffer timeout (%.1fs) — flushing.", elapsed)
            self._flush_sentence_buffer()

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _should_reject_from_asr_info(info: object | None) -> bool:
        """Return True when ASR metadata indicates likely no-speech hallucination."""
        if info is None:
            return False

        no_speech_prob = float(getattr(info, "no_speech_prob", 0.0) or 0.0)
        avg_logprob = float(getattr(info, "avg_logprob", 0.0) or 0.0)
        compression_ratio = float(getattr(info, "compression_ratio", 0.0) or 0.0)

        likely_silence = (
            no_speech_prob >= config.ASR_NO_SPEECH_PROB_THRESHOLD
            and avg_logprob <= config.ASR_LOW_LOGPROB_THRESHOLD
        )
        likely_hallucination = (
            compression_ratio >= config.ASR_COMPRESSION_RATIO_THRESHOLD
            and avg_logprob <= config.ASR_LOW_LOGPROB_THRESHOLD
        )
        return likely_silence or likely_hallucination

    def _transcribe_with_info(self, audio: np.ndarray) -> tuple[str, object | None]:
        """Run faster-whisper on a float32 16 kHz numpy array.

        The generator returned by model.transcribe() MUST be fully consumed
        before the next call — partial iteration can corrupt CTranslate2 state.
        """
        result = self.model.transcribe(
            audio,
            beam_size=self._beam_size,
            language=self._asr_language,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        # faster-whisper normally returns ``(segments, info)``, but some
        # edge cases can yield just the segments iterable. Normalize both.
        if isinstance(result, tuple):
            segments_gen = result[0]
            info = result[1] if len(result) > 1 else None
        else:
            segments_gen = result
            info = None
        # Drain the generator completely
        text = "".join(seg.text for seg in segments_gen).strip()
        return text, info

    def _transcribe(self, audio: np.ndarray) -> str:
        """Backward-compatible text-only wrapper around _transcribe_with_info()."""
        text, _ = self._transcribe_with_info(audio)
        return text

    def _transcribe_with_timestamps_and_info(
        self, audio: np.ndarray
    ) -> tuple[list[tuple[str, float, float]], object | None]:
        """Run faster-whisper with word_timestamps=True.

        Returns a list of ``(word, start_s, end_s)`` tuples where *start_s*
        and *end_s* are **chunk-relative** seconds.  The caller adds
        ``AudioChunk.global_offset`` to convert to global audio-stream time.

        The generator is fully drained before returning to avoid corrupting
        CTranslate2 state across calls.
        """
        result = self.model.transcribe(
            audio,
            beam_size=self._beam_size,
            language=self._asr_language,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=True,
        )
        if isinstance(result, tuple):
            segments_gen = result[0]
            info = result[1] if len(result) > 1 else None
        else:
            segments_gen = result
            info = None
        words: list[tuple[str, float, float]] = []
        for seg in segments_gen:
            if seg.words:
                for w in seg.words:
                    word_text = w.word.strip()
                    if word_text:
                        words.append((word_text, float(w.start), float(w.end)))
        return words, info

    def _transcribe_with_timestamps(
        self, audio: np.ndarray
    ) -> list[tuple[str, float, float]]:
        """Backward-compatible word-timestamp wrapper without metadata."""
        words, _ = self._transcribe_with_timestamps_and_info(audio)
        return words

    def _put(self, segment: TextSegment) -> None:
        """Push to text_queue with drop-oldest strategy on Full.

        Never blocks — if the queue is full the oldest pending transcription
        is evicted so the translator always sees the most recent text.
        """
        try:
            self._text_queue.put_nowait(segment)
        except queue.Full:
            try:
                dropped = self._text_queue.get_nowait()
                logger.warning(
                    "text_queue full — evicted oldest chunk #%d to insert chunk #%d",
                    dropped.chunk_id,
                    segment.chunk_id,
                )
            except queue.Empty:
                pass
            try:
                self._text_queue.put_nowait(segment)
            except queue.Full:
                logger.warning(
                    "text_queue still full — dropping chunk #%d", segment.chunk_id
                )
