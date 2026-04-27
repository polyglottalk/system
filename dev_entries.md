# dev_entries.md — PolyglotTalk development log

Tracks meaningful changes per branch, following the `vex` chess engine convention.

---

## [fix/asr-tail-correction] 2026-04-20

### Short-term: tail-replaceable sentence buffer (issue #15)

**Problem:** The sentence buffer (`_sentence_buf`) was append-only. When a later
ASR chunk re-transcribed the overlap region with a slightly different (often worse)
wording, both versions were appended and both appeared in the final `SENT` output.

**Example from session log:**
```
[ASR   #   3] and it is all
[ASR   #   4] and it is also converting it
[SENT  #   0] … and it is all and it is also converting it …
```

**Changes:**
- Added `_word_overlap_ratio(a, b)` static method to `ASREngine`: Jaccard overlap
  between two word lists (case-insensitive), denominator = `min(|a|, |b|)`.
- Replaced `_sentence_buf.append(deduped_text)` with tail-correction logic:
  - Inspect the last `ASR_TAIL_WINDOW=12` words of the buffer as the *tail slice*.
  - If the new chunk's Jaccard overlap with the tail slice exceeds
    `ASR_TAIL_OVERLAP_THRESHOLD=0.60` **and** the new chunk is ≥ 70% the size of
    the tail slice (`ASR_TAIL_MIN_SIZE_RATIO=0.70`), replace the tail instead
    of appending.
- Near-duplicate guard now compares `deduped_text` (new words only, after overlap
  dedup) against the previous chunk's full transcript, instead of comparing raw
  chunk text. Threshold raised from `0.85` → `ASR_NEAR_DUP_THRESHOLD=0.92` to
  allow corrections through.

**New config constants** (all in `config.py`):
- `ASR_TAIL_WINDOW: int = 12`
- `ASR_TAIL_OVERLAP_THRESHOLD: float = 0.60`
- `ASR_TAIL_MIN_SIZE_RATIO: float = 0.70`
- `ASR_NEAR_DUP_THRESHOLD: float = 0.92`

---

### Medium-term: timestamp-based incremental decoding (issue #15)

**Problem:** Text-space dedup is fragile because it ignores *when* each word was
spoken. Two chunks covering a shared overlap region can produce different words for
the same audio, and no text heuristic reliably selects the correct take.

**Approach:** Use faster-whisper's `word_timestamps=True` to get per-word
chunk-relative `(start_s, end_s)`. Convert to global audio-stream offsets using
`AudioChunk.global_offset`. Accept only words whose audio midpoint is strictly
beyond `_committed_cutoff + ε`.

**Changes:**
- `models.py` — Added `global_offset: float = 0.0` to `AudioChunk`.
- `audio_capture.py` — Added `_global_sample_offset: int` counter. Set
  `AudioChunk.global_offset = _global_sample_offset / sample_rate` per chunk;
  advance by `STRIDE_SAMPLES` after each emit.
- `asr_engine.py`:
  - Added `_committed_cutoff: float = 0.0` field.
  - Added `_transcribe_with_timestamps(audio)` method: calls `model.transcribe()`
    with `word_timestamps=True`, fully drains the generator, returns
    `list[tuple[str, float, float]]` of `(word, start_s, end_s)`.
  - `run()` is now bifurcated:
    - **Timestamp path** (default, `ASR_USE_WORD_TIMESTAMPS=True`): calls
      `_transcribe_with_timestamps()`, filters words by `midpoint > cutoff + ε`,
      updates `_committed_cutoff`. Skips `_deduplicate_overlap()` and the
      near-duplicate guard entirely.
    - **Text path** (fallback, `ASR_USE_WORD_TIMESTAMPS=False`): original
      `_deduplicate_overlap()` + near-duplicate guard (threshold 0.92, compares
      `deduped_text`).
  - Both paths apply the tail-correction logic to the sentence buffer.

**New config constants**:
- `ASR_USE_WORD_TIMESTAMPS: bool = True`
- `ASR_TIMESTAMP_EPSILON: float = 0.05`

---

### Regression tests (`tests/test_asr_engine.py`)

New test file covering:
- `TestWordOverlapRatio` — `_word_overlap_ratio()` edge cases (empty lists, case,
  partial, min-denominator).
- `TestTailCorrection` — high-overlap triggers replacement; low-overlap appends;
  empty buffer always appends.
- `TestTailCorrectionRegression` — exact log pattern `"and it is all"` /
  `"and it is also converting it"`: asserts final sentence does NOT contain both.
- `TestCommittedCutoff` — `_committed_cutoff` starts at zero, advances to last word
  end-time, correctly rejects overlap-region words in subsequent chunks, respects
  `global_offset`, does not advance when all words are rejected.
