"""
config.py — Global constants and environment variables for PolyglotTalk.

IMPORTANT: This module sets os.environ keys for CTranslate2 / OpenMP
BEFORE any faster_whisper or argostranslate imports happen anywhere
in the process. Import this module first in every entry point.
"""

import os

# ── Thread-count caps (must be set before importing CTranslate2 libs) ──────
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CT2_INTER_THREADS", "1")

# ── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16000          # Hz — Whisper expects 16 kHz
CHUNK_DURATION: float = 2.5       # seconds per ASR chunk
BLOCK_SIZE: int = int(SAMPLE_RATE * CHUNK_DURATION)  # 40 000 samples

# ── Overlapping chunks ──────────────────────────────────────────────────────
# Consecutive audio chunks share CHUNK_OVERLAP seconds of audio so that
# words at chunk boundaries are never cut.  The stride (new audio per chunk)
# is CHUNK_DURATION − CHUNK_OVERLAP.
#
# Research basis:
#   • Whispy (Bevilacqua et al., 2024) — shifting buffer with Levenshtein
#     deduplication achieves <2 % WER degradation vs offline Whisper.
#   • Whisper-Streaming (Machácek et al., 2023) — LocalAgreement-2 policy
#     with overlapping re-transcription achieves 3.3 s latency.
#   • Whisper long-form (OpenAI) — overlapping 30 s windows with timestamp-
#     based stitching avoid mid-word cuts.
CHUNK_OVERLAP: float = 1.0        # seconds of overlap between consecutive chunks
OVERLAP_SAMPLES: int = int(SAMPLE_RATE * CHUNK_OVERLAP)  # 16 000 samples (~2.5 words)
STRIDE_SAMPLES: int = BLOCK_SIZE - OVERLAP_SAMPLES        # 24 000 samples

# WSLg RDP bridge attenuates mic amplitude heavily (speech RMS ~0.0003).
# Native laptop/desktop mics sit much higher (speech RMS ~0.03–0.15,
# noise floor ~0.001–0.005). Default 0.008 suits native mics; export
# POLYGLOT_TALK_RMS_THRESHOLD=0.0001 when running under WSLg.
def _get_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid value for {name}: {raw_value!r}. Expected a floating-point number."
        ) from exc


RMS_SILENCE_THRESHOLD: float = _get_float_env(
    "POLYGLOT_TALK_RMS_THRESHOLD", 0.008
)

# ── Microphone / PortAudio stability ────────────────────────────────────────
# WSLg and some Linux PulseAudio devices expose a 44.1 kHz native mic even
# though the ASR pipeline consumes 16 kHz. AudioCapture will open at the native
# device rate when needed and resample down to SAMPLE_RATE.
AUDIO_INPUT_DEVICE: str | int | None = os.environ.get("POLYGLOT_TALK_AUDIO_DEVICE") or None
AUDIO_INPUT_LATENCY: str = os.environ.get("POLYGLOT_TALK_AUDIO_LATENCY", "high")

# ── Sentence accumulation ───────────────────────────────────────────────────
# ASR fragments are buffered until a natural sentence boundary is detected.
# This reduces the number of MT calls and produces better TTS prosody.
SENTENCE_BUFFER_TIMEOUT: float = 2.2   # shorter hold reduces end-to-end conversational latency
SENTENCE_BUFFER_MAXWORDS: int = 16     # flush earlier so MT/TTS process smaller segments

# ── Queue ───────────────────────────────────────────────────────────────────
QUEUE_MAXSIZE: int = 2            # backpressure limit per inter-stage queue
QUEUE_PUT_TIMEOUT: float = 1.0    # seconds before a blocked put retries
QUEUE_GET_TIMEOUT: float = 0.5    # seconds before a blocked get retries

# ── ASR (faster-whisper) ────────────────────────────────────────────────────
# Source ASR routing map (addition-friendly): source language → Whisper model.
# Add a new source language by appending to this map and ASR_TRANSCRIBE_LANG_MAP.
ASR_MODEL_MAP: dict[str, str] = {
    "en": "base.en",
}

# Source language → Whisper `language=` code passed to transcribe().
ASR_TRANSCRIBE_LANG_MAP: dict[str, str] = {
    "en": "en",
}

ASR_SUPPORTED_SOURCE_LANGS: frozenset[str] = frozenset(ASR_MODEL_MAP)

# Default source language and default model route.
SOURCE_LANG: str = "en"       # ISO 639-1 — shared by ASR and MT source side
ASR_MODEL_SIZE: str = ASR_MODEL_MAP[SOURCE_LANG]
ASR_COMPUTE_TYPE: str = "int8"
ASR_DEVICE: str = "auto"
ASR_BEAM_SIZE: int = 1
ASR_LANGUAGE: str = ASR_TRANSCRIBE_LANG_MAP[SOURCE_LANG]  # skip language-detection for speed

ASR_STRIP_TRAILING_PERIOD: bool = True

# ── ASR confidence gating ───────────────────────────────────────────────────
# Suppress hallucinated English text that can appear during silence or when
# source speech is in an unsupported language.  These thresholds mirror
# Whisper-style heuristics: only reject when the model simultaneously signals
# high no-speech probability and low text confidence.
ASR_NO_SPEECH_PROB_THRESHOLD: float = 0.60
ASR_LOW_LOGPROB_THRESHOLD: float = -1.00
ASR_COMPRESSION_RATIO_THRESHOLD: float = 2.40

# ── ASR tail-correction — short-term fix (issue #15) ───────────────────────
# When a new ASR chunk has high word-level Jaccard overlap with the tail of
# the sentence buffer, the tail is *replaced* rather than appended.  This
# prevents Whisper's garbled re-transcription of the overlap region from
# doubling content in the final SENT output.
#
# TAIL_WINDOW:            number of buffer-tail words inspected for overlap.
# TAIL_OVERLAP_THRESHOLD: Jaccard ratio above which replacement is triggered.
# TAIL_MIN_SIZE_RATIO:    new chunk must be ≥ this fraction of the tail slice
#                         size (prevents tiny fragments from wiping long tails).
# NEAR_DUP_THRESHOLD:     raised from 0.85 → 0.92 so that chunks containing a
#                         genuine correction at the tail (slight wording change)
#                         are not discarded by the near-duplicate guard.
ASR_TAIL_WINDOW: int = 12
ASR_TAIL_OVERLAP_THRESHOLD: float = 0.60
ASR_TAIL_MIN_SIZE_RATIO: float = 0.70
ASR_NEAR_DUP_THRESHOLD: float = 0.92   # was 0.85 — relaxed to allow tail corrections

# ── ASR timestamp-based dedup — medium-term fix (issue #15) ────────────────
# When enabled, faster-whisper is asked for per-word timestamps.  Each word's
# audio midpoint is compared against _committed_cutoff (the global audio-stream
# offset of the last committed word).  Only words beyond the cutoff are kept.
# This replaces suffix/prefix text dedup and the near-duplicate guard for the
# time-axis overlap problem; the tail-correction path remains active for the
# sentence buffer.
#
# ASR_USE_WORD_TIMESTAMPS: set False to fall back to the text-dedup path.
# ASR_TIMESTAMP_EPSILON:   50 ms lookahead so boundary words are not
#                          accidentally rejected.
ASR_USE_WORD_TIMESTAMPS: bool = True
ASR_TIMESTAMP_EPSILON: float = 0.05   # seconds; words with midpoint ≤ cutoff + ε are rejected

# ── Language codes — IMPORTANT: two different namespaces are in use ─────────
# MMS-TTS uses ISO 639-3 (three-letter):  "hin", "tam", "tel", "kan", "ben", …
# Argos Translate uses ISO 639-1 (two-letter): "hi" only (for Indian langs)
# MarianMT (Helsinki-NLP) uses ISO 639-1 (two-letter): "ta", "te", "kn", …
# TARGET_LANG is always the ISO 639-3 key used by MMS-TTS.
# SOURCE_LANG stays ISO 639-1 ("en") because Argos, MarianMT, and Whisper
# all use it for the source side.

# ── Translation backend routing ─────────────────────────────────────────────

# Active output language.  Must be a key in both MMS_TTS_MODEL_MAP and
# ARGOS_LANG_MAP or MARIANMT_MODEL_MAP.  Changing only this constant
# switches the full pipeline (ASR → MT → TTS).
TARGET_LANG: str = "hin"      # ISO 639-3 — used as primary language key

# Human-friendly labels used by CLI prompts and setup summaries.
TARGET_LANG_LABELS: dict[str, str] = {
    "hin": "Hindi",
    "tam": "Tamil",
    "tel": "Telugu",
    "kan": "Kannada",
    "ben": "Bengali",
    "mal": "Malayalam",
    "mar": "Marathi",
    "guj": "Gujarati",
}

# Languages for which Argos Translate publishes an en→xx offline package.
# As of 2025, only Hindi is available for Indian languages via argospm.
# All others fall through to MarianMT.
ARGOS_SUPPORTED_LANGS: frozenset[str] = frozenset({"hin"})

# ISO 639-3 → ISO 639-1 bridge for Argos Translate (Hindi only).
ARGOS_LANG_MAP: dict[str, str] = {
    "hin": "hi",   # Hindi — only Indian language with an Argos en→xx package
}

# ISO 639-3 → HuggingFace MarianMT checkpoint.
# Helsinki-NLP only publishes en→xx opus-mt packages for a small subset of Indian
# languages (verified 2025-01).  Marathi and Malayalam have confirmed checkpoints;
# Tamil, Telugu, Kannada, Bengali, and Gujarati do NOT — those fall through to
# the NLLB-200 backend below.
MARIANMT_MODEL_MAP: dict[str, str] = {
    "mal": "Helsinki-NLP/opus-mt-en-ml",   # Malayalam — confirmed on HuggingFace
    "mar": "Helsinki-NLP/opus-mt-en-mr",   # Marathi  — confirmed on HuggingFace
}

# ISO 639-3 → FLORES-200 / NLLB language tag for the five Indian languages that
# have no Helsinki-NLP opus-mt checkpoint.  Used with facebook/nllb-200-distilled-600M
# which ships within the already-installed transformers library — no new dependency.
NLLB_MODEL_ID: str = "facebook/nllb-200-distilled-600M"
NLLB_MAX_LENGTH: int = 160   # cap decode length to reduce MT latency on long streaming segments
NLLB_LANG_MAP: dict[str, str] = {
    "tam": "tam_Taml",   # Tamil
    "tel": "tel_Telu",   # Telugu
    "kan": "kan_Knda",   # Kannada
    "ben": "ben_Beng",   # Bengali
    "guj": "guj_Gujr",   # Gujarati
}


def get_mt_backend(target_lang: str) -> str:
    """Return MT backend for a target language key.

    Values:
      - "argos"  for languages in ARGOS_SUPPORTED_LANGS
      - "marian" for languages in MARIANMT_MODEL_MAP
      - "nllb"   otherwise
    """
    if target_lang in ARGOS_SUPPORTED_LANGS:
        return "argos"
    if target_lang in MARIANMT_MODEL_MAP:
        return "marian"
    return "nllb"


def get_supported_target_langs() -> tuple[str, ...]:
    """Return target languages sorted by stable label then key."""
    return tuple(
        sorted(
            MMS_TTS_MODEL_MAP,
            key=lambda code: (TARGET_LANG_LABELS.get(code, code), code),
        )
    )

# MT_BACKEND is derived automatically from TARGET_LANG — do not set manually.
# Values: "argos" (Hindi) | "marian" (Marathi/Malayalam) | "nllb" (all others)
MT_BACKEND: str = get_mt_backend(TARGET_LANG)

CONTEXT_MAXLEN: int = 2           # rolling source-segment window for prefix

# ── TTS (Facebook MMS-TTS, VITS-based) ──────────────────────────────────────
TTS_OUTPUT_DIR: str = "output"          # directory for saved TTS WAV files

# MMS-TTS model routing: maps TARGET_LANG (ISO 639-3) → HuggingFace checkpoint.
# Each language has its own VITS weights; all use the same VitsModel interface.
# To add a new language: add one entry here AND one entry in either
# ARGOS_LANG_MAP (if Argos supports it) or MARIANMT_MODEL_MAP (otherwise).
MMS_TTS_MODEL_MAP: dict[str, str] = {
    "hin": "facebook/mms-tts-hin",   # Hindi
    "tam": "facebook/mms-tts-tam",   # Tamil
    "tel": "facebook/mms-tts-tel",   # Telugu
    "kan": "facebook/mms-tts-kan",   # Kannada
    "ben": "facebook/mms-tts-ben",   # Bengali
    "mal": "facebook/mms-tts-mal",   # Malayalam
    "mar": "facebook/mms-tts-mar",   # Marathi
    "guj": "facebook/mms-tts-guj",   # Gujarati
}

# Validate TARGET_LANG at import time — fail fast rather than deep in a thread.
assert TARGET_LANG in MMS_TTS_MODEL_MAP, (
    f"TARGET_LANG={TARGET_LANG!r} has no MMS-TTS checkpoint. "
    f"Valid values: {sorted(MMS_TTS_MODEL_MAP)}"
)

# Validate source language route at import-time as well.
assert SOURCE_LANG in ASR_MODEL_MAP, (
    f"SOURCE_LANG={SOURCE_LANG!r} has no ASR model route. "
    f"Valid values: {sorted(ASR_MODEL_MAP)}"
)
assert SOURCE_LANG in ASR_TRANSCRIBE_LANG_MAP, (
    f"SOURCE_LANG={SOURCE_LANG!r} has no ASR transcribe language route. "
    f"Valid values: {sorted(ASR_TRANSCRIBE_LANG_MAP)}"
)

# Every language must have an Argos, MarianMT, or NLLB entry — never none.
_ALL_MT_LANGS = set(ARGOS_LANG_MAP) | set(MARIANMT_MODEL_MAP) | set(NLLB_LANG_MAP)
assert set(MMS_TTS_MODEL_MAP).issubset(_ALL_MT_LANGS), (
    f"These TTS languages have no MT backend: "
    f"{set(MMS_TTS_MODEL_MAP) - _ALL_MT_LANGS}"
)

# Device for MMS-TTS inference.  "auto" → cuda if available, else cpu.
MMS_TTS_DEVICE: str = "auto"

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_FORMAT: str = "[%(asctime)s %(threadName)s] %(levelname)s %(message)s"
LOG_LEVEL: str = "INFO"

# ── ASR hallucination blocklist ──────────────────────────────────────────────
ASR_HALLUCINATION_BLOCKLIST: frozenset = frozenset({
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "you",
    "bye",
    "bye bye",
    "goodbye",
    "please subscribe",
    "like and subscribe",
    "see you next time",
    ".",
    "",
})
