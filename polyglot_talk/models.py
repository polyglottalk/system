"""
models.py — Shared dataclasses passed through the pipeline queues.

Each dataclass carries a chunk_id and a timestamp so that end-to-end
latency can be computed across threads without shared state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioChunk:
    """One 2.5-second audio frame from the microphone."""

    chunk_id: int
    audio: np.ndarray          # float32, shape (BLOCK_SIZE,)
    timestamp: float = field(default_factory=time.perf_counter)
    # Monotonically increasing audio-stream offset (seconds) of the first
    # sample in this chunk.  Set by AudioCapture from the running sample
    # counter so that word timestamps from faster-whisper can be converted
    # to global audio offsets for _committed_cutoff comparison.
    global_offset: float = 0.0


@dataclass
class TextSegment:
    """Raw transcription produced by the ASR engine."""

    chunk_id: int
    text: str
    timestamp: float = field(default_factory=time.perf_counter)
    capture_timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class TranslatedSegment:
    """Translated text ready to be spoken aloud."""

    chunk_id: int
    text: str
    timestamp: float = field(default_factory=time.perf_counter)
    capture_timestamp: float = field(default_factory=time.perf_counter)
