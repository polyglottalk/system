"""
audio_capture.py — Microphone input thread for PolyglotTalk.

Opens a sounddevice InputStream and assembles raw callback blocks into
fixed-size AudioChunk objects, pushing them onto audio_queue.

Overlapping chunks
------------------
Consecutive chunks share ``config.CHUNK_OVERLAP`` seconds of audio so that
words straddling a chunk boundary are never cut mid-phoneme.  The effective
stride (new audio per chunk) is ``CHUNK_DURATION − CHUNK_OVERLAP``.

    chunk 0:  [0 ────────── BLOCK_SIZE]
    chunk 1:      [STRIDE ────────── STRIDE+BLOCK_SIZE]
    chunk 2:          [2·STRIDE ──── 2·STRIDE+BLOCK_SIZE]

Research basis  (see config.py for full citations):
  • Whispy  — shifting buffer with re-transcription + Levenshtein dedup
  • Whisper-Streaming — LocalAgreement-2 over overlapping re-transcriptions
  • Whisper long-form — overlapping 30 s windows with timestamp stitching

Backpressure: if audio_queue is full the current chunk is DROPPED and a
warning is logged — we never accumulate unbounded audio memory.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from . import config
from .models import AudioChunk

logger = logging.getLogger(__name__)


def _get_pause_event():
    """Return the broadcaster's pause_event, or None if dashboard not active."""
    try:
        from dashboard_server import broadcaster  # noqa: PLC0415
        return broadcaster.pause_event
    except ImportError:
        return None


class AudioCapture:
    """Captures microphone audio in overlapping chunks.

    Architecture
    ------------
    sounddevice callback (internal SD thread)
        → self._raw_q  (thread-safe Queue of raw 1-D float32 blocks)
            → run() assembles blocks into BLOCK_SIZE chunks with CHUNK_OVERLAP
                → audio_queue (shared pipeline queue)

    Each emitted chunk is BLOCK_SIZE samples long.  Consecutive chunks share
    OVERLAP_SAMPLES samples, so the stride (new audio per chunk) equals
    STRIDE_SAMPLES = BLOCK_SIZE − OVERLAP_SAMPLES.

    The double-queue pattern avoids numpy operations inside the real-time
    callback.  All assembly happens in the run() thread.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        stop_event: threading.Event,
        sample_rate: int = config.SAMPLE_RATE,
        chunk_duration: float = config.CHUNK_DURATION,
        audio_stop_event: threading.Event | None = None,
    ) -> None:
        self._audio_queue = audio_queue
        self._stop_event = stop_event
        # Dedicated event that drain() sets to stop capture without killing
        # the downstream ASR/Translator/TTS drain.  Falls back to stop_event
        # if not supplied (e.g. in unit tests).
        self._audio_stop_event = audio_stop_event if audio_stop_event is not None else stop_event
        self._sample_rate = sample_rate
        self._block_size = int(sample_rate * chunk_duration)
        self._overlap_samples = config.OVERLAP_SAMPLES
        self._stride_samples = config.STRIDE_SAMPLES

        # Internal queue: callback → assembly loop (no lock needed)
        self._raw_q: queue.Queue[np.ndarray] = queue.Queue()

        self._chunk_id: int = 0
        self._global_sample_offset: int = 0  # total samples emitted so far (for global_offset)
        self._stream_ready = threading.Event()
        self._startup_failed = threading.Event()

    def _candidate_stream_params(self) -> list[tuple[int | str | None, int, str]]:
        """Return candidate (device, samplerate, label) tuples for mic startup.

        We try the requested pipeline rate first (16 kHz), then the device's
        native sample rate as a fallback.  On WSLg/PulseAudio this avoids the
        intermittent PortAudio startup timeout seen on some machines.
        """
        candidates: list[tuple[int | str | None, int, str]] = []
        seen: set[tuple[str, int]] = set()

        default_device = (
            sd.default.device[0]
            if isinstance(sd.default.device, (list, tuple))
            else sd.default.device
        )
        for device in (config.AUDIO_INPUT_DEVICE, default_device, "default", "pulse", None):
            label = "<default>" if device is None else str(device)
            try:
                info = sd.query_devices(device, "input")
            except Exception:
                continue
            if int(info.get("max_input_channels", 0)) < 1:
                continue

            native_rate = int(info.get("default_samplerate") or self._sample_rate)
            for rate in (self._sample_rate, native_rate):
                key = (label, rate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((device, rate, label))

        if not candidates:
            candidates.append((config.AUDIO_INPUT_DEVICE, self._sample_rate, "<default>"))
        return candidates

    def _resample_block(
        self,
        block: np.ndarray,
        from_rate: int,
        to_rate: int,
    ) -> np.ndarray:
        """Resample a mono float32 block to the pipeline's target rate."""
        if from_rate == to_rate or len(block) == 0:
            return block.astype(np.float32, copy=False)

        new_len = max(1, int(round(len(block) * to_rate / from_rate)))
        old_x = np.linspace(0.0, 1.0, num=len(block), endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        return np.interp(new_x, old_x, block).astype(np.float32, copy=False)

    # ── sounddevice callback (runs in SD internal thread) ──────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("sounddevice status: %s", status)
        # indata shape: (frames, channels) — take channel 0, make 1-D copy
        self._raw_q.put_nowait(indata[:, 0].copy())

    # ── Thread target ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Open microphone stream, assemble overlapping chunks, push to audio_queue.

        Overlap strategy
        ----------------
        A rolling buffer accumulates raw blocks.  Whenever it holds at least
        BLOCK_SIZE samples a full chunk is emitted.  Instead of discarding
        the entire chunk (no-overlap, old behaviour), we only advance by
        STRIDE_SAMPLES, keeping the last OVERLAP_SAMPLES as the start of the
        next chunk.  This guarantees that words at boundaries appear in TWO
        consecutive chunks, giving Whisper enough acoustic context to
        transcribe them correctly on at least one side.
        """
        buffer: list[np.ndarray] = []
        buffer_samples: int = 0

        stream: sd.InputStream | None = None
        stream_rate = self._sample_rate
        stream_label = "<default>"
        last_error: Exception | None = None

        for device, candidate_rate, label in self._candidate_stream_params():
            try:
                logger.info(
                    "Opening microphone stream (device=%s, samplerate=%d Hz, latency=%s)…",
                    label,
                    candidate_rate,
                    config.AUDIO_INPUT_LATENCY,
                )
                sd.check_input_settings(
                    device=device,
                    samplerate=candidate_rate,
                    channels=1,
                    dtype="float32",
                )
                stream = sd.InputStream(
                    device=device,
                    samplerate=candidate_rate,
                    channels=1,
                    dtype="float32",
                    latency=config.AUDIO_INPUT_LATENCY,
                    callback=self._audio_callback,
                )
                stream_rate = candidate_rate
                stream_label = label
                break
            except (sd.PortAudioError, OSError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Microphone open attempt failed (device=%s, samplerate=%d Hz): %s",
                    label,
                    candidate_rate,
                    exc,
                )

        if stream is None:
            self._startup_failed.set()
            self._stream_ready.set()
            self._audio_stop_event.set()
            self._stop_event.set()
            logger.error(
                "Could not start microphone stream. Verify your input device and the "
                "PulseAudio/WSLg bridge. Hint: `pactl list sources short` or set "
                "POLYGLOT_TALK_AUDIO_DEVICE=pulse` before running the app.",
                exc_info=last_error,
            )
            try:
                self._audio_queue.put_nowait(None)
            except queue.Full:
                pass
            return

        with stream:
            self._stream_ready.set()
            logger.info(
                "Microphone stream opened (device=%s, stream=%d Hz → pipeline=%d Hz, overlap=%dms, stride=%dms)",
                stream_label,
                stream_rate,
                self._sample_rate,
                int(self._overlap_samples / self._sample_rate * 1000),
                int(self._stride_samples / self._sample_rate * 1000),
            )

            while not self._stop_event.is_set() and not self._audio_stop_event.is_set():
                # ── Pause support ────────────────────────────────────────
                pause_ev = _get_pause_event()
                if pause_ev is not None and pause_ev.is_set():
                    # Drain incoming mic data so buffer doesn't grow while paused
                    try:
                        self._raw_q.get_nowait()
                    except queue.Empty:
                        pass
                    time.sleep(0.05)
                    continue

                # Drain one raw block from the callback queue
                try:
                    block = self._raw_q.get(timeout=config.QUEUE_GET_TIMEOUT)
                except queue.Empty:
                    continue

                if stream_rate != self._sample_rate:
                    block = self._resample_block(block, from_rate=stream_rate, to_rate=self._sample_rate)

                buffer.append(block)
                buffer_samples += len(block)

                # Emit overlapping chunks as they become available
                while buffer_samples >= self._block_size:
                    full = np.concatenate(buffer)
                    chunk_audio = full[: self._block_size]

                    # Advance by stride (keep overlap for next chunk)
                    remainder = full[self._stride_samples :]

                    buffer = [remainder] if len(remainder) > 0 else []
                    buffer_samples = len(remainder)

                    item = AudioChunk(
                        chunk_id=self._chunk_id,
                        audio=chunk_audio,
                        timestamp=time.perf_counter(),
                        global_offset=self._global_sample_offset / self._sample_rate,
                    )
                    self._chunk_id += 1
                    self._global_sample_offset += self._stride_samples
                    self._push(item)

        logger.info("AudioCapture stopped.")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _push(self, item: AudioChunk) -> None:
        """Push chunk to audio_queue with drop-oldest strategy on Full.

        If the queue is full the oldest unprocessed chunk is evicted so
        ASR always sees the most recent audio.  This method never blocks,
        preserving true pipeline parallelism.
        """
        try:
            self._audio_queue.put_nowait(item)
        except queue.Full:
            try:
                dropped = self._audio_queue.get_nowait()
                logger.warning(
                    "audio_queue full — evicted oldest chunk #%d to insert chunk #%d",
                    dropped.chunk_id,
                    item.chunk_id,
                )
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(item)
            except queue.Full:
                logger.warning(
                    "audio_queue still full — dropping chunk #%d", item.chunk_id
                )
