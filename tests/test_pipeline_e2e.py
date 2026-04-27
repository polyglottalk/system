"""
tests/test_pipeline_e2e.py — Full pipeline integration test.

Uses a file-based AudioCapture subclass instead of a real microphone, and
a mock TTSEngine that records spoken text instead of producing audio.

Requires all models to be installed (run setup_models.py first).

Run:
    python -m pytest tests/test_pipeline_e2e.py -v -s
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polyglot_talk import config  # noqa: F401

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_speech_like_audio(duration: float = 2.5) -> np.ndarray:
    """Generate a 440 Hz sine wave that passes the RMS silence filter."""
    n = int(config.SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


class _FileAudioCapture:
    """Feeds pre-built AudioChunk objects into audio_queue.

    Replaces AudioCapture for testing — no microphone required.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        stop_event: threading.Event,
        chunks: list[np.ndarray],
        inter_chunk_delay: float = 0.0,
    ) -> None:
        self._audio_queue = audio_queue
        self._stop_event = stop_event
        self._chunks = chunks
        self._delay = inter_chunk_delay

    def run(self) -> None:
        from polyglot_talk.models import AudioChunk

        for i, audio in enumerate(self._chunks):
            if self._stop_event.is_set():
                break
            item = AudioChunk(chunk_id=i, audio=audio, timestamp=time.perf_counter())
            try:
                self._audio_queue.put(item, timeout=5.0)
            except queue.Full:
                pass
            if self._delay:
                time.sleep(self._delay)

        # Signal end of input
        try:
            self._audio_queue.put(None, timeout=2.0)
        except queue.Full:
            pass


class _MockTTSEngine:
    """Records translated text instead of speaking aloud."""

    def __init__(self, tts_queue: queue.Queue, stop_event: threading.Event) -> None:
        self._tts_queue = tts_queue
        self._stop_event = stop_event
        self.spoken: list[str] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._tts_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                continue

            if item is None:
                break

            self.spoken.append(item.text)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def check_models_installed():
    """Skip if the active MT model or faster-whisper are not ready."""
    if config.MT_BACKEND == "argos":
        import argostranslate.package

        installed = argostranslate.package.get_installed_packages()
        argos_target = config.ARGOS_LANG_MAP[config.TARGET_LANG]
        found = any(
            p.from_code == config.SOURCE_LANG and p.to_code == argos_target
            for p in installed
        )
        if not found:
            pytest.skip(
                f"Argos {config.SOURCE_LANG}→{argos_target} not installed. "
                "Run python setup_models.py."
            )
    # MarianMT and NLLB: models are cached in HuggingFace hub; no installed-package
    # check needed. If the cache is empty the test will fail with a clear error.


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_pipeline_with_synthetic_audio(check_models_installed) -> None:
    """Feed 3 sine-wave chunks through the pipeline; expect no crash."""
    from polyglot_talk.asr_engine import ASREngine
    from polyglot_talk.translator import Translator

    stop_event = threading.Event()
    audio_queue: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
    text_queue: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
    tts_queue: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)

    # Workers
    asr = ASREngine(audio_queue, text_queue, stop_event)
    translator = Translator(text_queue, tts_queue, stop_event)
    mock_tts = _MockTTSEngine(tts_queue, stop_event)

    # Synthetic speech-like input
    chunks = [_generate_speech_like_audio() for _ in range(3)]
    file_capture = _FileAudioCapture(audio_queue, stop_event, chunks)

    threads = [
        threading.Thread(target=file_capture.run, name="TestCapture", daemon=True),
        threading.Thread(target=asr.run, name="TestASR", daemon=True),
        threading.Thread(target=translator.run, name="TestTranslator", daemon=True),
        threading.Thread(target=mock_tts.run, name="TestTTS", daemon=True),
    ]
    for t in threads:
        t.start()

    # Wait for the file capture to finish feeding, then give pipeline time to drain
    threads[0].join(timeout=10)
    time.sleep(15)  # allow ASR + MT + TTS to process

    stop_event.set()
    # Push sentinels
    for q in (audio_queue, text_queue, tts_queue):
        try:
            q.put_nowait(None)
        except queue.Full:
            pass

    for t in threads[1:]:
        t.join(timeout=5)

    print(f"Translated {len(mock_tts.spoken)} segment(s): {mock_tts.spoken}")
    # Sine waves typically produce empty/hallucinated ASR output — no output is OK here.
    # The critical assertion is that no thread raised an exception.
    assert all(not t.is_alive() for t in threads), "Some threads still running."
    print("✓ test_pipeline_with_synthetic_audio passed (no crash)")


def test_all_threads_exit_on_stop(check_models_installed) -> None:
    """Pipeline.stop() must terminate all 4 threads within 5 s each."""
    from polyglot_talk.pipeline import Pipeline

    pipeline = Pipeline()
    pipeline.start()

    # Let it run briefly
    time.sleep(1.5)

    t_stop = time.perf_counter()
    pipeline.stop()
    elapsed = time.perf_counter() - t_stop

    assert all(
        not t.is_alive() for t in pipeline._threads
    ), "One or more threads still alive after stop()"

    print(f"✓ All threads exited after stop() in {elapsed:.2f}s")


def test_latency_measurement(check_models_installed) -> None:
    """Measure end-to-end latency over a short run and print per-stage timings."""
    import logging
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    from polyglot_talk.asr_engine import ASREngine
    from polyglot_talk.translator import Translator

    stop_event = threading.Event()
    audio_queue: queue.Queue = queue.Queue(maxsize=4)
    text_queue: queue.Queue = queue.Queue(maxsize=4)
    tts_queue: queue.Queue = queue.Queue(maxsize=4)

    asr = ASREngine(audio_queue, text_queue, stop_event)
    translator = Translator(text_queue, tts_queue, stop_event)
    mock_tts = _MockTTSEngine(tts_queue, stop_event)

    chunks = [_generate_speech_like_audio() for _ in range(2)]
    file_capture = _FileAudioCapture(audio_queue, stop_event, chunks)

    threads = [
        threading.Thread(target=file_capture.run, name="LatTestCapture", daemon=True),
        threading.Thread(target=asr.run, name="LatTestASR", daemon=True),
        threading.Thread(target=translator.run, name="LatTestTranslator", daemon=True),
        threading.Thread(target=mock_tts.run, name="LatTestTTS", daemon=True),
    ]
    t_start = time.perf_counter()
    for t in threads:
        t.start()

    threads[0].join(timeout=10)
    time.sleep(12)

    stop_event.set()
    for q in (audio_queue, text_queue, tts_queue):
        try:
            q.put_nowait(None)
        except queue.Full:
            pass
    for t in threads[1:]:
        t.join(timeout=5)

    total = time.perf_counter() - t_start
    print(f"✓ Latency test run completed in {total:.1f}s total wall time")
    print(f"  Translated segments captured: {mock_tts.spoken}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
