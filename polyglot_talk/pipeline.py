"""
pipeline.py — Orchestrates the four-thread PolyglotTalk pipeline.

Thread layout (all daemon threads):

    AudioCaptureThread → [audio_queue] → ASRThread
                                            → [text_queue] → TranslatorThread
                                                                → [tts_queue] → TTSThread

Model loading policy:
  • ASREngine and Translator each load their model in __init__ (on the main
    thread) so both models are fully in memory before any thread is started.
  • TTSEngine loads MMS-TTS inside TTSEngine.run() (on TTSThread).
    Pipeline.start() waits on ``_tts_engine._model_ready`` before opening the
    microphone so audio is not captured while TTS is still warming up.

Shutdown sequence (via stop()):
    1. stop_event.set()
    2. Push None sentinel into each queue (unblocks any waiting get())
    3. Join threads in reverse order with timeout=5 s each
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from . import config
from .audio_capture import AudioCapture
from .asr_engine import ASREngine
from .translator import Translator
from .tts_engine import TTSEngine

logger = logging.getLogger(__name__)


class Pipeline:
    """Builds, starts, and cleanly shuts down the full S2ST pipeline."""

    def __init__(self, source_lang: str = config.SOURCE_LANG, target_lang: str = config.TARGET_LANG) -> None:
        logger.info("Creating pipeline: %s → %s", source_lang, target_lang)

        # ── Shared synchronisation ────────────────────────────────────────
        self._stop_event = threading.Event()
        # Separate event used by drain() to stop AudioCapture without
        # killing the downstream ASR → Translator → TTS drain chain.
        self._audio_stop_event = threading.Event()

        # ── Inter-stage queues ────────────────────────────────────────────
        self.audio_queue: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.text_queue: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.tts_queue: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)

        # ── Worker instances (models loaded here, in main thread) ─────────
        # NOTE: ASREngine and Translator each load their model in __init__.
        logger.info("Loading ASR model…")
        self._asr_engine = ASREngine(
            audio_queue=self.audio_queue,
            text_queue=self.text_queue,
            stop_event=self._stop_event,
            source_lang=source_lang,
        )

        logger.info("Loading translation model (%s → %s)…", source_lang, target_lang)
        self._translator = Translator(
            text_queue=self.text_queue,
            tts_queue=self.tts_queue,
            stop_event=self._stop_event,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # AudioCapture and TTSEngine have zero-cost __init__
        self._audio_capture = AudioCapture(
            audio_queue=self.audio_queue,
            stop_event=self._stop_event,
            audio_stop_event=self._audio_stop_event,
        )
        self._tts_engine = TTSEngine(
            tts_queue=self.tts_queue,
            stop_event=self._stop_event,
            target_lang=target_lang,
        )

        # Thread handles (created in start())
        self._threads: list[threading.Thread] = []

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Create and start all four daemon threads.

        TTSThread is started first and the other threads are held until
        the TTS model finishes loading (signalled via
        ``_tts_engine._model_ready``).  This prevents the ASR → Translator
        stages from flooding the tts_queue while MMS-TTS is still warming up.
        """
        # ── Start TTS first and wait for model to be ready ────────────────
        tts_thread = threading.Thread(
            target=self._tts_engine.run, name="TTSThread", daemon=True
        )
        self._threads.append(tts_thread)
        tts_thread.start()
        logger.info("Started TTSThread — waiting for MMS-TTS model to load…")

        _TTS_WARMUP_TIMEOUT = 120  # seconds — generous for first-run GPU transfer
        ready = self._tts_engine._model_ready.wait(timeout=_TTS_WARMUP_TIMEOUT)
        if not ready:
            logger.warning(
                "TTS model did not finish loading within %ds — "
                "continuing anyway; first chunks may be dropped.",
                _TTS_WARMUP_TIMEOUT,
            )
        elif self._tts_engine._startup_failed.is_set():
            self.stop()
            raise RuntimeError(
                "TTSEngine failed to load the MMS-TTS model. "
                "Check the log above for details and run setup_models.py to "
                "download missing model files."
            )

        # ── Start microphone first and wait for it to come up cleanly ─────
        audio_thread = threading.Thread(
            target=self._audio_capture.run, name="AudioCaptureThread", daemon=True
        )
        self._threads.append(audio_thread)
        audio_thread.start()
        logger.info("Started AudioCaptureThread — waiting for microphone stream…")

        _AUDIO_STARTUP_TIMEOUT = 10
        self._audio_capture._stream_ready.wait(timeout=_AUDIO_STARTUP_TIMEOUT)
        if self._audio_capture._startup_failed.is_set():
            self.stop()
            raise RuntimeError(
                "Microphone stream failed to start. See the audio log above for device details."
            )

        # ── Start ASR and Translator once the mic is live ──────────────────
        remaining = [
            ("ASRThread", self._asr_engine.run),
            ("TranslatorThread", self._translator.run),
        ]
        for name, target in remaining:
            t = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(t)
            t.start()
            logger.info("Started %s", name)

        print("✓ Pipeline ready. Speak now…")

    def stop(self) -> None:
        """Signal all threads to exit and wait for them to finish."""
        logger.info("Stopping pipeline…")
        self._audio_stop_event.set()
        self._stop_event.set()

        # Push one sentinel per queue to unblock any thread stuck in get()
        for q in (self.audio_queue, self.text_queue, self.tts_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                # Drain one item to make room, then insert sentinel
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass

        # Join in reverse order (downstream first so upstream can drain)
        for t in reversed(self._threads):
            t.join(timeout=5)
            if t.is_alive():
                logger.warning("Thread %s did not stop in time.", t.name)

        print("Pipeline stopped.")

    def drain(self) -> None:
        """Graceful stop: halt audio capture and let queued work finish.

        Sends a sentinel only to audio_queue.  The None propagates naturally
        through ASR → Translator → TTS so every thread exits after draining
        its own queue.  Does NOT set stop_event, so threads process every item
        already enqueued before they see the sentinel.
        """
        logger.info("Draining pipeline — finishing queued TTS…")
        print("\nStopping capture — finishing queued translations…")
        # Stop AudioCapture immediately — no more mic audio needed
        self._audio_stop_event.set()
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(None)
            except queue.Full:
                pass

        # Wait for all threads to finish draining naturally
        for t in self._threads:
            t.join(timeout=30)
            if t.is_alive():
                logger.warning("Thread %s still alive after drain timeout.", t.name)

        # Ensure everything is fully torn down
        self._stop_event.set()
        print("Pipeline stopped.")

    def wait(self) -> None:
        """Block until Enter (graceful drain), Ctrl+C (hard stop), or auto-stop."""
        print("  Press Enter to stop recording and finish remaining TTS.")
        print("  Press Ctrl+C to stop immediately.")

        # Thread that watches for Enter key and triggers a graceful drain
        _drain_requested = threading.Event()

        def _key_watcher() -> None:
            try:
                input()  # blocks until Enter
                _drain_requested.set()
            except EOFError:
                pass  # non-interactive stdin — ignore

        key_thread = threading.Thread(target=_key_watcher, name="KeyWatcher", daemon=True)
        key_thread.start()

        try:
            while not self._stop_event.is_set() and not _drain_requested.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nInterrupt received — shutting down…")
            self.stop()
            key_thread.join(timeout=1)  # clean up key watcher
            return

        if _drain_requested.is_set():
            self.drain()
        else:
            # stop_event was set internally (e.g. silence auto-stop)
            self.stop()

        # Ensure key watcher thread is fully cleaned up before exiting
        key_thread.join(timeout=1)
