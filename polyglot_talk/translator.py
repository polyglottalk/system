"""
translator.py — Machine-translation thread with backend dispatch.

Backend selection
-----------------
Argos Translate is used for Hindi (the only Indian language for which
an offline Argos en→xx package is published on argospm).

Helsinki-NLP MarianMT (via HuggingFace transformers) is used for Indian
languages that have a confirmed Helsinki-NLP opus-mt checkpoint (currently
Marathi and Malayalam).

Facebook NLLB-200 (via HuggingFace transformers) is used for the remaining
Indian languages that lack a MarianMT checkpoint (Tamil, Telugu, Kannada,
Bengali, Gujarati).  All three backends are loaded ONCE in __init__;
run() never imports or loads anything.

Context continuity
------------------
Maintains a rolling deque of the last CONTEXT_MAXLEN source-language
segments.  Before each translation the previous segments are prepended
to the new text so the model sees sentence-boundary context.  After
translation the re-translated prefix is trimmed from the output (exact
match first, fuzzy difflib fallback).

The backend-specific objects are stored as self._argos_translate_fn,
self._marian_pipeline, or self._nllb_pipeline; _translate() dispatches
to whichever is active.  Everything else (context window, prefix trim,
queue discipline) is identical regardless of backend.
"""

from __future__ import annotations

import collections
import difflib
import logging
import queue
import threading
import time
from typing import Any, Callable, Deque

from . import config
from .models import TextSegment, TranslatedSegment

logger = logging.getLogger(__name__)


def _get_broadcaster():
    """Lazy import — no-op when --dashboard is not used."""
    try:
        from dashboard_server import broadcaster  # noqa: PLC0415
        return broadcaster
    except ImportError:
        return None


class Translator:
    """Translates TextSegment objects into TranslatedSegment objects.

    Context continuity
    ------------------
    self._context_source      — deque[str], last N source-text segments
    self._context_translated  — deque[str], last N translated segments
    """

    def __init__(
        self,
        text_queue: queue.Queue,
        tts_queue: queue.Queue,
        stop_event: threading.Event,
        source_lang: str = config.SOURCE_LANG,
        target_lang: str = config.TARGET_LANG,
        context_maxlen: int = config.CONTEXT_MAXLEN,
    ) -> None:
        self._text_queue = text_queue
        self._tts_queue = tts_queue
        self._stop_event = stop_event
        self._source_lang = source_lang
        # _target_lang is the ISO 639-3 code used for display (e.g. "hin", "tam").
        self._target_lang = target_lang
        # Derive backend from the *instance* target language.
        self._mt_backend: str = config.get_mt_backend(target_lang)

        self._context_source: Deque[str] = collections.deque(maxlen=context_maxlen)
        self._context_translated: Deque[str] = collections.deque(maxlen=context_maxlen)

        # Backend-specific objects — set by _load_model()
        self._argos_translate_fn: Callable[[str], str] | None = None
        self._marian_pipeline: Any | None = None
        self._nllb_pipeline: Any | None = None

        logger.info(
            "Loading translation model (%s → %s) via %s…",
            source_lang, target_lang, self._mt_backend,
        )
        t0 = time.perf_counter()
        self._load_model()
        logger.info("Translation model loaded in %.1fs", time.perf_counter() - t0)

    # ── Thread target ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Consume TextSegments, translate with context, push TranslatedSegments."""
        while not self._stop_event.is_set():
            try:
                item = self._text_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                continue

            if item is None:  # shutdown sentinel
                try:
                    self._tts_queue.put_nowait(None)
                except queue.Full:
                    pass
                break

            assert isinstance(item, TextSegment)

            if not item.text.strip():
                logger.debug("Chunk #%d skipped — empty text.", item.chunk_id)
                continue

            t0 = time.perf_counter()
            translated = self._translate_with_context(item.text)
            elapsed = time.perf_counter() - t0

            if not translated:
                logger.warning(
                    "Chunk #%d produced empty translation, skipping.", item.chunk_id
                )
                continue

            logger.debug(
                "Translation done (%.3fs) chunk #%d: %r",
                elapsed, item.chunk_id, translated,
            )
            print(f"[\u2192{self._target_lang.upper()}  #{item.chunk_id:>4d}] {translated}", flush=True)
            _bc = _get_broadcaster()
            if _bc is not None:
                _bc.emit({
                    "type": "translation_done",
                    "chunk_id": item.chunk_id,
                    "text": translated,
                    "lang": self._target_lang,
                })

            segment = TranslatedSegment(
                chunk_id=item.chunk_id,
                text=translated,
                timestamp=time.perf_counter(),
                capture_timestamp=item.capture_timestamp,
            )
            self._put(segment)

        logger.info("Translator stopped.")

    # ── Context-aware translation ────────────────────────────────────────────

    def _translate_with_context(self, new_text: str) -> str:
        """Translate new_text with rolling context prefix for continuity.

        Steps
        -----
        1. Build ``prefix_source`` from _context_source deque.
        2. Build ``prefix_translated`` from _context_translated deque
           (actual previous outputs — no extra translation call needed).
        3. Concatenate: ``combined_input = prefix_source + " " + new_text``.
        4. Translate ``combined_input`` (ONE backend call per segment).
        5. Strip ``prefix_translated`` from start of full output
           (exact match first, fuzzy difflib fallback).
        6. Update both context deques.
        7. Return trimmed translation (fallback to full if trim fails).
        """
        prefix_source = " ".join(self._context_source).strip()
        prefix_translated = " ".join(self._context_translated).strip()

        combined_input = f"{prefix_source} {new_text}" if prefix_source else new_text

        full_translation = self._translate(combined_input)

        trimmed = self._trim_prefix(full_translation, prefix_translated) if prefix_translated else full_translation

        self._context_source.append(new_text)
        result = trimmed.strip() if trimmed.strip() else full_translation.strip()
        self._context_translated.append(result)
        return result

    def _trim_prefix(self, full: str, prefix_tr: str) -> str:
        """Remove translated prefix from start of full translation.

        1. Exact match: strip ``prefix_tr`` from start of ``full``.
        2. Fuzzy fallback: use difflib to find longest matching prefix.
        3. If overlap < 30% of prefix_tr length, return ``full`` unchanged.
        """
        if not prefix_tr:
            return full

        if full.startswith(prefix_tr):
            return full[len(prefix_tr):].strip()

        matcher = difflib.SequenceMatcher(None, full, prefix_tr, autojunk=False)
        blocks = matcher.get_matching_blocks()
        trimmed_end = 0
        for block in blocks:
            if block.a == trimmed_end and block.b == 0:
                trimmed_end = block.a + block.size
            else:
                break

        overlap_ratio = trimmed_end / max(len(prefix_tr), 1)
        if trimmed_end > 0 and overlap_ratio >= 0.30:
            logger.debug(
                "Fuzzy prefix trim: removed %d chars (%.0f%% overlap)",
                trimmed_end, overlap_ratio * 100,
            )
            return full[trimmed_end:].strip()

        logger.debug("Prefix trim skipped — overlap %.0f%% < 30%%", overlap_ratio * 100)
        return full

    # ── Translation dispatch ────────────────────────────────────────────────

    def _translate(self, text: str) -> str:
        """Dispatch to the active MT backend (Argos, MarianMT, or NLLB)."""
        if self._mt_backend == "argos":
            assert self._argos_translate_fn is not None
            return self._argos_translate_fn(text)
        elif self._mt_backend == "marian":
            assert self._marian_pipeline is not None
            result = self._marian_pipeline(text)
            # transformers pipeline returns list[dict] with key "translation_text"
            return result[0]["translation_text"]
        else:  # nllb
            assert self._nllb_pipeline is not None
            result = self._nllb_pipeline(text)
            return result[0]["translation_text"]

    def _load_model(self) -> None:
        """Load the appropriate MT backend based on config.MT_BACKEND.

        Argos path
        ----------
        Verifies the installed Argos package for en→{argos_code} and
        binds a closure over argostranslate.translate.translate.

        MarianMT path
        -------------
        Loads Helsinki-NLP/opus-mt-en-{xx} via transformers pipeline.
        Used for Marathi (mr) and Malayalam (ml) which have confirmed checkpoints.

        NLLB path
        ---------
        Loads facebook/nllb-200-distilled-600M for the five Indian languages
        that lack a Helsinki-NLP opus-mt checkpoint (Tamil, Telugu, Kannada,
        Bengali, Gujarati).  Uses the same transformers pipeline API.

        Raises
        ------
        RuntimeError
            Argos path: if the required language package is not installed.
            Run 'python setup_models.py' first.
        """
        if self._mt_backend == "argos":
            import argostranslate.package
            import argostranslate.translate

            argos_code = config.ARGOS_LANG_MAP[self._target_lang]
            installed = argostranslate.package.get_installed_packages()
            found = any(
                p.from_code == self._source_lang and p.to_code == argos_code
                for p in installed
            )
            if not found:
                raise RuntimeError(
                    f"Argos Translate package not found for "
                    f"{self._source_lang!r} → {argos_code!r}. "
                    f"Run 'python setup_models.py' first."
                )
            # Capture the resolved codes in a closure so _translate() is simple.
            src = self._source_lang
            tgt = argos_code
            self._argos_translate_fn = lambda text: argostranslate.translate.translate(text, src, tgt)
            logger.debug("Argos backend ready: %s → %s", src, tgt)

        elif self._mt_backend == "marian":
            from transformers import pipeline as hf_pipeline

            model_id = config.MARIANMT_MODEL_MAP[self._target_lang]
            # device=0 → first CUDA GPU; device=-1 → CPU.
            # "auto" is not accepted by the text2text-generation pipeline;
            # resolve manually to match MMS_TTS_DEVICE behaviour.
            import torch
            device = 0 if torch.cuda.is_available() else -1
            self._marian_pipeline = hf_pipeline(
                "translation",
                model=model_id,
                device=device,
            )
            logger.debug("MarianMT backend ready: %s (device=%d)", model_id, device)

        else:  # nllb
            from transformers import pipeline as hf_pipeline

            nllb_tgt = config.NLLB_LANG_MAP[self._target_lang]
            import torch
            device = 0 if torch.cuda.is_available() else -1
            self._nllb_pipeline = hf_pipeline(
                "translation",
                model=config.NLLB_MODEL_ID,
                src_lang="eng_Latn",
                tgt_lang=nllb_tgt,
                device=device,
                max_length=config.NLLB_MAX_LENGTH,
            )
            logger.debug(
                "NLLB backend ready: %s → %s (device=%d)",
                config.NLLB_MODEL_ID, nllb_tgt, device,
            )

    # ── Queue helper ────────────────────────────────────────────────────────

    def _put(self, segment: TranslatedSegment) -> None:
        """Push to tts_queue with drop-oldest strategy on Full.

        Never blocks — if the queue is full the oldest pending translation
        is evicted so TTS always speaks the most recent output.
        """
        try:
            self._tts_queue.put_nowait(segment)
        except queue.Full:
            try:
                dropped = self._tts_queue.get_nowait()
                logger.warning(
                    "tts_queue full — evicted oldest chunk #%d to insert chunk #%d",
                    dropped.chunk_id, segment.chunk_id,
                )
            except queue.Empty:
                pass
            try:
                self._tts_queue.put_nowait(segment)
            except queue.Full:
                logger.warning("tts_queue still full — dropping chunk #%d", segment.chunk_id)
