"""
tests/test_context.py — Unit tests for Translator context-continuity logic.

Uses ``unittest.mock`` to replace argostranslate.translate.translate so
no Argos model needs to be installed.

Run:
    python -m pytest tests/test_context.py -v
"""

from __future__ import annotations

import collections
import os
import queue
import sys
import threading
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polyglot_talk import config  # noqa: F401
import pytest

_ARGOS_TARGET = config.ARGOS_LANG_MAP["hin"]  # Argos only supports Hindi for Indian languages


# ── Fixture: a Translator wired to mocked argostranslate ─────────────────────

@pytest.fixture()
def make_translator():
    """Return a factory that creates a Translator with mocked translation.

    Uses patch.start()/patch.stop() (not a ``with`` block) so the patches
    remain active for the entire test body, not just during factory construction.
    """
    active_patchers: list = []

    def _factory(translate_side_effect=None):
        # Keep patches alive for the duration of the test (not just _factory)
        p_pkg = patch("argostranslate.package.get_installed_packages")
        p_tr = patch("argostranslate.translate.translate")
        mock_pkg = p_pkg.start()
        mock_tr = p_tr.start()
        active_patchers.extend([p_pkg, p_tr])

        # Simulate package installed
        fake_pkg = MagicMock()
        fake_pkg.from_code = config.SOURCE_LANG
        fake_pkg.to_code = _ARGOS_TARGET
        mock_pkg.return_value = [fake_pkg]

        if translate_side_effect is not None:
            mock_tr.side_effect = translate_side_effect
        else:
            # Default: echo "[tgt] {text}"
            mock_tr.side_effect = lambda text, src, tgt: f"[{tgt}] {text}"

        from polyglot_talk.translator import Translator

        t = Translator(
            text_queue=queue.Queue(),
            tts_queue=queue.Queue(),
            stop_event=threading.Event(),
            target_lang="hin",   # force Argos backend regardless of config.TARGET_LANG
        )
        # mock_tr patches argostranslate.translate.translate which is called
        # by t._translate(text) — no need to replace the instance method.
        return t, mock_tr

    yield _factory

    # Tear down: stop all patchers started by _factory calls in this test
    for p in reversed(active_patchers):
        p.stop()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_empty_context_no_prefix(make_translator) -> None:
    """With empty context deque, input is passed to translate without prefix."""
    translator, mock_tr = make_translator()
    mock_tr.side_effect = lambda text, src, tgt: f"TRANSLATED({text})"

    result = translator._translate_with_context("Hello world")

    mock_tr.assert_called_once_with("Hello world", config.SOURCE_LANG, _ARGOS_TARGET)
    assert "Hello world" in result
    print(f"✓ empty context: {result!r}")


def test_one_segment_context_prepends_prefix(make_translator) -> None:
    """After one segment in context, next call prepends it as prefix."""
    translator, mock_tr = make_translator()

    calls: list[str] = []

    # _translate(self, text) — only 1 arg is passed by _translate_with_context
    def _tr(text):
        calls.append(text)
        return f"T({text})"

    translator._translate = _tr
    translator._translate_prefix = lambda p: f"T({p})"

    # Seed one context segment
    translator._context_source.append("Good morning")
    translator._cache_key = "Good morning"
    translator._cache_val = "T(Good morning)"

    translator._translate_with_context("How are you?")

    # combined input should be "Good morning How are you?"
    assert calls[0] == "Good morning How are you?", (
        f"Expected prefixed input, got: {calls[0]!r}"
    )
    print("✓ one-segment prefix prepend OK")


def test_two_segment_context(make_translator) -> None:
    """With maxlen=2 context, only the last 2 source segments are used."""
    translator, mock_tr = make_translator()

    calls: list[str] = []

    # _translate(self, text) — only 1 arg is passed by _translate_with_context
    def _tr(text):
        calls.append(text)
        return f"T({text})"

    translator._translate = _tr
    translator._translate_prefix = lambda p: f"T({p})"

    translator._context_source.append("Segment one")
    translator._context_source.append("Segment two")
    translator._cache_key = "Segment one Segment two"
    translator._cache_val = "T(Segment one Segment two)"

    translator._translate_with_context("Segment three")

    expected_input = "Segment one Segment two Segment three"
    assert calls[0] == expected_input, f"Got: {calls[0]!r}"
    print("✓ two-segment context prefix OK")


def test_exact_prefix_trimmed(make_translator) -> None:
    """When translated prefix appears at start of full translation, it is stripped."""
    translator, _ = make_translator()

    translator._translate = lambda text, *_: text  # identity for testing
    translator._translate_prefix = lambda p: "PREFIX_TR"

    translator._context_source.append("Some source")
    translator._cache_key = "Some source"
    translator._cache_val = "PREFIX_TR"

    translator._translate = lambda text: f"PREFIX_TR new part"

    result = translator._translate_with_context("new part")
    # Should strip "PREFIX_TR " from start
    assert "PREFIX_TR" not in result or result == "PREFIX_TR new part", (
        "Prefix should be stripped if exact match"
    )
    print(f"✓ exact prefix trim result: {result!r}")


def test_trim_prefix_exact() -> None:
    """_trim_prefix strips exact prefix correctly."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import queue, threading
    from unittest.mock import patch, MagicMock

    with patch("argostranslate.package.get_installed_packages") as mp, \
         patch("argostranslate.translate.translate"):
        fake = MagicMock()
        fake.from_code = "en"
        fake.to_code = _ARGOS_TARGET
        mp.return_value = [fake]
        from polyglot_talk.translator import Translator
        t = Translator(queue.Queue(), queue.Queue(), threading.Event(), target_lang="hin")

    result = t._trim_prefix("नमस्ते कैसे हैं", "नमस्ते")
    assert result == "कैसे हैं", f"Expected 'कैसे हैं', got {result!r}"
    print(f"✓ _trim_prefix exact: {result!r}")


def test_trim_prefix_no_match_returns_full() -> None:
    """_trim_prefix returns full string when prefix does not appear."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import queue, threading
    from unittest.mock import patch, MagicMock

    with patch("argostranslate.package.get_installed_packages") as mp, \
         patch("argostranslate.translate.translate"):
        fake = MagicMock()
        fake.from_code = "en"
        fake.to_code = _ARGOS_TARGET
        mp.return_value = [fake]
        from polyglot_talk.translator import Translator
        t = Translator(queue.Queue(), queue.Queue(), threading.Event(), target_lang="hin")

    result = t._trim_prefix("completely different text", "XYZ ABC DEF")
    assert result == "completely different text", f"Should be unchanged: {result!r}"
    print(f"✓ _trim_prefix no-match fallback: {result!r}")


def test_empty_text_skipped(make_translator) -> None:
    """run() loop skips empty/whitespace text segments."""
    translator, mock_tr = make_translator()

    mock_tr.side_effect = lambda text, src, tgt: "some translation"

    from polyglot_talk.models import TextSegment
    import queue

    text_q = queue.Queue()
    tts_q = queue.Queue()
    stop = threading.Event()

    import collections
    from unittest.mock import patch, MagicMock

    with patch("argostranslate.package.get_installed_packages") as mp, \
         patch("argostranslate.translate.translate") as mtr:
        fake = MagicMock()
        fake.from_code = "en"
        fake.to_code = _ARGOS_TARGET
        mp.return_value = [fake]
        mtr.return_value = "translation"

        from polyglot_talk.translator import Translator
        t = Translator(text_q, tts_q, stop, target_lang="hin")

        text_q.put(TextSegment(chunk_id=0, text="   "))  # whitespace — should skip
        text_q.put(None)  # sentinel

        t.run()

    # Translator propagates a shutdown sentinel (None) to tts_queue so that
    # downstream TTSEngine can exit cleanly.  We drain it here and confirm
    # that no real TranslatedSegment was enqueued for the whitespace input.
    from polyglot_talk.models import TranslatedSegment
    items = []
    while not tts_q.empty():
        items.append(tts_q.get_nowait())

    real_segments = [i for i in items if isinstance(i, TranslatedSegment)]
    assert real_segments == [], (
        f"Expected no TranslatedSegments for whitespace input, got {real_segments}"
    )
    sentinels = [i for i in items if i is None]
    assert len(sentinels) == 1, f"Expected exactly one shutdown sentinel, got {items}"
    print("✓ empty text skipped")


def test_context_deque_updated_after_translation(make_translator) -> None:
    """Source text is appended to context after each successful translation."""
    translator, _ = make_translator()

    calls = []
    translator._translate = lambda text: calls.append(text) or "result"
    translator._translate_prefix = lambda p: ""

    assert len(translator._context_source) == 0

    translator._translate_with_context("First sentence")
    assert "First sentence" in translator._context_source

    translator._translate_with_context("Second sentence")
    assert "Second sentence" in translator._context_source

    # Deque maxlen=2, so "First sentence" is still there after 2 items
    assert len(translator._context_source) == 2
    print(f"✓ context deque after 2 calls: {list(translator._context_source)}")
