"""
tests/test_translator.py — Verify Argos Translate en→hi translation.

Requires the Argos en→hi package to be installed (run setup_models.py
first).

Run:
    python -m tests.test_translator
    python -m pytest tests/test_translator.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polyglot_talk import config  # noqa: F401

import pytest

_ARGOS_TARGET = config.ARGOS_LANG_MAP["hin"]  # Argos only supports Hindi for Indian languages


@pytest.fixture(scope="module")
def check_model_installed():
    """Skip all tests if the Argos package is not installed."""
    import argostranslate.package

    installed = argostranslate.package.get_installed_packages()
    found = any(
        p.from_code == config.SOURCE_LANG and p.to_code == _ARGOS_TARGET
        for p in installed
    )
    if not found:
        pytest.skip(
            f"Argos {config.SOURCE_LANG}→{_ARGOS_TARGET} not installed. "
            "Run python setup_models.py first."
        )


def test_hello_translation(check_model_installed) -> None:
    """'Hello, how are you?' → non-empty Devanagari string."""
    import argostranslate.translate

    result = argostranslate.translate.translate(
        "Hello, how are you?", config.SOURCE_LANG, _ARGOS_TARGET
    )
    print(f"Translation: {result!r}")

    assert result and result.strip(), "Translation is empty"

    # Devanagari Unicode range: U+0900–U+097F
    has_devanagari = any("\u0900" <= ch <= "\u097F" for ch in result)
    assert has_devanagari, (
        f"Expected Devanagari script in translation, got: {result!r}"
    )
    assert len(result) > 5, f"Translation suspiciously short: {result!r}"
    print("✓ test_hello_translation passed")


def test_translation_non_empty(check_model_installed) -> None:
    """A variety of sentences should all produce non-empty output."""
    import argostranslate.translate

    sentences = [
        "Good morning.",
        "What is your name?",
        "The weather is nice today.",
        "Please speak slowly.",
    ]
    for sent in sentences:
        result = argostranslate.translate.translate(
            sent, config.SOURCE_LANG, _ARGOS_TARGET
        )
        assert result and result.strip(), f"Empty translation for: {sent!r}"
        print(f"  {sent!r} → {result!r}")

    print("✓ test_translation_non_empty passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
