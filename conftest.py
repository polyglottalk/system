"""
conftest.py — pytest configuration for PolyglotTalk.

Adds the project root to sys.path so all modules (config, models,
audio_capture, etc.) are importable without installing the package.
Also ensures config.py (which sets OS env vars) is imported before any
test module loads faster_whisper or argostranslate.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is first on sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Set CTranslate2 thread limits before any test imports CT2 libs
from polyglot_talk import config  # noqa: F401, E402
