"""
setup_models.py — One-time model download and verification script.

Run this script ONCE while online to download all required models.
After that, PolyglotTalk runs fully offline.

What it downloads
-----------------
1. faster-whisper base.en (int8) — ~150 MB
   Cached to:  ~/.cache/huggingface/hub/  (or WHISPER_MODELS_DIR)

2a. Argos Translate en→hi — ~100 MB  [Hindi only]
    Installed to:  ~/.local/share/argos-translate/packages/  (Linux)
                   %LOCALAPPDATA%\\argos-translate\\packages\\  (Windows)

2b. Helsinki-NLP MarianMT checkpoints — ~300 MB each
    [Marathi and Malayalam only — only Indian languages with confirmed checkpoints]
    Cached to:  ~/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-{xx}/

2c. facebook/nllb-200-distilled-600M — ~1.2 GB
    [Tamil, Telugu, Kannada, Bengali, Gujarati — no Helsinki-NLP checkpoint exists]
    Cached to:  ~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/

3.  Facebook MMS-TTS models for all supported targets — ~150 MB each
    Cached to:  ~/.cache/huggingface/hub/models--facebook--mms-tts-{lang}/

Total:  multi-GB on first run (downloads all supported language assets).

Note on MT backend
------------------
Argos Translate only publishes an en→hi offline package for Indian languages.
Marathi and Malayalam use Helsinki-NLP MarianMT (confirmed checkpoints exist).
Tamil, Telugu, Kannada, Bengali, and Gujarati use facebook/nllb-200-distilled-600M
because Helsinki-NLP does not publish en→{ta,te,kn,bn,gu} opus-mt checkpoints.
This script installs all backends required by currently configured supported languages.

Usage
-----
    python setup_models.py
    python setup_models.py --skip-verify   # download only, skip smoke tests
"""

from __future__ import annotations

# config MUST be imported first — sets OMP_NUM_THREADS / CT2_INTER_THREADS
from polyglot_talk import config  # noqa: F401

import argparse
import sys
import time


# ── ANSI helpers ─────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _info(msg: str) -> None:
    print(f"  \u2192 {msg}")


def _warn(msg: str) -> None:
    print(f"  ! {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}", file=sys.stderr)


# ── Step 1: faster-whisper ────────────────────────────────────────────────────

def download_asr_models() -> dict[str, object]:
    """Download all configured ASR model routes once each.

    Returns a map {source_lang: WhisperModel} for smoke tests.
    """
    print("\n[1/3] Downloading faster-whisper model(s)…")

    from faster_whisper import WhisperModel

    models: dict[str, object] = {}
    seen_model_ids: set[str] = set()
    for source_lang in sorted(config.ASR_MODEL_MAP):
        model_id = config.ASR_MODEL_MAP[source_lang]
        _info(
            f"Source={source_lang}  model={model_id}  compute={config.ASR_COMPUTE_TYPE}  device={config.ASR_DEVICE}"
        )
        if model_id in seen_model_ids:
            _ok(f"ASR model {model_id} already loaded via another source route — skipping.")
            continue
        t0 = time.perf_counter()
        model = WhisperModel(
            model_id,
            device=config.ASR_DEVICE,
            compute_type=config.ASR_COMPUTE_TYPE,
        )
        elapsed = time.perf_counter() - t0
        models[source_lang] = model
        seen_model_ids.add(model_id)
        _ok(f"ASR model {model_id} loaded/verified in {elapsed:.1f}s")
    return models


def verify_asr_models(models: dict[str, object]) -> None:
    _info("Smoke-testing ASR model route(s) (1 second of silence)…")
    import numpy as np

    silence = np.zeros(config.SAMPLE_RATE, dtype="float32")
    for source_lang in sorted(config.ASR_MODEL_MAP):
        model_id = config.ASR_MODEL_MAP[source_lang]
        model = models.get(source_lang)
        if model is None:
            # Shared model id already verified by another route.
            continue
        asr_lang = config.ASR_TRANSCRIBE_LANG_MAP[source_lang]
        segments_gen, _info_obj = model.transcribe(
            silence,
            beam_size=config.ASR_BEAM_SIZE,
            language=asr_lang,
            vad_filter=False,
        )
        _ = list(segments_gen)
        _ok(f"ASR smoke test passed ({source_lang} / {model_id})")


# ── Step 2a: Argos Translate (Hindi only) ────────────────────────────────────

def download_argos_models() -> bool:
    """Download all Argos packages required by ARGOS_LANG_MAP.

    Returns False only if at least one required package is missing upstream.
    """
    print("\n[2a/3] Downloading Argos Translate language pack(s)…")

    import argostranslate.package

    if not config.ARGOS_LANG_MAP:
        _ok("No Argos language routes configured — skipping.")
        return True

    _info("Fetching Argos package index (requires internet)…")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    installed = argostranslate.package.get_installed_packages()

    ok = True
    for target_lang, argos_code in sorted(config.ARGOS_LANG_MAP.items()):
        pair_label = f"{config.SOURCE_LANG}→{argos_code} ({target_lang})"

        already = any(
            p.from_code == config.SOURCE_LANG and p.to_code == argos_code
            for p in installed
        )
        if already:
            _ok(f"Argos package {pair_label} already installed — skipping.")
            continue

        pkg = next(
            (
                p
                for p in available
                if p.from_code == config.SOURCE_LANG and p.to_code == argos_code
            ),
            None,
        )
        if pkg is None:
            ok = False
            _warn(
                f"No Argos package found for {pair_label} in the upstream index.\n"
                f"  Check https://www.argosopentech.com/argospm/index/ for available pairs."
            )
            continue

        _info(f"Downloading {pkg.from_name} → {pkg.to_name} (version {pkg.package_version})…")
        t0 = time.perf_counter()
        download_path = pkg.download()
        argostranslate.package.install_from_path(download_path)
        elapsed = time.perf_counter() - t0
        _ok(f"Argos package {pair_label} installed in {elapsed:.1f}s")

    return ok


# ── Step 2b: MarianMT (all non-Hindi languages) ──────────────────────────────

def download_marian_models() -> None:
    """Download all configured Helsinki-NLP MarianMT checkpoints."""
    print("\n[2b/3] Downloading MarianMT translation model(s)…")
    if not config.MARIANMT_MODEL_MAP:
        _ok("No MarianMT routes configured — skipping.")
        return

    from transformers import MarianMTModel, MarianTokenizer

    for target_lang, model_id in sorted(config.MARIANMT_MODEL_MAP.items()):
        _info(f"Target={target_lang}  model={model_id}")
        t0 = time.perf_counter()
        _tokenizer = MarianTokenizer.from_pretrained(model_id)
        _model = MarianMTModel.from_pretrained(model_id)
        elapsed = time.perf_counter() - t0
        del _tokenizer, _model  # free memory — just needed for cache warm-up
        _ok(f"MarianMT model {model_id} downloaded/verified in {elapsed:.1f}s")


def download_nllb_model() -> None:
    """Download facebook/nllb-200-distilled-600M once if any NLLB route exists.

    Used for Tamil, Telugu, Kannada, Bengali, and Gujarati — languages that
    lack a Helsinki-NLP opus-mt checkpoint.  The model is a ~1.2 GB download
    cached to ~/.cache/huggingface/hub/ and shared across all NLLB languages.
    This is a no-op when no NLLB language routes are configured.
    """
    if not config.NLLB_LANG_MAP:
        _ok("No NLLB routes configured — skipping.")
        return

    print("\n[2c/3] Downloading NLLB-200 translation model…")
    _info(
        f"Model: {config.NLLB_MODEL_ID}  targets={','.join(sorted(config.NLLB_LANG_MAP))}"
    )

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    t0 = time.perf_counter()
    _tokenizer = AutoTokenizer.from_pretrained(config.NLLB_MODEL_ID)
    _model = AutoModelForSeq2SeqLM.from_pretrained(config.NLLB_MODEL_ID)
    elapsed = time.perf_counter() - t0
    del _tokenizer, _model  # free memory — just needed for cache warm-up
    _ok(f"NLLB-200 model downloaded/verified in {elapsed:.1f}s")


def verify_translation_model(target_lang: str) -> None:
    """Smoke-test translation route for one target language."""
    backend = config.get_mt_backend(target_lang)
    _info(f'Smoke-testing translation model for target={target_lang} (backend={backend})…')

    if backend == "argos":
        import argostranslate.translate
        argos_target = config.ARGOS_LANG_MAP[target_lang]
        result = argostranslate.translate.translate("Hello", config.SOURCE_LANG, argos_target)
    elif backend == "marian":
        from transformers import pipeline as hf_pipeline
        import torch
        model_id = config.MARIANMT_MODEL_MAP[target_lang]
        device = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline("translation", model=model_id, device=device)
        result = pipe("Hello")[0]["translation_text"]
    else:  # nllb
        from transformers import pipeline as hf_pipeline
        import torch
        nllb_tgt = config.NLLB_LANG_MAP[target_lang]
        device = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline(
            "translation",
            model=config.NLLB_MODEL_ID,
            src_lang="eng_Latn",
            tgt_lang=nllb_tgt,
            device=device,
            max_length=400,
        )
        result = pipe("Hello")[0]["translation_text"]

    if not result or not result.strip():
        _fail("Translation smoke test failed — empty output!")
        sys.exit(1)
    _ok(f'Translation smoke test passed ({target_lang}): "Hello" → "{result.strip()}"')


# ── Step 3: Facebook MMS-TTS ─────────────────────────────────────────────────

def download_tts_models() -> None:
    """Download all configured MMS-TTS checkpoints to HuggingFace cache."""
    print("\n[3/3] Downloading Facebook MMS-TTS model(s)…")

    from transformers import VitsModel, VitsTokenizer  # noqa: PLC0415

    for target_lang, model_id in sorted(config.MMS_TTS_MODEL_MAP.items()):
        _info(f"Target={target_lang}  model={model_id}  device={config.MMS_TTS_DEVICE}")
        t0 = time.perf_counter()
        _tokenizer = VitsTokenizer.from_pretrained(model_id)
        _model = VitsModel.from_pretrained(model_id)
        elapsed = time.perf_counter() - t0
        del _tokenizer, _model
        _ok(f"MMS-TTS model {model_id} downloaded/verified in {elapsed:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PolyglotTalk model setup")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip smoke tests (download-only mode)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" PolyglotTalk — Model Setup")
    print(f" Source routes: {', '.join(sorted(config.ASR_MODEL_MAP))}")
    print(f" Target routes: {', '.join(config.get_supported_target_langs())}")
    print("=" * 60)

    asr_models = download_asr_models()
    if not args.skip_verify:
        verify_asr_models(asr_models)

    # MT backend assets: Argos + MarianMT + NLLB routes
    download_argos_models()
    download_marian_models()
    download_nllb_model()

    if not args.skip_verify:
        # Verify one representative target per backend to keep setup bounded.
        verified_backends: set[str] = set()
        for target_lang in config.get_supported_target_langs():
            backend = config.get_mt_backend(target_lang)
            if backend in verified_backends:
                continue
            verify_translation_model(target_lang)
            verified_backends.add(backend)

    download_tts_models()

    print("\n" + "=" * 60)
    print(" \u2713 All models ready for offline use.")
    print(" Run 'python main.py' to start PolyglotTalk.")
    print("=" * 60)


if __name__ == "__main__":
    main()
