"""
main.py — Entry point for PolyglotTalk.

Usage
-----
    python main.py                          # CLI only, interactive target prompt
    python main.py --source en --target hin
    python main.py --dashboard              # dashboard-only, start from UI
    python main.py --dashboard --target hin # dashboard + auto-start pipeline
    python main.py --no-prompt
    python main.py --log-level DEBUG

IMPORTANT: config is imported first so that OMP_NUM_THREADS and
CT2_INTER_THREADS are set in os.environ before any CTranslate2 library
(faster-whisper, argostranslate) is imported anywhere in the process.
"""

from __future__ import annotations

# ── config MUST be the first project import ──────────────────────────────────
from polyglot_talk import config  # noqa: F401 — sets os.environ before CT2 is imported

import argparse
import logging
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polyglot-talk",
        description="Real-time offline Speech-to-Speech Translation",
    )
    p.add_argument(
        "--source",
        default=config.SOURCE_LANG,
        metavar="LANG",
        help=f"Source language code (default: {config.SOURCE_LANG})",
    )
    p.add_argument(
        "--target",
        default=None,
        metavar="LANG",
        help=(
            "Target language ISO 639-3 code. If omitted in CLI mode, prompts "
            "interactively. If omitted in --dashboard mode, pipeline starts from UI."
        ),
    )
    p.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive target-language prompt and use config default.",
    )
    p.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    p.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Start the real-time WebSocket dashboard server",
    )
    p.add_argument(
        "--dashboard-port",
        type=int,
        default=8765,
        metavar="PORT",
        help="Port for the dashboard WebSocket server (default: 8765)",
    )
    return p


def _prompt_target_language(default_target: str) -> str:
    """Prompt user to choose a target language code from supported routes."""
    supported = config.get_supported_target_langs()

    print("\nSelect target language for English speech translation:")
    for idx, code in enumerate(supported, start=1):
        label = config.TARGET_LANG_LABELS.get(code, code)
        backend = config.get_mt_backend(code)
        default_tag = "  (default)" if code == default_target else ""
        print(f"  {idx}. {label} [{code}]  MT={backend}{default_tag}")

    while True:
        raw = input(f"Choose 1-{len(supported)} or code [{default_target}]: ").strip().lower()
        if not raw:
            return default_target
        if raw.isdigit():
            pos = int(raw)
            if 1 <= pos <= len(supported):
                return supported[pos - 1]
        if raw in supported:
            return raw
        print("Invalid selection. Enter a number from the list or a language code.")


def _resolve_target_language(args: argparse.Namespace) -> str:
    """Resolve target language from args, prompt rules, and runtime mode."""
    if args.target is not None:
        return args.target.strip().lower()

    # Dashboard mode keeps source/target selection in the UI unless explicitly set.
    if args.dashboard:
        return config.TARGET_LANG

    # CLI mode can prompt interactively unless disabled.
    if args.no_prompt or not sys.stdin.isatty():
        return config.TARGET_LANG
    return _prompt_target_language(default_target=config.TARGET_LANG)


def main() -> None:
    args = _build_parser().parse_args()
    source_lang = args.source.strip().lower()
    target_lang = _resolve_target_language(args)

    if source_lang not in config.ASR_MODEL_MAP:
        valid = ", ".join(sorted(config.ASR_MODEL_MAP))
        raise SystemExit(f"Unsupported --source {source_lang!r}. Valid values: {valid}")

    if target_lang not in config.MMS_TTS_MODEL_MAP:
        valid = ", ".join(config.get_supported_target_langs())
        raise SystemExit(f"Unsupported --target {target_lang!r}. Valid values: {valid}")

    # ── Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=config.LOG_FORMAT,
        stream=sys.stdout,
    )
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    # ── Resolve device label for banner ───────────────────────────────────
    device = config.MMS_TTS_DEVICE
    if device == "auto":
        try:
            import torch  # noqa: PLC0415
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print("=" * 60)
    print(" PolyglotTalk v0.1 — Offline Speech-to-Speech Translation")
    print(f" {source_lang.upper()} → {target_lang.upper()}  |  TTS: MMS-TTS ({device})  |  No cloud APIs")
    print(f" TTS output saved to: {config.TTS_OUTPUT_DIR}/chunk_NNNN.wav")
    print("=" * 60)

    # ── Clean up old output chunks ────────────────────────────────────────
    output_dir = Path(config.TTS_OUTPUT_DIR)
    if output_dir.exists():
        for wav_file in output_dir.glob("chunk_*.wav"):
            wav_file.unlink()
            logging.getLogger(__name__).debug("Removed old chunk: %s", wav_file.name)

    # ══════════════════════════════════════════════════════════════════════
    # DASHBOARD MODE
    # ══════════════════════════════════════════════════════════════════════
    if args.dashboard:
        import threading
        from dashboard_server import pipeline_manager, run_server

        dash_thread = threading.Thread(
            target=run_server,
            kwargs={"host": "0.0.0.0", "port": args.dashboard_port},
            name="DashboardServer",
            daemon=True,
        )
        dash_thread.start()
        time.sleep(1.0)

        print(
            f"  Dashboard: http://localhost:{args.dashboard_port}  "
            f"(WS: ws://localhost:{args.dashboard_port}/ws)"
        )

        if args.target:
            print(f"  Auto-starting pipeline: {source_lang} → {target_lang}")
            pipeline_manager.start(source_lang=source_lang, target_lang=target_lang)
        else:
            print("  Open the dashboard and press ▶ Start to begin.")

        print("  Press Ctrl+C to shut down.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nInterrupt received — shutting down…")
            pipeline_manager.stop()
            time.sleep(1.0)
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════
    # CLI-ONLY MODE (no dashboard)
    # ══════════════════════════════════════════════════════════════════════
    from polyglot_talk.pipeline import Pipeline  # noqa: PLC0415

    pipeline = Pipeline(source_lang=source_lang, target_lang=target_lang)

    try:
        pipeline.start()
        pipeline.wait()
    except RuntimeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
