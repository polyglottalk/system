"""
benchmark_e2e.py — End-to-end pipeline latency benchmark.

Measures per-stage latency (ASR, MT, TTS) and total E2E time by
feeding test clips through each stage sequentially.

Usage
-----
    python benchmarks/benchmark_e2e.py

Output
------
    results/e2e/e2e_latency_mms.csv
"""

from __future__ import annotations

import os
import random
import sys
import time

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from polyglot_talk import config  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import system_meta  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
LIBRISPEECH_DIR = os.path.join(PROJECT_ROOT, "data", "dev-clean", "LibriSpeech", "dev-clean")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "e2e")
# Output CSV paths are computed per-machine inside run_benchmark().

NUM_TRIALS = 20

# Maximum audio clips available for the random trial pool (>= NUM_TRIALS).
# A random seed is drawn once per run and stored in the output metadata.
MAX_CLIPS: int = 500


def _load_librispeech() -> list[tuple[str, str]]:
    """Return list of (flac_path, reference_text) from LibriSpeech dev-clean.

    Walks LIBRISPEECH_DIR, reads every ``*.trans.txt`` file, and maps each
    ``UTTID TEXT`` line to its corresponding .flac file on disk.
    Reference text is lowercased from the original ALL-CAPS transcripts.
    """
    import pathlib

    entries: list[tuple[str, str]] = []
    for trans_file in sorted(pathlib.Path(LIBRISPEECH_DIR).rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uttid, text = line.split(" ", 1)
                flac_path = str(chapter_dir / f"{uttid}.flac")
                if os.path.exists(flac_path):
                    entries.append((flac_path, text.lower()))
    return entries


def _load_audio(audio_path: str) -> np.ndarray:
    """Load a FLAC or WAV file as float32 16 kHz mono."""
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != 16000:
        from scipy.signal import resample
        num_samples = int(len(data) * 16000 / sr)
        data = resample(data, num_samples).astype(np.float32) # type: ignore

    return data


def run_benchmark() -> None:
    machine_dir = os.path.join(RESULTS_DIR, system_meta.machine_slug())
    os.makedirs(machine_dir, exist_ok=True)
    results_csv = os.path.join(machine_dir, "e2e_latency_mms.csv")

    all_entries = _load_librispeech()
    print(f"Discovered {len(all_entries)} clips in LibriSpeech dev-clean.\n")

    if not all_entries:
        print(f"ERROR: No FLAC files found under {LIBRISPEECH_DIR}")
        sys.exit(1)

    # ── Random sample for this run ─────────────────────────────────────────
    seed = random.randrange(2 ** 32)
    rng = random.Random(seed)
    pool = rng.sample(all_entries, min(MAX_CLIPS, len(all_entries)))
    sampled = rng.sample(pool, min(NUM_TRIALS, len(pool)))
    print(f"Random seed: {seed}  |  Pool: {len(pool)}  |  Trials: {len(sampled)}\n")

    # Collect system / config snapshot (before model loads)
    meta = system_meta.collect()
    meta["dataset"] = "LibriSpeech dev-clean"
    meta["dataset_total_clips"] = str(len(all_entries))
    meta["max_clip_pool"] = str(MAX_CLIPS)
    meta["num_trials"] = str(len(sampled))
    meta["random_seed"] = str(seed)

    # Pre-load audio for the sampled clips
    audio_clips: list[tuple[str, np.ndarray]] = []
    for flac_path, _text in sampled:
        audio_clips.append((flac_path, _load_audio(flac_path)))

    print(f"Pre-loaded {len(audio_clips)} audio files.\n")

    # ── Load models (once) ───────────────────────────────────────────────────
    print("Loading ASR model...")
    from faster_whisper import WhisperModel
    asr_model = WhisperModel(
        config.ASR_MODEL_SIZE,
        device=config.ASR_DEVICE,
        compute_type=config.ASR_COMPUTE_TYPE,
    )
    print(f"  ✓ ASR model loaded ({config.ASR_MODEL_SIZE})")

    print("Loading translation model...")

    import argostranslate.translate
    # Verify Argos model is installed
    import argostranslate.package

    _argos_target = config.ARGOS_LANG_MAP[config.TARGET_LANG]
    installed = argostranslate.package.get_installed_packages()

    if not any(p.from_code == "en" and p.to_code == _argos_target for p in installed):
        print(f"ERROR: Argos en→{_argos_target} package not installed. Run setup_models.py.")
        sys.exit(1)

    print(f"  ✓ Translation model loaded (Argos en→{_argos_target})")

    print("Loading MMS-TTS model...")

    import torch
    import soundfile as sf
    from transformers import VitsModel, VitsTokenizer

    tts_device = config.MMS_TTS_DEVICE
    if tts_device == "auto":
        tts_device = "cuda" if torch.cuda.is_available() else "cpu"

    tts_model_id = config.MMS_TTS_MODEL_MAP[config.TARGET_LANG]
    tts_tokenizer = VitsTokenizer.from_pretrained(tts_model_id)
    tts_model = VitsModel.from_pretrained(tts_model_id)
    tts_model = tts_model.to(tts_device) # type: ignore
    tts_model.eval()

    tts_output_dir = os.path.join(PROJECT_ROOT, config.TTS_OUTPUT_DIR, "e2e_benchmark")
    os.makedirs(tts_output_dir, exist_ok=True)
    print(f"  ✓ MMS-TTS model loaded (device={tts_device}, model={tts_model_id})")

    print(f"\nRunning {len(audio_clips)} E2E trials...\n")

    # ── Run trials ────────────────────────────────────────────────────────
    all_rows: list[dict] = []

    for trial, (flac_path, audio) in enumerate(audio_clips, 1):
        uttid = os.path.splitext(os.path.basename(flac_path))[0]

        # ── ASR stage ────────────────────────────────────────────────────────
        t_asr_start = time.perf_counter()
        segments_gen, _info = asr_model.transcribe(
            audio,
            beam_size=config.ASR_BEAM_SIZE,
            language=config.ASR_LANGUAGE,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        transcript = "".join(seg.text for seg in segments_gen).strip()
        t_asr_end = time.perf_counter()
        asr_time = t_asr_end - t_asr_start

        if not transcript:
            transcript = "hello"  # fallback for empty transcripts

        # ── MT stage ─────────────────────────────────────────────────────────
        t_mt_start = time.perf_counter()
        translated = argostranslate.translate.translate(transcript, "en", _argos_target)
        t_mt_end = time.perf_counter()
        mt_time = t_mt_end - t_mt_start

        # ── TTS stage ────────────────────────────────────────────────────────
        out_path = os.path.join(tts_output_dir, f"bench_chunk_{trial:04d}.wav")
        t_tts_start = time.perf_counter()
        
        try:
            inputs = tts_tokenizer(translated, return_tensors="pt")
            inputs = {k: v.to(tts_device) for k, v in inputs.items()}
            with torch.no_grad():
                audio_out = tts_model(**inputs)
            waveform = audio_out.waveform[0].squeeze().cpu().numpy()
            sf.write(out_path, waveform.astype(np.float32),
                     samplerate=tts_model.config.sampling_rate)
            
        except Exception as exc:
            print(f"  WARNING: MMS-TTS synthesis failed for trial {trial}: {exc}")
            
        tts_time = time.perf_counter() - t_tts_start
        total_e2e = asr_time + mt_time + tts_time

        all_rows.append({
            "trial": trial,
            "clip": uttid,
            "transcript": transcript,
            "translation": translated,
            "asr_time_s": f"{asr_time:.4f}",
            "mt_time_s": f"{mt_time:.4f}",
            "tts_time_s": f"{tts_time:.4f}",
            "total_e2e_s": f"{total_e2e:.4f}",
        })

        print(f"  Trial {trial:>2}: ASR={asr_time:.3f}s  MT={mt_time:.3f}s  "
              f"TTS={tts_time:.3f}s  E2E={total_e2e:.3f}s")

    # ── Statistics ───────────────────────────────────────────────────────────
    asr_times = [float(r["asr_time_s"]) for r in all_rows]
    mt_times = [float(r["mt_time_s"]) for r in all_rows]
    tts_times = [float(r["tts_time_s"]) for r in all_rows]
    e2e_times = [float(r["total_e2e_s"]) for r in all_rows]

    stats = {
        "asr": (np.mean(asr_times), np.std(asr_times)),
        "mt": (np.mean(mt_times), np.std(mt_times)),
        "tts": (np.mean(tts_times), np.std(tts_times)),
        "e2e": (np.mean(e2e_times), np.std(e2e_times)),
    }

    # Append MEAN/STD summary rows to the data block
    all_rows.append({
        "trial": "MEAN", "clip": "", "transcript": "", "translation": "",
        "asr_time_s": f"{stats['asr'][0]:.4f}",
        "mt_time_s": f"{stats['mt'][0]:.4f}",
        "tts_time_s": f"{stats['tts'][0]:.4f}",
        "total_e2e_s": f"{stats['e2e'][0]:.4f}",
    })
    all_rows.append({
        "trial": "STD", "clip": "", "transcript": "", "translation": "",
        "asr_time_s": f"{stats['asr'][1]:.4f}",
        "mt_time_s": f"{stats['mt'][1]:.4f}",
        "tts_time_s": f"{stats['tts'][1]:.4f}",
        "total_e2e_s": f"{stats['e2e'][1]:.4f}",
    })

    # ── Write CSV (metadata in .meta.json sidecar) ──────────────────────────────────────
    system_meta.write_csv(
        results_csv,
        fieldnames=["trial", "clip", "transcript", "translation",
                    "asr_time_s", "mt_time_s", "tts_time_s", "total_e2e_s"],
        rows=all_rows,
        meta=meta,
    )
    print(f"✓ Results saved to {results_csv}")
    print(f"  (metadata sidecar: {results_csv.replace('.csv', '.meta.json')})")

    # Print paper-ready table
    print(f"\n{'=' * 60}")
    print("  E2E Latency Table (for Paper Section 5)")
    print(f"{'=' * 60}")
    print(f"  {'Stage':<12} {'Mean (s)':>12} {'Std (s)':>12}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 12}")
    for stage, (mean, std) in stats.items():
        print(f"  {stage.upper():<12} {mean:>12.4f} {std:>12.4f}")


if __name__ == "__main__":
    run_benchmark()
