"""
benchmarks/system_meta.py — Reproducibility helpers for benchmark output.

Provides two public functions used by every benchmark:

  collect()          → dict[str, str]  hardware + config.py snapshot
  write_csv()        → writes a clean CSV file + companion <basename>.meta.json

CSV and metadata format
-----------------------
The output CSV files are clean (no comment header):

    model,clip,wer,latency_s
    tiny.en,1272-128104-0000,0.0500,0.2341
    ...

Metadata is stored in a companion ``<stem>.meta.json`` file.  To load both::

    import pandas as pd
    import json
    
    df = pd.read_csv("results/asr_results.csv")
    with open("results/asr_results.meta.json") as f:
        meta = json.load(f)
        print(f"Run timestamp: {meta['run_timestamp']}")
        print(f"ASR model: {meta['config_asr_model_size']}")
"""

from __future__ import annotations

import csv
import json
import os
import platform
from typing import Any


# ── Public API ───────────────────────────────────────────────────────────────

def collect() -> dict[str, str]:
    """Return a flat dict of hardware info and config.py settings.

    Imports are lazy so that this module can be imported cheaply without
    pulling in torch or config until ``collect()`` is actually called.
    """
    import datetime

    meta: dict[str, str] = {}

    # ── Timestamp / Python ────────────────────────────────────────────────
    meta["run_timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["python_version"] = platform.python_version()
    meta["hostname"] = platform.node()
    meta["os"] = f"{platform.system()} {platform.release()}"

    # ── CPU ───────────────────────────────────────────────────────────────
    meta["cpu_arch"] = platform.machine()
    meta["cpu_logical_cores"] = str(os.cpu_count() or "unknown")
    meta["cpu_model"] = _cpu_model_name()

    # ── RAM ───────────────────────────────────────────────────────────────
    meta["ram_total_gb"] = _ram_total_gb()

    # ── GPU / CUDA ────────────────────────────────────────────────────────
    gpu_info = _gpu_info()
    meta.update(gpu_info)

    # ── Thread / parallelism env vars (set by config.py) ─────────────────
    meta["env_OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "unset")
    meta["env_CT2_INTER_THREADS"] = os.environ.get("CT2_INTER_THREADS", "unset")

    # ── config.py settings ────────────────────────────────────────────────
    # Import lazily — config.py sets env vars on load, which must already
    # have happened before this point in the benchmark entry points.
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from polyglot_talk import config  # noqa: PLC0415

        meta["config_asr_model_size"] = config.ASR_MODEL_SIZE
        meta["config_asr_compute_type"] = config.ASR_COMPUTE_TYPE
        meta["config_asr_device"] = config.ASR_DEVICE
        meta["config_asr_beam_size"] = str(config.ASR_BEAM_SIZE)
        meta["config_asr_language"] = config.ASR_LANGUAGE
        meta["config_asr_strip_trailing_period"] = str(config.ASR_STRIP_TRAILING_PERIOD)
        meta["config_sample_rate_hz"] = str(config.SAMPLE_RATE)
        meta["config_chunk_duration_s"] = str(config.CHUNK_DURATION)
        meta["config_chunk_overlap_s"] = str(config.CHUNK_OVERLAP)
        meta["config_rms_silence_threshold"] = str(config.RMS_SILENCE_THRESHOLD)
        meta["config_source_lang"] = config.SOURCE_LANG
        meta["config_target_lang"] = config.TARGET_LANG
        meta["config_context_maxlen"] = str(config.CONTEXT_MAXLEN)
        meta["config_tts_model"] = config.MMS_TTS_MODEL_MAP[config.TARGET_LANG]
        meta["config_tts_device"] = config.MMS_TTS_DEVICE
    except Exception as exc:
        meta["config_load_error"] = str(exc)

    return meta


def machine_slug() -> str:
    """Return a filesystem-safe identifier for the current machine.

    Format: ``<hostname>_<GPU_slug>``  (e.g. ``Renegade_RTX4060``)
    or      ``<hostname>_CPU``         (when no CUDA GPU is detected).

    Used as a per-machine subdirectory under each ``results/<benchmark>/``
    folder so that runs on different machines never overwrite each other.
    """
    import re

    hostname = re.sub(r"[^A-Za-z0-9_-]", "_", platform.node() or "unknown")
    gpu = _gpu_info()
    if gpu.get("cuda_available") == "true":
        raw = gpu.get("gpu_name", "GPU")
        # Keep the last two space-separated tokens and strip non-alnum chars.
        # "NVIDIA GeForce RTX 4060" → "RTX4060"
        # "NVIDIA A100"             → "NVIDIAA100"
        tokens = raw.split()
        gpu_slug = re.sub(r"[^A-Za-z0-9]", "", "".join(tokens[-2:]))
    else:
        gpu_slug = "CPU"
    return f"{hostname}_{gpu_slug}"


def write_csv(
    path: str,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    meta: dict[str, str],
    extra_meta: dict[str, str] | None = None,
) -> None:
    """Write *rows* to *path* as a clean CSV file.

    Metadata is written to a companion ``<stem>.meta.json`` sidecar file for
    clean CSV files that work seamlessly with pandas and other tools.

    Parameters
    ----------
    path:        Absolute or relative path to the output ``.csv`` file.
    fieldnames:  Column names passed to :class:`csv.DictWriter`.
    rows:        Data rows (list of dicts matching *fieldnames*).
    meta:        Base metadata dict from :func:`collect`.
    extra_meta:  Optional additional keys merged on top of *meta*
                 (e.g. ``{"random_seed": "42", "num_clips": "500"}``).
    """
    all_meta = dict(meta)
    if extra_meta:
        all_meta.update(extra_meta)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # ── JSON sidecar ──────────────────────────────────────────────────────
    _write_meta_json(path, all_meta)


# ── Private helpers ──────────────────────────────────────────────────────────

def _cpu_model_name() -> str:
    """Best-effort CPU model string across platforms."""
    # Linux — /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    # Fallback: platform.processor()
    return platform.processor() or platform.machine() or "unknown"


def _ram_total_gb() -> str:
    """Total system RAM in GB as a string."""
    # Try psutil first
    try:
        import psutil  # noqa: PLC0415
        gb = psutil.virtual_memory().total / (1024 ** 3)
        return f"{gb:.1f}"
    except ImportError:
        pass
    # Linux fallback — /proc/meminfo
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024 ** 2:.1f}"
    except OSError:
        pass
    return "unknown"


def _gpu_info() -> dict[str, str]:
    """Return CUDA/GPU details if available, otherwise mark as absent."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "cuda_available": "true",
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": f"{props.total_memory / 1024 ** 3:.1f}",
                "cuda_version": torch.version.cuda or "unknown",
            }
        else:
            return {
                "cuda_available": "false",
                "gpu_name": "N/A",
                "gpu_memory_gb": "N/A",
                "cuda_version": "N/A",
            }
    except ImportError:
        return {
            "cuda_available": "unknown (torch not importable)",
            "gpu_name": "unknown",
            "gpu_memory_gb": "unknown",
            "cuda_version": "unknown",
        }


def _write_meta_json(csv_path: str, meta: dict[str, str]) -> None:
    """Write ``<csv_path stem>.meta.json`` alongside the CSV."""
    stem = csv_path
    if stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    json_path = stem + ".meta.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
