# PolyglotTalk — Benchmarks

All benchmarks are run from the **project root** with the virtual environment activated.

---

## Prerequisites

### 1. Install Python dependencies

```bash
uv venv --python 3.11.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Download models (one-time)

```bash
python setup_models.py
```

### 3. Download the LibriSpeech dev-clean dataset

Required by `benchmark_asr.py`, `benchmark_e2e.py`, and `benchmark_context.py`.

```bash
wget https://openslr.trmal.net/resources/12/dev-clean.tar.gz
mkdir -p ../data && tar -xzf dev-clean.tar.gz -C ../data
rm dev-clean.tar.gz
```

Extracts to `data/dev-clean/LibriSpeech/dev-clean/` in the project root (~350 MB, 2 703 utterances).

### 4. Prepare MT test sentences

Required by `benchmark_mt.py`. The file `data/test_sentences/sentences.txt` is already included in the repo (25 English–Hindi pairs, `|`-separated).

---

## Running the Benchmarks

Each script is self-contained and writes its output to `results/`.

### ASR — Word Error Rate & Latency

Compares `faster-whisper` model sizes (`tiny.en`, `base.en`, `small.en`) on randomly sampled LibriSpeech clips.

```bash
python benchmarks/benchmark_asr.py
```

| Output file | Description |
|---|---|
| `results/asr/<machine>/asr_results.csv` | Per-clip WER and latency for every model |
| `results/asr/<machine>/asr_summary.csv` | Per-model average WER, average latency, std latency |
| `results/asr/<machine>/asr_summary.meta.json` | System snapshot + config used for the run |

**Key parameters** (edit in `benchmark_asr.py`):

| Constant | Default | Effect |
|---|---|---|
| `MAX_CLIPS` | `500` | Clips randomly sampled per model (≤ 2 703 available) |
| `MODEL_SIZES` | `["tiny.en", "base.en", "small.en"]` | Models to compare |

---

### MT — BLEU Score & Latency

Compares Argos Translate vs MarianMT (`Helsinki-NLP/opus-mt-en-hi`) on 25 test sentences.

> **Note:** MarianMT requires `transformers` and `sentencepiece` (`pip install sentencepiece`). The benchmark skips it gracefully if they are absent.

```bash
python benchmarks/benchmark_mt.py
```

| Output file | Description |
|---|---|
| `results/mt/<machine>/mt_results.csv` | Per-sentence BLEU and latency for every model |
| `results/mt/<machine>/mt_summary.csv` | Per-model average BLEU, average latency, std latency |

---

### E2E — End-to-End Pipeline Latency

Measures per-stage latency (ASR → MT → TTS) on 20 randomly sampled LibriSpeech clips, processed **sequentially** (no threading). The pipeline parallelism of `main.py` means real perceived latency is lower than the sum of stage means.

```bash
python benchmarks/benchmark_e2e.py
```

| Output file | Description |
|---|---|
| `results/e2e/<machine>/e2e_latency_mms.csv` | Per-trial stage times and total E2E time |
| `results/e2e/<machine>/e2e_latency_mms.meta.json` | System snapshot + config |

WAV files synthesised during the run are written to `output/e2e_benchmark/`.

**Key parameters** (edit in `benchmark_e2e.py`):

| Constant | Default | Effect |
|---|---|---|
| `NUM_TRIALS` | `20` | Number of clips to run end-to-end |
| `MAX_CLIPS` | `500` | Pool size for random sampling |

---

### Context — Translation Continuity

Runs 10 consecutive LibriSpeech sentences through the translator with and without the rolling context window (`CONTEXT_MAXLEN = 2`), counting translation repetitions and grammar breaks.

```bash
python benchmarks/benchmark_context.py
```

| Output file | Description |
|---|---|
| `results/context/<machine>/context_results.csv` | Summary: repetitions, grammar breaks, avg latency per condition |
| `results/context/<machine>/context_detail.csv` | Per-sentence outputs for both conditions |
| `results/context/<machine>/context_results.meta.json` | System snapshot + config |

---

## Results

All results below were obtained on the following machine:

| Property | Value |
|---|---|
| OS | Linux 5.15.167.4 (WSL2 on Windows 11) |
| CPU | AMD Ryzen 7 7700X (8 cores / 16 threads) |
| RAM | 15.2 GB |
| GPU | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| CUDA | 12.8 |
| Python | 3.11.9 |
| `OMP_NUM_THREADS` | 2 |
| `CT2_INTER_THREADS` | 1 |

Shared config across all runs: `ASR_COMPUTE_TYPE=int8`, `ASR_BEAM_SIZE=1`, `ASR_LANGUAGE=en`, `CONTEXT_MAXLEN=2`.

---

### ASR Results (500 clips × 3 models, LibriSpeech dev-clean)

| Model | Avg WER | Avg Latency (s) | Std Latency (s) |
|---|---|---|---|
| `tiny.en` | **7.87 %** | 0.064 | 0.043 |
| `base.en` | **6.03 %** | 0.108 | 0.107 |
| `small.en` | **4.65 %** | 0.198 | 0.097 |

**Takeaway:** `base.en` (the production default) achieves 6.03 % WER at ~0.11 s per clip — 3× faster than `small.en` for only 1.4 pp more WER. `tiny.en` is ~1.7× faster still but adds another 1.8 pp WER.

---

### MT Results (25 sentences, en → hi)

| Model | Avg BLEU | Avg Latency (s) | Std Latency (s) |
|---|---|---|---|
| `argos_translate` | **0.464** | 0.088 | 0.227 |
| `marianmt_opus-mt-en-hi` | **0.149** | 0.202 | 0.093 |

**Takeaway:** Argos Translate scores 3× higher BLEU than MarianMT on this test set and is 2.3× faster per sentence. The high std for Argos is due to a cold-start on the first sentence; subsequent calls are ~0.02–0.04 s.

---

### E2E Latency Results (20 trials, sequential, MMS-TTS on CUDA)

ASR config: `base.en`, int8, beam_size=1. TTS: `facebook/mms-tts-hin` on CUDA (RTX 4060).

| Stage | Mean (s) | Std (s) |
|---|---|---|
| ASR (`base.en`, CPU int8) | 0.126 | 0.081 |
| MT (Argos Translate) | 0.104 | 0.147 |
| TTS (MMS-TTS, CUDA) | 0.136 | 0.156 |
| **Total E2E (sequential)** | **0.366** | **0.374** |

> These are **sequential** measurements (one stage finishes before the next starts). In the live pipeline (`main.py`) all four stages run in parallel threads, so perceived end-to-end latency is dominated by the longest single stage rather than the sum — approximately **0.13–0.20 s** per utterance at steady state.

---

### Context Continuity Results (10 sentences, Argos en → hi)

| Metric | With Context | Without Context |
|---|---|---|
| Translation repetitions | 0 | 0 |
| Grammar breaks | 1 | 1 |
| Avg latency (s) | 0.348 | 0.105 |

> The context window (`CONTEXT_MAXLEN=2`) prepends the last two source segments before each translation call, which increases per-call latency because more text is fed to the model. Both conditions show zero repetitions on this 10-sentence sample; grammar breaks (a single short/failed translation) are equal, indicating the context window adds no regressions on clean input.

---

## Output Files Reference

Output files are written to a **per-machine subdirectory** named `<hostname>_<GPU>` (e.g. `Renegade_RTX4060`) or `<hostname>_CPU` when no CUDA GPU is present. This ensures results from different machines sit side-by-side without overwriting each other.

```
results/
├── asr/
│   └── Renegade_RTX4060/
│       ├── asr_results.csv
│       ├── asr_results.meta.json
│       ├── asr_summary.csv
│       └── asr_summary.meta.json
├── mt/
│   └── Renegade_RTX4060/
│       ├── mt_results.csv
│       └── mt_summary.csv
├── context/
│   └── Renegade_RTX4060/
│       ├── context_results.csv
│       ├── context_results.meta.json
│       ├── context_detail.csv
│       └── context_detail.meta.json
└── e2e/
    └── Renegade_RTX4060/
        ├── e2e_latency_mms.csv
        └── e2e_latency_mms.meta.json
```

Every `*.meta.json` sidecar records the full system snapshot (CPU, GPU, CUDA version, RAM) and the exact `config.py` values active at the time of the run, making results reproducible.
