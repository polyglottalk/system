# PolyglotTalk

Real-time, fully **offline** Speech-to-Speech Translation (S2ST) — no cloud APIs, no data leaves your machine.
ASR and translation run on **CPU**; TTS (MMS-TTS) automatically uses **CUDA** if a GPU is present, falling back to CPU.

```
Microphone → [Whisper ASR] → [Argos Translate] → [MMS-TTS] → WAV Files
```

Speak English into your microphone; translated speech is synthesised and saved to `output/chunk_*.wav` files (preventing mic feedback). Each stage runs in its own thread so recording, transcribing, translating, and synthesis all happen **simultaneously**. Supports 8 Indian languages (Hindi, Tamil, Telugu, Kannada, Bengali, Malayalam, Marathi, Gujarati).

---

## Features

- **Fully offline** — all models run locally after a one-time download
- **CPU + optional CUDA** — ASR and translation always run on CPU; TTS (MMS-TTS) auto-detects CUDA and uses it if available, otherwise falls back to CPU
- **Overlapping audio chunks** — consecutive 2.5 s windows share 1.0 s of audio so words at chunk boundaries are never cut off
- **Sentence buffering** — ASR fragments accumulate into natural sentences before translation, giving the MT model better context
- **Hyphen-aware deduplication** — overlap text is removed at word boundaries even when Whisper transcribes the same words differently across chunks (e.g. "real time" vs "real-time")
- **True pipeline parallelism** — 4 threads run concurrently with ~4 s end-to-end latency
- **No mic feedback** — TTS output saved to files, not played through speakers
- **Auto-stop** — pipeline exits cleanly after 2.5 s of silence
- **Live console output** — transcription, translation, and TTS file paths printed as they happen
- **Configurable** — source/target language, TTS speed, and output directory via CLI flags
- **Hallucination filtering** — common Whisper silence-artifacts ("Thank you", "Bye") are blocked

---

## Demo Output

```
============================================================
 PolyglotTalk v0.1 — Offline Speech-to-Speech Translation
 EN → HI  |  TTS: MMS-TTS (cuda)  |  No cloud APIs
 TTS output saved to: output/chunk_NNNN.wav
============================================================
✓ Pipeline ready. Speak now… (Ctrl+C to stop)
[ASR   #1] Hello, how are you today?
[→HI  #1] नमस्ते, आप आज कैसे हैं?
[TTS  #1] saved → output/chunk_0001.wav
[ASR   #2] I'm doing well, thank you.
[→HI  #2] मैं ठीक हूँ, धन्यवाद।
[TTS  #2] saved → output/chunk_0002.wav
Pipeline stopped.
```

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.11.x |
| uv _(recommended)_ | latest |

### System packages (Linux / WSL2)

```bash
sudo apt-get install -y libportaudio2 portaudio19-dev pulseaudio libpulse0 alsa-utils ffmpeg
```

| Package | Purpose |
|---|---|
| `libportaudio2` | PortAudio C library — required by `sounddevice` |
| `pulseaudio` / `libpulse0` | Audio routing (WSL2: connects to WSLg's RDP microphone) |
| `alsa-utils` | ALSA audio utilities — needed for audio device enumeration |

### HuggingFace CLI

MMS-TTS models for 8 Indian languages are public, ungated repositories — no authentication required.

> **WSL2 users:** Audio is bridged via WSLg on Windows 11. After installing
> `libpulse0`, verify your mic appears with:
> ```bash
> PULSE_SERVER=unix:/mnt/wslg/PulseServer pactl list sources short
> ```
> You should see `RDPSource` listed. If not, run `wsl --update` from PowerShell.

> **TTS output files:** Synthesised speech is saved to `output/chunk_NNNN.wav`
> files (where N is the chunk ID). This avoids microphone feedback during
> live translation. TTS uses Facebook MMS-TTS, a fast non-autoregressive
> VITS-based model with per-language checkpoints. MMS-TTS automatically runs
> on **CUDA** when a compatible GPU is detected (`MMS_TTS_DEVICE = "auto"`),
> falling back to CPU otherwise. Unlike flow-matching models, its latency does
> not scale with text length.

---

## Quick Start

### 1. Create environment and install Python dependencies

```bash
# Using uv (recommended)
uv venv --python 3.11.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Download models (one-time, requires internet, ~2.5–3.5 GB)

```bash
python setup_models.py
```

This downloads:
- `faster-whisper` `base.en` int8 model (~150 MB) → `~/.cache/huggingface/hub/`
- Argos Translate language packs for all 8 supported languages (~800 MB) → `~/.local/share/argos-translate/`
- Facebook MMS-TTS model for the configured target language (~150 MB, cached by transformers) → `~/.cache/huggingface/hub/`

### 3. Run

```bash
python main.py
```

Speak into your microphone. The pipeline stops automatically after ~2.5 s of silence. Press **Ctrl+C** to stop at any time.

---

## Usage

```
python main.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--source LANG` | `en` | Source language (ISO 639-1) |
| `--target LANG` | `hin` | Target language (ISO 639-3 code; must be in `MMS_TTS_MODEL_MAP`) |
| `--log-level LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Examples

```bash
# English → Hindi (ISO 639-3 code)
python main.py --source en --target hin

# English → Tamil
python main.py --source en --target tam

# Debug mode (shows all internal logs)
python main.py --log-level DEBUG
```

> **Target language codes:** Use ISO 639-3 codes (`hin`, `tam`, `tel`, `kan`, `ben`, `mal`, `mar`, `guj`).
> These are the keys in `config.py`'s `MMS_TTS_MODEL_MAP` and `ARGOS_LANG_MAP`.
> Internally, the pipeline bridges ISO 639-3 to ISO 639-1 for Argos Translate.
> Run `python setup_models.py` after changing the target language to download the translation and TTS models.

---

## Project Structure

```
polyglot-talk/
├── polyglot_talk/         ← core package (7 modules, relative imports)
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── pipeline.py
│   ├── audio_capture.py
│   ├── asr_engine.py
│   ├── translator.py
│   └── tts_engine.py
├── benchmarks/            ← unchanged
├── tests/                 ← unchanged
├── data/                  ← datasets
│   ├── dev-clean/
│   └── test_sentences/
├── results/
│   └── deprecated/        ← was results_deprecated/
├── output/
├── docs/                  ← planning & contribution docs
│   ├── CONTRIBUTING.md
│   ├── IMPLEMENTATION_PLAN.md
│   └── PolyglotTalk_Task_List.md
├── main.py
├── setup_models.py
├── conftest.py
└── requirements.txt
```

---

## How It Works

```
┌──────────────┐  audio_queue  ┌──────────────┐  text_queue  ┌──────────────┐  tts_queue  ┌──────────────┐
│ AudioCapture │  (maxsize=2)  │  ASREngine   │  (maxsize=2) │  Translator  │ (maxsize=2) │  TTSEngine   │
│  Thread 1    │ ───────────►  │  Thread 2    │ ───────────► │  Thread 3    │ ──────────► │  Thread 4    │
│  sounddevice │  AudioChunk   │faster-whisper│  TextSegment │argos-translate  TranslatedSeg│  MMS-TTS     │
└──────────────┘               └──────────────┘              └──────────────┘              └──────────────┘
```

All four threads run simultaneously. At steady state:

- **Thread 1** records the next chunk (2.5 s window, 1.5 s stride) while Thread 2 is still transcribing the current one. Each chunk overlaps the previous by 1.0 s so words at boundaries are never split.
- **Thread 2** transcribes each chunk and deduplicates the overlapping text. Fragments accumulate in a sentence buffer and are only forwarded to Thread 3 when a natural sentence boundary is detected (silence gap ≥ 5 s or buffer ≥ 25 words).
- **Thread 3** translates while Thread 4 synthesises and saves the chunk before that.

End-to-end latency ≈ `stride_duration + ASR_time + MT_time` ≈ **4–6 seconds** — not the sum of all stage durations.

**Backpressure:** All queues use a drop-oldest strategy — if a downstream stage falls behind, the oldest unprocessed item is evicted to make room for the freshest one, so the pipeline always stays current.

**Context continuity:** The translator maintains a rolling window of the last 2 source segments, prepending them as context to each new translation to reduce sentence-boundary errors.

**Audio isolation:** TTS output is saved to `output/chunk_NNNN.wav` files instead of being played through speakers. This prevents synthesised speech from feeding back into the microphone, which would corrupt future ASR and create feedback loops.

---

## Configuration

Key values in [config.py](polyglot_talk/config.py):

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | `16000` Hz | Whisper requires 16 kHz mono |
| `CHUNK_DURATION` | `2.5` s | Total audio window sent to Whisper per call |
| `CHUNK_OVERLAP` | `1.0` s | Seconds of audio shared with the previous chunk; prevents words being cut at boundaries |
| `SENTENCE_BUFFER_TIMEOUT` | `5.0` s | Flush the sentence buffer after this many seconds without new ASR text (must exceed CPU transcription gap) |
| `SENTENCE_BUFFER_MAXWORDS` | `25` | Force-flush the sentence buffer when it reaches this many words |
| `ASR_STRIP_TRAILING_PERIOD` | `True` | Remove the period Whisper auto-appends to every chunk; prevents MT treating mid-sentence fragments as complete sentences |
| `RMS_SILENCE_THRESHOLD` | `0.0001` | Below this RMS, audio is treated as silence |
| `ASR_MODEL_SIZE` | `"base.en"` | Whisper model variant |
| `ASR_COMPUTE_TYPE` | `"int8"` | Quantization (int8 = fastest on CPU) |
| `ASR_BEAM_SIZE` | `1` | Beam width (1 = greedy, fastest) |
| `QUEUE_MAXSIZE` | `2` | Max items per inter-thread queue |
| `CONTEXT_MAXLEN` | `2` | Number of past segments used as MT context |
| `TTS_OUTPUT_DIR` | `"output"` | Directory where synthesised WAV files are saved |
| `TARGET_LANG` | `"hin"` | ISO 639-3 target language code (used to select TTS model from `MMS_TTS_MODEL_MAP`) |
| `MMS_TTS_DEVICE` | `"auto"` | Device for MMS-TTS (`"auto"`, `"cuda"`, or `"cpu"`) |

---

## Running Tests

```bash
# Run all tests
.venv/bin/python -m pytest -v

# Run with skip summary
.venv/bin/python -m pytest -v -rs

# Run a specific test file
.venv/bin/python -m pytest tests/test_translator.py -v
```

| Test | Requires | Notes |
|---|---|---|
| `test_audio_capture.py` | Live microphone | Skipped in CI or if no audio device |
| `test_asr.py` | LibriSpeech dev-clean dataset | WER evaluation on 100 utterances; skipped if dataset absent |
| `test_translator.py` | Models installed | Verifies Devanagari output |
| `test_tts.py` | Models installed | Checks WAV output at `model.config.sampling_rate`; no reference audio needed |
| `test_context.py` | Nothing | Fully mocked — runs anywhere |
| `test_pipeline_e2e.py` | Models installed | Uses synthetic audio, no mic needed |

### Dataset Setup (ASR tests and benchmarks)

`test_asr.py` and the ASR benchmarks require the [LibriSpeech](https://www.openslr.org/12) `dev-clean` dataset (~350 MB). Download and extract it into the project root:

```bash
wget https://openslr.trmal.net/resources/12/dev-clean.tar.gz
mkdir -p data && tar -xzf dev-clean.tar.gz -C data
rm dev-clean.tar.gz
```

This extracts to `data/dev-clean/LibriSpeech/dev-clean/` relative to the project root, which is where the benchmarks and tests expect it.

---

## Dependencies

```
faster-whisper==1.1.0       # ASR — CTranslate2-optimised Whisper
argostranslate==1.9.6       # Machine translation — OpenNMT + CTranslate2
transformers>=4.49.0        # VitsModel + VitsTokenizer — MMS-TTS per-language models
sounddevice==0.5.1          # Microphone input via PortAudio
soundfile==0.13.1           # FLAC/WAV I/O for ASR benchmarking + TTS output
jiwer==4.0.0                # WER calculation for ASR evaluation
numpy>=1.24,<2.0            # Audio arrays
scipy>=1.11                 # Signal processing utilities
pytest==8.3.4               # Testing
torch==2.10.0               # PyTorch (CPU ok, CUDA optional)
```

---

## Adding a New Language

To add support for a new Indian language to MMS-TTS:

1. Verify Facebook publishes an MMS-TTS checkpoint for that language at `facebook/mms-tts-{lang}` (e.g., `facebook/mms-tts-pan` for Punjabi)
2. Verify Argos Translate has an `en→{lang}` package at [argosopentech.com](https://www.argosopentech.com/argospm/index/)
3. In `polyglot_talk/config.py`, add entries to both maps:
   ```python
   "pan": "facebook/mms-tts-pan",  # Punjabi → MMS_TTS_MODEL_MAP
   "pan": "pa",                     # Punjabi → ARGOS_LANG_MAP (ISO 639-3 → ISO 639-1)
   ```
4. Run `python setup_models.py` after updating `TARGET_LANG` to the ISO 639-3 code
5. Test with `python main.py --target pan`

---

## Future Upgrades

- **Custom voice cloning:** Support custom TTS voice cloning with reference audio
- **Live playback:** Optional mode to play TTS output to speakers (separate input device to prevent feedback)
- **Improved VAD:** Add Silero-VAD before `ASREngine` to pre-filter silence, eliminating ASR hallucinations
