# PolyglotTalk Dashboard

Real-time browser dashboard for the PolyglotTalk offline speech-to-speech pipeline.
Connects over WebSocket to a FastAPI backend that bridges the 4-thread Python pipeline.

```
Microphone → [ASR] → [Translator] → [TTS]
                 ↘                ↘         ↘
              ws broadcast    ws broadcast   ws broadcast + /audio/{file}
                              ↗
                   Browser dashboard (React + Tailwind)
```

There are **two ways to run the dashboard**. Choose whichever fits:

| | Dev mode | Prod mode |
|---|---|---|
| Requires | Node.js + running Vite dev server | `npm run build` once, no Node.js at runtime |
| Browser opens | `http://localhost:5173` | `http://localhost:8765` (or custom port) |
| Ports needed | 5173 (Vite) + 8765 (FastAPI) | 8765 only |
| Recommended for | Frontend development | Normal use on WSL2 / Linux |

> **WSL2 users**: See the WSL2 note below before starting.

---

## Prerequisites

| Tool | Purpose |
|---|---|
| Python (≥3.11) + venv | Pipeline and FastAPI backend |
| Node.js (≥18) + npm | Only needed to build or dev the frontend |

### Python backend deps
These are already installed if you ran `pip install -r requirements.txt`:
```
fastapi, uvicorn[standard]
```

### Installing Node.js on WSL2 / Ubuntu (one-time)
```bash
# Install fnm (fast Node manager) — no sudo needed
curl -fsSL https://fnm.vercel.app/install | bash
source ~/.bashrc           # reload PATH so fnm is available

fnm install 22             # download Node 22 LTS
fnm use 22
node --version             # should print v22.x.x
```

---

## Mode A — Production (recommended for WSL2)

Build the React app once, then FastAPI serves everything from a single port.

### Step 1 — Build the frontend
```bash
cd dashboard
npm install        # first time only
npm run build      # outputs to dashboard/dist/
cd ..
```

### Step 2 — Start the backend
```bash
source .venv/bin/activate
python main.py --dashboard            # opens http://localhost:8765
# or with a language pre-selected:
python main.py --dashboard --target hin
# or on a custom port:
python main.py --dashboard --dashboard-port 9000   # opens http://localhost:9000
```

Open **http://localhost:8765** (or your custom port) in your browser.
The dashboard and WebSocket are served from the **same** port — no Vite needed.

---

## Mode B — Development (hot-reload frontend)

Run both servers simultaneously. Vite proxies API/WS traffic to FastAPI so you
only need one browser port (5173).

### Terminal 1 — FastAPI backend
```bash
source .venv/bin/activate
python main.py --dashboard --no-prompt
```

### Terminal 2 — Vite dev server
```bash
cd dashboard
npm install       # first time only
npm run dev       # starts at http://localhost:5173
```

Open **http://localhost:5173** in your browser.

> **Custom backend port:** If you pass `--dashboard-port 9000`, start Vite with:
> ```bash
> VITE_API_PORT=9000 npm run dev
> ```
> This tells the Vite proxy where to forward API/WS requests.

### WSL2 note
The Vite dev server now binds to `0.0.0.0` (all interfaces), so your Windows
browser can reach it at `http://localhost:5173`. If you still can't connect,
check that WSL2 mirrored-mode networking is enabled in `.wslconfig`, or use
the WSL2 machine's IP address instead of `localhost`.

---

## Why WebSocket?

The pipeline is 4 continuously-running threads that push events (ASR chunks,
translations, TTS completions) as they happen. HTTP poll-based approaches
would introduce hundreds of milliseconds of extra latency and waste CPU.
WebSocket gives a persistent bidirectional channel so the server can **push**
events to the browser the instant they occur — with sub-millisecond overhead.

Each event is a small JSON object (< 200 bytes). The connection auto-reconnects
every 2 seconds if it drops, and the server re-sends the full pipeline state on
reconnect so the dashboard always reflects the real state.

---

## Dashboard Layout

```
┌─────────────────────────────────────────────┬──────────────┐
│  Live Transcription                         │              │
│  (growing partial text + blinking cursor)   │  Audio Files │
│─────────────────────────────────────────────│  (WAV player)│
│  Translation (→HIN / →TAM / etc.)           │              │
│  (fades in on each new translation)         │──────────────│
│                                             │  Event Log   │
│                                             │  (chunk log) │
├─────────────────────────────────────────────┴──────────────┤
│  Stats: WS · Mic · Chunks · Sentences · Avg Latency · TTS  │
└────────────────────────────────────────────────────────────┘
```

### Panels

| Panel | Description |
|---|---|
| **Live Transcription** | ASR chunks accumulate word-by-word with a blinking cursor. When a sentence is flushed it fades out. Older confirmed sentences appear muted above. |
| **Translation** | Latest translated text fades in on each `translation_done` event. Supports RTL scripts (Gujarati, Hindi, etc.) via `dir="auto"`. |
| **Audio Files** | Each saved `chunk_NNNN.wav` appears as a playable row. Click ▶ to play audio directly in the browser via the `/audio/` API endpoint. |
| **Event Log** | Colour-coded scrolling log of every pipeline event: `ASR` (blue), `SENT` (purple), `TRANS` (amber), `TTS` (green). |
| **Stats Bar** | Chunk count, sentence count, translation count, TTS file count, rolling average end-to-end latency (ms), mic status. |

---

## WebSocket Events

All events are JSON objects pushed on the `/ws` endpoint.

| Type | Fields | Description |
|---|---|---|
| `asr_chunk` | `chunk_id`, `text` | Raw deduplicated ASR fragment |
| `sentence_flushed` | `chunk_id`, `text` | Complete sentence flushed from buffer |
| `translation_done` | `chunk_id`, `text`, `lang` | Translated text (ISO 639-3 `lang`) |
| `tts_saved` | `chunk_id`, `filename`, `latency_ms` | WAV file written; `latency_ms` = capture→file |
| `pipeline_status` | `status` | `"ready"`, `"loading"`, `"paused"`, `"stopped"`, `"error"` |
| `connected` | — | Sent once on WebSocket handshake with current pipeline state |

All events also carry a `ts` field (Unix timestamp seconds).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `WS` | `/ws` | WebSocket event stream |
| `GET` | `/audio/{filename}` | Serve `output/chunk_NNNN.wav` |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/pipeline/state` | Current status, langs, paused flag |
| `POST` | `/pipeline/start` | Body: `{source_lang, target_lang}` |
| `POST` | `/pipeline/stop` | Stop pipeline (keeps server alive) |
| `POST` | `/pipeline/pause` | Pause microphone capture |
| `POST` | `/pipeline/resume` | Resume microphone capture |

---

## Ports

| Mode | Port(s) | Change how |
|---|---|---|
| Prod (FastAPI static) | `8765` only | `--dashboard-port N` |
| Dev (Vite proxy) | `5173` (browser) + `8765` (FastAPI) | `VITE_API_PORT=N npm run dev` + `--dashboard-port N` |

---

## How `--dashboard-port` works

`--dashboard-port` sets the port that **uvicorn** (FastAPI) listens on.

- **Prod mode**: the browser opens that port directly — one port, no confusion.
- **Dev mode**: the browser always uses Vite's port (5173). The `VITE_API_PORT`
  env var tells Vite's proxy which port to forward API/WS requests to.
  If you don't set `VITE_API_PORT`, the proxy defaults to `8765`.

