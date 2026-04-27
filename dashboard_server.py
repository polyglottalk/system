"""
dashboard_server.py — Real-time WebSocket dashboard server for PolyglotTalk.

Architecture
------------
DashboardBroadcaster  — thread-safe asyncio bridge; pipeline threads call emit().
PipelineManager       — controls Pipeline lifecycle (start / stop / restart) from
                        HTTP endpoints; runs model loading in a background thread so
                        FastAPI stays responsive.

Endpoints
---------
  WS   /ws                   — WebSocket event stream
  GET  /audio/{fname}        — Serve output/chunk_NNNN.wav
  GET  /health               — {"status": "ok"}
  GET  /pipeline/state       — current status, langs, paused flag
  POST /pipeline/start       — body: {source_lang, target_lang}
  POST /pipeline/stop        — stop pipeline (keeps server alive)
  POST /pipeline/pause       — pause microphone capture
  POST /pipeline/resume      — resume microphone capture
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DashboardBroadcaster — thread-safe asyncio bridge
# ---------------------------------------------------------------------------

class DashboardBroadcaster:
    """Thread-safe event broadcaster for the pipeline threads."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Pause state — set = pipeline is paused; read by AudioCapture
        self.pause_event: threading.Event = threading.Event()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def emit(self, event: dict) -> None:
        """Fire-and-forget emit from any thread. Silent no-op if no clients."""
        if self._loop is None or not self._clients:
            return
        event.setdefault("ts", time.time())
        payload = json.dumps(event, ensure_ascii=False)
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: str) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        logger.info("Dashboard client connected (total=%d)", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        logger.info("Dashboard client disconnected (total=%d)", len(self._clients))


# Module-level singleton — imported by pipeline modules
broadcaster = DashboardBroadcaster()


# ---------------------------------------------------------------------------
# PipelineManager — controls Pipeline lifecycle from REST endpoints
# ---------------------------------------------------------------------------

class PipelineManager:
    """Start, stop, and restart the pipeline from outside main.py.

    Model loading is blocking (~5-15 s) so it always runs in a daemon thread.
    Status is communicated back to all WebSocket clients via broadcaster.emit().
    """

    # Human-readable status values broadcast as pipeline_status events
    IDLE    = "idle"
    LOADING = "loading"
    READY   = "ready"
    PAUSED  = "paused"
    STOPPED = "stopped"
    ERROR   = "error"

    def __init__(self) -> None:
        self._pipeline: Any = None           # polyglot_talk.pipeline.Pipeline | None
        self._source_lang: str = "en"
        self._target_lang: str = "hin"
        self._status: str = self.IDLE
        self._error_msg: str = ""
        self._lock: threading.Lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    def register_external_pipeline(
        self, pipeline: Any, source_lang: str, target_lang: str
    ) -> None:
        """Called from main.py after pipeline.start() to mark it as ready."""
        with self._lock:
            self._pipeline = pipeline
            self._source_lang = source_lang
            self._target_lang = target_lang
            self._status = self.READY

    def start(self, source_lang: str = "en", target_lang: str = "hin") -> None:
        """Start (or restart) the pipeline in a background thread."""
        t = threading.Thread(
            target=self._start_worker,
            args=(source_lang, target_lang),
            name="PipelineStartWorker",
            daemon=True,
        )
        t.start()

    def stop(self) -> None:
        """Stop the running pipeline (non-blocking)."""
        with self._lock:
            p = self._pipeline
            self._pipeline = None
            self._status = self.STOPPED
        if p is not None:
            threading.Thread(target=p.stop, name="PipelineStopWorker", daemon=True).start()
        broadcaster.pause_event.clear()
        broadcaster.emit({"type": "pipeline_status", "status": self.STOPPED})

    @property
    def state(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "running": self._status == self.READY,
                "loading": self._status == self.LOADING,
                "paused": broadcaster.pause_event.is_set(),
                "source_lang": self._source_lang,
                "target_lang": self._target_lang,
                "error": self._error_msg,
            }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _start_worker(self, source_lang: str, target_lang: str) -> None:
        with self._lock:
            if self._status == self.LOADING:
                logger.warning("Pipeline already loading — ignoring start request.")
                return
            # Stop any existing pipeline first
            old = self._pipeline
            self._pipeline = None
            self._status = self.LOADING
            self._error_msg = ""

        broadcaster.pause_event.clear()
        broadcaster.emit({
            "type": "pipeline_status",
            "status": self.LOADING,
            "source_lang": source_lang,
            "target_lang": target_lang,
        })

        if old is not None:
            try:
                old.stop()
            except Exception:
                logger.exception("Error stopping old pipeline")

        try:
            # config MUST be imported first (sets OMP_NUM_THREADS etc.)
            from polyglot_talk import config as _config  # noqa: F401
            from polyglot_talk.pipeline import Pipeline

            p = Pipeline(source_lang=source_lang, target_lang=target_lang)
            p.start()

            with self._lock:
                self._pipeline = p
                self._source_lang = source_lang
                self._target_lang = target_lang
                self._status = self.READY

            broadcaster.emit({
                "type": "pipeline_status",
                "status": self.READY,
                "source_lang": source_lang,
                "target_lang": target_lang,
            })
            logger.info("Pipeline started: %s → %s", source_lang, target_lang)

        except Exception as e:
            msg = str(e)
            with self._lock:
                self._status = self.ERROR
                self._error_msg = msg
            broadcaster.emit({
                "type": "pipeline_status",
                "status": self.ERROR,
                "message": msg,
            })
            logger.exception("Pipeline start failed")


# Module-level singleton
pipeline_manager = PipelineManager()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="PolyglotTalk Dashboard", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_OUTPUT_DIR = Path("output")
# Compiled React build — populated by `npm run build` inside dashboard/
_DIST_DIR   = Path("dashboard/dist")


@app.on_event("startup")
async def _startup() -> None:
    broadcaster.set_loop(asyncio.get_running_loop())
    logger.info("DashboardBroadcaster event loop registered.")


@app.get("/")
async def root():
    # Production mode: serve the pre-built React SPA directly.
    # Run `npm run build` inside the dashboard/ directory first.
    dist_index = _DIST_DIR / "index.html"
    if dist_index.exists():
        return FileResponse(str(dist_index))
    # Development mode: redirect to the Vite dev server.
    # Make sure `npm run dev` is running in dashboard/ (requires Node.js).
    return RedirectResponse(url="http://localhost:5173")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "ts": time.time()})


# ── Pipeline state & control ─────────────────────────────────────────────────

@app.get("/pipeline/state")
async def get_pipeline_state():
    return JSONResponse(pipeline_manager.state)


class StartBody(BaseModel):
    source_lang: str = "en"
    target_lang: str = "hin"


@app.post("/pipeline/start")
async def start_pipeline(body: StartBody):
    pipeline_manager.start(source_lang=body.source_lang, target_lang=body.target_lang)
    return JSONResponse({"status": "loading", "source_lang": body.source_lang, "target_lang": body.target_lang})


@app.post("/pipeline/stop")
async def stop_pipeline():
    pipeline_manager.stop()
    return JSONResponse({"status": "stopped"})


@app.post("/pipeline/pause")
async def pause_pipeline():
    broadcaster.pause_event.set()
    broadcaster.emit({
        "type": "pipeline_status",
        "status": "paused",
        "source_lang": pipeline_manager.state["source_lang"],
        "target_lang": pipeline_manager.state["target_lang"],
    })
    return JSONResponse({"status": "paused"})


@app.post("/pipeline/resume")
async def resume_pipeline():
    broadcaster.pause_event.clear()
    broadcaster.emit({
        "type": "pipeline_status",
        "status": "ready",
        "source_lang": pipeline_manager.state["source_lang"],
        "target_lang": pipeline_manager.state["target_lang"],
    })
    return JSONResponse({"status": "ready"})


# ── Audio file serving ────────────────────────────────────────────────────────

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    if not filename.startswith("chunk_") or not filename.endswith(".wav"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    path = _OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(str(path), media_type="audio/wav")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await broadcaster.connect(ws)
    # Send current state on connect so client syncs immediately
    state = pipeline_manager.state
    await ws.send_text(json.dumps({"type": "connected", "ts": time.time(), **state}))
    try:
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                pass  # client is receive-only; keep the connection alive
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.disconnect(ws)


# ── Production SPA static files ──────────────────────────────────────────────
# Mount the built React app so FastAPI can serve it without a separate Vite
# process.  This mount is registered AFTER all explicit API/WS routes so it
# only catches requests that nothing else matched.
# When dist/ does NOT exist (dev mode), the mount is skipped and the root()
# redirect to Vite handles it instead.
if _DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="spa")


# ── Server launcher ───────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Start uvicorn in the calling thread (blocking). Use a daemon thread."""
    import uvicorn

    cfg = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        loop="asyncio",
    )
    uvicorn.Server(cfg).run()
