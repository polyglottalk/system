import { useCallback, useEffect, useRef, useState } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { TranscriptFeed }  from './components/TranscriptFeed'
import { ChunkLog }        from './components/ChunkLog'
import { AudioSidebar }    from './components/AudioSidebar'
import { StatsBar }        from './components/StatsBar'
import { PipelineControl } from './components/PipelineControl'

const MAX_LOG_EVENTS = 200
// Use relative URLs so the app works regardless of which port serves it:
// - dev mode:  Vite proxy forwards /pipeline/* and /audio/* to FastAPI
// - prod mode: FastAPI serves the built files AND the API on the same port
const API_BASE = ''

function initStats() {
  return {
    chunkCount:       0,
    sentenceCount:    0,
    translationCount: 0,
    ttsCount:         0,
    latencySamples:   [],
  }
}

export default function App() {
  // ── Connection ──────────────────────────────────────────────────
  const [connected, setConnected] = useState(false)

  // ── Pipeline state ──────────────────────────────────────────────
  // status: 'idle' | 'loading' | 'ready' | 'paused' | 'stopped' | 'error' | 'offline'
  const [pipelineStatus, setPipelineStatus] = useState('idle')
  const [paused, setPaused]                 = useState(false)
  const [sourceLang, setSourceLang]         = useState('en')
  const [targetLang, setTargetLang]         = useState('hin')
  const targetLangRef                        = useRef(targetLang)
  const pendingTranslationsRef               = useRef(new Map())

  useEffect(() => {
    targetLangRef.current = targetLang
  }, [targetLang])

  // ── Transcription ───────────────────────────────────────────────
  const [partialText, setPartialText] = useState('')
  const [flushing, setFlushing]       = useState(false)
  // Each entry: { chunkId, text, translationText, targetLang, ttsFile, latencyMs, status }
  // status: 'translating' | 'tts' | 'done'
  const [confirmedEntries, setConfirmedEntries] = useState([])

  // ── Log & audio ─────────────────────────────────────────────────
  const [events, setEvents]         = useState([])
  const [audioFiles, setAudioFiles] = useState([])

  // ── Stats ────────────────────────────────────────────────────────
  const [stats, setStats] = useState(initStats)

  // ── Drawer (AudioSidebar + EventLog) ─────────────────────────────
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerTab, setDrawerTab]   = useState('audio') // 'audio' | 'log'

  // ── Helper: apply state from server ─────────────────────────────
  const applyServerState = useCallback((state) => {
    if (state.status)      setPipelineStatus(state.status)
    if (state.paused != null) setPaused(state.paused)
    if (state.source_lang) setSourceLang(state.source_lang)
    if (state.target_lang) setTargetLang(state.target_lang)
  }, [])

  // ── WebSocket message handler ─────────────────────────────────────
  const handleMessage = useCallback((event) => {
    // Always log (capped)
    setEvents(prev => {
      const next = [...prev, event]
      return next.length > MAX_LOG_EVENTS ? next.slice(-MAX_LOG_EVENTS) : next
    })

    switch (event.type) {
      case 'connected': {
        // Server sends current state on first connect + every reconnect
        applyServerState(event)
        break
      }

      case 'asr_chunk': {
        setPartialText(prev => prev ? `${prev} ${event.text}` : event.text)
        setStats(s => ({ ...s, chunkCount: s.chunkCount + 1 }))
        break
      }

      case 'sentence_flushed': {
        setFlushing(true)
        setTimeout(() => {
          const pending = pendingTranslationsRef.current.get(event.chunk_id)
          if (pending) pendingTranslationsRef.current.delete(event.chunk_id)

          setConfirmedEntries(prev => [...prev, {
            chunkId:   event.chunk_id,
            text:      event.text,
            translationText: pending?.text ?? null,
            targetLang: pending?.lang ?? targetLangRef.current,
            ttsFile:   null,
            latencyMs: null,
            status:    pending ? 'tts' : 'translating',
          }])
          setPartialText('')
          setFlushing(false)
        }, 120)
        setStats(s => ({ ...s, sentenceCount: s.sentenceCount + 1 }))
        break
      }

      case 'translation_done': {
        // Attach translation to the matching confirmed entry (by chunk_id)
        setConfirmedEntries(prev => {
          const idx = prev.findLastIndex?.(e => e.chunkId === event.chunk_id)
            ?? [...prev].reverse().findIndex(e => e.chunkId === event.chunk_id)
          if (idx === -1) {
            // sentence_flushed card may not exist yet (UI insertion delay)
            pendingTranslationsRef.current.set(event.chunk_id, {
              text: event.text,
              lang: event.lang ?? targetLangRef.current,
            })
            return prev
          }
          const realIdx = typeof prev.findLastIndex === 'function' ? idx : prev.length - 1 - idx
          return prev.map((e, i) => i === realIdx
            ? { ...e, translationText: event.text, targetLang: event.lang ?? e.targetLang ?? targetLangRef.current, status: 'tts' }
            : e)
        })
        setStats(s => ({ ...s, translationCount: s.translationCount + 1 }))
        break
      }

      case 'tts_saved': {
        setAudioFiles(prev => [...prev, { filename: event.filename, chunkId: event.chunk_id }])
        setConfirmedEntries(prev => {
          const idx = prev.findLastIndex?.(e => e.chunkId === event.chunk_id)
            ?? [...prev].reverse().findIndex(e => e.chunkId === event.chunk_id)
          if (idx === -1) {
            // If card was not inserted yet, keep translation payload for later attach.
            if (event.text) {
              pendingTranslationsRef.current.set(event.chunk_id, {
                text: event.text,
                lang: event.lang ?? targetLangRef.current,
              })
            }
            return prev
          }
          const realIdx = typeof prev.findLastIndex === 'function' ? idx : prev.length - 1 - idx
          return prev.map((e, i) =>
            i === realIdx
              ? {
                ...e,
                translationText: e.translationText ?? event.text ?? e.translationText,
                targetLang: e.targetLang ?? event.lang ?? targetLangRef.current,
                ttsFile: event.filename,
                latencyMs: event.latency_ms ?? null,
                status: 'done',
              }
              : e
          )
        })
        setStats(s => ({
          ...s,
          ttsCount: s.ttsCount + 1,
          latencySamples: event.latency_ms
            ? [...s.latencySamples.slice(-19), event.latency_ms]
            : s.latencySamples,
        }))
        break
      }

      case 'pipeline_status': {
        const status = event.status
        setPipelineStatus(status)
        if (event.source_lang) setSourceLang(event.source_lang)
        if (event.target_lang) setTargetLang(event.target_lang)

        if (status === 'paused') {
          setPaused(true)
        } else if (status === 'ready') {
          setPaused(false)
        } else if (status === 'stopped' || status === 'idle') {
          setPaused(false)
        }

        // Reset UI when a fresh pipeline becomes ready
        if (status === 'ready') {
          setStats(initStats())
          setPartialText('')
          setConfirmedEntries([])
          setAudioFiles([])
          pendingTranslationsRef.current.clear()
        }
        break
      }

      default: break
    }
  }, [applyServerState])

  // ── WebSocket status handler ──────────────────────────────────────
  const handleStatus = useCallback((isConnected) => {
    setConnected(isConnected)
    if (!isConnected) {
      setPipelineStatus(prev => prev === 'ready' || prev === 'paused' ? prev : 'offline')
      // Note: we don't reset paused/langs — server re-sends full state on reconnect
    }
  }, [])

  useWebSocket(handleMessage, handleStatus)

  // ── Pipeline control actions ──────────────────────────────────────
  const handleStart = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/pipeline/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_lang: sourceLang, target_lang: targetLang }),
      })
    } catch (e) { console.error('Start failed:', e) }
  }, [sourceLang, targetLang])

  const handleStop = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/pipeline/stop`, { method: 'POST' })
    } catch (e) { console.error('Stop failed:', e) }
  }, [])

  const handlePauseToggle = useCallback(async () => {
    const endpoint = paused ? '/pipeline/resume' : '/pipeline/pause'
    try {
      await fetch(`${API_BASE}${endpoint}`, { method: 'POST' })
    } catch (e) { console.error('Pause toggle failed:', e) }
  }, [paused])

  // ── Page title ────────────────────────────────────────────────────
  useEffect(() => {
    const statusIcon =
      pipelineStatus === 'ready'   ? '● '  :
      pipelineStatus === 'paused'  ? '⏸ '  :
      pipelineStatus === 'loading' ? '⏳ '  : ''
    document.title = `${statusIcon}PolyglotTalk Dashboard`
  }, [pipelineStatus])

  // ── Layout ────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col" style={{ height: '100vh', background: 'var(--bg-base)' }}>

      {/* ── Header ─────────────────────────────────────────────── */}
      <header
        className="flex items-center justify-between px-5 py-3 shrink-0 gap-4"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 shrink-0">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center text-sm shrink-0"
            style={{ background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-purple))' }}
          >
            🗣
          </div>
          <div>
            <h1 className="text-sm font-semibold text-[var(--text-primary)] leading-none">
              PolyglotTalk
            </h1>
            <p className="text-[10px] text-[var(--text-muted)] leading-tight mt-0.5">
              Offline Speech-to-Speech Dashboard
            </p>
          </div>
        </div>

        {/* Pipeline controls (center) */}
        <div className="flex-1 flex justify-center">
          <PipelineControl
            connected={connected}
            pipelineStatus={pipelineStatus}
            paused={paused}
            sourceLang={sourceLang}
            targetLang={targetLang}
            onTargetLangChange={setTargetLang}
            onStart={handleStart}
            onStop={handleStop}
            onPauseToggle={handlePauseToggle}
          />
        </div>

        {/* WS connection badge */}
        <div
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium shrink-0"
          style={{
            background: connected ? 'rgba(62,201,124,0.12)' : 'rgba(255,95,95,0.12)',
            color: connected ? 'var(--accent-green)' : 'var(--accent-red)',
            border: `1px solid ${connected ? 'rgba(62,201,124,0.3)' : 'rgba(255,95,95,0.3)'}`,
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: connected ? 'var(--accent-green)' : 'var(--accent-red)' }}
          />
          {connected ? 'WS Connected' : 'Reconnecting…'}
        </div>

        {/* Drawer toggle button */}
        <button
          onClick={() => setDrawerOpen(o => !o)}
          className="shrink-0 flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors"
          style={{
            background: drawerOpen ? 'rgba(79,142,247,0.15)' : 'var(--bg-card)',
            color: drawerOpen ? 'var(--accent-blue)' : 'var(--text-muted)',
            border: `1px solid ${drawerOpen ? 'rgba(79,142,247,0.35)' : 'var(--border)'}`,
          }}
          aria-label="Toggle sidebar drawer"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <rect x="1" y="1" width="12" height="12" rx="2"/>
            <line x1="9" y1="1" x2="9" y2="13"/>
          </svg>
          Logs
        </button>
      </header>

      {/* ── Main content ─────────────────────────────────────────── */}
      <main className="flex flex-1 overflow-hidden">

        {/* TranscriptFeed — full width, paired sentence cards */}
        <div className="flex-1 overflow-hidden p-5">
          <TranscriptFeed
            confirmedEntries={confirmedEntries}
            partialText={partialText}
            flushing={flushing}
          />
        </div>

        {/* Collapsible right drawer: AudioSidebar + EventLog */}
        <div
          className="flex flex-col shrink-0 overflow-hidden"
          style={{
            width: drawerOpen ? '320px' : '0',
            transition: 'width 0.25s ease',
            borderLeft: drawerOpen ? '1px solid var(--border)' : 'none',
          }}
        >
          {drawerOpen && (
            <>
              {/* Drawer tab bar */}
              <div
                className="flex shrink-0"
                style={{ borderBottom: '1px solid var(--border)' }}
              >
                {['audio', 'log'].map(tab => (
                  <button
                    key={tab}
                    onClick={() => setDrawerTab(tab)}
                    className="flex-1 py-2 text-xs font-medium transition-colors"
                    style={{
                      color: drawerTab === tab ? 'var(--accent-blue)' : 'var(--text-muted)',
                      borderBottom: drawerTab === tab ? '2px solid var(--accent-blue)' : '2px solid transparent',
                      background: 'none',
                      border: 'none',
                      borderBottom: drawerTab === tab ? `2px solid var(--accent-blue)` : '2px solid transparent',
                      cursor: 'pointer',
                    }}
                  >
                    {tab === 'audio' ? 'Audio Files' : 'Event Log'}
                  </button>
                ))}
              </div>

              {/* Drawer content */}
              <div className="flex-1 overflow-hidden p-4">
                {drawerTab === 'audio'
                  ? <AudioSidebar audioFiles={audioFiles} />
                  : <ChunkLog events={events} />
                }
              </div>
            </>
          )}
        </div>
      </main>

      {/* ── Stats bar ─────────────────────────────────────────────── */}
      <StatsBar
        stats={stats}
        connected={connected}
        pipelineStatus={pipelineStatus}
      />
    </div>
  )
}
