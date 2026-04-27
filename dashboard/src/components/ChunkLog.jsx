import { useEffect, useRef } from 'react'

const TAG_META = {
  asr_chunk:        { label: 'ASR',  cls: 'tag-asr' },
  sentence_flushed: { label: 'SENT', cls: 'tag-sent' },
  translation_done: { label: 'TRANS', cls: 'tag-trans' },
  tts_saved:        { label: 'TTS',  cls: 'tag-tts' },
  pipeline_status:  { label: 'SYS',  cls: 'tag-status' },
  connected:        { label: 'SYS',  cls: 'tag-status' },
}

function tag(event) {
  return TAG_META[event.type] ?? { label: event.type.toUpperCase(), cls: 'tag-status' }
}

function eventText(event) {
  switch (event.type) {
    case 'asr_chunk':        return `[#${event.chunk_id}] ${event.text}`
    case 'sentence_flushed': return `[#${event.chunk_id}] ${event.text}`
    case 'translation_done': return `[#${event.chunk_id}] (→${event.lang?.toUpperCase()}) ${event.text}`
    case 'tts_saved':        return `[#${event.chunk_id}] ${event.filename}  (${event.latency_ms}ms e2e)`
    case 'pipeline_status':  return `Pipeline ${event.status}`
    case 'connected':        return 'WebSocket connected'
    default: return JSON.stringify(event)
  }
}

/**
 * ChunkLog
 *
 * Auto-scrolling list of events, colour-coded by type.
 */
export function ChunkLog({ events }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium tracking-widest uppercase text-[var(--text-muted)]">
            Event Log
          </span>
        </div>
        <span className="text-xs text-[var(--text-muted)] font-mono">
          {events.length} events
        </span>
      </div>

      {/* Scrolling list */}
      <div className="flex-1 overflow-y-auto space-y-1 pr-1">
        {events.length === 0 ? (
          <p className="text-[var(--text-muted)] text-sm italic">No events yet…</p>
        ) : (
          events.map((evt, i) => {
            const { label, cls } = tag(evt)
            return (
              <div
                key={i}
                className="flex items-start gap-2 animate-slide-up py-0.5"
              >
                <span
                  className={`shrink-0 text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded ${cls}`}
                >
                  {label}
                </span>
                <span className="text-[var(--text-secondary)] text-sm font-mono leading-5 break-all">
                  {eventText(evt)}
                </span>
              </div>
            )
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
