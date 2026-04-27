/**
 * TranscriptFeed
 *
 * Unified paired-sentence view.  Replaces the separate LiveTranscript and
 * TranslationPanel components.  Each committed sentence pair is rendered as a
 * TranscriptCard; the in-progress partial transcription is pinned at the bottom
 * with a subtle highlight.  The feed auto-scrolls to the latest card.
 *
 * Props
 * -----
 * confirmedEntries {Array}   [{chunkId, text, translationText, targetLang, ttsFile, latencyMs, status}]
 * partialText      {string}  Accumulating words for the current sentence
 * flushing         {boolean} True for 300ms while sentence animates out
 */
import { useEffect, useRef } from 'react'
import { TranscriptCard } from './TranscriptCard'

export function TranscriptFeed({ confirmedEntries, partialText, flushing }) {
  const bottomRef = useRef(null)

  // Auto-scroll to the bottom whenever entries or partial text change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [confirmedEntries, partialText])

  return (
    <div className="flex flex-col h-full overflow-hidden">

      {/* ── Label ──────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 mb-3 shrink-0">
        <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent-blue)] animate-pulse-glow" />
        <span className="text-xs font-medium tracking-widest uppercase text-[var(--text-muted)]">
          Live Transcription
        </span>
      </div>

      {/* ── Committed sentence history ──────────────────────────────── */}
      <div className="flex-1 overflow-y-auto pr-1 space-y-2 pb-2">
        {confirmedEntries.length === 0 && !partialText && (
          <p className="text-sm italic px-1 py-2" style={{ color: 'var(--text-muted)' }}>
            Sentence pairs will appear here once speech is detected…
          </p>
        )}

        {confirmedEntries.map((entry, i) => {
          // Fade older entries; newest is full opacity
          const opacity = 0.4 + (i / Math.max(confirmedEntries.length - 1, 1)) * 0.6
          return (
            <TranscriptCard
              key={entry.chunkId ?? i}
              entry={entry}
              opacity={opacity}
            />
          )
        })}

        <div ref={bottomRef} />
      </div>

      {/* ── In-progress row (current live capture) ─────────────────── */}
      {(partialText || flushing) && (
        <div
          className={`shrink-0 rounded-xl p-4 mt-2 ${flushing ? 'animate-fade-out' : ''}`}
          style={{
            background: 'rgba(79,142,247,0.06)',
            border: '1px solid rgba(79,142,247,0.2)',
          }}
        >
          <div className="flex items-start gap-2">
            <span
              className="text-[10px] font-mono font-semibold mt-0.5 shrink-0 w-6 text-center rounded"
              style={{ color: 'var(--accent-blue)', background: 'rgba(79,142,247,0.12)', padding: '1px 4px' }}
            >
              EN
            </span>
            <p
              className="font-medium leading-tight"
              style={{ fontSize: '1.15rem', color: 'var(--text-primary)', lineHeight: 1.35 }}
            >
              {partialText}
              <span
                className="inline-block w-0.5 h-5 ml-1 align-middle animate-blink"
                style={{ background: 'var(--accent-blue)' }}
              />
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
