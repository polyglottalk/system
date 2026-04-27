/**
 * LiveTranscript
 *
 * Shows the growing partial ASR text (word-by-word) plus a history of
 * confirmed sentences.  Each confirmed sentence shows its translation
 * inline — in a contrasting colour — as soon as the translation_done
 * event arrives (before TTS audio is generated).
 *
 * Props
 * -----
 * partialText    {string}   Accumulating words for the current sentence
 * flushing       {boolean}  True for 300ms while sentence animates out
 * confirmedEntries {Array}  [{chunkId, text, translation: string|null}]
 */
import { useRef } from 'react'

const MAX_HISTORY = 5  // how many past sentences to keep visible

export function LiveTranscript({ partialText, flushing, confirmedEntries }) {
  const bottomRef = useRef(null)

  const visible = confirmedEntries.slice(-MAX_HISTORY)

  return (
    <div className="flex flex-col h-full overflow-hidden">

      {/* ── Label ──────────────────────────────────────────── */}
      <div className="flex items-center gap-2 mb-3 shrink-0">
        <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent-blue)] animate-pulse-glow" />
        <span className="text-xs font-medium tracking-widest uppercase text-[var(--text-muted)]">
          Live Transcription
        </span>
      </div>

      {/* ── Confirmed sentence history ─────────────────────── */}
      <div className="flex-1 overflow-y-auto pr-1 space-y-3 pb-2">
        {visible.map((entry, i) => {
          // Fade older entries more
          const opacity = 0.35 + (i / Math.max(visible.length - 1, 1)) * 0.4

          return (
            <div
              key={entry.chunkId ?? i}
              className="space-y-0.5 animate-slide-up"
              style={{ opacity }}
            >
              {/* Original (English) sentence */}
              <p className="text-[var(--text-secondary)] text-base font-normal leading-snug">
                {entry.text}
              </p>

              {/* Translation — appears as soon as translation_done fires */}
              {entry.translation ? (
                <p
                  className="font-medium leading-snug animate-fade-in"
                  style={{
                    fontSize: '1.1rem',
                    color: 'var(--accent-purple)',
                    direction: 'auto',
                  }}
                  dir="auto"
                >
                  {entry.translation}
                </p>
              ) : (
                /* Skeleton shimmer while waiting for translation */
                <div className="flex items-center gap-1.5 mt-1">
                  <span
                    className="inline-block w-2 h-2 rounded-full"
                    style={{ background: 'var(--accent-purple)', opacity: 0.4, animation: 'blink 1s step-end infinite' }}
                  />
                  <span
                    className="inline-block w-2 h-2 rounded-full"
                    style={{ background: 'var(--accent-purple)', opacity: 0.4, animation: 'blink 1s step-end infinite 0.3s' }}
                  />
                  <span
                    className="inline-block w-2 h-2 rounded-full"
                    style={{ background: 'var(--accent-purple)', opacity: 0.4, animation: 'blink 1s step-end infinite 0.6s' }}
                  />
                </div>
              )}
            </div>
          )
        })}

        <div ref={bottomRef} />
      </div>

      {/* ── Active partial line (current speech) ──────────── */}
      <div
        className={`shrink-0 pt-3 min-h-[4rem] ${flushing ? 'animate-fade-out' : ''}`}
        style={{ borderTop: visible.length > 0 ? '1px solid var(--border)' : 'none' }}
      >
        {partialText ? (
          <p
            className="text-[var(--text-primary)] font-medium leading-tight"
            style={{ fontSize: '2rem', lineHeight: 1.3 }}
          >
            {partialText}
            <span
              className="animate-blink ml-1 inline-block w-0.5 h-7 bg-[var(--accent-blue)] align-middle"
              aria-hidden="true"
            />
          </p>
        ) : (
          <p
            className="text-[var(--text-muted)] font-light"
            style={{ fontSize: '2rem', lineHeight: 1.3 }}
          >
            Listening…
          </p>
        )}
      </div>
    </div>
  )
}
