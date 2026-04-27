/**
 * TranscriptCard
 *
 * Renders one committed sentence pair: English source on top, target-language
 * translation below.  The card reflects pipeline stage via a status badge:
 *
 *   translating  → target-lang row shows a three-dot shimmer skeleton
 *   tts          → target-lang row visible, TTS row shows a spinner
 *   done         → target-lang row visible, TTS row shows filename + latency badge
 *
 * Props
 * -----
 * entry   {object}   { chunkId, text, translationText, targetLang, ttsFile, latencyMs, status }
 *   status: 'translating' | 'tts' | 'done'
 * opacity {number}   0–1 fade applied to older cards
 */
export function TranscriptCard({ entry, opacity = 1 }) {
  const { text, translationText, targetLang, ttsFile, latencyMs, status } = entry

  const uiTagMap = {
    hin: 'HI',
    guj: 'GU',
    tam: 'TA',
    tel: 'TE',
    kan: 'KA',
    ben: 'BN',
    mal: 'ML',
    mar: 'MR',
    en: 'EN',
    hi: 'HI',
    gu: 'GU',
    ta: 'TA',
    te: 'TE',
    ka: 'KA',
    bn: 'BN',
    ml: 'ML',
    mr: 'MR',
  }
  const langTag = uiTagMap[String(targetLang || '').toLowerCase()] ?? String(targetLang || '--').slice(0, 2).toUpperCase()

  return (
    <div
      className="rounded-xl p-4 space-y-2 animate-slide-up"
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        opacity,
        transition: 'opacity 0.3s ease',
      }}
    >
      {/* ── EN row ─────────────────────────────────────────────────── */}
      <div className="flex items-start gap-2">
        <span
          className="text-[10px] font-mono font-semibold mt-0.5 shrink-0 w-6 text-center rounded"
          style={{ color: 'var(--accent-blue)', background: 'rgba(79,142,247,0.12)', padding: '1px 4px' }}
        >
          EN
        </span>
        <p className="text-sm leading-snug" style={{ color: 'var(--text-secondary)' }}>
          {text}
        </p>
      </div>

      {/* ── Target-language row ─────────────────────────────────────── */}
      <div className="flex items-start gap-2">
        <span
          className="text-[10px] font-mono font-semibold mt-0.5 shrink-0 w-6 text-center rounded"
          style={{ color: 'var(--accent-purple)', background: 'rgba(155,109,255,0.12)', padding: '1px 4px' }}
        >
          {langTag}
        </span>
        {status === 'translating' ? (
          /* Shimmer skeleton while translation is pending */
          <div className="flex items-center gap-1.5 mt-1">
            {[0, 0.3, 0.6].map((delay) => (
              <span
                key={delay}
                className="inline-block w-2 h-2 rounded-full"
                style={{
                  background: 'var(--accent-purple)',
                  opacity: 0.4,
                  animation: `blink 1s step-end infinite ${delay}s`,
                }}
              />
            ))}
          </div>
        ) : (
          <p
            className="text-sm font-medium leading-snug animate-fade-in"
            style={{ color: 'var(--accent-purple)', direction: 'auto' }}
            dir="auto"
          >
            {translationText}
          </p>
        )}
      </div>

      {/* ── TTS row ────────────────────────────────────────────────── */}
      {status === 'tts' && (
        <div className="flex items-center gap-2 pl-8 mt-1">
          {/* Spinner */}
          <span
            className="animate-spin inline-block w-3 h-3 rounded-full border-2"
            style={{
              borderColor: 'var(--accent-amber) transparent transparent transparent',
            }}
          />
          <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
            TTS rendering…
          </span>
        </div>
      )}

      {status === 'done' && ttsFile && (
        <div className="flex items-center gap-2 pl-8 mt-1">
          {/* Checkmark */}
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <circle cx="6" cy="6" r="5.5" stroke="var(--accent-green)" strokeWidth="1"/>
            <polyline
              points="3.5,6 5.2,7.8 8.5,4"
              stroke="var(--accent-green)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="text-[10px] font-mono truncate" style={{ color: 'var(--text-muted)' }}>
            {ttsFile}
          </span>
          {latencyMs != null && (
            <span
              className="ml-auto text-[10px] font-mono px-1.5 py-0.5 rounded"
              style={{
                color: 'var(--accent-amber)',
                background: 'rgba(245,166,35,0.12)',
                whiteSpace: 'nowrap',
              }}
            >
              {latencyMs}ms
            </span>
          )}
        </div>
      )}
    </div>
  )
}
