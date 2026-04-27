import { useEffect, useRef } from 'react'

/**
 * TranslationPanel
 *
 * Shows the latest translated text.  Each new translation fades in.
 * The text is large (≥ 28px) and supports RTL scripts naturally via
 * CSS `dir="auto"`.
 */
export function TranslationPanel({ translationText, lang }) {
  const keyRef = useRef(0)

  // Bump the key each time text changes so the fade-in re-triggers
  useEffect(() => { keyRef.current++ }, [translationText])

  const langLabel = lang ? lang.toUpperCase() : '—'

  return (
    <div className="flex flex-col h-full">
      {/* Label */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent-purple)]" />
          <span className="text-xs font-medium tracking-widest uppercase text-[var(--text-muted)]">
            Translation
          </span>
        </div>
        {lang && (
          <span className="text-xs px-2 py-0.5 rounded-full font-mono font-medium tag-trans">
            → {langLabel}
          </span>
        )}
      </div>

      {/* Translation text */}
      <div className="flex-1 flex items-center">
        {translationText ? (
          <p
            key={keyRef.current}
            className="animate-fade-in text-[var(--text-primary)] font-light w-full"
            style={{ fontSize: '1.75rem', lineHeight: 1.5, direction: 'auto' }}
            dir="auto"
          >
            {translationText}
          </p>
        ) : (
          <p
            className="text-[var(--text-muted)] font-light"
            style={{ fontSize: '1.75rem', lineHeight: 1.5 }}
          >
            Translation will appear here…
          </p>
        )}
      </div>
    </div>
  )
}
