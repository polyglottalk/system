/**
 * PipelineControl
 *
 * Compact header control bar containing:
 *   - Source/target language selectors (target is editable)
 *   - Start / Stop button (changes based on pipeline state)
 *   - Pause / Resume button (only when pipeline is running)
 */

const SOURCE_LANGUAGES = [
  { code: 'en', name: 'English' },
]

const TARGET_LANGUAGES = [
  { code: 'hin', name: 'Hindi',     native: 'हिन्दी'   },
  { code: 'guj', name: 'Gujarati',  native: 'ગુજરાતી'  },
  { code: 'tam', name: 'Tamil',     native: 'தமிழ்'   },
  { code: 'tel', name: 'Telugu',    native: 'తెలుగు'   },
  { code: 'kan', name: 'Kannada',   native: 'ಕನ್ನಡ'   },
  { code: 'ben', name: 'Bengali',   native: 'বাংলা'   },
  { code: 'mal', name: 'Malayalam', native: 'മലയാളം'  },
  { code: 'mar', name: 'Marathi',   native: 'मराठी'   },
]

export { TARGET_LANGUAGES }

const selectStyle = {
  background: 'var(--bg-hover)',
  border: '1px solid var(--border)',
  borderRadius: 8,
  color: 'var(--text-primary)',
  padding: '4px 28px 4px 10px',
  fontSize: 12,
  fontFamily: 'Inter, system-ui, sans-serif',
  fontWeight: 500,
  cursor: 'pointer',
  outline: 'none',
  appearance: 'none',
  WebkitAppearance: 'none',
  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%238b90a7' d='M6 8L1 3h10z'/%3E%3C/svg%3E")`,
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'right 8px center',
  transition: 'border-color 0.2s',
}

function IconPlay() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="currentColor">
      <polygon points="1,0.5 8.5,4.5 1,8.5"/>
    </svg>
  )
}

function IconStop() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="currentColor">
      <rect x="0.5" y="0.5" width="8" height="8" rx="1.5"/>
    </svg>
  )
}

function IconPause() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="currentColor">
      <rect x="1" y="0.5" width="2.5" height="8" rx="0.8"/>
      <rect x="5.5" y="0.5" width="2.5" height="8" rx="0.8"/>
    </svg>
  )
}

function IconResume() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="currentColor">
      <polygon points="1,0.5 8.5,4.5 1,8.5"/>
    </svg>
  )
}

function ControlBtn({ onClick, color, borderColor, bg, children, title, id }) {
  return (
    <button
      id={id}
      onClick={onClick}
      title={title}
      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold transition-all active:scale-95 select-none"
      style={{ color, border: `1px solid ${borderColor}`, background: bg }}
    >
      {children}
    </button>
  )
}

export function PipelineControl({
  connected,
  pipelineStatus,
  paused,
  sourceLang,
  targetLang,
  onTargetLangChange,
  onStart,
  onStop,
  onPauseToggle,
}) {
  const isRunning  = pipelineStatus === 'ready' || paused
  const isLoading  = pipelineStatus === 'loading'
  const isStopped  = !isRunning && !isLoading
  const canPause   = connected && isRunning
  const canStart   = connected && isStopped && !isLoading
  const canStop    = connected && (isRunning || isLoading)

  const targetLangObj = TARGET_LANGUAGES.find(l => l.code === targetLang) ?? TARGET_LANGUAGES[0]

  return (
    <div className="flex items-center gap-2">

      {/* ── Language pill ─────────────────────────────────── */}
      <div
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg"
        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
      >
        {/* Source — fixed (EN only for now) */}
        <span className="text-[10px] font-semibold text-[var(--text-muted)] uppercase tracking-wide">
          EN
        </span>

        {/* Arrow */}
        <span className="text-[var(--text-muted)] text-xs">→</span>

        {/* Target dropdown */}
        <div className="relative">
          <select
            id="select-target-lang"
            value={targetLang}
            disabled={isLoading || (isRunning && !paused)}
            onChange={e => onTargetLangChange(e.target.value)}
            style={{
              ...selectStyle,
              opacity: (isLoading || (isRunning && !paused)) ? 0.5 : 1,
              cursor: (isLoading || (isRunning && !paused)) ? 'not-allowed' : 'pointer',
            }}
          >
            {TARGET_LANGUAGES.map(l => (
              <option key={l.code} value={l.code}>
                {l.name} — {l.native}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Start button ──────────────────────────────────── */}
      {canStart && (
        <ControlBtn
          id="btn-start-pipeline"
          onClick={onStart}
          color="var(--accent-green)"
          borderColor="rgba(62,201,124,0.4)"
          bg="rgba(62,201,124,0.12)"
          title="Start pipeline"
        >
          <IconPlay /> Start
        </ControlBtn>
      )}

      {/* ── Loading spinner ──────────────────────────────── */}
      {isLoading && (
        <div
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold"
          style={{
            color: 'var(--accent-amber)',
            border: '1px solid rgba(245,166,35,0.35)',
            background: 'rgba(245,166,35,0.10)',
          }}
        >
          <span className="inline-block w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
          Loading…
        </div>
      )}

      {/* ── Stop button (shown while running or loading) ── */}
      {canStop && (
        <ControlBtn
          id="btn-stop-pipeline"
          onClick={onStop}
          color="var(--accent-red)"
          borderColor="rgba(255,95,95,0.35)"
          bg="rgba(255,95,95,0.10)"
          title="Stop pipeline"
        >
          <IconStop /> Stop
        </ControlBtn>
      )}

      {/* ── Pause / Resume button ─────────────────────────── */}
      {canPause && (
        <ControlBtn
          id="btn-pause-resume"
          onClick={onPauseToggle}
          color={paused ? 'var(--accent-green)' : 'var(--accent-blue)'}
          borderColor={paused ? 'rgba(62,201,124,0.4)' : 'rgba(79,142,247,0.4)'}
          bg={paused ? 'rgba(62,201,124,0.12)' : 'rgba(79,142,247,0.10)'}
          title={paused ? 'Resume mic capture' : 'Pause mic capture'}
        >
          {paused ? <><IconResume /> Resume</> : <><IconPause /> Pause</>}
        </ControlBtn>
      )}
    </div>
  )
}
