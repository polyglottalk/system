import { useRef, useState } from 'react'

function AudioRow({ filename, chunkId }) {
  const [playing, setPlaying] = useState(false)
  const audioRef = useRef(null)

  const toggle = () => {
    if (!audioRef.current) {
      // Use a root-relative path so the URL always targets the same origin
      // that served the dashboard — works in both Vite dev mode (proxied)
      // and production (FastAPI static files).
      audioRef.current = new Audio(`/audio/${filename}`)
      audioRef.current.onended = () => setPlaying(false)
      audioRef.current.onerror = () => setPlaying(false)
    }
    if (playing) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      setPlaying(false)
    } else {
      audioRef.current.play().catch(() => setPlaying(false))
      setPlaying(true)
    }
  }

  return (
    <div
      className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-[var(--bg-hover)] transition-colors group"
      key={filename}
    >
      {/* Play / Pause button */}
      <button
        onClick={toggle}
        className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 transition-all"
        style={{
          background: playing ? 'var(--accent-green)' : 'var(--bg-hover)',
          border: '1px solid',
          borderColor: playing ? 'var(--accent-green)' : 'var(--border)',
        }}
        aria-label={playing ? 'Pause' : 'Play'}
      >
        {playing ? (
          // Pause icon
          <svg width="10" height="10" viewBox="0 0 10 10" fill="var(--bg-base)">
            <rect x="1.5" y="1" width="2.5" height="8" rx="0.5"/>
            <rect x="6" y="1" width="2.5" height="8" rx="0.5"/>
          </svg>
        ) : (
          // Play icon
          <svg width="10" height="10" viewBox="0 0 10 10" fill={playing ? 'var(--bg-base)' : 'var(--accent-green)'}>
            <polygon points="2,1 9,5 2,9"/>
          </svg>
        )}
      </button>

      {/* File info */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-mono text-[var(--text-secondary)] truncate">
          {filename}
        </p>
        <p className="text-[10px] text-[var(--text-muted)]">
          chunk #{String(chunkId).padStart(4, '0')}
        </p>
      </div>
    </div>
  )
}

/**
 * AudioSidebar
 *
 * Lists saved TTS WAV files as they are emitted.  Each entry has a
 * play/pause button that streams the file from the FastAPI audio endpoint.
 */
export function AudioSidebar({ audioFiles }) {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 shrink-0">
        <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent-green)]" />
        <span className="text-xs font-medium tracking-widest uppercase text-[var(--text-muted)]">
          Audio Files
        </span>
        <span className="ml-auto text-xs font-mono text-[var(--text-muted)]">
          {audioFiles.length} saved
        </span>
      </div>

      {/* File list */}
      <div className="flex-1 overflow-y-auto space-y-0.5">
        {audioFiles.length === 0 ? (
          <p className="text-[var(--text-muted)] text-sm italic px-3 py-2">
            Audio files appear here after TTS…
          </p>
        ) : (
          [...audioFiles].reverse().map((f) => (
            <AudioRow key={f.filename} filename={f.filename} chunkId={f.chunkId} />
          ))
        )}
      </div>
    </div>
  )
}
