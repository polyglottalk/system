/**
 * StatsBar
 *
 * Shows live pipeline statistics: chunk count, sentence count,
 * average e2e latency, mic status, and WebSocket connection state.
 */
export function StatsBar({ stats, connected, pipelineStatus }) {
  const latency = stats.latencySamples.length > 0
    ? Math.round(stats.latencySamples.reduce((a, b) => a + b, 0) / stats.latencySamples.length)
    : null

  const micColor =
    pipelineStatus === 'ready'   ? 'var(--accent-green)' :
    pipelineStatus === 'stopped' ? 'var(--accent-red)'   :
    connected                    ? 'var(--accent-amber)'  :
                                   'var(--text-muted)'

  const micLabel =
    pipelineStatus === 'ready'   ? 'Live' :
    pipelineStatus === 'stopped' ? 'Stopped' :
    connected                    ? 'Waiting' :
                                   'Offline'

  return (
    <div
      className="flex items-center gap-6 px-4 py-2.5 shrink-0"
      style={{ borderTop: '1px solid var(--border)' }}
    >
      {/* WS connection */}
      <Stat
        icon={
          <span
            className="w-2 h-2 rounded-full"
            style={{
              background: connected ? 'var(--accent-green)' : 'var(--accent-red)',
              display: 'inline-block',
              boxShadow: connected ? '0 0 6px var(--accent-green)' : 'none',
            }}
          />
        }
        label={connected ? 'Connected' : 'Disconnected'}
      />

      {/* Mic / pipeline status */}
      <Stat
        icon={
          <span style={{ color: micColor, fontSize: '1rem' }}>
            {pipelineStatus === 'ready' ? '🎤' : '🔇'}
          </span>
        }
        label={micLabel}
      />

      <div className="w-px h-4 bg-[var(--border)]" />

      <Stat label="Chunks"    value={stats.chunkCount} />
      <Stat label="Sentences" value={stats.sentenceCount} />
      <Stat label="Avg Latency" value={latency != null ? `${latency} ms` : '—'} />
      <Stat label="Translations" value={stats.translationCount} />
      <Stat label="TTS Saved" value={stats.ttsCount} />
    </div>
  )
}

function Stat({ icon, label, value }) {
  return (
    <div className="flex items-center gap-1.5">
      {icon}
      <span className="text-xs text-[var(--text-muted)]">{label}</span>
      {value !== undefined && (
        <span className="text-xs font-semibold font-mono text-[var(--text-secondary)]">
          {value}
        </span>
      )}
    </div>
  )
}
