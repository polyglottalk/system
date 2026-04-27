import { useEffect, useRef, useCallback } from 'react'

// Always connect WebSocket back to the same host+port that served this page.
// In dev mode (Vite proxy) that's ws://localhost:5173/ws → proxied to FastAPI.
// In prod mode (FastAPI static) that's ws://localhost:<dashboard-port>/ws directly.
const _proto  = window.location.protocol === 'https:' ? 'wss' : 'ws'
const WS_URL  = `${_proto}://${window.location.host}/ws`
const RECONNECT_DELAY_MS = 2000

/**
 * useWebSocket — auto-reconnecting WebSocket hook.
 *
 * @param {(event: object) => void} onMessage  Called for every parsed JSON message.
 * @param {(connected: boolean) => void} onStatusChange  Called on connect/disconnect.
 */
export function useWebSocket(onMessage, onStatusChange) {
  const wsRef = useRef(null)
  const timerRef = useRef(null)
  const mountedRef = useRef(true)

  const onMessageRef = useRef(onMessage)
  const onStatusRef  = useRef(onStatusChange)
  onMessageRef.current = onMessage
  onStatusRef.current  = onStatusChange

  const connect = useCallback(() => {
    if (!mountedRef.current) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return }
      onStatusRef.current(true)
    }

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        onMessageRef.current(data)
      } catch { /* ignore bad JSON */ }
    }

    ws.onclose  = () => {
      if (!mountedRef.current) return
      onStatusRef.current(false)
      timerRef.current = setTimeout(connect, RECONNECT_DELAY_MS)
    }

    ws.onerror = () => ws.close()
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      clearTimeout(timerRef.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [connect])
}
