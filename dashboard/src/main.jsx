import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// StrictMode is intentionally omitted: it double-invokes effects in dev,
// which causes the WebSocket hook to open two simultaneous connections.
createRoot(document.getElementById('root')).render(<App />)
