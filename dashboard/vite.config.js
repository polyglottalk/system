import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// FastAPI backend port — override with VITE_API_PORT env var when using a
// custom --dashboard-port, e.g.: VITE_API_PORT=9000 npm run dev
const API_PORT = process.env.VITE_API_PORT || '8765'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    // host: true  binds to 0.0.0.0 so the dev server is reachable from the
    // Windows browser when running inside WSL2 (required for mirrored-mode or
    // localhost relay to work reliably).
    host: true,
    port: 5173,
    proxy: {
      // All WebSocket and REST traffic is proxied to the FastAPI backend so
      // the browser only ever needs ONE port (5173) in dev mode.
      '/ws': {
        target: `ws://localhost:${API_PORT}`,
        ws: true,
        changeOrigin: true,
      },
      '/audio': {
        target: `http://localhost:${API_PORT}`,
        changeOrigin: true,
      },
      '/health': {
        target: `http://localhost:${API_PORT}`,
        changeOrigin: true,
      },
      '/pipeline': {
        target: `http://localhost:${API_PORT}`,
        changeOrigin: true,
      },
    },
  },
})
