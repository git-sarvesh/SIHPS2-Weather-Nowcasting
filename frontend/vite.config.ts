/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Strict: if 5173 is occupied, fail loudly instead of silently drifting to
    // 5174. A drifting port breaks the HMR websocket and, more confusingly, moves
    // the app to an origin the backend's CORS allow-list does not include, so the
    // UI silently loses API access while still rendering.
    port: 5173,
    strictPort: true,
    host: '127.0.0.1',
    proxy: {
      // In dev the dashboard calls the backend through this same-origin proxy
      // (see VITE_API_BASE_URL in .env.example), which avoids CORS entirely.
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // MapLibre and Recharts are large and change rarely; splitting them keeps
        // the app chunk small enough to iterate on quickly.
        manualChunks: {
          maplibre: ['maplibre-gl'],
          charts: ['recharts'],
        },
      },
    },
  },
})
