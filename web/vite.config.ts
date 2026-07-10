import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Build straight into ../Public — the Swift server's static root.
// base './' keeps asset URLs relative so the same build works on any
// host/port (neuramesh host, LAN IP, localhost).
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: '../Public',
    emptyOutDir: true,
  },
  server: {
    // Dev mode: proxy API + WebSocket to a running nmp-dashboard.
    proxy: {
      '/api': 'http://localhost:3000',
      '/health': 'http://localhost:3000',
      '/ws': { target: 'ws://localhost:3000', ws: true },
    },
  },
});
