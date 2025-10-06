import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',           // serve your edge_detection.html at /
  publicDir: '../public',
  base: './',            // so your index.html references assets relatively
  appType: 'mpa',        // ← no SPA fallback  
});
