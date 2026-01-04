import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
    plugins: [react()],
    base: process.env.GITHUB_PAGES ? '/weatherflow/' : '/',
    build: {
        outDir: 'dist',
        sourcemap: true,
        rollupOptions: {
            output: {
                manualChunks: {
                    'react-vendor': ['react', 'react-dom'],
                    'plotly-vendor': ['plotly.js-dist-min', 'react-plotly.js'],
                    'three-vendor': ['three']
                }
            }
        }
    },
    server: {
        proxy: {
            '/api': {
                target: process.env.VITE_API_URL || 'http://localhost:8000',
                changeOrigin: true
            }
        }
    }
});
