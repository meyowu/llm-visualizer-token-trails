import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

// `--mode artifact` inlines everything into one HTML file for publishing as a claude.ai Artifact.
export default defineConfig(({ mode }) => ({
  base: './',
  plugins: mode === 'artifact' ? [viteSingleFile()] : [],
  build: mode === 'artifact' ? { outDir: 'dist-artifact' } : {},
}))
