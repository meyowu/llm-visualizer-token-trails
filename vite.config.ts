import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

// The site lives at the root of its domain with a real path per page (/anatomy/attention/), so assets load from
// '/'. `--mode artifact` inlines everything into one HTML file for publishing as a claude.ai Artifact.
export default defineConfig(({ mode }) => ({
  base: mode === 'artifact' ? './' : '/',
  plugins: mode === 'artifact' ? [viteSingleFile()] : [],
  build: mode === 'artifact' ? { outDir: 'dist-artifact' } : {},
}))
