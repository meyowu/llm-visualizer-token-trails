import { defineConfig, type Plugin } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

/**
 * Cloudflare Web Analytics (no cookies, no personal data), on the site build only: not on the dev server and not in
 * the single-file preview. Page changes through the History API are counted automatically; step changes use
 * replaceState and are not. The token identifies the site in the Cloudflare account and is public by design.
 */
const analytics = (): Plugin => ({
  name: 'cloudflare-web-analytics',
  apply: 'build',
  transformIndexHtml: {
    order: 'post',
    handler: () => [{
      tag: 'script', injectTo: 'body',
      attrs: { type: 'module', src: 'https://static.cloudflareinsights.com/beacon.min.js', 'data-cf-beacon': '{"token": "cfd87153b63a4090a371263cde133586"}' },
    }],
  },
})

// The site lives at the root of its domain with a real path per page (/anatomy/attention/), so assets load from
// '/'. `--mode artifact` inlines everything into one HTML file for publishing as a claude.ai Artifact.
export default defineConfig(({ mode }) => ({
  base: mode === 'artifact' ? './' : '/',
  plugins: mode === 'artifact' ? [viteSingleFile()] : [analytics()],
  build: mode === 'artifact' ? { outDir: 'dist-artifact' } : {},
}))
