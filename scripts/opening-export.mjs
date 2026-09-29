/*
 * The one pass the opening animation draws, cut from src/data/gpt2.json so the home page does not download the
 * rest: GPT-2 small on "The cat sat on the" (first pass), its tokens, attention, and the top next-token logits.
 * Logits beyond the first 8 are folded into the tail's log-sum-exp at every temperature of the grid, so the
 * probabilities it computes are the same as the full file's. Rerun after scripts/gpt2-export.ts:
 *
 *   node scripts/opening-export.mjs      → src/data/opening.json
 */
import fs from 'node:fs'

const KEEP = 8
const src = new URL('../src/data/gpt2.json', import.meta.url), out = new URL('../src/data/opening.json', import.meta.url)
const R = JSON.parse(fs.readFileSync(src, 'utf8'))
const pass = R.presets.find((p) => p.key === 'cat').passes[0], n = pass.next
const lse = (xs) => { const m = Math.max(...xs); return m + Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0)) }
const tailLse = R.tGrid.map((T, t) => lse([n.tailLse[t], ...n.top.slice(KEEP).map((c) => (c.z - n.zmax) / T)]))
const opening = {
  model: R.model, prompt: 'cat', tGrid: R.tGrid,
  ids: pass.ids, syms: pass.syms, attn: pass.attn,
  next: { top: n.top.slice(0, KEEP), zmax: n.zmax, tailLse: tailLse.map((v) => Math.round(v * 1e6) / 1e6), tailCount: n.tailCount + n.top.length - KEEP, tailQuantiles: [] },
}
fs.writeFileSync(out, JSON.stringify(opening) + '\n')
console.log(`wrote ${out.pathname}: ${(fs.statSync(out).size / 1024).toFixed(1)} KiB`)
