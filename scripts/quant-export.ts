/*
 * Quantizes the real GPT-2 small, offline, and measures what it costs: every linear layer's weights
 * rounded to int8 or int4 with one scale per tensor, per output channel or per group of 128, then the
 * model run again. Exports the loss on a paragraph, the next-token distribution for a prompt, one real
 * weight column, and one real activation vector with its outliers. The browser never runs the model.
 *
 *   node scripts/quant-export.ts           (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/gpt2/ (model.safetensors, merges.txt, as for gpt2-export.ts).
 * Writes src/data/quant.json.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { Gpt2Bpe } from '../src/lib/gpt2/bpe.ts'
import { D, loadSafetensors, model, type Weights } from './gpt2-model.ts'

const DIR = `${process.env.HOME}/.cache/token-trails/gpt2`
const bpe = new Gpt2Bpe(readFileSync(`${DIR}/merges.txt`, 'utf8'))
const BASE = loadSafetensors(`${DIR}/model.safetensors`)
const LINEAR = (l: number) => [`h.${l}.attn.c_attn.weight`, `h.${l}.attn.c_proj.weight`, `h.${l}.mlp.c_fc.weight`, `h.${l}.mlp.c_proj.weight`]
const shapeOf = (name: string): [number, number] => (name.endsWith('c_attn.weight') ? [768, 2304] : name.includes('mlp.c_fc') ? [768, 3072] : name.includes('mlp.c_proj') ? [3072, 768] : [768, 768])

type Scheme = { name: string; bits: number; scale: 'tensor' | 'channel' | 'group' }
/** Symmetric absmax rounding; GPT-2 stores weights [in × out], so a channel is a column and a group is 128 rows of one column. */
function quantize(Wt: Float32Array, rows: number, cols: number, s: Scheme): Float32Array {
  const q = (1 << (s.bits - 1)) - 1, out = new Float32Array(Wt.length)
  const block = (idx: (k: number) => number, n: number) => {
    let m = 0
    for (let k = 0; k < n; k++) m = Math.max(m, Math.abs(Wt[idx(k)]))
    const sc = m / q || 1
    for (let k = 0; k < n; k++) out[idx(k)] = Math.round(Wt[idx(k)] / sc) * sc
  }
  if (s.scale === 'tensor') block((k) => k, Wt.length)
  else for (let c = 0; c < cols; c++) {
    if (s.scale === 'channel') block((k) => k * cols + c, rows)
    else for (let g = 0; g < rows; g += 128) block((k) => (g + k) * cols + c, Math.min(128, rows - g))
  }
  return out
}
const SCHEMES: Scheme[] = [
  { name: 'fp32', bits: 32, scale: 'tensor' },
  { name: 'int8 · per channel', bits: 8, scale: 'channel' },
  { name: 'int8 · per tensor', bits: 8, scale: 'tensor' },
  { name: 'int4 · groups of 128', bits: 4, scale: 'group' },
  { name: 'int4 · per channel', bits: 4, scale: 'channel' },
  { name: 'int4 · per tensor', bits: 4, scale: 'tensor' },
]
const TEXT = 'The history of computing began long before the first electronic machines. People used tables, slide rules and mechanical calculators to do arithmetic, and the first programmable computers filled whole rooms. Today a phone holds more memory than those machines could ever address.'
const ids = bpe.encode(TEXT), PROMPT = bpe.encode('The cat sat on the')
const r3 = (v: number) => Math.round(v * 1000) / 1000, r4 = (v: number) => Math.round(v * 10000) / 10000
let baseNext: Float32Array | null = null
const results: any[] = []
let probe: Float32Array | null = null
for (const s of SCHEMES) {
  const W: Weights = new Map(BASE)
  let err = 0, tot = 0
  if (s.bits < 32) for (let l = 0; l < 12; l++) for (const name of LINEAR(l)) {
    const [rows, cols] = shapeOf(name), orig = BASE.get(name)!, qd = quantize(orig, rows, cols, s)
    for (let i = 0; i < orig.length; i++) { err += (qd[i] - orig[i]) ** 2; tot += orig[i] ** 2 }
    W.set(name, qd)
  }
  const m = model(W, 12, s.bits === 32 ? (l, rows) => { if (l === 5) probe = rows[rows.length - 1] } : undefined)
  const lp = m(ids)
  let ce = 0
  for (let i = 0; i < ids.length - 1; i++) ce -= lp[i][ids[i + 1]]
  ce /= ids.length - 1
  const next = m(PROMPT).at(-1)!
  if (!baseNext) baseNext = next
  let kl = 0
  for (let v = 0; v < next.length; v++) kl += Math.exp(baseNext[v]) * (baseNext[v] - next[v])
  const top = [...next.keys()].sort((a, b) => next[b] - next[a]).slice(0, 6).map((v) => ({ s: bpe.symbolOf(v), p: r3(Math.exp(next[v])) }))
  results.push({ name: s.name, bits: s.bits, scale: s.scale, loss: r3(ce), ppl: Math.round(Math.exp(ce) * 10) / 10, kl: r4(kl), relErr: r4(Math.sqrt(err / (tot || 1))), top })
  console.log(`${s.name.padEnd(22)} loss ${ce.toFixed(3)} ppl ${Math.exp(ce).toFixed(1)} KL ${kl.toFixed(4)} rel.err ${Math.sqrt(err / (tot || 1)).toFixed(4)} · ${top.map((t) => `${t.s} ${(t.p * 100).toFixed(1)}%`).join(', ')}`)
}
// one real weight column (block 6's MLP up-projection, output channel 0), for the rounding picture
const fc = BASE.get('h.5.mlp.c_fc.weight')!, col = Array.from({ length: 768 }, (_, k) => fc[k * 3072])
const colMax = Math.max(...col.map(Math.abs))
// the activation going into that MLP for the last prompt token: a few dimensions are huge
const act = Array.from(probe!), top = [...act.keys()].sort((a, b) => Math.abs(act[b]) - Math.abs(act[a])).slice(0, 4)
console.log(`weight column max ${colMax.toFixed(3)}; activation max dims ${top.map((k) => `${k}: ${act[k].toFixed(1)}`).join(', ')}; median |x| ${[...act].map(Math.abs).sort((a, b) => a - b)[384].toFixed(3)}`)
writeFileSync(new URL('../src/data/quant.json', import.meta.url), JSON.stringify({
  model: 'GPT-2 small (openai-community/gpt2)', text: TEXT, tokens: ids.length, prompt: 'The cat sat on the',
  results, column: col.map(r4), activation: act.map(r3),
}))
console.log('wrote src/data/quant.json')
