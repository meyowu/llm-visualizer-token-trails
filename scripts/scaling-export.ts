/*
 * A small scaling experiment for real, offline: GPT-2 small, medium and large (the same data and recipe at three
 * sizes) score the same text, which none of them can have seen (this site's own glossary, written in 2026), and the
 * mean next-token loss of each is exported with its parameter count. The browser never runs a model.
 *
 *   node scripts/scaling-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs model.safetensors of openai-community/gpt2, gpt2-medium and gpt2-large in ~/.cache/token-trails/gpt2,
 * gpt2-medium and gpt2-large (merges.txt in gpt2). Writes src/data/scaling.json.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { TERMS } from '../src/core/glossary.ts'
import { Gpt2Bpe } from '../src/lib/gpt2/bpe.ts'
import { loadSafetensors, model } from './gpt2-model.ts'

const HOME = `${process.env.HOME}/.cache/token-trails`
const bpe = new Gpt2Bpe(readFileSync(`${HOME}/gpt2/merges.txt`, 'utf8'))
// a few definitions from this site's glossary: ordinary English that postdates GPT-2's training data
const PICK = ['Attention', 'Residual stream', 'KV cache', 'Speculative decoding', 'Quantization', 'Gradient', 'Backpropagation', 'Fine-tuning', 'In-context learning', 'Tool call']
const text = PICK.map((t) => { const d = TERMS.find((x) => x.term === t); if (!d) throw new Error(t); return d.def }).join(' ')
const ids = bpe.encode(text)
const SIZES = [
  { name: 'GPT-2 small', dir: 'gpt2', L: 12, D: 768, H: 12 },
  { name: 'GPT-2 medium', dir: 'gpt2-medium', L: 24, D: 1024, H: 16 },
  { name: 'GPT-2 large', dir: 'gpt2-large', L: 36, D: 1280, H: 20 },
]
const r4 = (v: number) => Number(v.toPrecision(4))
const out = SIZES.map((s) => {
  const t0 = Date.now(), lp = model(loadSafetensors(`${HOME}/${s.dir}/model.safetensors`), s.L, undefined, undefined, { D: s.D, H: s.H })(ids)
  const losses = ids.slice(1).map((t, i) => -lp[i][t]), loss = losses.reduce((a, b) => a + b, 0) / losses.length
  const nonEmb = 12 * s.L * s.D * s.D, total = nonEmb + 50257 * s.D + 1024 * s.D
  console.log(`${s.name}: loss ${loss.toFixed(4)} over ${losses.length} tokens (${((Date.now() - t0) / 1000).toFixed(0)} s)`)
  return { name: s.name, layers: s.L, width: s.D, params: total, nonEmbedding: nonEmb, loss: r4(loss), losses: losses.map(r4) }
})
writeFileSync(new URL('../src/data/scaling.json', import.meta.url), JSON.stringify({ text, tokens: ids.length, terms: PICK, models: out }))
console.log('wrote src/data/scaling.json')
