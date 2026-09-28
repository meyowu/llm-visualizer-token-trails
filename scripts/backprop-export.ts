/*
 * Backpropagation through GPT-2 small, for real, offline: the forward pass on the Training sentence (the same one
 * the Next-token loss page scores), the mean next-token loss, and its gradient with respect to every weight, by the
 * chain rule layer by layer, in float64. Exports the gradient's size for every weight matrix, the gradient reaching
 * the residual stream between blocks, a slice of one linear layer's backward pass (dW = Xᵀ·dY), which positions each
 * position's loss sends gradient to, and a check of three weights against finite differences.
 *
 *   node scripts/backprop-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/gpt2/ (model.safetensors, merges.txt, as for gpt2-export.ts). Writes src/data/backprop.json.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { Gpt2Bpe } from '../src/lib/gpt2/bpe.ts'
import { D, L, gpt2, len } from './gpt2-grad.ts'
import { loadSafetensors } from './gpt2-model.ts'

const HOME = `${process.env.HOME}/.cache/token-trails`
const bpe = new Gpt2Bpe(readFileSync(`${HOME}/gpt2/merges.txt`, 'utf8'))
const W = new Map([...loadSafetensors(`${HOME}/gpt2/model.safetensors`)].map(([k, v]) => [k, Float64Array.from(v)]))
const { forward: fwd, backward: bwd, w } = gpt2(W)
const text = JSON.parse(readFileSync(new URL('../src/data/gpt2.json', import.meta.url), 'utf8')).training.text as string
const ids = bpe.encode(text), N = ids.length - 1, inp = ids.slice(0, N)
const forward = () => fwd(ids)
const backward = (f: ReturnType<typeof forward>, weights: number[]) => bwd(f, weights)

const r4 = (v: number) => Number(v.toPrecision(4))
const f = forward()
console.log(`${JSON.stringify(text)}: mean loss ${f.loss.toFixed(4)}`, f.losses.map((v) => v.toFixed(3)).join(' '))
const b = backward(f, inp.map(() => 1 / N))
const norm = (n: string) => r4(len(b.G.get(n)!))
const KINDS = ['ln_1.weight', 'attn.c_attn.weight', 'attn.c_proj.weight', 'ln_2.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
const layers = Array.from({ length: L }, (_, l) => Object.fromEntries(KINDS.map((k) => [k, norm(`h.${l}.${k}`)])))
console.log('gradient norms, c_fc per block:', layers.map((x) => x['mlp.c_fc.weight']).join(' '))

// which input positions each position's loss reaches (the causal mask, backwards)
const blame = inp.map((_, i) => {
  const bi = backward(f, inp.map((__, j) => (j === i ? 1 : 0)))
  return bi.dx0.map((d) => r4(len(d)))
})
console.log('blame (row: loss at position i, col: embedding at j):', blame.map((r) => r.join(' ')).join(' | '))

// finite differences on three weights: (L(w + ε) − L(w − ε)) / 2ε against the chain rule
const CHECKS: [string, number][] = [[`h.${L - 1}.mlp.c_proj.weight`, 5 * D + 17], [`h.0.attn.c_attn.weight`, 100 * 3 * D + 900], ['wpe.weight', 2 * D + 40]]
const eps = 1e-4
const checks = CHECKS.map(([name, idx]) => {
  const t = w(name), v = t[idx]
  t[idx] = v + eps; const up = forward().loss
  t[idx] = v - eps; const down = forward().loss
  t[idx] = v
  const fd = (up - down) / (2 * eps), bp = b.G.get(name)![idx]
  console.log(`${name}[${idx}]: backprop ${bp.toExponential(6)} · finite difference ${fd.toExponential(6)}`)
  return { name, index: idx, row: Math.floor(idx / (name.includes('c_attn') ? 3 * D : D)), col: idx % (name.includes('c_attn') ? 3 * D : D), backprop: Number(bp.toPrecision(6)), finite: Number(fd.toPrecision(6)) }
})

// one linear layer's backward pass: block 12's MLP output projection, dims 1–8 of its 3072 inputs and 768 outputs
const sx = b.slice.mlp.x as number[][], sdy = b.slice.mlp.dy as number[][]
const dW = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => sx.reduce((a, r, n) => a + r[i] * sdy[n][j], 0)))
const full = b.G.get(`h.${L - 1}.mlp.c_proj.weight`)!
const agree = dW.every((r, i) => r.every((v, j) => Math.abs(v - full[i * D + j]) < 1e-9 * Math.max(1, Math.abs(v))))
console.log('slice dW matches the full gradient:', agree)

writeFileSync(new URL('../src/data/backprop.json', import.meta.url), JSON.stringify({
  model: 'GPT-2 small', text, tokens: ids.map((i) => bpe.symbolOf(i)), loss: r4(f.loss), losses: f.losses.map(r4),
  params: Object.fromEntries(KINDS.map((k) => [k, w(`h.0.${k}`).length])),
  layers, embed: { wte: norm('wte.weight'), wpe: norm('wpe.weight'), ln_f: norm('ln_f.weight') },
  stream: b.streamNorm.map(r4), blame,
  linear: { x: sx.map((r) => r.map(r4)), dy: sdy.map((r) => r.map((v) => Number(v.toPrecision(3)))), dW: dW.map((r) => r.map((v) => Number(v.toPrecision(3)))) },
  checks,
}))
console.log('wrote src/data/backprop.json')
