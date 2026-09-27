/*
 * Runs speculative decoding for real, offline: distilgpt2 (6 layers) drafts tokens for GPT-2 small
 * (12 layers), which checks them. Exports every round (drafted tokens, the target's own choice at each
 * position, how many were accepted, the correction) and both models' probabilities for the drafted
 * tokens. The browser never runs a model.
 *
 *   node scripts/spec-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/gpt2/ (model.safetensors, merges.txt, as for gpt2-export.ts) and
 * ~/.cache/token-trails/distilgpt2/model.safetensors from https://huggingface.co/distilbert/distilgpt2.
 * Writes src/data/spec.json.
 */
import { readFileSync, writeFileSync, openSync, readSync } from 'node:fs'
import { Gpt2Bpe } from '../src/lib/gpt2/bpe.ts'

const HOME = `${process.env.HOME}/.cache/token-trails`
const D = 768, H = 12, DH = 64, V = 50257, FF = 3072

function loadSafetensors(path: string): Map<string, Float32Array> {
  const fd = openSync(path, 'r')
  const lenBuf = Buffer.alloc(8)
  readSync(fd, lenBuf, 0, 8, 0)
  const hlen = Number(lenBuf.readBigUInt64LE(0))
  const hbuf = Buffer.alloc(hlen)
  readSync(fd, hbuf, 0, hlen, 8)
  const header = JSON.parse(hbuf.toString('utf8'))
  const all = readFileSync(path)
  const out = new Map<string, Float32Array>()
  for (const [name, t] of Object.entries<any>(header)) {
    if (name === '__metadata__' || t.dtype !== 'F32') continue
    const [a, b] = t.data_offsets as [number, number]
    const copy = new Float32Array((b - a) / 4)
    Buffer.from(copy.buffer).set(all.subarray(8 + hlen + a, 8 + hlen + b))
    out.set(name.replace(/^transformer\./, ''), copy)
  }
  return out
}
function layerNorm(x: Float32Array, g: Float32Array, b: Float32Array): Float32Array {
  let m = 0
  for (let i = 0; i < x.length; i++) m += x[i]
  m /= x.length
  let v = 0
  for (let i = 0; i < x.length; i++) v += (x[i] - m) ** 2
  const s = Math.sqrt(v / x.length + 1e-5), out = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = ((x[i] - m) / s) * g[i] + b[i]
  return out
}
/** x · W + b with W stored [in × out] (GPT-2's Conv1D). */
function linear(x: Float32Array, Wt: Float32Array, b: Float32Array, out: number): Float32Array {
  const y = Float32Array.from(b)
  for (let i = 0; i < x.length; i++) { const xi = x[i], row = i * out; if (xi) for (let j = 0; j < out; j++) y[j] += xi * Wt[row + j] }
  return y
}
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))

/** A GPT-2-shaped model; returns log-probabilities for the next token at every position. */
function model(path: string, L: number) {
  const W = loadSafetensors(path), w = (n: string) => { const t = W.get(n); if (!t) throw new Error(n); return t }
  const wte = w('wte.weight'), wpe = w('wpe.weight')
  return (ids: number[]): Float32Array[] => {
    const N = ids.length
    let x = ids.map((id, p) => { const v = new Float32Array(D); for (let k = 0; k < D; k++) v[k] = wte[id * D + k] + wpe[p * D + k]; return v })
    for (let l = 0; l < L; l++) {
      const P = `h.${l}.`
      const qkv = x.map((r) => linear(layerNorm(r, w(P + 'ln_1.weight'), w(P + 'ln_1.bias')), w(P + 'attn.c_attn.weight'), w(P + 'attn.c_attn.bias'), 3 * D))
      const o = x.map(() => new Float32Array(D))
      for (let h = 0; h < H; h++) for (let i = 0; i < N; i++) {
        const sc: number[] = []
        for (let j = 0; j <= i; j++) { let s = 0; for (let k = 0; k < DH; k++) s += qkv[i][h * DH + k] * qkv[j][D + h * DH + k]; sc.push(s / 8) }
        const m = Math.max(...sc), e = sc.map((s) => Math.exp(s - m)), z = e.reduce((a, b) => a + b, 0)
        for (let j = 0; j <= i; j++) { const p = e[j] / z; for (let k = 0; k < DH; k++) o[i][h * DH + k] += p * qkv[j][2 * D + h * DH + k] }
      }
      x = x.map((r, i) => { const p = linear(o[i], w(P + 'attn.c_proj.weight'), w(P + 'attn.c_proj.bias'), D); return r.map((v, k) => v + p[k]) })
      x = x.map((r) => { const hid = linear(layerNorm(r, w(P + 'ln_2.weight'), w(P + 'ln_2.bias')), w(P + 'mlp.c_fc.weight'), w(P + 'mlp.c_fc.bias'), FF).map(gelu); const p = linear(hid, w(P + 'mlp.c_proj.weight'), w(P + 'mlp.c_proj.bias'), D); return r.map((v, k) => v + p[k]) })
    }
    return x.map((r) => {
      const f = layerNorm(r, w('ln_f.weight'), w('ln_f.bias')), z = new Float32Array(V)
      for (let t = 0; t < V; t++) { let s = 0; for (let k = 0; k < D; k++) s += f[k] * wte[t * D + k]; z[t] = s }
      let m = -Infinity
      for (const v of z) if (v > m) m = v
      let Z = 0
      for (const v of z) Z += Math.exp(v - m)
      const lz = m + Math.log(Z)
      return z.map((v) => v - lz)
    })
  }
}

const bpe = new Gpt2Bpe(readFileSync(`${HOME}/gpt2/merges.txt`, 'utf8'))
const sym = (id: number) => bpe.symbolOf(id)
const target = model(`${HOME}/gpt2/model.safetensors`, 12), draft = model(`${HOME}/distilgpt2/model.safetensors`, 6)
const argmax = (v: Float32Array) => { let b = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i; return b }
const r3 = (v: number) => Math.round(v * 1000) / 1000

const PROMPT = process.env.PROMPT ?? 'The cat sat on the', K = 4, ROUNDS = 5
const prompt = bpe.encode(PROMPT)
let ctx = [...prompt]
const rounds: any[] = []
let targetPasses = 0, draftPasses = 0
for (let r = 0; r < ROUNDS; r++) {
  // the draft model proposes K tokens, one pass each
  const drafted: number[] = [], q: number[] = []
  let d = [...ctx]
  for (let i = 0; i < K; i++) { const lp = draft(d); draftPasses++; const t = argmax(lp[d.length - 1]); drafted.push(t); q.push(Math.exp(lp[d.length - 1][t])); d = [...d, t] }
  // the target model scores all of them in one pass
  const lp = target([...ctx, ...drafted]); targetPasses++
  const choice = drafted.map((_, i) => argmax(lp[ctx.length - 1 + i])), p = drafted.map((t, i) => Math.exp(lp[ctx.length - 1 + i][t]))
  let acc = 0
  while (acc < K && choice[acc] === drafted[acc]) acc++
  const bonus = acc < K ? choice[acc] : argmax(lp[ctx.length - 1 + K])
  rounds.push({ ctx: ctx.length, drafted: drafted.map(sym), choice: choice.map(sym), p: p.map(r3), q: q.map(r3), accepted: acc, next: sym(bonus) })
  console.log(`round ${r + 1}: drafted ${drafted.map((t) => JSON.stringify(sym(t))).join(' ')} | target ${choice.map((t) => JSON.stringify(sym(t))).join(' ')} → accept ${acc}, then ${JSON.stringify(sym(bonus))}`)
  ctx = [...ctx, ...drafted.slice(0, acc), bonus]
}
// the same number of tokens by plain greedy decoding with the target alone
let plain = [...prompt]
while (plain.length < ctx.length) { const lp = target(plain); plain = [...plain, argmax(lp[plain.length - 1])] }
const same = plain.every((t, i) => t === ctx[i])
console.log(`speculative: ${bpe.decode(ctx)}\nplain:       ${bpe.decode(plain)}\nidentical: ${same} · ${ctx.length - prompt.length} tokens in ${targetPasses} target passes (+ ${draftPasses} draft passes)`)
writeFileSync(new URL('../src/data/spec.json', import.meta.url), JSON.stringify({
  target: 'GPT-2 small (openai-community/gpt2)', draft: 'distilgpt2 (distilbert/distilgpt2)', k: K,
  prompt: prompt.map(sym), rounds, output: ctx.slice(prompt.length).map(sym), identical: same, targetPasses, draftPasses,
}))
console.log('wrote src/data/spec.json')
