/*
 * Runs real Mamba-130m once, offline, and exports what the Mamba page shows: its next-token guesses
 * (to set beside GPT-2 small's, a model of the same size) and the input-dependent step size Δ that
 * makes the state-space model selective. The browser never runs the model.
 *
 *   node scripts/mamba-export.ts           (Node 23+, which runs TypeScript directly)
 *
 * Needs, in ~/.cache/token-trails/mamba/: model.safetensors and tokenizer.json
 * from https://huggingface.co/state-spaces/mamba-130m-hf. Writes src/data/mamba.json.
 */
import { readFileSync, writeFileSync, openSync, readSync } from 'node:fs'

const DIR = `${process.env.HOME}/.cache/token-trails/mamba`
const D = 768, DI = 1536, N = 16, R = 48, L = 24, K = 4

/* ---------- weights ---------- */
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
    out.set(name, copy)
  }
  return out
}
const W = loadSafetensors(`${DIR}/model.safetensors`)
const w = (name: string) => { const t = W.get(name); if (!t) throw new Error(name); return t }

/* ---------- GPT-NeoX's byte-level BPE (same scheme as GPT-2, its own merges) ---------- */
const tj = JSON.parse(readFileSync(`${DIR}/tokenizer.json`, 'utf8'))
const vocab: Record<string, number> = tj.model.vocab
const ranks = new Map<string, number>((tj.model.merges as (string | string[])[]).map((m, i) => [Array.isArray(m) ? m.join(' ') : m, i]))
const idTok = new Map(Object.entries(vocab).map(([t, i]) => [i, t]))
const byteChar = (() => {
  const bs: number[] = []
  for (let i = 33; i <= 126; i++) bs.push(i)
  for (let i = 161; i <= 172; i++) bs.push(i)
  for (let i = 174; i <= 255; i++) bs.push(i)
  const cs = [...bs]
  let n = 0
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++) }
  return new Map(bs.map((b, i) => [b, String.fromCharCode(cs[i])]))
})()
function bpe(parts: string[]): string[] {
  while (parts.length > 1) {
    let best = -1, bestRank = Infinity
    for (let i = 0; i < parts.length - 1; i++) { const r = ranks.get(parts[i] + ' ' + parts[i + 1]); if (r !== undefined && r < bestRank) { bestRank = r; best = i } }
    if (best < 0) break
    parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)]
  }
  return parts
}
function encode(text: string): number[] {
  const ids: number[] = []
  for (const m of text.normalize('NFC').matchAll(/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu))
    for (const t of bpe([...Buffer.from(m[0], 'utf8')].map((b) => byteChar.get(b)!))) ids.push(vocab[t])
  return ids
}

/* ---------- math ---------- */
function rmsNorm(x: Float32Array, g: Float32Array): Float32Array {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i] * x[i]
  const r = 1 / Math.sqrt(s / x.length + 1e-5)
  return x.map((v, i) => v * r * g[i])
}
/** y = W x (+ b), W stored [out × in]. */
function linear(x: Float32Array, Wt: Float32Array, out: number, b?: Float32Array): Float32Array {
  const y = b ? Float32Array.from(b) : new Float32Array(out), n = x.length
  for (let j = 0; j < out; j++) { let s = 0; const row = j * n; for (let i = 0; i < n; i++) s += x[i] * Wt[row + i]; y[j] += s }
  return y
}
const silu = (v: number) => v / (1 + Math.exp(-v))
const softplus = (v: number) => (v > 20 ? v : Math.log1p(Math.exp(v)))

/* ---------- forward, keeping Δ for every layer ---------- */
function forward(ids: number[]) {
  const emb = w('backbone.embeddings.weight'), T = ids.length
  let hid = ids.map((id) => emb.slice(id * D, id * D + D))
  const deltas: number[][][] = [] // [layer][t][channel]
  for (let l = 0; l < L; l++) {
    const P = `backbone.layers.${l}.mixer.`
    const u = hid.map((r) => rmsNorm(r, w(`backbone.layers.${l}.norm.weight`)))
    const xz = u.map((r) => linear(r, w(P + 'in_proj.weight'), 2 * DI))
    const cw = w(P + 'conv1d.weight'), cb = w(P + 'conv1d.bias')
    // causal depthwise convolution of width 4, then SiLU
    const x = xz.map((_, t) => {
      const o = new Float32Array(DI)
      for (let c = 0; c < DI; c++) {
        let s = cb[c]
        for (let k = 0; k < K; k++) { const tt = t - (K - 1) + k; if (tt >= 0) s += cw[c * K + k] * xz[tt][c] }
        o[c] = silu(s)
      }
      return o
    })
    const Alog = w(P + 'A_log'), Dp = w(P + 'D')
    const h = new Float32Array(DI * N), ys: Float32Array[] = [], dl: number[][] = []
    for (let t = 0; t < T; t++) {
      const dbl = linear(x[t], w(P + 'x_proj.weight'), R + 2 * N)
      const dt = linear(dbl.slice(0, R), w(P + 'dt_proj.weight'), DI, w(P + 'dt_proj.bias')).map(softplus)
      const B = dbl.slice(R, R + N), C = dbl.slice(R + N)
      const y = new Float32Array(DI)
      for (let c = 0; c < DI; c++) {
        let s = 0
        for (let n = 0; n < N; n++) {
          const A = -Math.exp(Alog[c * N + n]), i = c * N + n
          h[i] = Math.exp(dt[c] * A) * h[i] + dt[c] * B[n] * x[t][c]
          s += h[i] * C[n]
        }
        y[c] = (s + Dp[c] * x[t][c]) * silu(xz[t][DI + c])
      }
      ys.push(y); dl.push(Array.from(dt))
    }
    deltas.push(dl)
    hid = hid.map((r, t) => { const o = linear(ys[t], w(P + 'out_proj.weight'), D); return r.map((v, d) => v + o[d]) })
  }
  const last = rmsNorm(hid[T - 1], w('backbone.norm_f.weight')), V = emb.length / D, z = new Float32Array(V)
  for (let v = 0; v < V; v++) { let s = 0; for (let d = 0; d < D; d++) s += last[d] * emb[v * D + d]; z[v] = s }
  return { deltas, z }
}
function top(z: Float32Array, k: number) {
  let m = -Infinity
  for (const v of z) if (v > m) m = v
  let Z = 0
  for (const v of z) Z += Math.exp(v - m)
  return [...z.keys()].sort((a, b) => z[b] - z[a]).slice(0, k).map((i) => ({ s: idTok.get(i) ?? '?', p: Math.round((Math.exp(z[i] - m) / Z) * 1000) / 1000 }))
}

/* ---------- export ---------- */
const PROMPT = 'The cat sat on the'
const p1 = forward(encode(PROMPT))
const next = top(p1.z, 8)
console.log(`${PROMPT} →`, next.map((t) => `${t.s} ${(t.p * 100).toFixed(1)}%`).join(', '))
const TEXT = process.env.TRY ?? 'Mary went to the market. John went to the park. Mary bought'
const ids = encode(TEXT), toks = ids.map((i) => idTok.get(i) ?? '?'), f = forward(ids)
const meanDelta = f.deltas.map((layer) => layer.map((ch) => ch.reduce((a, b) => a + b, 0) / ch.length))
for (let l = 0; l < L; l++) console.log(`layer ${String(l + 1).padStart(2)}  ${meanDelta[l].map((v) => v.toFixed(3)).join(' ')}`)
console.log(toks.join(' | '))
console.log('next:', top(f.z, 5).map((t) => `${t.s} ${(t.p * 100).toFixed(1)}%`).join(', '))
const r3 = (v: number) => Math.round(v * 1000) / 1000
writeFileSync(new URL('../src/data/mamba.json', import.meta.url), JSON.stringify({
  model: 'Mamba-130m (state-spaces/mamba-130m-hf)',
  prompt: PROMPT, next,
  text: TEXT, toks, textNext: top(f.z, 5),
  /** Mean Δ over the 1,536 channels, per layer and token. */
  meanDelta: meanDelta.map((l) => l.map(r3)),
}))
console.log('wrote src/data/mamba.json')
