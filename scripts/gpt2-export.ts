/*
 * Runs real GPT-2 small once, offline, and exports the small slices of its activations the site
 * shows: attention patterns, per-layer predictions (logit lens), the next-token distribution and
 * the vectors behind the Unembed page. The browser never runs the model.
 *
 *   node scripts/gpt2-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs, in ~/.cache/token-trails/gpt2/: model.safetensors, merges.txt, vocab.json
 * from https://huggingface.co/openai-community/gpt2. Writes src/data/gpt2.json.
 */
import { readFileSync, writeFileSync, openSync, readSync } from 'node:fs'
import { Gpt2Bpe } from '../src/lib/gpt2/bpe.ts'

const DIR = `${process.env.HOME}/.cache/token-trails/gpt2`
const D = 768, H = 12, DH = 64, L = 12, V = 50257, FF = 3072

/* ---------- weights ---------- */
function loadSafetensors(path: string): Map<string, { shape: number[]; data: Float32Array }> {
  const fd = openSync(path, 'r')
  const lenBuf = Buffer.alloc(8)
  readSync(fd, lenBuf, 0, 8, 0)
  const hlen = Number(lenBuf.readBigUInt64LE(0))
  const hbuf = Buffer.alloc(hlen)
  readSync(fd, hbuf, 0, hlen, 8)
  const header = JSON.parse(hbuf.toString('utf8'))
  const all = readFileSync(path)
  const out = new Map<string, { shape: number[]; data: Float32Array }>()
  for (const [name, t] of Object.entries<any>(header)) {
    if (name === '__metadata__') continue
    const [a, b] = t.data_offsets as [number, number]
    const slice = all.subarray(8 + hlen + a, 8 + hlen + b)
    const copy = new Float32Array(slice.byteLength / 4)
    Buffer.from(copy.buffer).set(slice)
    out.set(name, { shape: t.shape, data: copy })
  }
  return out
}
const W = loadSafetensors(`${DIR}/model.safetensors`)
const w = (name: string) => { const t = W.get(name); if (!t) throw new Error(name); return t.data }
const bpe = new Gpt2Bpe(readFileSync(`${DIR}/merges.txt`, 'utf8'))

/* ---------- math ---------- */
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
/** x [in] · W [in × out] + b, with W row-major as stored by GPT-2's Conv1D. */
function linear(x: Float32Array, Wt: Float32Array, b: Float32Array, out: number): Float32Array {
  const y = Float32Array.from(b)
  for (let i = 0; i < x.length; i++) {
    const xi = x[i], row = i * out
    if (xi === 0) continue
    for (let j = 0; j < out; j++) y[j] += xi * Wt[row + j]
  }
  return y
}
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
const wte = w('wte.weight'), wpe = w('wpe.weight')
function logitsOf(x: Float32Array): Float32Array {
  const z = new Float32Array(V)
  for (let v = 0; v < V; v++) {
    let s = 0
    const o = v * D
    for (let k = 0; k < D; k++) s += x[k] * wte[o + k]
    z[v] = s
  }
  return z
}

interface Forward {
  /** att[layer][head][i][j] for j ≤ i. */
  att: number[][][][]
  /** Last position's residual after each block (before ln_f). */
  resid: Float32Array[]
  /** ln_f output at the last position. */
  xf: Float32Array
  logits: Float32Array
  /** Block 1's MLP hidden activations (after GELU), one 3,072-vector per position. */
  mlp0: Float32Array[]
  /** The final residual stream at every position (before ln_f). */
  last: Float32Array[]
}
function forward(ids: number[]): Forward {
  const N = ids.length
  let x = ids.map((id, p) => { const v = new Float32Array(D); for (let k = 0; k < D; k++) v[k] = wte[id * D + k] + wpe[p * D + k]; return v })
  const att: number[][][][] = [], resid: Float32Array[] = []
  let mlp0: Float32Array[] = []
  for (let l = 0; l < L; l++) {
    const P = `h.${l}.`
    const a = x.map((r) => layerNorm(r, w(P + 'ln_1.weight'), w(P + 'ln_1.bias')))
    const qkv = a.map((r) => linear(r, w(P + 'attn.c_attn.weight'), w(P + 'attn.c_attn.bias'), 3 * D))
    const heads: number[][][] = []
    const o = x.map(() => new Float32Array(D))
    for (let h = 0; h < H; h++) {
      const A: number[][] = []
      for (let i = 0; i < N; i++) {
        const sc: number[] = []
        for (let j = 0; j <= i; j++) {
          let s = 0
          for (let k = 0; k < DH; k++) s += qkv[i][h * DH + k] * qkv[j][D + h * DH + k]
          sc.push(s / Math.sqrt(DH))
        }
        const m = Math.max(...sc), e = sc.map((s) => Math.exp(s - m)), z = e.reduce((p, q) => p + q, 0)
        const row = e.map((v) => v / z)
        A.push(row)
        for (let j = 0; j <= i; j++) for (let k = 0; k < DH; k++) o[i][h * DH + k] += row[j] * qkv[j][2 * D + h * DH + k]
      }
      heads.push(A)
    }
    att.push(heads)
    x = x.map((r, i) => { const p = linear(o[i], w(P + 'attn.c_proj.weight'), w(P + 'attn.c_proj.bias'), D); return r.map((v, k) => v + p[k]) })
    const hids: Float32Array[] = []
    x = x.map((r) => {
      const m = layerNorm(r, w(P + 'ln_2.weight'), w(P + 'ln_2.bias'))
      const hid = linear(m, w(P + 'mlp.c_fc.weight'), w(P + 'mlp.c_fc.bias'), FF).map(gelu)
      hids.push(hid)
      const p = linear(hid, w(P + 'mlp.c_proj.weight'), w(P + 'mlp.c_proj.bias'), D)
      return r.map((v, k) => v + p[k])
    })
    if (l === 0) mlp0 = hids
    resid.push(x[N - 1])
  }
  const xf = layerNorm(x[N - 1], w('ln_f.weight'), w('ln_f.bias'))
  return { att, resid, xf, logits: logitsOf(xf), mlp0, last: x }
}

/* ---------- export helpers ---------- */
const TGRID = Array.from({ length: 37 }, (_, i) => Math.round((0.2 + i * 0.05) * 100) / 100)
const TOPK = 256
const sym = (id: number) => bpe.symbolOf(id)
const r3 = (x: number) => Math.round(x * 1000) / 1000
function softmaxTop(z: Float32Array, k: number) {
  const idx = Array.from({ length: V }, (_, i) => i).sort((a, b) => z[b] - z[a])
  const m = z[idx[0]]
  let Z = 0
  for (let i = 0; i < V; i++) Z += Math.exp(z[i] - m)
  return { idx, top: idx.slice(0, k).map((id) => ({ id, s: sym(id), p: Math.exp(z[id] - m) / Z })) }
}
/** Next-token data: top-256 logits exactly, plus the rest summarised so any T on the slider grid is exact. */
function nextData(z: Float32Array) {
  const { idx } = softmaxTop(z, 1)
  const top = idx.slice(0, TOPK).map((id) => ({ id, s: sym(id), z: r3(z[id]) }))
  const rest = idx.slice(TOPK)
  const zmax = z[idx[0]]
  // log Σ_{rank > 256} exp((z − zmax)/T) for each T on the slider grid
  const tailLse = TGRID.map((T) => { let s = 0; for (const id of rest) s += Math.exp((z[id] - zmax) / T); return r3(Math.log(s)) })
  // the shape of the long tail, for drawing: 64 quantiles of the remaining logits
  const quant = Array.from({ length: 64 }, (_, q) => r3(z[rest[Math.min(rest.length - 1, Math.floor((q / 63) * (rest.length - 1)))]]))
  return { top, zmax: r3(zmax), tailLse, tailCount: rest.length, tailQuantiles: quant }
}
/** Attention for all layers and heads, rows j ≤ i, quantised to one byte each and base64-encoded. */
function packAttention(att: number[][][][]): string {
  const bytes: number[] = []
  for (const layer of att) for (const head of layer) for (const row of head) for (const v of row) bytes.push(Math.round(v * 255))
  return Buffer.from(Uint8Array.from(bytes)).toString('base64')
}
function lens(resid: Float32Array[]) {
  return resid.map((r) => {
    const xf = layerNorm(r, w('ln_f.weight'), w('ln_f.bias'))
    return softmaxTop(logitsOf(xf), 3).top.map((t) => ({ id: t.id, s: t.s, p: r3(t.p) }))
  })
}
/** A vector as base64 float32, so x · W_E[id] reproduces the logit exactly (int8 drifted by ~4 logits over 768 terms). */
const f32 = (v: Float32Array) => Buffer.from(Float32Array.from(v).buffer).toString('base64')

const EMB_DIMS = 16
/** The first dimensions of a 768-vector, for the overview's embedding strip. */
const head = (m: Float32Array, i: number) => Array.from(m.subarray(i * D, i * D + EMB_DIMS), r3)

/* ---------- presets ---------- */
const PRESETS = [
  { key: 'cat', text: 'The cat sat on the', note: 'a simple sentence' },
  { key: 'france', text: 'The capital of France is', note: 'recalling a fact' },
  { key: 'count', text: 'one two three four one two three', note: 'copying a pattern (induction heads)' },
]
const PASSES = 3

const out: any = { model: 'GPT-2 small (openai-community/gpt2)', tGrid: TGRID, topK: TOPK, presets: [] as any[] }
for (const pr of PRESETS) {
  let ids = bpe.encode(pr.text)
  const passes: any[] = []
  for (let p = 0; p < PASSES; p++) {
    const t0 = Date.now()
    const f = forward(ids)
    const nd = nextData(f.logits)
    const greedy = nd.top[0].id
    // block 1's MLP: the 24 neurons whose activations vary most across the context. Position 0 is left
    // out of the choice because GPT-2 gives the first token a few huge activations whatever it is.
    const NEUR = 24
    const variance = Array.from({ length: FF }, (_, n) => {
      const a = f.mlp0.slice(1).map((h) => h[n]), m = a.reduce((p2, q) => p2 + q, 0) / a.length
      return a.reduce((p2, q) => p2 + (q - m) ** 2, 0)
    })
    const picks = Array.from({ length: FF }, (_, n) => n).sort((a, b) => variance[b] - variance[a]).slice(0, NEUR).sort((a, b) => a - b)
    const pass: any = {
      ids, syms: ids.map(sym), attn: packAttention(f.att), lens: lens(f.resid), next: nd, greedy,
      wte: ids.map((id) => head(wte, id)), wpe: ids.map((_, i) => head(wpe, i)),
      neurons: { idx: picks, act: f.mlp0.map((h) => picks.map((n) => r3(h[n]))) },
    }
    if (p === 0) {
      // the Unembed page: the last position before and after ln_f, and the embedding rows of the top candidates, so x · W_E[id] can be shown for real
      const cands = nd.top.slice(0, 6).map((t) => t.id)
      pass.unembed = { h: f32(f.resid[L - 1]), xf: f32(f.xf), rows: cands.map((id) => ({ id, v: f32(wte.subarray(id * D, id * D + D)) })) }
    }
    passes.push(pass)
    console.log(`${pr.key} pass ${p}: ${JSON.stringify(bpe.decode(ids))} → ${JSON.stringify(sym(greedy))} (${Date.now() - t0} ms)`)
    ids = [...ids, greedy]
  }
  out.presets.push({ key: pr.key, text: pr.text, note: pr.note, passes })
}
// training: every position of one sentence predicts the token after it, all at once
{
  const text = 'The cat sat on the floor', ids = bpe.encode(text), f = forward(ids)
  const positions = ids.slice(0, -1).map((_, i) => {
    const z = logitsOf(layerNorm(f.last[i], w('ln_f.weight'), w('ln_f.bias'))), top = softmaxTop(z, 3)
    const m = Math.max(...z)
    let Z = 0
    for (let v = 0; v < V; v++) Z += Math.exp(z[v] - m)
    const target = ids[i + 1]
    return { target, p: Math.exp(z[target] - m) / Z, top: top.top.map((t) => ({ id: t.id, s: t.s, p: r3(t.p) })) }
  })
  out.training = { text, ids, syms: ids.map(sym), positions }
  console.log('training:', positions.map((q) => `${sym(q.target)} ${(q.p * 100).toFixed(1)}%`).join(' · '))
}
// a map of token embeddings: words from a few everyday categories, projected on the first two
// principal components of their (unit-length) W_E rows, with each word's nearest neighbours among
// the ~12,000 most common word tokens
{
  const CATS: Record<string, string[]> = {
    animals: 'cat dog horse cow pig bird fish mouse rabbit lion tiger bear wolf fox sheep duck'.split(' '),
    numbers: 'one two three four five six seven eight nine ten hundred thousand million'.split(' '),
    colours: 'red blue green yellow black white orange purple pink brown'.split(' '),
    days: 'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split(' '),
    months: 'January February March April May June July August September October November December'.split(' '),
    family: 'mother father brother sister son daughter wife husband uncle aunt'.split(' '),
    furniture: 'floor bed couch table chair sofa desk bench ground edge'.split(' '),
    verbs: 'run walk eat drink sleep write read speak think'.split(' '),
    'function words': 'the a and of to in is was that it for on with as'.split(' '),
    countries: 'France Germany Italy Spain China Japan India Russia Canada Mexico'.split(' '),
  }
  const unit = (id: number) => { const v = wte.subarray(id * D, id * D + D); let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n); return Float32Array.from(v, (x) => x / n) }
  const items = Object.entries(CATS).flatMap(([cat, ws]) => ws.map((w) => ({ w, cat, ids: bpe.encode(' ' + w) }))).filter((t) => t.ids.length === 1).map((t) => ({ ...t, id: t.ids[0], v: unit(t.ids[0]) }))
  const mu = new Float32Array(D)
  for (const t of items) t.v.forEach((x, k) => (mu[k] += x / items.length))
  const C = items.map((t) => t.v.map((x, k) => x - mu[k]))
  const dot = (p: ArrayLike<number>, q: ArrayLike<number>) => { let r = 0; for (let k = 0; k < D; k++) r += p[k] * q[k]; return r }
  function pc(avoid: Float32Array[]) {
    let w = Float32Array.from({ length: D }, (_, k) => Math.sin(k * 1.7 + avoid.length))
    for (let it = 0; it < 100; it++) {
      const nw = new Float32Array(D)
      C.forEach((v) => { const s2 = dot(v, w); for (let k = 0; k < D; k++) nw[k] += v[k] * s2 })
      for (const u of avoid) { const d2 = dot(nw, u); for (let k = 0; k < D; k++) nw[k] -= d2 * u[k] }
      const n = Math.sqrt(dot(nw, nw)); w = nw.map((x) => x / n)
    }
    return w
  }
  const p1 = pc([]), p2 = pc([p1])
  const pool: number[] = []
  for (let id = 256; id < 50257 && pool.length < 12000; id++) if (/^Ġ[A-Za-z]{2,}$/.test(sym(id))) pool.push(id)
  const poolV = pool.map(unit)
  out.embeddingMap = items.map((t, i) => {
    const sims = pool.map((id, j) => [id, dot(t.v, poolV[j])] as [number, number]).filter(([id]) => id !== t.id).sort((x, y) => y[1] - x[1]).slice(0, 5)
    return { s: sym(t.id), cat: t.cat, x: r3(dot(C[i], p1)), y: r3(dot(C[i], p2)), nn: sims.map(([id, c]) => [sym(id), r3(c)]) }
  })
  console.log('map:', items.length, 'words;', out.embeddingMap.find((m: any) => m.s === 'Ġcat').nn.map((n: any) => n[0]).join(' '))
}
writeFileSync(new URL('../src/data/gpt2.json', import.meta.url), JSON.stringify(out))
console.log('wrote src/data/gpt2.json')
