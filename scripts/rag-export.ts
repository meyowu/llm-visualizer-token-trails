/*
 * Retrieval-augmented generation run for real, offline. The corpus is this site's own text: every glossary
 * definition and every line of the visual-language legend. Each is embedded with all-MiniLM-L6-v2 (a 6-layer
 * BERT, mean-pooled and normalised); questions are embedded the same way and ranked by cosine similarity.
 * Qwen3-1.7B (thinking off) then answers with and without the retrieved passages in its prompt.
 * Also exports a 2D projection (the top two principal components) of every passage. The browser never runs a model.
 *
 *   node scripts/rag-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/minilm/ (model.safetensors, vocab.txt from
 * https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and ~/.cache/token-trails/qwen3/
 * (see scripts/qwen-model.ts). Writes src/data/rag.json.
 */
import { openSync, readFileSync, readSync, writeFileSync } from 'node:fs'
import { TERMS } from '../src/core/glossary.ts'
import { LEGEND } from '../src/core/legend.ts'
import { Qwen, chatPrompt } from './qwen-model.ts'

const DIR = `${process.env.HOME}/.cache/token-trails/minilm`
const D = 384, H = 12, DH = 32, L = 6, FF = 1536

/* ---------- the embedding model: a small BERT ---------- */
function loadSafetensors(path: string): Map<string, Float32Array> {
  const fd = openSync(path, 'r'), lenBuf = Buffer.alloc(8)
  readSync(fd, lenBuf, 0, 8, 0)
  const hlen = Number(lenBuf.readBigUInt64LE(0)), hbuf = Buffer.alloc(hlen)
  readSync(fd, hbuf, 0, hlen, 8)
  const header = JSON.parse(hbuf.toString('utf8')), all = readFileSync(path), out = new Map<string, Float32Array>()
  for (const [name, t] of Object.entries<any>(header)) {
    if (name === '__metadata__' || t.dtype !== 'F32') continue
    const [a, b] = t.data_offsets as [number, number], copy = new Float32Array((b - a) / 4)
    Buffer.from(copy.buffer).set(all.subarray(8 + hlen + a, 8 + hlen + b))
    out.set(name, copy)
  }
  return out
}
const W = loadSafetensors(`${DIR}/model.safetensors`)
const w = (n: string) => { const t = W.get(n); if (!t) throw new Error(n); return t }
const vocab = readFileSync(`${DIR}/vocab.txt`, 'utf8').replace(/\n$/, '').split('\n').map((s) => s.replace(/\r$/, ''))
const ids = new Map(vocab.map((t, i) => [t, i]))
const isPunct = (c: string) => /[!-/:-@[-`{-~]/.test(c) || /\p{P}/u.test(c)
function wordpiece(word: string): string[] {
  const pieces: string[] = []
  let start = 0
  while (start < word.length) {
    let end = word.length, piece = ''
    while (start < end) { const sub = (start > 0 ? '##' : '') + word.slice(start, end); if (ids.has(sub)) { piece = sub; break } end-- }
    if (!piece) return ['[UNK]']
    pieces.push(piece); start = end
  }
  return pieces
}
function tokenize(text: string): string[] {
  const out = ['[CLS]']
  for (const word of text.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu, '').split(/\s+/).filter(Boolean)) {
    let cur = ''
    for (const c of word) { if (isPunct(c)) { if (cur) out.push(...wordpiece(cur)); out.push(...wordpiece(c)); cur = '' } else cur += c }
    if (cur) out.push(...wordpiece(cur))
  }
  return [...out.slice(0, 255), '[SEP]']
}
function layerNorm(x: Float32Array, g: Float32Array, b: Float32Array) {
  let m = 0
  for (const v of x) m += v
  m /= x.length
  let s = 0
  for (const v of x) s += (v - m) ** 2
  const r = 1 / Math.sqrt(s / x.length + 1e-12), o = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) o[i] = (x[i] - m) * r * g[i] + b[i]
  return o
}
function linear(x: Float32Array, Wt: Float32Array, b: Float32Array, out: number) {
  const y = Float32Array.from(b), n = x.length
  for (let j = 0; j < out; j++) { let s = 0; const r = j * n; for (let i = 0; i < n; i++) s += x[i] * Wt[r + i]; y[j] += s }
  return y
}
const erf = (x: number) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y }
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2))
/** Mean of the last layer over every token, normalised to length 1 (sentence-transformers' pooling). */
function embed(text: string): Float32Array {
  const toks = tokenize(text), N = toks.length, we = w('embeddings.word_embeddings.weight'), pe = w('embeddings.position_embeddings.weight'), te = w('embeddings.token_type_embeddings.weight')
  let x = toks.map((t, p) => { const id = ids.get(t) ?? ids.get('[UNK]')!, v = new Float32Array(D); for (let k = 0; k < D; k++) v[k] = we[id * D + k] + pe[p * D + k] + te[k]; return layerNorm(v, w('embeddings.LayerNorm.weight'), w('embeddings.LayerNorm.bias')) })
  for (let l = 0; l < L; l++) {
    const P = `encoder.layer.${l}.`
    const q = x.map((r) => linear(r, w(P + 'attention.self.query.weight'), w(P + 'attention.self.query.bias'), D))
    const k = x.map((r) => linear(r, w(P + 'attention.self.key.weight'), w(P + 'attention.self.key.bias'), D))
    const v = x.map((r) => linear(r, w(P + 'attention.self.value.weight'), w(P + 'attention.self.value.bias'), D))
    const c = x.map(() => new Float32Array(D))
    for (let h = 0; h < H; h++) for (let i = 0; i < N; i++) {
      const sc = new Float64Array(N)
      let m = -Infinity
      for (let j = 0; j < N; j++) { let s = 0; for (let d = 0; d < DH; d++) s += q[i][h * DH + d] * k[j][h * DH + d]; sc[j] = s / Math.sqrt(DH); if (sc[j] > m) m = sc[j] }
      let z = 0
      for (let j = 0; j < N; j++) { sc[j] = Math.exp(sc[j] - m); z += sc[j] }
      for (let j = 0; j < N; j++) for (let d = 0; d < DH; d++) c[i][h * DH + d] += (sc[j] / z) * v[j][h * DH + d]
    }
    x = x.map((r, i) => { const o = linear(c[i], w(P + 'attention.output.dense.weight'), w(P + 'attention.output.dense.bias'), D); return layerNorm(r.map((a, d) => a + o[d]), w(P + 'attention.output.LayerNorm.weight'), w(P + 'attention.output.LayerNorm.bias')) })
    x = x.map((r) => { const hid = linear(r, w(P + 'intermediate.dense.weight'), w(P + 'intermediate.dense.bias'), FF).map(gelu), o = linear(hid, w(P + 'output.dense.weight'), w(P + 'output.dense.bias'), D); return layerNorm(r.map((a, d) => a + o[d]), w(P + 'output.LayerNorm.weight'), w(P + 'output.LayerNorm.bias')) })
  }
  const mean = new Float32Array(D)
  for (const r of x) for (let d = 0; d < D; d++) mean[d] += r[d] / N
  const norm = Math.hypot(...mean)
  return mean.map((v) => v / norm)
}
const dot = (a: Float32Array, b: Float32Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s }
const r3 = (v: number) => Math.round(v * 1000) / 1000

/* ---------- the corpus: this site's glossary and legend ---------- */
const docs = [
  ...TERMS.map((t) => ({ title: t.term, kind: 'glossary', text: `${t.term}: ${t.def}` })),
  ...LEGEND.map(([, t], i) => ({ title: `Legend ${i + 1}`, kind: 'legend', text: `How to read the pictures on this site: ${t}` })),
]
console.log(`${docs.length} passages`)
const E = docs.map((d) => embed(d.text))

// two principal components, for the map (power iteration on the centred embeddings)
const mean = new Float32Array(D)
for (const e of E) for (let d = 0; d < D; d++) mean[d] += e[d] / E.length
const Xc = E.map((e) => e.map((v, d) => v - mean[d]))
function pc(prev: Float32Array[]) {
  let v = Float32Array.from({ length: D }, (_, i) => Math.sin(i * 12.9898 + prev.length) + 0.5)
  for (let it = 0; it < 200; it++) {
    const nv = new Float32Array(D)
    for (const x of Xc) { const s = dot(x, v); for (let d = 0; d < D; d++) nv[d] += s * x[d] }
    for (const u of prev) { const s = dot(nv, u); for (let d = 0; d < D; d++) nv[d] -= s * u[d] }
    const n = Math.hypot(...nv); v = nv.map((a) => a / n)
  }
  return v
}
const pc1 = pc([]), pc2 = pc([pc1])
const proj = (e: Float32Array) => { const c = e.map((v, d) => v - mean[d]); return [r3(dot(c, pc1)), r3(dot(c, pc2))] }

const QUESTIONS = (process.env.Q ?? [
  'On this site, what does a hatched cell mean?',
  'What trick lets a big model check a small model\'s guesses?',
  'How does the model know which word came first?',
  'Who proposed PagedAttention, and in which year?',
].join('|')).split('|')
const K = 3
const ranked = QUESTIONS.map((question) => {
  const e = embed(question), scores = E.map((d) => dot(d, e)), order = [...scores.keys()].sort((a, b) => scores[b] - scores[a])
  console.log(`\n${question}`)
  order.slice(0, 5).forEach((i) => console.log(`  ${scores[i].toFixed(3)}  ${docs[i].title}`))
  return { question, vec: [...e.slice(0, 16)].map(r3), xy: proj(e), top: order.slice(0, 8).map((i) => ({ doc: i, score: r3(scores[i]) })) }
})
if (process.env.Q) process.exit(0)

/* ---------- answers, with and without the retrieved passages ---------- */
const q = await Qwen.load(`${process.env.HOME}/.cache/token-trails/qwen3`)
const ragPrompt = (question: string, ctx: number[]) => `Answer the question using only the context below. If the context does not contain the answer, say so.\n\nContext:\n${ctx.map((i, k) => `[${k + 1}] ${docs[i].text}`).join('\n')}\n\nQuestion: ${question}`
const answers = ranked.map((r) => {
  q.reset()
  const bare = q.generate(chatPrompt([{ role: 'user', content: r.question }]), { max: 90 }).text
  q.reset()
  const user = ragPrompt(r.question, r.top.slice(0, K).map((t) => t.doc)), chat = chatPrompt([{ role: 'user', content: user }])
  const grounded = q.generate(chat, { max: 90 }).text
  console.log(`\n${r.question}\n  bare: ${bare}\n  with context: ${grounded}`)
  return { bare, grounded, user, tokens: q.tok.encode(chat).length }
})
q.close()

writeFileSync(new URL('../src/data/rag.json', import.meta.url), JSON.stringify({
  embedder: 'all-MiniLM-L6-v2', dims: D, model: 'Qwen3-1.7B', k: K,
  docs: docs.map((d, i) => ({ ...d, xy: proj(E[i]), tokens: tokenize(d.text).length })),
  sample: { doc: ranked[0].top[0].doc, vec: [...E[ranked[0].top[0].doc].slice(0, 16)].map(r3) },
  queries: ranked.map((r, i) => ({ ...r, ...answers[i] })),
}))
console.log('wrote src/data/rag.json')
