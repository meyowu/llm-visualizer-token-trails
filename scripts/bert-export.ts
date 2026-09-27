/*
 * Runs real BERT-base (uncased) once, offline, and exports what the BERT page shows: masked-word
 * predictions that use both sides of the blank, and one attention head that looks ahead.
 * The browser never runs the model.
 *
 *   node scripts/bert-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs, in ~/.cache/token-trails/bert/: model.safetensors and vocab.txt
 * from https://huggingface.co/google-bert/bert-base-uncased. Writes src/data/bert.json.
 */
import { readFileSync, writeFileSync, openSync, readSync } from 'node:fs'

const DIR = `${process.env.HOME}/.cache/token-trails/bert`
const D = 768, H = 12, DH = 64, L = 12, FF = 3072

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
    if (name === '__metadata__') continue
    if (t.dtype !== 'F32') throw new Error(`${name}: ${t.dtype}`)
    const [a, b] = t.data_offsets as [number, number]
    const copy = new Float32Array((b - a) / 4)
    Buffer.from(copy.buffer).set(all.subarray(8 + hlen + a, 8 + hlen + b))
    out.set(name, copy)
  }
  return out
}
const W = loadSafetensors(`${DIR}/model.safetensors`)
/** Old checkpoints name LayerNorm parameters gamma/beta instead of weight/bias. */
const w = (name: string) => {
  const t = W.get(name) ?? W.get(name.replace('LayerNorm.weight', 'LayerNorm.gamma').replace('LayerNorm.bias', 'LayerNorm.beta'))
  if (!t) throw new Error(name)
  return t
}
const vocab = readFileSync(`${DIR}/vocab.txt`, 'utf8').replace(/\n$/, '').split('\n').map((s) => s.replace(/\r$/, ''))
const ids = new Map(vocab.map((t, i) => [t, i]))

/* ---------- WordPiece ---------- */
const isPunct = (c: string) => /[!-/:-@[-`{-~]/.test(c) || /\p{P}/u.test(c)
function basic(text: string): string[] {
  const t = text.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu, '')
  const out: string[] = []
  for (const word of t.split(/\s+/).filter(Boolean)) {
    let cur = ''
    for (const c of word) {
      if (isPunct(c)) { if (cur) out.push(cur); out.push(c); cur = '' } else cur += c
    }
    if (cur) out.push(cur)
  }
  return out
}
function wordpiece(word: string): string[] {
  const pieces: string[] = []
  let start = 0
  while (start < word.length) {
    let end = word.length, piece = ''
    while (start < end) {
      const sub = (start > 0 ? '##' : '') + word.slice(start, end)
      if (ids.has(sub)) { piece = sub; break }
      end--
    }
    if (!piece) return ['[UNK]']
    pieces.push(piece)
    start = end
  }
  return pieces
}
/** Tokens with [CLS] and [SEP]; the literal [MASK] is kept as one token. */
function tokenize(text: string): string[] {
  const parts = text.split('[MASK]')
  const toks: string[] = ['[CLS]']
  parts.forEach((part, k) => {
    for (const word of basic(part)) toks.push(...wordpiece(word))
    if (k < parts.length - 1) toks.push('[MASK]')
  })
  toks.push('[SEP]')
  return toks
}

/* ---------- math ---------- */
function layerNorm(x: Float32Array, g: Float32Array, b: Float32Array): Float32Array {
  let m = 0
  for (let i = 0; i < x.length; i++) m += x[i]
  m /= x.length
  let v = 0
  for (let i = 0; i < x.length; i++) v += (x[i] - m) ** 2
  const s = Math.sqrt(v / x.length + 1e-12), out = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = ((x[i] - m) / s) * g[i] + b[i]
  return out
}
/** PyTorch Linear: y = W x + b with W stored [out × in]. */
function linear(x: Float32Array, Wt: Float32Array, b: Float32Array, out: number): Float32Array {
  const y = Float32Array.from(b), n = x.length
  for (let j = 0; j < out; j++) {
    let s = 0
    const row = j * n
    for (let i = 0; i < n; i++) s += x[i] * Wt[row + i]
    y[j] += s
  }
  return y
}
function erf(x: number) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return x >= 0 ? y : -y
}
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2))

/* ---------- forward ---------- */
function forward(toks: string[]) {
  const N = toks.length, tid = toks.map((t) => ids.get(t) ?? ids.get('[UNK]')!)
  const we = w('bert.embeddings.word_embeddings.weight'), pe = w('bert.embeddings.position_embeddings.weight'), te = w('bert.embeddings.token_type_embeddings.weight')
  let x = tid.map((id, p) => {
    const v = new Float32Array(D)
    for (let k = 0; k < D; k++) v[k] = we[id * D + k] + pe[p * D + k] + te[k]
    return layerNorm(v, w('bert.embeddings.LayerNorm.weight'), w('bert.embeddings.LayerNorm.bias'))
  })
  const att: number[][][][] = []
  for (let l = 0; l < L; l++) {
    const P = `bert.encoder.layer.${l}.`
    const q = x.map((r) => linear(r, w(P + 'attention.self.query.weight'), w(P + 'attention.self.query.bias'), D))
    const k = x.map((r) => linear(r, w(P + 'attention.self.key.weight'), w(P + 'attention.self.key.bias'), D))
    const v = x.map((r) => linear(r, w(P + 'attention.self.value.weight'), w(P + 'attention.self.value.bias'), D))
    const ctx = x.map(() => new Float32Array(D)), heads: number[][][] = []
    for (let h = 0; h < H; h++) {
      const A: number[][] = []
      for (let i = 0; i < N; i++) {
        const sc: number[] = []
        for (let j = 0; j < N; j++) { let s = 0; for (let d = 0; d < DH; d++) s += q[i][h * DH + d] * k[j][h * DH + d]; sc.push(s / Math.sqrt(DH)) }
        const m = Math.max(...sc), e = sc.map((s) => Math.exp(s - m)), z = e.reduce((a, b) => a + b, 0), row = e.map((s) => s / z)
        A.push(row)
        for (let j = 0; j < N; j++) for (let d = 0; d < DH; d++) ctx[i][h * DH + d] += row[j] * v[j][h * DH + d]
      }
      heads.push(A)
    }
    att.push(heads)
    // post-LN: normalise after each residual add
    x = x.map((r, i) => { const o = linear(ctx[i], w(P + 'attention.output.dense.weight'), w(P + 'attention.output.dense.bias'), D); return layerNorm(r.map((val, d) => val + o[d]), w(P + 'attention.output.LayerNorm.weight'), w(P + 'attention.output.LayerNorm.bias')) })
    x = x.map((r) => {
      const hid = linear(r, w(P + 'intermediate.dense.weight'), w(P + 'intermediate.dense.bias'), FF).map(gelu)
      const o = linear(hid, w(P + 'output.dense.weight'), w(P + 'output.dense.bias'), D)
      return layerNorm(r.map((val, d) => val + o[d]), w(P + 'output.LayerNorm.weight'), w(P + 'output.LayerNorm.bias'))
    })
  }
  return { x, att }
}
/** The masked-LM head: transform, then scores against every word embedding (tied) plus a bias. */
function mlm(h: Float32Array, k: number) {
  const t = layerNorm(linear(h, w('cls.predictions.transform.dense.weight'), w('cls.predictions.transform.dense.bias'), D).map(gelu), w('cls.predictions.transform.LayerNorm.weight'), w('cls.predictions.transform.LayerNorm.bias'))
  const we = w('bert.embeddings.word_embeddings.weight'), bias = w('cls.predictions.bias'), V = vocab.length
  const z = new Float32Array(V)
  for (let j = 0; j < V; j++) { let s = bias[j]; for (let d = 0; d < D; d++) s += t[d] * we[j * D + d]; z[j] = s }
  let m = -Infinity
  for (let j = 0; j < V; j++) if (z[j] > m) m = z[j]
  let Z = 0
  for (let j = 0; j < V; j++) Z += Math.exp(z[j] - m)
  return [...z.keys()].sort((a, b) => z[b] - z[a]).slice(0, k).map((j) => ({ t: vocab[j], p: Math.round((Math.exp(z[j] - m) / Z) * 1000) / 1000 }))
}

/* ---------- export ---------- */
const r3 = (v: number) => Math.round(v * 1000) / 1000
/** Sentences where the words after the blank matter. `left` is what a left-to-right model sees (gpt2-export.ts scores it). TRY=… tests others. */
const EXAMPLES = process.env.TRY ? process.env.TRY.split('|').map((text) => ({ text, left: '' })) : [
  { text: 'The [MASK] barked and wagged its tail.', left: 'The' },
  { text: 'I went to the [MASK] to borrow a book.', left: 'I went to the' },
  { text: 'She played the [MASK] in the orchestra.', left: 'She played the' },
]
const out: any = { model: 'BERT-base uncased (google-bert/bert-base-uncased)', examples: [] }
for (const ex of EXAMPLES) {
  const toks = tokenize(ex.text), f = forward(toks), mi = toks.indexOf('[MASK]')
  const top = mlm(f.x[mi], 8)
  out.examples.push({ text: ex.text, left: ex.left, toks, mask: mi, top })
  console.log(ex.text, '→', top.map((t) => `${t.t} ${(t.p * 100).toFixed(1)}%`).join(', '))
}
// the head that most looks one token ahead, on a plain sentence: something GPT-2's causal mask forbids
{
  const text = 'The cat sat on the floor.', toks = tokenize(text), f = forward(toks), N = toks.length
  let best = { l: 0, h: 0, s: -1 }
  for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) {
    let s = 0
    for (let i = 1; i < N - 2; i++) s += f.att[l][h][i][i + 1]
    s /= N - 3
    if (s > best.s) best = { l, h, s }
  }
  out.ahead = { toks, layer: best.l, head: best.h, score: r3(best.s), A: f.att[best.l][best.h].map((row) => row.map(r3)) }
  console.log(`look-ahead head: layer ${best.l + 1} head ${best.h + 1}, mean weight on the next token ${best.s.toFixed(2)}`)
}
// a sentence pair as BERT reads it: [CLS] A [SEP] B [SEP], with segment ids
{
  const a = tokenize('The dog barked at the mailman.'), b = tokenize('It was hungry.').slice(1)
  out.pair = { toks: [...a, ...b], seg: [...a.map(() => 0), ...b.map(() => 1)] }
  out.cls = tokenize('The movie was great!')
  console.log('pair:', out.pair.toks.join(' '))
}
writeFileSync(new URL('../src/data/bert.json', import.meta.url), JSON.stringify(out))
console.log('wrote src/data/bert.json')
