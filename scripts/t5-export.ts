/*
 * Runs real T5-small once, offline, and exports what the T5 page shows: text-to-text answers to a few
 * task prefixes, a span-corruption fill, and the learned relative-position biases.
 * The browser never runs the model.
 *
 *   node scripts/t5-export.ts              (Node 23+, which runs TypeScript directly)
 *
 * Needs, in ~/.cache/token-trails/t5/: model.safetensors and tokenizer.json
 * from https://huggingface.co/google-t5/t5-small. Writes src/data/t5.json.
 */
import { readFileSync, writeFileSync, openSync, readSync } from 'node:fs'

const DIR = `${process.env.HOME}/.cache/token-trails/t5`
const D = 512, H = 8, DK = 64, L = 6, FF = 2048, BUCKETS = 32, MAX_DIST = 128

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
    const [a, b] = t.data_offsets as [number, number]
    const copy = new Float32Array((b - a) / 4)
    Buffer.from(copy.buffer).set(all.subarray(8 + hlen + a, 8 + hlen + b))
    out.set(name, copy)
  }
  return out
}
const W = loadSafetensors(`${DIR}/model.safetensors`)
const w = (name: string) => { const t = W.get(name); if (!t) throw new Error(name); return t }

/* ---------- SentencePiece unigram tokenizer ---------- */
const tok = JSON.parse(readFileSync(`${DIR}/tokenizer.json`, 'utf8'))
const pieces: [string, number][] = tok.model.vocab
const pieceId = new Map(pieces.map(([p], i) => [p, i]))
for (const a of tok.added_tokens) pieceId.set(a.content, a.id)
const idPiece = new Map([...pieceId].map(([p, i]) => [i, p]))
const UNK = 2, EOS = 1, PAD = 0
/** Best segmentation of one ▁-prefixed word by total piece score (Viterbi). */
function viterbi(word: string): number[] {
  const n = word.length, best = Array(n + 1).fill(-Infinity), back: [number, number][] = Array(n + 1)
  best[0] = 0
  for (let i = 0; i < n; i++) {
    if (best[i] === -Infinity) continue
    for (let j = i + 1; j <= Math.min(n, i + 24); j++) {
      const id = pieceId.get(word.slice(i, j))
      const s = id !== undefined && id >= 3 && id < pieces.length ? pieces[id][1] : j === i + 1 ? -100 : -Infinity
      if (best[i] + s > best[j]) { best[j] = best[i] + s; back[j] = [i, id !== undefined && id >= 3 && id < pieces.length ? id : UNK] }
    }
  }
  const ids: number[] = []
  for (let j = n; j > 0; j = back[j][0]) ids.unshift(back[j][1])
  return ids
}
function encode(text: string): number[] {
  const ids: number[] = []
  for (const part of text.split(/(<extra_id_\d+>)/)) {
    if (/^<extra_id_\d+>$/.test(part)) { ids.push(pieceId.get(part)!); continue }
    for (const word of part.split(/\s+/).filter(Boolean)) ids.push(...viterbi('▁' + word))
  }
  return [...ids, EOS]
}
const piece = (id: number) => idPiece.get(id) ?? '<unk>'
const decode = (ids: number[]) => ids.filter((i) => i > 2).map(piece).join('').replace(/▁/g, ' ').replace(/ ?(<extra_id_\d+>) ?/g, ' $1 ').trim()

/* ---------- math ---------- */
/** T5's LayerNorm: scale only, no mean and no bias (RMSNorm). */
function rmsNorm(x: Float32Array, g: Float32Array): Float32Array {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i] * x[i]
  const r = 1 / Math.sqrt(s / x.length + 1e-6), out = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = x[i] * r * g[i]
  return out
}
/** y = W x with W stored [out × in]; T5 has no biases. */
function linear(x: Float32Array, Wt: Float32Array, out: number): Float32Array {
  const y = new Float32Array(out), n = x.length
  for (let j = 0; j < out; j++) { let s = 0; const row = j * n; for (let i = 0; i < n; i++) s += x[i] * Wt[row + i]; y[j] = s }
  return y
}
/** T5's relative-position bucket (HF _relative_position_bucket): rel = key position − query position. */
function bucket(rel: number, bidirectional: boolean) {
  let nb = BUCKETS, ret = 0, n = rel
  if (bidirectional) { nb /= 2; if (n > 0) ret += nb; n = Math.abs(n) } else n = Math.max(-n, 0)
  const exact = nb / 2
  if (n < exact) return ret + n
  return ret + Math.min(nb - 1, exact + Math.trunc((Math.log(n / exact) / Math.log(MAX_DIST / exact)) * (nb - exact)))
}
function attend(q: Float32Array[], k: Float32Array[], v: Float32Array[], bias: ((h: number, i: number, j: number) => number) | null, causal: boolean) {
  const out = q.map(() => new Float32Array(H * DK))
  for (let h = 0; h < H; h++) for (let i = 0; i < q.length; i++) {
    const sc: number[] = []
    for (let j = 0; j < k.length; j++) {
      if (causal && j > i) { sc.push(-Infinity); continue }
      let s = 0
      for (let d = 0; d < DK; d++) s += q[i][h * DK + d] * k[j][h * DK + d]
      sc.push(s + (bias ? bias(h, i, j) : 0)) // no ÷√d: T5 folds it into the initialisation
    }
    const m = Math.max(...sc), e = sc.map((s) => Math.exp(s - m)), z = e.reduce((a, b) => a + b, 0)
    for (let j = 0; j < k.length; j++) { const p = e[j] / z; if (p) for (let d = 0; d < DK; d++) out[i][h * DK + d] += p * v[j][h * DK + d] }
  }
  return out
}
const relBias = (table: Float32Array, bidirectional: boolean) => (h: number, i: number, j: number) => table[bucket(j - i, bidirectional) * H + h]

/* ---------- forward ---------- */
const emb = w('shared.weight')
const embed = (ids: number[]) => ids.map((id) => emb.slice(id * D, id * D + D))
function encoder(ids: number[]) {
  let x = embed(ids)
  const bias = relBias(w('encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'), true)
  for (let l = 0; l < L; l++) {
    const P = `encoder.block.${l}.layer.`
    const n1 = x.map((r) => rmsNorm(r, w(P + '0.layer_norm.weight')))
    const a = attend(n1.map((r) => linear(r, w(P + '0.SelfAttention.q.weight'), D)), n1.map((r) => linear(r, w(P + '0.SelfAttention.k.weight'), D)), n1.map((r) => linear(r, w(P + '0.SelfAttention.v.weight'), D)), bias, false)
    x = x.map((r, i) => { const o = linear(a[i], w(P + '0.SelfAttention.o.weight'), D); return r.map((v, d) => v + o[d]) })
    x = x.map((r) => { const n = rmsNorm(r, w(P + '1.layer_norm.weight')), hid = linear(n, w(P + '1.DenseReluDense.wi.weight'), FF).map((v) => Math.max(0, v)), o = linear(hid, w(P + '1.DenseReluDense.wo.weight'), D); return r.map((v, d) => v + o[d]) })
  }
  return x.map((r) => rmsNorm(r, w('encoder.final_layer_norm.weight')))
}
function decoderLogits(mem: Float32Array[], ids: number[]) {
  let y = embed(ids)
  const bias = relBias(w('decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'), false)
  for (let l = 0; l < L; l++) {
    const P = `decoder.block.${l}.layer.`
    const n1 = y.map((r) => rmsNorm(r, w(P + '0.layer_norm.weight')))
    const a = attend(n1.map((r) => linear(r, w(P + '0.SelfAttention.q.weight'), D)), n1.map((r) => linear(r, w(P + '0.SelfAttention.k.weight'), D)), n1.map((r) => linear(r, w(P + '0.SelfAttention.v.weight'), D)), bias, true)
    y = y.map((r, i) => { const o = linear(a[i], w(P + '0.SelfAttention.o.weight'), D); return r.map((v, d) => v + o[d]) })
    // cross-attention: queries from the decoder, keys and values from the encoder output, no position bias
    const n2 = y.map((r) => rmsNorm(r, w(P + '1.layer_norm.weight')))
    const c = attend(n2.map((r) => linear(r, w(P + '1.EncDecAttention.q.weight'), D)), mem.map((r) => linear(r, w(P + '1.EncDecAttention.k.weight'), D)), mem.map((r) => linear(r, w(P + '1.EncDecAttention.v.weight'), D)), null, false)
    y = y.map((r, i) => { const o = linear(c[i], w(P + '1.EncDecAttention.o.weight'), D); return r.map((v, d) => v + o[d]) })
    y = y.map((r) => { const n = rmsNorm(r, w(P + '2.layer_norm.weight')), hid = linear(n, w(P + '2.DenseReluDense.wi.weight'), FF).map((v) => Math.max(0, v)), o = linear(hid, w(P + '2.DenseReluDense.wo.weight'), D); return r.map((v, d) => v + o[d]) })
  }
  // the output matrix is the embedding, tied, with the stream scaled by d_model^−½ first
  const last = rmsNorm(y[y.length - 1], w('decoder.final_layer_norm.weight')).map((v) => v * D ** -0.5)
  const V = emb.length / D, z = new Float32Array(V)
  for (let t = 0; t < V; t++) { let s = 0; for (let d = 0; d < D; d++) s += last[d] * emb[t * D + d]; z[t] = s }
  return z
}
function generate(text: string, max = 40) {
  const src = encode(text), mem = encoder(src), out = [PAD]
  for (let step = 0; step < max; step++) {
    const z = decoderLogits(mem, out)
    let best = 0
    for (let t = 1; t < z.length; t++) if (z[t] > z[best]) best = t
    out.push(best)
    if (best === EOS) break
  }
  return { src: src.map(piece), out: out.slice(1).map(piece), text: decode(out) }
}

/* ---------- export ---------- */
/** Task prefixes T5 was trained on; TRY=a|b tests others. */
const TASKS = process.env.TRY ? process.env.TRY.split('|') : [
  'translate English to German: I have seen the cat.',
  'translate English to French: The weather is nice today.',
  'summarize: The Transformer is a neural network that reads a whole sentence at once. It was introduced in 2017 for translation and now powers most language models, from BERT and T5 to GPT.',
  'question: When was the Transformer introduced? context: The Transformer was introduced in 2017 for translation and now powers most language models.',
  'sst2 sentence: This movie was a waste of two hours.',
  'stsb sentence1: A man is playing a guitar. sentence2: A person plays the guitar.',
]
const out: any = { model: 'T5-small (google-t5/t5-small)', tasks: [] }
for (const t of TASKS) {
  const r = generate(t)
  out.tasks.push({ input: t, src: r.src, out: r.out, text: r.text })
  console.log(`${t}\n  → ${r.text}`)
}
{
  // the example from the T5 docs: sentinels mark the dropped spans, the decoder writes them out
  const text = 'The <extra_id_0> walks in <extra_id_1> park', r = generate(text)
  out.spans = { input: text, src: r.src, out: r.out, text: r.text }
  console.log(`${text}\n  → ${r.text}`)
}
const r3 = (v: number) => Math.round(v * 1000) / 1000
out.bias = {
  encoder: Array.from({ length: BUCKETS }, (_, b) => Array.from(w('encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight').slice(b * H, b * H + H), r3)),
  decoder: Array.from({ length: BUCKETS }, (_, b) => Array.from(w('decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight').slice(b * H, b * H + H), r3)),
}
writeFileSync(new URL('../src/data/t5.json', import.meta.url), JSON.stringify(out))
console.log('wrote src/data/t5.json')
