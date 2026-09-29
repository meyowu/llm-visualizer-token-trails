/*
 * Runs real CLIP ViT-B/32 once, offline, on four simple images drawn here (coloured shapes) and a few
 * captions, and exports what the CLIP page shows: the image–text similarity matrix, zero-shot scores,
 * and a slice of each embedding. The browser never runs the model.
 *
 *   node scripts/clip-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs, in ~/.cache/token-trails/clip/: model.safetensors (the safetensors conversion of
 * openai/clip-vit-base-patch32, revision refs/pr/21), vocab.json, merges.txt. Writes src/data/clip.json.
 */
import { readFileSync, writeFileSync, openSync, readSync } from 'node:fs'

const DIR = `${process.env.HOME}/.cache/token-trails/clip`

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
    if (t.dtype !== 'F32') continue // position_ids buffers (I64) are not weights
    const [a, b] = t.data_offsets as [number, number]
    const copy = new Float32Array((b - a) / 4)
    Buffer.from(copy.buffer).set(all.subarray(8 + hlen + a, 8 + hlen + b))
    out.set(name, copy)
  }
  return out
}
const W = loadSafetensors(`${DIR}/model.safetensors`)
const w = (name: string) => { const t = W.get(name); if (!t) throw new Error(name); return t }

/* ---------- CLIP's BPE: lower-cased, byte-level, with an end-of-word marker ---------- */
const vocab: Record<string, number> = JSON.parse(readFileSync(`${DIR}/vocab.json`, 'utf8'))
const ranks = new Map(readFileSync(`${DIR}/merges.txt`, 'utf8').split('\n').slice(1).filter(Boolean).map((l, i) => [l, i]))
const byteChar = (() => {
  const bs: number[] = []
  for (let i = 33; i <= 126; i++) bs.push(i)
  for (let i = 161; i <= 172; i++) bs.push(i)
  for (let i = 174; i <= 255; i++) bs.push(i)
  const cs = [...bs]
  let n = 0
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++) }
  const m = new Map<number, string>()
  bs.forEach((b, i) => m.set(b, String.fromCharCode(cs[i])))
  return m
})()
function bpe(word: string[]): string[] {
  let parts = word
  while (parts.length > 1) {
    let best = -1, bestRank = Infinity
    for (let i = 0; i < parts.length - 1; i++) { const r = ranks.get(parts[i] + ' ' + parts[i + 1]); if (r !== undefined && r < bestRank) { bestRank = r; best = i } }
    if (best < 0) break
    parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)]
  }
  return parts
}
const SOT = vocab['<|startoftext|>'], EOT = vocab['<|endoftext|>']
function tokenize(text: string): { ids: number[]; toks: string[] } {
  const clean = text.toLowerCase().replace(/\s+/g, ' ').trim()
  const toks: string[] = []
  for (const m of clean.matchAll(/'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+/gu)) {
    const chars = [...Buffer.from(m[0], 'utf8')].map((b) => byteChar.get(b)!)
    chars[chars.length - 1] += '</w>'
    toks.push(...bpe(chars))
  }
  return { ids: [SOT, ...toks.map((t) => vocab[t]), EOT], toks }
}

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
/** PyTorch Linear: W stored [out × in]. */
function linear(x: Float32Array, Wt: Float32Array, b: Float32Array | null, out: number): Float32Array {
  const y = b ? Float32Array.from(b) : new Float32Array(out), n = x.length
  for (let j = 0; j < out; j++) { let s = 0; const row = j * n; for (let i = 0; i < n; i++) s += x[i] * Wt[row + i]; y[j] += s }
  return y
}
const quickGelu = (x: number) => x / (1 + Math.exp(-1.702 * x))
/** One pre-LN Transformer layer, as in both CLIP towers. */
function layer(x: Float32Array[], P: string, D: number, H: number, causal: boolean): Float32Array[] {
  const DH = D / H, n = x.map((r) => layerNorm(r, w(P + 'layer_norm1.weight'), w(P + 'layer_norm1.bias')))
  const q = n.map((r) => linear(r, w(P + 'self_attn.q_proj.weight'), w(P + 'self_attn.q_proj.bias'), D))
  const k = n.map((r) => linear(r, w(P + 'self_attn.k_proj.weight'), w(P + 'self_attn.k_proj.bias'), D))
  const v = n.map((r) => linear(r, w(P + 'self_attn.v_proj.weight'), w(P + 'self_attn.v_proj.bias'), D))
  const ctx = x.map(() => new Float32Array(D))
  for (let h = 0; h < H; h++) for (let i = 0; i < x.length; i++) {
    const sc: number[] = []
    for (let j = 0; j < x.length; j++) {
      if (causal && j > i) { sc.push(-Infinity); continue }
      let s = 0
      for (let d = 0; d < DH; d++) s += q[i][h * DH + d] * k[j][h * DH + d]
      sc.push(s / Math.sqrt(DH))
    }
    const m = Math.max(...sc), e = sc.map((s) => Math.exp(s - m)), z = e.reduce((a, b) => a + b, 0)
    for (let j = 0; j < x.length; j++) { const p = e[j] / z; if (p) for (let d = 0; d < DH; d++) ctx[i][h * DH + d] += p * v[j][h * DH + d] }
  }
  x = x.map((r, i) => { const o = linear(ctx[i], w(P + 'self_attn.out_proj.weight'), w(P + 'self_attn.out_proj.bias'), D); return r.map((val, d) => val + o[d]) })
  return x.map((r) => {
    const m = layerNorm(r, w(P + 'layer_norm2.weight'), w(P + 'layer_norm2.bias'))
    const hid = linear(m, w(P + 'mlp.fc1.weight'), w(P + 'mlp.fc1.bias'), 4 * D).map(quickGelu)
    const o = linear(hid, w(P + 'mlp.fc2.weight'), w(P + 'mlp.fc2.bias'), D)
    return r.map((val, d) => val + o[d])
  })
}
const unit = (v: Float32Array) => { const n = Math.hypot(...v); return v.map((x) => x / n) }

/** Text tower: 12 causal layers of width 512; the embedding is read at the end-of-text token. */
function encodeText(text: string) {
  const { ids, toks } = tokenize(text), te = w('text_model.embeddings.token_embedding.weight'), pe = w('text_model.embeddings.position_embedding.weight'), D = 512
  let x = ids.map((id, p) => { const v = new Float32Array(D); for (let k = 0; k < D; k++) v[k] = te[id * D + k] + pe[p * D + k]; return v })
  for (let l = 0; l < 12; l++) x = layer(x, `text_model.encoder.layers.${l}.`, D, 8, true)
  const last = layerNorm(x[ids.indexOf(EOT)], w('text_model.final_layer_norm.weight'), w('text_model.final_layer_norm.bias'))
  return { toks, emb: unit(linear(last, w('text_projection.weight'), null, 512)) }
}

/* ---------- images: coloured shapes, drawn with 4 × 4 supersampling ---------- */
const S = 224
type Shape = (x: number, y: number) => boolean
const star = (x: number, y: number) => {
  const a = Math.atan2(y, x) + Math.PI / 2, r = Math.hypot(x, y), k = ((a % (2 * Math.PI / 5)) + 2 * Math.PI / 5) % (2 * Math.PI / 5)
  const t = Math.abs(k - Math.PI / 5) / (Math.PI / 5)
  return r < 34 + (82 - 34) * t ** 1.6
}
/** The classic implicit heart, (u² + v² − 1)³ < u² v³, about 140 px across (image y points down). */
const heart = (x: number, y: number) => {
  const u = x / 62, v = -y / 62 + 0.125, q = u * u + v * v - 1
  return q * q * q < u * u * v * v * v
}
// no red disc: on white it reads as a national flag
const SHAPES: Record<string, Shape> = {
  heart,
  square: (x, y) => Math.abs(x) < 64 && Math.abs(y) < 64,
  triangle: (x, y) => y < 62 && y > -80 + Math.abs(x) * 1.75,
  star,
}
const COLOURS: Record<string, [number, number, number]> = { red: [220, 40, 40], blue: [40, 80, 220], green: [40, 170, 60], yellow: [245, 200, 30] }
function draw(shape: string, colour: string): Uint8ClampedArray {
  const img = new Uint8ClampedArray(S * S * 3).fill(255), f = SHAPES[shape], c = COLOURS[colour]
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let cover = 0
    for (let sy = 0; sy < 4; sy++) for (let sx = 0; sx < 4; sx++) if (f(x + (sx + 0.5) / 4 - S / 2, y + (sy + 0.5) / 4 - S / 2)) cover++
    const a = cover / 16
    for (let ch = 0; ch < 3; ch++) img[(y * S + x) * 3 + ch] = Math.round(255 * (1 - a) + c[ch] * a)
  }
  return img
}
/** Image tower: 7 × 7 patches of 32 × 32, a class token, 12 layers of width 768, then a projection to 512. */
function encodeImage(img: Uint8ClampedArray) {
  const mean = [0.48145466, 0.4578275, 0.40821073], std = [0.26862954, 0.26130258, 0.27577711], D = 768, PS = 32, G = S / PS
  const px = (ch: number, y: number, x: number) => (img[(y * S + x) * 3 + ch] / 255 - mean[ch]) / std[ch]
  const pw = w('vision_model.embeddings.patch_embedding.weight'), pos = w('vision_model.embeddings.position_embedding.weight')
  const tokens = [Float32Array.from(w('vision_model.embeddings.class_embedding'))]
  for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
    const v = new Float32Array(D)
    for (let o = 0; o < D; o++) {
      let s = 0
      for (let ch = 0; ch < 3; ch++) for (let y = 0; y < PS; y++) for (let x = 0; x < PS; x++) s += pw[((o * 3 + ch) * PS + y) * PS + x] * px(ch, gy * PS + y, gx * PS + x)
      v[o] = s
    }
    tokens.push(v)
  }
  let x = tokens.map((t, p) => layerNorm(t.map((v, d) => v + pos[p * D + d]), w('vision_model.pre_layrnorm.weight'), w('vision_model.pre_layrnorm.bias')))
  for (let l = 0; l < 12; l++) x = layer(x, `vision_model.encoder.layers.${l}.`, D, 12, false)
  const cls = layerNorm(x[0], w('vision_model.post_layernorm.weight'), w('vision_model.post_layernorm.bias'))
  return unit(linear(cls, w('visual_projection.weight'), null, 512))
}

/* ---------- export ---------- */
const r3 = (v: number) => Math.round(v * 1000) / 1000
const PAIRS: [string, string][] = [['red', 'heart'], ['blue', 'square'], ['green', 'triangle'], ['yellow', 'star']]
const scale = Math.exp(w('logit_scale')[0])
const images = PAIRS.map(([c, s]) => { const img = draw(s, c); console.log(`image: ${c} ${s}`); return { c, s, img, emb: encodeImage(img) } })
const caption = (texts: string[]) => texts.map((t) => ({ t, ...encodeText(t) }))
const captions = caption(PAIRS.map(([c, s]) => `a ${c} ${s}`))
const shapes = caption(Object.keys(SHAPES).map((s) => `a photo of a ${s}`))
const colours = caption(Object.keys(COLOURS).map((c) => `a photo of something ${c}`))
const cos = (a: Float32Array, b: Float32Array) => a.reduce((s, v, i) => s + v * b[i], 0)
const matrix = (texts: { emb: Float32Array }[]) => images.map((im) => texts.map((t) => r3(cos(im.emb, t.emb))))
/** A small preview of each image (56 × 56 RGB, 4 × 4 pixels averaged), for the page to draw. */
const preview = (img: Uint8ClampedArray) => {
  const P = 56, out = new Uint8Array(P * P * 3), f = S / P
  for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) for (let ch = 0; ch < 3; ch++) {
    let s = 0
    for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) s += img[((y * f + dy) * S + x * f + dx) * 3 + ch]
    out[(y * P + x) * 3 + ch] = Math.round(s / (f * f))
  }
  return Buffer.from(out).toString('base64')
}
/** The 8 embeddings (4 images, then 4 captions) on their top two principal components. */
const pca = (() => {
  const X = [...images.map((im) => im.emb), ...captions.map((c) => c.emb)], D = 512
  const mean = new Float64Array(D)
  for (const x of X) for (let k = 0; k < D; k++) mean[k] += x[k] / X.length
  const C = X.map((x) => Float64Array.from(x, (v, k) => v - mean[k]))
  const comps: Float64Array[] = []
  for (let c = 0; c < 2; c++) {
    let v = Float64Array.from({ length: D }, (_, k) => Math.sin(k * 1.3 + c))
    for (let it = 0; it < 200; it++) {
      const u = new Float64Array(D)
      for (const x of C) { const d = x.reduce((s, xv, k) => s + xv * v[k], 0); for (let k = 0; k < D; k++) u[k] += d * x[k] }
      for (const q of comps) { const d = u.reduce((s, uv, k) => s + uv * q[k], 0); for (let k = 0; k < D; k++) u[k] -= d * q[k] }
      const n = Math.hypot(...u); v = u.map((x) => x / n)
    }
    comps.push(v)
  }
  return C.map((x) => comps.map((q) => r3(x.reduce((s, xv, k) => s + xv * q[k], 0))))
})()
const out = {
  model: 'CLIP ViT-B/32 (openai/clip-vit-base-patch32)',
  pca,
  scale: r3(scale),
  images: images.map((im) => ({ label: `${im.c} ${im.s}`, px: preview(im.img), head: Array.from(im.emb.slice(0, 8), r3) })),
  captions: captions.map((c) => ({ text: c.t, toks: c.toks, head: Array.from(c.emb.slice(0, 8), r3) })),
  sim: matrix(captions),
  shapes: { texts: shapes.map((c) => c.t), sim: matrix(shapes) },
  colours: { texts: colours.map((c) => c.t), sim: matrix(colours) },
}
for (const [name, m] of [['captions', out.sim], ['shapes', out.shapes.sim], ['colours', out.colours.sim]] as const) {
  console.log(name)
  m.forEach((row, i) => { const e = row.map((v) => Math.exp(v * scale)), z = e.reduce((a, b) => a + b, 0); console.log(`  ${out.images[i].label.padEnd(16)} ${row.map((v) => v.toFixed(3)).join(' ')}   softmax ${e.map((v) => (v / z).toFixed(2)).join(' ')}`) })
}
writeFileSync(new URL('../src/data/clip.json', import.meta.url), JSON.stringify(out))
console.log('wrote src/data/clip.json', JSON.stringify(out).length, 'bytes; logit scale', scale.toFixed(1))
