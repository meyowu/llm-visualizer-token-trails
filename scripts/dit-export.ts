/*
 * Reads a few tensors of the real DiT-XL/2 (facebook/DiT-XL-2-256) and exports what the DiT page shows:
 * the adaLN-Zero modulation (shift, scale and gate for attention and the MLP) that the timestep and the
 * class produce in three of its 28 blocks. The browser never runs the model.
 *
 *   node scripts/dit-export.ts             (Node 23+, which runs TypeScript directly)
 *
 * Fetches only those tensors (and single rows of the class table) with HTTP range requests from the
 * safetensors conversion (revision refs/pr/1), cached in ~/.cache/token-trails/dit/. Writes src/data/dit.json.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'

const URL_ = 'https://huggingface.co/facebook/DiT-XL-2-256/resolve/refs%2Fpr%2F1/transformer/diffusion_pytorch_model.safetensors'
const DIR = `${process.env.HOME}/.cache/token-trails/dit`
mkdirSync(DIR, { recursive: true })
const D = 1152, BLOCKS = [0, 13, 27], CLASSES = [207, 1000] // golden retriever, and the "no class" label used for guidance

async function range(a: number, b: number): Promise<Buffer> {
  const r = await fetch(URL_, { headers: { Range: `bytes=${a}-${b}` } })
  if (r.status !== 206) throw new Error(`range ${a}-${b}: HTTP ${r.status}`)
  return Buffer.from(await r.arrayBuffer())
}
let header: any = null, hlen = 0
async function meta(name: string) {
  if (!header) { hlen = Number((await range(0, 7)).readBigUInt64LE(0)); header = JSON.parse((await range(8, 8 + hlen - 1)).toString('utf8')) }
  const t = header[name]
  if (!t || t.dtype !== 'F32') throw new Error(`${name}: ${t?.dtype}`)
  return t as { shape: number[]; data_offsets: [number, number] }
}
/** A whole tensor, or rows [r0, r1) of a 2-D one; cached on disk. */
async function tensor(name: string, rows?: [number, number]): Promise<Float32Array> {
  const file = `${DIR}/${name}${rows ? `.${rows[0]}-${rows[1]}` : ''}.f32`
  if (existsSync(file)) return new Float32Array(readFileSync(file).buffer.slice(0))
  const t = await meta(name), [a, b] = t.data_offsets, rowBytes = rows ? (t.shape.slice(1).reduce((x, y) => x * y, 1) * 4) : 0
  const s = rows ? a + rows[0] * rowBytes : a, e = rows ? a + rows[1] * rowBytes : b
  const buf = await range(8 + hlen + s, 8 + hlen + e - 1)
  writeFileSync(file, buf)
  const out = new Float32Array(buf.byteLength / 4)
  Buffer.from(out.buffer).set(buf)
  return out
}
/** y = W x + b, W stored [out × in]. */
function linear(x: Float32Array, W: Float32Array, b: Float32Array, out: number) {
  const y = Float32Array.from(b), n = x.length
  for (let j = 0; j < out; j++) { let s = 0; const row = j * n; for (let i = 0; i < n; i++) s += x[i] * W[row + i]; y[j] += s }
  return y
}
const silu = (v: Float32Array) => v.map((x) => x / (1 + Math.exp(-x)))
/** diffusers' timestep embedding: 256 sinusoids, cosine half first (flip_sin_to_cos, downscale_freq_shift = 1). */
function tEmb(t: number) {
  const half = 128, out = new Float32Array(256)
  for (let i = 0; i < half; i++) { const f = Math.exp((-Math.log(10000) * i) / (half - 1)); out[i] = Math.cos(t * f); out[half + i] = Math.sin(t * f) }
  return out
}

const TS = Array.from({ length: 41 }, (_, i) => Math.min(999, i * 25))
const NAMES = ['shift_msa', 'scale_msa', 'gate_msa', 'shift_mlp', 'scale_mlp', 'gate_mlp']
const mods: Record<string, Float32Array[][]> = {} // `${block}/${class}` → [t][chunk] vectors
for (const b of BLOCKS) {
  const P = `transformer_blocks.${b}.norm1.`
  const l1w = await tensor(P + 'emb.timestep_embedder.linear_1.weight'), l1b = await tensor(P + 'emb.timestep_embedder.linear_1.bias')
  const l2w = await tensor(P + 'emb.timestep_embedder.linear_2.weight'), l2b = await tensor(P + 'emb.timestep_embedder.linear_2.bias')
  const lw = await tensor(P + 'linear.weight'), lb = await tensor(P + 'linear.bias')
  for (const y of CLASSES) {
    const cls = await tensor(P + 'emb.class_embedder.embedding_table.weight', [y, y + 1])
    mods[`${b}/${y}`] = TS.map((t) => {
      const te = linear(silu(linear(tEmb(t), l1w, l1b, D)), l2w, l2b, D)
      const c = te.map((v, i) => v + cls[i])
      const m = linear(silu(c), lw, lb, 6 * D)
      return NAMES.map((_, k) => m.slice(k * D, (k + 1) * D))
    })
  }
  console.log(`block ${b + 1}: done`)
}
const mean = (v: Float32Array) => v.reduce((s, x) => s + x, 0) / v.length
const meanAbs = (v: Float32Array) => v.reduce((s, x) => s + Math.abs(x), 0) / v.length
for (const b of BLOCKS) for (const y of CLASSES) {
  const m = mods[`${b}/${y}`]
  console.log(`block ${b + 1} class ${y}`)
  for (const k of [1, 2, 4, 5]) console.log(`  ${NAMES[k].padEnd(10)} ${[0, 10, 20, 30, 40].map((i) => `t${TS[i]} μ ${mean(m[i][k]).toFixed(3)} |·| ${meanAbs(m[i][k]).toFixed(3)}`).join('  ')}`)
}
const r3 = (v: number) => Math.round(v * 1000) / 1000
/** Values scaled to [−1, 1] by `s`, as signed bytes, base64. */
const i8 = (v: ArrayLike<number>, s: number) => Buffer.from(Int8Array.from(v, (x) => Math.max(-127, Math.min(127, Math.round((x / s) * 127)))).buffer).toString('base64')
const SHOW = 48
const out = {
  model: 'DiT-XL/2 (facebook/DiT-XL-2-256)',
  blocks: BLOCKS, classes: CLASSES, ts: TS, names: NAMES,
  /** Per block and class: mean |value| of each of the 6 modulation vectors at each timestep. */
  meanAbs: Object.fromEntries(Object.entries(mods).map(([key, m]) => [key, NAMES.map((_, k) => m.map((row) => r3(meanAbs(row[k]))))])),
  /** Block 14 (index 13), golden retriever: the first 48 numbers of each vector at each timestep, int8 / 127 × range. */
  heat: (() => {
    const m = mods[`13/207`], range_ = Math.max(...m.flatMap((row) => row.flatMap((v) => Array.from(v.slice(0, SHOW), Math.abs))))
    return { block: 13, cls: 207, n: SHOW, range: r3(range_), data: NAMES.map((_, k) => i8(m.flatMap((row) => Array.from(row[k].slice(0, SHOW))), range_)) }
  })(),
}
writeFileSync(new URL('../src/data/dit.json', import.meta.url), JSON.stringify(out))
console.log('wrote src/data/dit.json', JSON.stringify(out).length, 'bytes')
