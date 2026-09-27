/*
 * Reads two tensors of the real ViT-B/16 (google/vit-base-patch16-224) and exports what the ViT page
 * shows: how similar each learned position embedding is to every other, and the top principal
 * components of the patch-embedding filters (both as in Dosovitskiy et al. 2020, figure 7).
 *
 *   node scripts/vit-export.ts             (Node 23+, which runs TypeScript directly)
 *
 * Fetches only those tensors, with HTTP range requests, and caches them in ~/.cache/token-trails/vit/.
 * Writes src/data/vit.json.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'

const URL_ = 'https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.safetensors'
const DIR = `${process.env.HOME}/.cache/token-trails/vit`
mkdirSync(DIR, { recursive: true })

async function range(a: number, b: number): Promise<Buffer> {
  const r = await fetch(URL_, { headers: { Range: `bytes=${a}-${b}` } })
  if (r.status !== 206) throw new Error(`range ${a}-${b}: HTTP ${r.status}`)
  return Buffer.from(await r.arrayBuffer())
}
async function tensor(name: string): Promise<{ shape: number[]; data: Float32Array }> {
  const file = `${DIR}/${name}.f32`, meta = `${DIR}/${name}.json`
  if (existsSync(file)) return { shape: JSON.parse(readFileSync(meta, 'utf8')), data: new Float32Array(readFileSync(file).buffer.slice(0)) }
  const hlen = Number((await range(0, 7)).readBigUInt64LE(0))
  const header = JSON.parse((await range(8, 8 + hlen - 1)).toString('utf8'))
  const t = header[name]
  if (!t || t.dtype !== 'F32') throw new Error(`${name}: ${t?.dtype}`)
  const [a, b] = t.data_offsets as [number, number]
  const buf = await range(8 + hlen + a, 8 + hlen + b - 1)
  writeFileSync(file, buf); writeFileSync(meta, JSON.stringify(t.shape))
  const data = new Float32Array(buf.byteLength / 4)
  Buffer.from(data.buffer).set(buf)
  return { shape: t.shape, data }
}
/** Values in [−1, 1] as signed bytes (×127), base64. */
const i8 = (v: ArrayLike<number>) => Buffer.from(Int8Array.from(v, (x) => Math.max(-127, Math.min(127, Math.round(x * 127)))).buffer).toString('base64')

const pos = await tensor('vit.embeddings.position_embeddings') // [1, 197, 768]
const D = pos.shape[2], P = pos.shape[1] - 1, G = Math.round(Math.sqrt(P))
// cosine similarity between the 196 patch positions ([CLS] at index 0 left out)
const rows = Array.from({ length: P }, (_, i) => pos.data.subarray((i + 1) * D, (i + 2) * D))
const norms = rows.map((r) => Math.hypot(...r))
const sim = new Float32Array(P * P)
for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) {
  let s = 0
  for (let k = 0; k < D; k++) s += rows[i][k] * rows[j][k]
  sim[i * P + j] = s / (norms[i] * norms[j])
}
console.log(`positions: ${P} (${G} × ${G}), similarity of (0,0) to its right neighbour ${sim[1].toFixed(2)}, to the one below ${sim[G].toFixed(2)}, to the far corner ${sim[P - 1].toFixed(2)}`)

// the patch-embedding filters: 768 of them, each 3 × 16 × 16; their top principal components
const wp = await tensor('vit.embeddings.patch_embeddings.projection.weight') // [768, 3, 16, 16]
const F = wp.shape[0], K = wp.shape[1] * wp.shape[2] * wp.shape[3], NC = 28
const mean = new Float64Array(K)
for (let f = 0; f < F; f++) for (let k = 0; k < K; k++) mean[k] += wp.data[f * K + k] / F
const X = Array.from({ length: F }, (_, f) => Float64Array.from({ length: K }, (_, k) => wp.data[f * K + k] - mean[k]))
const cov = Array.from({ length: K }, () => new Float64Array(K))
for (const x of X) for (let a = 0; a < K; a++) { const xa = x[a]; if (!xa) continue; const row = cov[a]; for (let b = 0; b < K; b++) row[b] += xa * x[b] }
const comps: Float64Array[] = [], vars: number[] = []
for (let c = 0; c < NC; c++) {
  // power iteration, deflated against the components already found
  let v = Float64Array.from({ length: K }, (_, k) => Math.sin(k * 12.9898 + c * 78.233))
  let lambda = 0
  for (let it = 0; it < 300; it++) {
    const u = new Float64Array(K)
    for (let a = 0; a < K; a++) { let s = 0; const row = cov[a]; for (let b = 0; b < K; b++) s += row[b] * v[b]; u[a] = s }
    for (const q of comps) { let d = 0; for (let k = 0; k < K; k++) d += u[k] * q[k]; for (let k = 0; k < K; k++) u[k] -= d * q[k] }
    lambda = Math.hypot(...u)
    v = u.map((x) => x / lambda)
  }
  comps.push(v); vars.push(lambda)
}
const total = cov.reduce((s, row, a) => s + row[a], 0)
console.log(`filters: ${F} × ${K}; top ${NC} components explain ${((vars.reduce((a, b) => a + b, 0) / total) * 100).toFixed(1)}% of the variance`)

const out = {
  model: 'ViT-B/16 (google/vit-base-patch16-224)',
  grid: G,
  /** P × P cosine similarities, row-major, as signed bytes / 127. */
  posSim: i8(sim),
  /** Top principal components of the patch filters, each 3 × 16 × 16 (channel-major), scaled to [−1, 1]. */
  filters: comps.map((v) => { const m = Math.max(...v.map(Math.abs)); return i8(v.map((x) => x / m)) }),
  /** Share of the filters' variance each component explains. */
  explained: vars.map((l) => Math.round((l / total) * 10000) / 10000),
}
writeFileSync(new URL('../src/data/vit.json', import.meta.url), JSON.stringify(out))
console.log('wrote src/data/vit.json', JSON.stringify(out).length, 'bytes')
