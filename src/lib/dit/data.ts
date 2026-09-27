/*
 * The adaLN-Zero modulation of the real DiT-XL/2, computed offline by scripts/dit-export.ts from its
 * timestep and class embedders: shift, scale and gate for attention and the MLP, in blocks 1, 14 and 28.
 */
import raw from '../../data/dit.json'

const R = raw as {
  model: string; blocks: number[]; classes: number[]; ts: number[]; names: string[]
  meanAbs: Record<string, number[][]>
  heat: { block: number; cls: number; n: number; range: number; data: string[] }
}
export const dit = R
/** Mean |value| of modulation vector `k` (0–5, see names) for a block and class, at each timestep in ts. */
export const meanAbs = (block: number, cls: number, k: number) => R.meanAbs[`${block}/${cls}`][k]
const bytes = (b64: string) => { const s = atob(b64), a = new Int8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i) << 24 >> 24; return a }
const heat = R.heat.data.map(bytes)
/** Block 14, golden retriever: number d (of the first n) of vector k at timestep index ti. */
export const heatAt = (k: number, ti: number, d: number) => (heat[k][ti * R.heat.n + d] / 127) * R.heat.range

/** DDPM's linear schedule, as DiT uses: β from 0.0001 to 0.02 over 1,000 steps; ᾱ_t = Π (1 − β). */
export const ALPHA_BAR = (() => { const a: number[] = []; let p = 1; for (let t = 0; t < 1000; t++) { p *= 1 - (1e-4 + ((0.02 - 1e-4) * t) / 999); a.push(p) } return a })()
