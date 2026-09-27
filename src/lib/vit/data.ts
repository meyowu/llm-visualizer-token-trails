/*
 * Two slices of the real ViT-B/16, exported offline by scripts/vit-export.ts: the cosine similarity
 * between its learned position embeddings, and the top principal components of its patch filters.
 */
import raw from '../../data/vit.json'

const R = raw as { model: string; grid: number; posSim: string; filters: string[]; explained: number[] }
const bytes = (b64: string) => { const s = atob(b64), a = new Int8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i) << 24 >> 24; return a }

/** Patch positions per side (14 for 224 / 16). */
export const GRID = R.grid
const sim = bytes(R.posSim)
/** Cosine similarity between the position embeddings of patches i and j (row-major, [CLS] excluded). */
export const posSim = (i: number, j: number) => sim[i * GRID * GRID + j] / 127
/** Principal component c of the patch filters: 3 × 16 × 16 values in [−1, 1], channel-major (R, G, B). */
export const filters = R.filters.map((f) => Float32Array.from(bytes(f), (v) => v / 127))
export const explained = R.explained
