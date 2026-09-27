/*
 * Real CLIP ViT-B/32 outputs for four drawn images and a few captions, exported offline by
 * scripts/clip-export.ts. The browser never runs the model.
 */
import raw from '../../data/clip.json'

interface Raw {
  model: string
  /** The learned logit scale (the inverse temperature), exp(logit_scale). */
  scale: number
  /** Top two principal components of the 8 embeddings: 4 images, then 4 captions. */
  pca: [number, number][]
  images: { label: string; px: string; head: number[] }[]
  captions: { text: string; toks: string[]; head: number[] }[]
  /** Cosine similarity, image i × caption j. */
  sim: number[][]
  shapes: { texts: string[]; sim: number[][] }
  colours: { texts: string[]; sim: number[][] }
}
export const clip = raw as unknown as Raw

/** An image's 56 × 56 RGB preview as a small canvas, drawn once. */
export function preview(i: number): HTMLCanvasElement {
  const s = atob(clip.images[i].px), cv = document.createElement('canvas')
  cv.width = 56; cv.height = 56
  const c = cv.getContext('2d')!, img = c.createImageData(56, 56)
  for (let p = 0; p < 56 * 56; p++) { for (let ch = 0; ch < 3; ch++) img.data[p * 4 + ch] = s.charCodeAt(p * 3 + ch); img.data[p * 4 + 3] = 255 }
  c.putImageData(img, 0, 0)
  return cv
}
export const softmax = (z: number[]) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s) }
