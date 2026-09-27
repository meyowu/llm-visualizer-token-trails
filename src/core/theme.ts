import { clamp, lerp } from './util'

export type RGB = [number, number, number]

export interface Palette {
  bg: RGB
  ink: RGB
  ink2: RGB
  mute: RGB
  faint: RGB
  neg: RGB
  /** Seven spectral token hues; token i uses tok[i % 7]. */
  tok: RGB[]
  dark: boolean
}

/** Canvas palette, read from the CSS tokens in styles.css and refreshed on theme change. */
export const C: Palette = { bg: [0, 0, 0], ink: [0, 0, 0], ink2: [0, 0, 0], mute: [0, 0, 0], faint: [0, 0, 0], neg: [0, 0, 0], tok: [], dark: true }

function hex(h: string): RGB {
  h = h.replace('#', '').trim()
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  const n = parseInt(h, 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

export function readTheme() {
  const cs = getComputedStyle(document.documentElement)
  const g = (n: string) => hex(cs.getPropertyValue(n))
  Object.assign(C, {
    bg: g('--bg'), ink: g('--ink'), ink2: g('--ink2'), mute: g('--mute'), faint: g('--faint'), neg: g('--neg'),
    tok: [0, 1, 2, 3, 4, 5, 6].map((i) => g('--t' + i)),
  })
  C.dark = (C.bg[0] + C.bg[1] + C.bg[2]) / 3 < 128
}

/** Re-read the palette whenever the theme changes; `onChange` lets still frames redraw. */
export function watchTheme(onChange: () => void = () => {}) {
  const update = () => { readTheme(); onChange() }
  update()
  matchMedia('(prefers-color-scheme: light)').addEventListener('change', update)
  new MutationObserver(update).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
}

export const rgba = (c: RGB, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${clamp(a)})`
export const mixc = (a: RGB, b: RGB, t: number): RGB => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
/** A brighter (dark theme) or deeper (light theme) version of a hue, for sparks and highlights. */
export const pop = (c: RGB): RGB => (C.dark ? mixc(c, [255, 255, 255], 0.45) : mixc(c, [0, 0, 0], 0.1))
/** Weighted blend of hues; weights should sum to 1. */
export function blend(cols: RGB[], w: number[]): RGB {
  let r = 0, g = 0, b = 0
  w.forEach((x, k) => { r += x * cols[k][0]; g += x * cols[k][1]; b += x * cols[k][2] })
  return [r, g, b]
}
