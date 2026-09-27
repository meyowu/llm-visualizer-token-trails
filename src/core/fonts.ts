/*
 * The site's three families ship with it as woff2 files (no Google Fonts request, which is slow
 * or blocked in some regions). Only the weights, styles and subsets the site uses are
 * registered; Latin Extended holds GPT-2's Ġ and Ċ, Greek holds the math letters.
 */
import f0 from '@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2?url'
import f1 from '@fontsource/jetbrains-mono/files/jetbrains-mono-latin-500-normal.woff2?url'
import f2 from '@fontsource/jetbrains-mono/files/jetbrains-mono-latin-ext-400-normal.woff2?url'
import f3 from '@fontsource/jetbrains-mono/files/jetbrains-mono-latin-ext-500-normal.woff2?url'
import f4 from '@fontsource/jetbrains-mono/files/jetbrains-mono-greek-400-normal.woff2?url'
import f5 from '@fontsource/jetbrains-mono/files/jetbrains-mono-greek-500-normal.woff2?url'
import f6 from '@fontsource/geist/files/geist-latin-400-normal.woff2?url'
import f7 from '@fontsource/geist/files/geist-latin-500-normal.woff2?url'
import f8 from '@fontsource/geist/files/geist-latin-600-normal.woff2?url'
import f9 from '@fontsource/geist/files/geist-latin-ext-400-normal.woff2?url'
import f10 from '@fontsource/eb-garamond/files/eb-garamond-latin-400-normal.woff2?url'
import f11 from '@fontsource/eb-garamond/files/eb-garamond-latin-500-normal.woff2?url'
import f12 from '@fontsource/eb-garamond/files/eb-garamond-latin-400-italic.woff2?url'
import f13 from '@fontsource/eb-garamond/files/eb-garamond-latin-500-italic.woff2?url'
import f14 from '@fontsource/eb-garamond/files/eb-garamond-greek-400-normal.woff2?url'
import f15 from '@fontsource/eb-garamond/files/eb-garamond-greek-400-italic.woff2?url'

const LATIN = 'U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD'
const LATIN_EXT = 'U+0100-02BA,U+02BD-02C5,U+02C7-02CC,U+02CE-02D7,U+02DD-02FF,U+0304,U+0308,U+0329,U+1D00-1DBF,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20C0,U+2113,U+2C60-2C7F,U+A720-A7FF'
const GREEK = 'U+0370-0377,U+037A-037F,U+0384-038A,U+038C,U+038E-03A1,U+03A3-03FF'

const FACES: [string, string, string, string, string][] = [
  ['JetBrains Mono', f0, '400', 'normal', LATIN],
  ['JetBrains Mono', f1, '500', 'normal', LATIN],
  ['JetBrains Mono', f2, '400', 'normal', LATIN_EXT],
  ['JetBrains Mono', f3, '500', 'normal', LATIN_EXT],
  ['JetBrains Mono', f4, '400', 'normal', GREEK],
  ['JetBrains Mono', f5, '500', 'normal', GREEK],
  ['Geist', f6, '400', 'normal', LATIN],
  ['Geist', f7, '500', 'normal', LATIN],
  ['Geist', f8, '600', 'normal', LATIN],
  ['Geist', f9, '400', 'normal', LATIN_EXT],
  ['EB Garamond', f10, '400', 'normal', LATIN],
  ['EB Garamond', f11, '500', 'normal', LATIN],
  ['EB Garamond', f12, '400', 'italic', LATIN],
  ['EB Garamond', f13, '500', 'italic', LATIN],
  ['EB Garamond', f14, '400', 'normal', GREEK],
  ['EB Garamond', f15, '400', 'italic', GREEK],
]

/** Register and load every face now: canvas text does not make the browser fetch a font by itself. */
export function registerFonts() {
  if (!('fonts' in document)) return
  for (const [family, url, weight, style, unicodeRange] of FACES) {
    const face = new FontFace(family, `url(${url}) format('woff2')`, { weight, style, unicodeRange, display: 'swap' })
    document.fonts.add(face)
    face.load().catch(() => { /* the fallback fonts stay in use */ })
  }
}
