import { ZH, ZH_TERMS } from '../locales/zh'
import { pref } from './prefs'

/*
 * English and Chinese. English is the source: every string in the code is English and is its own key. Chinese
 * comes from the locale files in src/locales/zh. t() translates a finished string, so it works on text built
 * from live values: numbers and quoted spans (“ Cairo”, ‘the’) in it become placeholders, a key reads like
 * 'loss {} after {} steps', and the Chinese puts the values back as {0}, {1}… in any order. A string with no
 * entry stays English.
 *
 * HTML text is translated where the frame, player, rail and pages set it. Canvas text is translated where it is
 * drawn (localizeCanvas patches fillText and measureText); token text and math are drawn inside raw(), untouched.
 */
export type Lang = 'en' | 'zh'
export const LANGS: Lang[] = ['en', 'zh']

/** Numbers (not inside names like GPT-2 or W_Q) and quoted spans. */
const PH = /“[^”]*”|‘[^’]*’|(?<![\w.\-−_])[−-]?\d+(?:[.,]\d+)*%?(?![\w]|[.,]\d)/g
/** A string as its key: placeholders for the numbers and quotes, which are returned as args. */
export function norm(s: string): { key: string; args: string[] } {
  const args: string[] = []
  const key = s.replace(PH, (m) => { args.push(m); return '{}' })
  return { key, args }
}

const lowerZH = new Map(Object.entries(ZH).map(([k, v]) => [k.toLowerCase(), v]))

function initial(): Lang {
  let q: string | null = null
  try { q = new URLSearchParams(location.hash.split('?')[1] ?? '').get('lang') } catch { /* no location */ }
  const saved = q ?? pref.get('lang')
  if (saved === 'en' || saved === 'zh') return saved
  return typeof navigator !== 'undefined' && /^zh/i.test(navigator.language) ? 'zh' : 'en'
}
export let lang: Lang = initial()
document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en'

const listeners = new Set<() => void>()
/** Switch language; listeners (the router) re-render. */
export function setLang(l: Lang) {
  if (l === lang) return
  lang = l
  pref.set('lang', l)
  document.documentElement.lang = l === 'zh' ? 'zh-CN' : 'en'
  cache.clear()
  for (const fn of listeners) fn()
}
export const onLang = (fn: () => void) => { listeners.add(fn); return () => listeners.delete(fn) }

/* ---------- translating ---------- */
const cache = new Map<string, string>()
/** Harvest mode (dev): every string asked for, by key, with an example and the routes it appeared on. */
export const harvest: { on: boolean; route: string; seen: Map<string, { ex: string; routes: Set<string> }> } = { on: import.meta.env.DEV && pref.get('i18n-harvest') === '1', route: '', seen: new Map() }

/** The current language's text for an English string. */
export function t(s: string): string {
  if (!s) return s
  if (harvest.on) record(s)
  if (lang === 'en') return s
  const hit = cache.get(s)
  if (hit !== undefined) return hit
  const out = translate(s)
  if (cache.size > 5000) cache.clear()
  cache.set(s, out)
  return out
}
function translate(s: string): string {
  if (!/[A-Za-z]/.test(s)) return s
  // keep leading and trailing spaces out of the key
  const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(s)!
  const core = m[2], { key, args } = norm(core)
  const exact: string | undefined = ZH[key]
  const shout = exact === undefined && core === core.toUpperCase()
  const v = exact ?? lowerZH.get(key.toLowerCase())
  if (v === undefined) return s
  let out = v.replace(/\{(\d+)\}/g, (_: string, i: string) => args[+i] ?? '')
  if (shout) out = out.toUpperCase()
  return m[1] + out + m[3]
}
function record(s: string) {
  if (!/[A-Za-z]{2}/.test(s)) return
  const core = s.trim(), { key } = norm(core)
  let e = harvest.seen.get(key)
  if (!e) harvest.seen.set(key, (e = { ex: core, routes: new Set() }))
  e.routes.add(harvest.route)
}

/** A glossary term in the current language: its name, the spellings to mark in captions, its definition. */
export function termText(term: string, def: string, match: string[]): { term: string; def: string; match: string[] } {
  if (lang === 'en') return { term, def, match }
  const z = ZH_TERMS[term]
  if (!z) return { term, def: t(def), match }
  const [name, zdef, ...spellings] = z
  // Chinese captions keep acronyms such as GELU or KV in Latin letters, so the English spellings stay too
  return { term: name, def: zdef, match: [...spellings, ...match] }
}

/* ---------- canvas ---------- */
let rawDepth = 0
/** Draw without translating (token text, math, model output). */
export function raw<T>(fn: () => T): T {
  rawDepth++
  try { return fn() } finally { rawDepth-- }
}
/** Translate every string this context draws or measures, unless drawn inside raw(). */
export function localizeCanvas(ctx: CanvasRenderingContext2D) {
  const fill = ctx.fillText.bind(ctx), measure = ctx.measureText.bind(ctx)
  const tr = (s: string) => (rawDepth ? s : t(String(s)))
  ctx.fillText = (s: string, x: number, y: number, w?: number) => (w === undefined ? fill(tr(s), x, y) : fill(tr(s), x, y, w))
  ctx.measureText = (s: string) => measure(tr(s))
}
