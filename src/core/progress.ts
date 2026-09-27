import { pref } from './prefs'

/*
 * What the reader has seen, kept in this browser: the steps of each page they have reached, and
 * where they were last, for the rail's checkmarks and "Resume" on the start page.
 */

type Seen = Record<string, { steps: string[]; of: number }>
const read = (): Seen => { try { return JSON.parse(pref.get('seen') ?? '{}') } catch { return {} } }

/** Record that a page's step was reached; `of` is how many steps the page has. */
export function markSeen(route: string, phase: string, of: number) {
  const s = read(), e = s[route] ?? { steps: [], of }
  if (!e.steps.includes(phase) || e.of !== of) { s[route] = { steps: [...new Set([...e.steps, phase])], of }; pref.set('seen', JSON.stringify(s)) }
  pref.set('last', JSON.stringify({ route, phase }))
}
/** How far the reader got on a page: steps seen and steps in all (0 of 0 if never opened). */
export function progress(route: string) {
  const e = read()[route]
  return e ? { seen: e.steps.length, of: e.of } : { seen: 0, of: 0 }
}
/** Where the reader was last: a page and a step. */
export function lastPlace(): { route: string; phase: string } | null {
  try { return JSON.parse(pref.get('last') ?? 'null') } catch { return null }
}
