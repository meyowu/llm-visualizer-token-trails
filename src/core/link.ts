/*
 * The address carries the state of the page: #/route?phase=softmax&head=6.6. Pages write their
 * own keys with setParams (history is replaced, not pushed) and read them back on mount.
 */

/** The query of the current route. */
export const getParams = () => new URLSearchParams(location.hash.split('?')[1] ?? '')

/** Set or clear keys in the address without adding a history entry. */
export function setParams(values: Record<string, string | null>) {
  const [path] = location.hash.replace(/^#\/?/, '').split('?')
  const q = getParams()
  for (const [k, v] of Object.entries(values)) if (v === null) q.delete(k); else q.set(k, v)
  const want = `#/${path}${q.size ? '?' + q.toString() : ''}`
  if (location.hash !== want) try { history.replaceState(history.state, '', want) } catch { /* sandboxed frames may refuse */ }
}
