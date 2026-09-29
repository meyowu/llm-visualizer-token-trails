/*
 * The address carries the page and its state: /anatomy/unembed/?phase=softmax&head=6.6 on the site, one prerendered
 * HTML file per page so each can be found and shared. The single-file preview has only one address, so there the
 * route lives in the hash instead: #/anatomy/unembed?phase=softmax. Pages write their own keys with setParams (history
 * is replaced, not pushed) and read them back on mount. Old #/ links still open, on either build.
 */

/** Routes in the hash (the single-file preview) or in the path (the site). */
export const HASH_ROUTES = import.meta.env.MODE === 'artifact'

/** The address of a route: /anatomy/unembed/ (the home page is /), or #/anatomy/unembed in the preview. */
export function routeHref(route: string, qs = ''): string {
  const q = qs ? '?' + qs : ''
  if (HASH_ROUTES) return `#/${route}${q}`
  return (route === 'home' ? '/' : `/${route}/`) + q
}

/** The route and query in the address, from its path or an old #/ link ('' when the address names no route). */
export function readAddress(): { route: string; qs: string } {
  const h = location.hash
  if (HASH_ROUTES || h.startsWith('#/')) {
    const [route, qs] = h.replace(/^#\/?/, '').split('?')
    if (route || HASH_ROUTES) return { route, qs: qs ?? '' }
  }
  return { route: location.pathname.replace(/^\/+|\/+$/g, ''), qs: location.search.slice(1) }
}

/** The query of the current route. */
export const getParams = () => new URLSearchParams(readAddress().qs)

/** Set or clear keys in the address without adding a history entry. */
export function setParams(values: Record<string, string | null>) {
  const { route } = readAddress(), q = getParams()
  for (const [k, v] of Object.entries(values)) if (v === null) q.delete(k); else q.set(k, v)
  const want = routeHref(route || 'home', q.toString())
  const now = HASH_ROUTES ? location.hash : location.pathname + location.search
  if (now !== want) try { history.replaceState(history.state, '', want) } catch { /* sandboxed frames may refuse */ }
}
