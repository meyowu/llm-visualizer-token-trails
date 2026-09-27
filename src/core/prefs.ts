/** Reader preferences shared by every exhibit (speed, pacing, temperature…), kept in this browser when storage is available. */
export const pref = {
  get(key: string): string | null {
    try { return localStorage.getItem('tt-' + key) } catch { return null }
  },
  set(key: string, v: string) {
    try { localStorage.setItem('tt-' + key, v) } catch { /* private mode or blocked storage */ }
  },
}
