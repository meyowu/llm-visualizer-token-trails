import { rich } from '../core/frame'
import { TERMS } from '../core/glossary'
import type { Nav } from './registry'

/** Every term of art on the site, with a one- or two-sentence definition. */
export function mountGlossary(root: HTMLElement, _nav: Nav): () => void {
  root.classList.add('start')
  const slug = (t: string) => t.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  root.innerHTML = `
    <header class="head"><div><p class="eyebrow">Reference</p><div class="titlebar"><h1>Glossary</h1><p class="sub">terms of art, in plain words</p></div></div></header>
    <div class="start-body glossary">
      <dl>${[...TERMS].sort((a, b) => a.term.localeCompare(b.term)).map((t) => `<div id="${slug(t.term)}"><dt>${rich(t.term)}</dt><dd>${rich(t.def)}</dd></div>`).join('')}</dl>
    </div>`
  return () => {}
}
