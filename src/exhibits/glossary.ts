import { rich } from '../core/frame'
import { TERMS } from '../core/glossary'
import { lang, t, termText } from '../core/i18n'
import type { Nav } from './registry'

/** Every term of art on the site, with a one- or two-sentence definition. */
export function mountGlossary(root: HTMLElement, _nav: Nav): () => void {
  root.classList.add('start')
  const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  // in Chinese, each term shows its English name beside it, and the list is sorted by the Chinese names
  const entries = TERMS.map((e) => ({ en: e.term, x: termText(e.term, e.def, e.match) })).sort((a, b) => a.x.term.localeCompare(b.x.term, lang === 'zh' ? 'zh-CN' : 'en'))
  root.innerHTML = `
    <header class="head"><div><p class="eyebrow">${t('Reference')}</p><div class="titlebar"><h1>${t('Glossary')}</h1><p class="sub">${t('terms of art, in plain words')}</p></div></div></header>
    <div class="start-body glossary">
      <dl>${entries.map(({ en, x }) => `<div id="${slug(en)}"><dt>${rich(x.term)}${x.term !== en ? ` <small lang="en">${rich(en)}</small>` : ''}</dt><dd>${rich(x.def)}</dd></div>`).join('')}</dl>
    </div>`
  return () => {}
}
