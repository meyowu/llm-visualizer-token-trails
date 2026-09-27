import { rich, toggle } from '../core/frame'
import { PROMPT_LABELS, nextDist, presets } from '../lib/gpt2/data'
import type { Nav } from './registry'

/** The route of each step of the tour, in teaching order. */
const PATH: [string, string, string][] = [
  ['anatomy', 'Forward pass', 'The whole trip on one screen, with the numbers of a real GPT-2 run. Come back to it as the map.'],
  ['anatomy/tokenizer', 'Tokenizer', 'How text becomes a list of token ids.'],
  ['anatomy/embedding', 'Embedding', 'How each id becomes a vector of 768 numbers.'],
  ['anatomy/layernorm', 'LayerNorm & Residual', 'The stream every layer reads from and adds to.'],
  ['anatomy/attention', 'Attention', 'How a token pulls in information from the tokens before it.'],
  ['anatomy/mlp', 'MLP', 'How each token is then transformed on its own.'],
  ['anatomy/unembed', 'Unembed & Sampling', 'How the last vector becomes probabilities, and one next token.'],
]

const LEGEND: [string, string][] = [
  ['<svg viewBox="0 0 64 20"><rect x="1" y="3" width="26" height="14" rx="4" fill="none" stroke="var(--t0)"/><rect x="35" y="3" width="26" height="14" rx="4" fill="none" stroke="var(--t2)"/></svg>', 'Each token keeps its own colour everywhere.'],
  ['<svg viewBox="0 0 64 20"><path d="M2 10h30" stroke="var(--t1)" stroke-width="2"/><path d="M32 10h30" stroke="var(--ink2)" stroke-width="2"/></svg>', 'A lane is one token’s vector flowing through the model; its colour blends as it takes in other tokens.'],
  ['<svg viewBox="0 0 64 20"><path d="M26 17l8-5V1l-8 5z" fill="none" stroke="var(--ink2)"/><path d="M40 17l8-5V1l-8 5z" fill="none" stroke="var(--mute)"/></svg>', 'A glass plate is a layer the lanes pass through.'],
  ['<svg viewBox="0 0 64 20"><rect x="14" y="4" width="11" height="11" fill="var(--ink2)"/><rect x="39" y="4.5" width="10" height="10" fill="none" stroke="var(--ink2)"/></svg>', 'In a matrix, a filled cell is positive and an outlined cell is negative.'],
  ['<svg viewBox="0 0 64 20"><text x="32" y="15" text-anchor="middle" fill="var(--ink)" font-size="14" font-family="var(--mono)">↗</text></svg>', 'Parts marked ↗ open a detail view. Hover any matrix cell for its formula.'],
]

/** The landing page: what the model does, a live next-token example, the path, and how to read the pictures. */
export function mountStart(root: HTMLElement, nav: Nav): () => void {
  root.classList.add('start')
  root.innerHTML = `
    <header class="head">
      <div>
        <p class="eyebrow">Start here</p>
        <h1>A language model predicts the next token<span class="sub">and nothing else</span></h1>
      </div>
    </header>
    <div class="start-body">
      <section class="st-demo" aria-labelledby="st-demo-h">
        <h2 id="st-demo-h">Try it: what comes next?</h2>
        <div class="st-prompt"></div>
        <p class="st-text"></p>
        <ol class="st-bars" aria-label="Most likely next tokens"></ol>
        <p class="st-note"></p>
      </section>
      <section class="st-what">
        <h2>What is a Transformer?</h2>
        <p>The model on this site is <b>GPT-2 small</b> (OpenAI, 2019, 124M parameters), a Transformer. It cuts text into tokens, turns each token into a vector of 768 numbers, and passes the vectors through 12 blocks. In each block, <b>attention</b> lets every token read from the tokens before it, and an <b>MLP</b> then works on each token alone. The last token's vector is finally turned into a probability for every possible next token.</p>
        <p><b>Decoder-only</b> means it reads left to right: a token never sees the ones after it. Generating text is just this, repeated: pick a token, append it, run again. LLaMA and most chat models keep the same design with a few changes (see Lineage).</p>
      </section>
      <section class="st-path" aria-labelledby="st-path-h">
        <h2 id="st-path-h">The path</h2>
        <ol></ol>
        <button type="button" class="st-go">Start the tour →</button>
      </section>
      <section class="st-read">
        <h2>How to read the pictures</h2>
        <ul class="st-legend"></ul>
        <p class="st-keys"><kbd>Space</kbd> play or pause · <kbd>←</kbd> <kbd>→</kbd> previous or next step. Each step pauses at its end so there is time to read; switch to <b>Auto</b> in the controls to play straight through.</p>
      </section>
      <section class="st-real">
        <h2>Real or toy?</h2>
        <p>The Forward pass and Unembed pages show the numbers of a real GPT-2 small run, computed offline. The detail views that animate every matrix product use a toy model (8 numbers per token instead of 768) so each cell fits on screen; they say <code>shown: toy</code> and give GPT-2's real sizes alongside.</p>
      </section>
    </div>`

  const q = <T extends HTMLElement>(s: string) => root.querySelector(s) as T
  const PRESETS = presets()
  let k = 0
  function demo() {
    const p = PRESETS[k].passes[0], d = nextDist(p.next, 1, 5), max = d.rows[0].p
    q('.st-text').innerHTML = `${escape(PRESETS[k].text)}<span class="caret"></span>`
    q('.st-bars').innerHTML = d.rows.map((r) => `
      <li><span class="tok">${tokHtml(r.text)}</span><span class="bar"><i style="width:${((r.p / max) * 100).toFixed(1)}%"></i></span><span class="p">${(r.p * 100).toFixed(1)}%</span></li>`).join('')
    q('.st-note').innerHTML = rich(`…and ${d.restCount.toLocaleString('en-US')} other tokens share the remaining ${(d.rest * 100).toFixed(0)}%. Real GPT-2 small output; Ġ marks a leading space.`)
  }
  toggle(q('.st-prompt'), 'Prompt', PRESETS.map((p) => PROMPT_LABELS[p.key] ?? p.text), 0, (i) => { k = i; demo() })
  demo()

  const ol = q('.st-path ol')
  PATH.forEach(([route, name, what], i) => {
    const li = document.createElement('li')
    li.innerHTML = `<a href="#/${route}"><span class="n">${i === 0 ? 'map' : String(i).padStart(2, '0')}</span><b></b><small></small></a>`
    li.querySelector('b')!.textContent = name
    li.querySelector('small')!.textContent = what
    li.querySelector('a')!.addEventListener('click', (e) => { e.preventDefault(); nav(route) })
    ol.appendChild(li)
  })
  q('.st-go').addEventListener('click', () => nav('anatomy'))
  q('.st-legend').innerHTML = LEGEND.map(([svg, t]) => `<li><span class="sw" aria-hidden="true">${svg}</span><span>${t}</span></li>`).join('')
  return () => {}
}

const escape = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
/** A token as GPT-2 spells it: a leading space shows as a faint Ġ. */
const tokHtml = (t: string) => (t.startsWith(' ') ? `<i>Ġ</i>${escape(t.slice(1))}` : escape(t.replace(/\n/g, 'Ċ')))
