import { rich, toggle } from '../core/frame'
import { t } from '../core/i18n'
import { legendList } from '../core/legend'
import { lastPlace } from '../core/progress'
import { PROMPT_LABELS, nextDist, presets } from '../lib/gpt2/data'
import { pageName, type Nav } from './registry'

/** The route of each step of the tour, in teaching order. */
const PATH: [string, string][] = [
  ['foundations', 'Foundations'],
  ['anatomy', 'Forward pass'],
  ['anatomy/tokenizer', 'Tokenizer'],
  ['anatomy/embedding', 'Embedding'],
  ['anatomy/layernorm', 'LayerNorm & Residual'],
  ['anatomy/attention', 'Attention'],
  ['anatomy/mlp', 'MLP'],
  ['anatomy/unembed', 'Unembed & Sampling'],
  ['training/loss', 'Training'],
  ['lineage/transformer-2017', 'Architectures'],
  ['serving/kv-cache', 'Serving'],
  ['agents/in-context', 'Agents'],
]


/** The landing page: what the model does, a live next-token example, the path, and how to read the pictures. */
export function mountStart(root: HTMLElement, nav: Nav): () => void {
  root.classList.add('start')
  root.innerHTML = `
    <header class="head">
      <div>
        <p class="eyebrow">${t(`Start here`)}</p>
        <div class="titlebar"><h1>${t(`A language model predicts the next token`)}</h1><p class="sub">${t(`and nothing else`)}</p></div>
      </div>
    </header>
    <div class="start-body">
      <section class="st-demo" aria-labelledby="st-demo-h">
        <h2 id="st-demo-h">${t(`Try it: what comes next?`)}</h2>
        <div class="st-prompt"></div>
        <p class="st-text"></p>
        <ol class="st-bars" aria-label="${t(`Most likely next tokens`)}"></ol>
        <p class="st-note"></p>
      </section>
      <section class="st-what">
        <h2>${t(`What is a Transformer?`)}</h2>
        <p>${t(`The model on this site is <b>GPT-2 small</b> (OpenAI, 2019, 124M parameters), a Transformer. It cuts text into tokens, turns each token into a vector of 768 numbers, and passes the vectors through 12 blocks. In each block, <b>attention</b> lets every token read from the tokens before it, and an <b>MLP</b> then works on each token alone. The last token's vector is finally turned into a probability for every possible next token.`)}</p>
        <p>${t(`<b>Decoder-only</b> means it reads left to right: a token never sees the ones after it. Generating text is just this, repeated: pick a token, append it, run again. LLaMA and most chat models keep the same design with a few changes (see Architectures).`)}</p>
      </section>
      <section class="st-scale">
        <h2>${t(`Where the 124M numbers live`)}</h2>
        <div class="st-bar" role="img" aria-label="${t(`GPT-2 small parameters: MLPs 56.7 million, token embeddings 38.6 million, attention 28.3 million, positions 0.8 million, LayerNorms 0.04 million`)}"></div>
        <ul class="st-bar-key"></ul>
        <p>${t(`Running it costs about 2 floating-point operations per weight per token, roughly 250M FLOPs for each new token, plus attention’s share, which grows with the context. Bigger models keep the same parts, only wider and deeper: GPT-2 XL has 1.5B weights, LLaMA 3 has 8B and 70B.`)}</p>
      </section>
      <section class="st-path" aria-labelledby="st-path-h">
        <h2 id="st-path-h">${t(`The path`)}</h2>
        <ol></ol>
        <button type="button" class="st-go">${t(`Start the tour →`)}</button>
      </section>
      <section class="st-read">
        <h2>${t(`How to read the pictures`)}</h2>
        ${legendList()}
        <p class="st-keys">${t(`<kbd>Space</kbd> play or pause · <kbd>←</kbd> <kbd>→</kbd> previous or next step. Each step pauses at its end so there is time to read; switch to <b>Auto</b> in the controls to play straight through.`)}</p>
      </section>
      <section class="st-train">
        <h2>${t(`Where do the numbers come from?`)}</h2>
        <p>${t(`Every weight (124M of them) was set by <b>training</b>: GPT-2 read about 40 GB of web text, predicted each next token, and after every batch nudged all its weights so the actual next token got a little more probability. Nothing in the model was written by hand. Even the tokenizer was learned, by counting which pairs of symbols appear together most often. The Next-token loss page shows the signal it learns from.`)}</p>
      </section>
      <section class="st-real">
        <h2>${t(`Real or toy?`)}</h2>
        <p>${t(`The Forward pass and Unembed pages show the numbers of a real GPT-2 small run, computed offline. The detail views that animate every matrix product use a toy model (8 numbers per token instead of 768) so each cell fits on screen; they say <code>shown: toy</code> and give GPT-2's real sizes alongside.`)}</p>
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
    q('.st-note').innerHTML = rich(t(`…and ${d.restCount.toLocaleString('en-US')} other tokens share the remaining ${(d.rest * 100).toFixed(0)}%. Real GPT-2 small output; Ġ marks a leading space.`))
  }
  toggle(q('.st-prompt'), 'Prompt', PRESETS.map((p) => PROMPT_LABELS[p.key] ?? p.text), 0, (i) => { k = i; demo() })
  demo()

  const ol = q('.st-path ol')
  PATH.forEach(([route, name], i) => {
    const li = document.createElement('li')
    const n = route === 'foundations' ? 'prep' : route === 'anatomy' ? 'map' : route.startsWith('training') ? 'then' : route.startsWith('lineage') || route.startsWith('serving') || route.startsWith('agents') ? 'more' : String(i - 1).padStart(2, '0')
    li.innerHTML = `<a href="#/${route}"><span class="n">${/\d/.test(n) ? n : t(n)}</span><b></b></a>`
    li.querySelector('b')!.textContent = t(name)
    li.querySelector('a')!.addEventListener('click', (e) => { e.preventDefault(); nav(route) })
    ol.appendChild(li)
  })
  // GPT-2 small's parameters by part (exact counts, biases included)
  const PARTS: [string, number, string][] = [
    ['MLPs, 12 × 4.72M', 56_669_184, 'var(--t4)'], ['token embeddings W_E (also the output)', 38_597_376, 'var(--t0)'],
    ['attention, 12 × 2.36M', 28_348_416, 'var(--t2)'], ['positions W_P', 786_432, 'var(--t3)'], ['LayerNorms', 38_400, 'var(--t5)'],
  ]
  const total = PARTS.reduce((a, [, n]) => a + n, 0)
  q('.st-bar').innerHTML = PARTS.map(([, n, col]) => `<i style="flex:${n} 0 1px;background:${col}"></i>`).join('')
  q('.st-bar-key').innerHTML = PARTS.map(([name, n, col]) => `<li><i style="background:${col}"></i>${rich(t(name))} <b>${n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : (n / 1e3).toFixed(0) + 'K'}</b> <small>${((n / total) * 100).toFixed(n / total < 0.01 ? 2 : 0)}%</small></li>`).join('')
  q('.st-go').addEventListener('click', () => nav('anatomy'))
  // pick up where this browser left off
  const last = lastPlace()
  if (last && last.route !== 'start') {
    const b = document.createElement('button')
    b.type = 'button'; b.className = 'st-go st-resume'
    const page = pageName(last.route) ?? last.route.split('/').pop()
    b.textContent = `${t('Resume')}: ${t(page ?? '')} ›`
    b.addEventListener('click', () => nav(`${last.route}?phase=${last.phase}`))
    q('.st-go').after(b)
  }
  return () => {}
}

const escape = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
/** A token as GPT-2 spells it: a leading space shows as a faint Ġ. */
const tokHtml = (t: string) => (t.startsWith(' ') ? `<i>Ġ</i>${escape(t.slice(1))}` : escape(t.replace(/\n/g, 'Ċ')))
