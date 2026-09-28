import { F, ctx } from '../../core/draw'
import { raw } from '../../core/i18n'
import { C, rgba } from '../../core/theme'
import { clamp, eout, lerp } from '../../core/util'
import { plainText, rag, type Query } from '../../lib/rag/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'
import { modelGlyph, runs, whoBar, wrap, type Who } from './common'

/*
 * Retrieval-augmented generation on this site's own text: every glossary definition and legend line is
 * embedded once; a question is embedded the same way, the closest passages by cosine similarity are put in
 * the prompt, and the model answers from them. Real embeddings (all-MiniLM-L6-v2) and real answers
 * (Qwen3-1.7B), exported offline by scripts/rag-export.ts.
 */

const PHASES = [
  { id: 'corpus', name: 'The knowledge, in passages', short: 'Passages', dur: 9 },
  { id: 'embed', name: 'Text becomes a vector', short: 'Embed', dur: 10 },
  { id: 'map', name: 'Close in meaning, close in space', short: 'Map', dur: 11 },
  { id: 'rank', name: 'Rank by cosine similarity', short: 'Retrieve', dur: 10 },
  { id: 'prompt', name: 'Passages into the prompt', short: 'Augment', dur: 9 },
  { id: 'answer', name: 'With and without retrieval', short: 'Generate', dur: 11 },
  { id: 'miss', name: 'When retrieval misses', short: 'Misses', dur: 12 },
]

const D = rag.docs, Qs = rag.queries, HIT = Qs[0], NEAR = Qs[1], MISS = Qs[2], GAP = Qs[3], K = rag.k
const nGloss = D.filter((d) => d.kind === 'glossary').length, nLegend = D.length - nGloss
const HIT_DOC = D[HIT.top[0].doc]
const rankOf = (q: Query, title: string) => q.top.findIndex((t) => D[t.doc].title === title) + 1
const NEAR_RANK = rankOf(NEAR, 'Speculative decoding')

const COMPARE: Record<string, [string, string]> = {
  corpus: ['the glossary these passages come from', 'glossary'],
  embed: ['BERT, the encoder family MiniLM belongs to', 'lineage/bert?phase=use'],
  map: ['CLIP’s shared embedding space', 'lineage/clip?phase=space'],
  rank: ['the dot product', 'foundations'],
  prompt: ['the context growing in a ReAct loop', 'agents/react?phase=context'],
  answer: ['instruction tuning', 'agents/in-context?phase=instruct'],
  miss: ['what a lookup tool returns', 'agents/react?phase=observe'],
}

const CAPS: Record<string, [string, string]> = {
  corpus: [`The knowledge to draw on is this site’s own text: ${nGloss} glossary definitions and ${nLegend} lines of the legend, ${D.length} passages. A model trained elsewhere has never read them.`, `${D.length} passages · ${nGloss} glossary + ${nLegend} legend`],
  embed: [`An embedding model reads each passage once and outputs one vector: ${rag.embedder}, a 6-layer BERT, averages its last layer over the tokens and scales the result to length 1. The question is embedded the same way.`, `text → ${rag.dims} numbers, length 1`],
  map: [`Passages about related things get similar vectors. Squeezed onto two dimensions (their top two principal components), related terms sit together and the question lands near the legend line that answers it. Two dimensions keep only part of the picture, so other close passages can look far.`, `2 of ${rag.dims} dimensions, for the eye only`],
  rank: [`Retrieval is a dot product: the question’s vector against every passage’s, highest first. The top ${K} are kept. For the second question the answer is only at rank ${NEAR_RANK}; a word overlap put “Few-shot prompting” first.`, `score = q · d (both length 1) · keep the top ${K}`],
  prompt: [`The program pastes the top passages into the prompt above the question, with an instruction to answer from them only. The model sees retrieved text exactly as it sees anything else in its context.`, `${HIT.tokens} tokens: instruction + ${K} passages + question`],
  answer: [`Without the passages, ${rag.model} guesses what a hatched cell might mean anywhere. With them, it answers from this site’s legend. Retrieval did not change the model; it changed what was in front of it.`, 'same model · with and without the passages'],
  miss: [`Retrieval fails quietly. “Which word came first” shares no words with the passages about position, and none comes back. A question can also find the right passage when that passage lacks the answer. Told to use only the context, the model says so, where on its own it made a name up.`, 'wrong passages in, weak answer out'],
}

export function mountRag(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: `Real embeddings from ${rag.embedder} and real answers from ${rag.model}, exported offline; the passages are this site’s glossary and legend.`,
      eyebrow: 'Agents · Retrieval',
      title: 'RAG Pipeline',
      subtitle: 'embed → retrieve → read',
      specs: [
        { label: 'passages', value: String(D.length), real: 'this site’s glossary and legend', realLabel: '' },
        { label: 'embedder', value: rag.embedder, real: `${rag.dims} dims`, realLabel: '' },
        { label: 'retrieve', value: `top ${K} by cosine` },
        { label: 'model', value: rag.model },
      ],
    },
    size: [1040, 480],
    aria: 'Retrieval-augmented generation: passages from this site are embedded as vectors; a question is embedded and compared with every passage; the closest ones are added to the prompt and the model answers from them; with and without retrieval, and two ways retrieval misses.',
    phases: PHASES, learn: 'rag', tokens: words(['query', 'passage', 'answer']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    hints: { map: 'Hover a dot to read its passage.' },
    still: ['map', 10],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption, arrow } = k
  const pad = 36, top = 56, f11 = F.mono(11), rowH = 19
  let mouse: [number, number] | null = null
  stage.canvas.addEventListener('pointermove', (e) => { mouse = stage.local(e) as [number, number] })
  stage.canvas.addEventListener('pointerleave', () => { mouse = null })

  function block(items: { text: string; who: Who; strong?: boolean }[], x: number, y: number, w: number, a: number, maxLines = 99) {
    let n = 0
    for (const it of items) {
      const y0 = y
      for (const l of wrap(it.text, w - 18, f11)) {
        if (n++ >= maxLines) break
        runs([[l, it.who === 'model' ? C.ink : C.ink2]], x + 14, y + rowH / 2, a, it.strong ? F.mono(11, 600) : f11); y += rowH
      }
      if (y > y0) whoBar(it.who, x, y0 + 3, y - 3, a)
    }
    return y
  }
  /** A vector as a row of cells: filled for positive, outlined for negative, size by magnitude. */
  function vecCells(v: number[], x: number, y: number, a: number, cs = 20) {
    const m = Math.max(...v.map(Math.abs))
    v.forEach((val, i) => {
      const s = 4 + (cs - 6) * Math.sqrt(Math.abs(val) / m), cx = x + i * (cs + 3) + cs / 2
      ctx.strokeStyle = rgba(C.faint, a); ctx.lineWidth = 1; ctx.strokeRect(x + i * (cs + 3) + 0.5, y + 0.5, cs - 1, cs - 1)
      if (val >= 0) { ctx.fillStyle = rgba(C.ink, 0.85 * a); ctx.fillRect(cx - s / 2, y + cs / 2 - s / 2, s, s) }
      else { ctx.strokeStyle = rgba(C.ink, 0.85 * a); ctx.strokeRect(cx - s / 2 + 0.5, y + cs / 2 - s / 2 + 0.5, s - 1, s - 1) }
    })
  }
  const fmt3 = (v: number) => v.toFixed(3)
  /** Mark an answer that stopped at the token limit. */
  const cut = (t: string) => (/[.!?)]$/.test(t) ? t : t + ' …')

  /* ---------- 1: the passages ---------- */
  function sceneCorpus(p: number) {
    const { W } = stage, x0 = pad, y0 = top + 30, colW = 150, rows = Math.ceil(D.length / 6)
    title(`${D.length} passages · tokens each`, x0, y0 - 16, 1)
    const n = Math.floor(clamp((p - 0.03) / 0.5) * D.length + 0.999)
    D.slice(0, n).forEach((d, i) => {
      const c = Math.floor(i / rows), r = i % rows, x = x0 + c * colW, y = y0 + r * 21
      const on = d === HIT_DOC
      ctx.fillStyle = rgba(C.ink, d.kind === 'legend' ? 0.7 : 0.3); ctx.fillRect(x, y + 4, Math.min(40, d.tokens * 0.5), 9)
      ctx.font = F.mono(10.5, on ? 600 : 400); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(on ? C.ink : C.ink2, 1)
      // passage titles are the retrieved data: never translated
      raw(() => ctx.fillText(d.title.length > 13 ? d.title.slice(0, 12) + '…' : d.title, x + 46, y + 9))
    })
    const ka = eout(clamp((p - 0.5) / 0.1)), ky = y0 + rows * 21 + 14
    ctx.fillStyle = rgba(C.ink, 0.3 * ka); ctx.fillRect(x0, ky, 14, 9); caption('glossary definition', x0 + 20, ky + 8, ka, C.mute, 'left')
    ctx.fillStyle = rgba(C.ink, 0.7 * ka); ctx.fillRect(x0 + 190, ky, 14, 9); caption('legend line', x0 + 210, ky + 8, ka, C.mute, 'left')
    // one passage in full
    const ea = eout(clamp((p - 0.6) / 0.12)), xr = x0 + 6 * colW - 40 + 60
    if (xr < W - 200) {
      title(`${HIT_DOC.title} · ${HIT_DOC.tokens} tokens`, xr, y0 - 16, ea)
      block([{ text: HIT_DOC.text, who: 'prompt' }], xr, y0, W - pad - xr, ea)
    }
    mk.formula = { segs: [['passage', C.ink2], [' = ', C.mute], ['one definition or one legend line', C.ink]], note: 'Real systems cut long documents into chunks of a few hundred tokens, often overlapping, so each chunk is about one thing and fits in the prompt.' }
  }

  /* ---------- 2: embedding ---------- */
  function sceneEmbed(p: number) {
    const x0 = pad, y1 = top + 60, y2 = top + 250, xM = 470, xV = 560
    const row = (label: string, text: string, vec: number[], y: number, a: number) => {
      title(label, x0, y - 40, a)
      block([{ text, who: 'prompt' }], x0, y - 26, 380, a, 5)
      modelGlyph(k, xM, y, rag.embedder.replace('all-', ''), '6 layers', a * 0.9)
      arrow([[x0 + 390, y], [xM - 30, y]], a); arrow([[xM + 30, y], [xV - 10, y]], a)
      vecCells(vec, xV, y - 10, a)
      caption(`dims 1–16 of ${rag.dims}`, xV, y + 30, a, C.mute, 'left')
    }
    row(`passage · ${HIT_DOC.title}`, HIT_DOC.text, rag.sample.vec, y1, 1)
    const qa = eout(clamp((p - 0.35) / 0.12))
    row('question', HIT.question, HIT.vec, y2, qa)
    const ca = eout(clamp((p - 0.65) / 0.12))
    runs([['cosine similarity  ', C.mute], [fmt3(HIT.top[0].score), C.ink]], xV, (y1 + y2) / 2 + 4, ca, F.mono(13, 600))
    caption('the dot product of the two unit vectors, over all 384 dimensions', xV, (y1 + y2) / 2 + 24, ca, C.mute, 'left')
    mk.formula = { segs: [['v', C.ink], [' = ', C.mute], ['mean(last layer over tokens) / ‖mean‖', C.ink2]], note: 'Embedding models are trained contrastively (as CLIP is) on pairs of texts that belong together, such as a question and its answer, so related texts point the same way.' }
  }

  /* ---------- 3: the map ---------- */
  const xs = D.map((d) => d.xy[0]).concat(Qs.map((q) => q.xy[0])), ys = D.map((d) => d.xy[1]).concat(Qs.map((q) => q.xy[1]))
  const bx = [Math.min(...xs), Math.max(...xs)], by = [Math.min(...ys), Math.max(...ys)]
  function sceneMap(p: number) {
    const { W, H } = stage, x0 = pad + 20, x1 = W * 0.62, y0 = top + 30, y1 = H - 60
    const X = (v: number) => lerp(x0, x1, (v - bx[0]) / (bx[1] - bx[0])), Y = (v: number) => lerp(y1, y0, (v - by[0]) / (by[1] - by[0]))
    title('passages, top two principal components', x0 - 20, y0 - 16, 1)
    const qa = eout(clamp((p - 0.3) / 0.12)), top3 = HIT.top.slice(0, K).map((t) => t.doc)
    let hover = -1, hd = 64
    D.forEach((d, i) => {
      const x = X(d.xy[0]), y = Y(d.xy[1]), on = qa > 0 && top3.includes(i)
      if (mouse) { const dd = (mouse[0] - x) ** 2 + (mouse[1] - y) ** 2; if (dd < hd) { hd = dd; hover = i } }
      ctx.beginPath(); ctx.arc(x, y, on ? 5 : 3.5, 0, 7)
      if (d.kind === 'legend') { ctx.fillStyle = rgba(C.ink, 0.8); ctx.fill() } else { ctx.strokeStyle = rgba(C.ink2, 0.7); ctx.lineWidth = 1; ctx.stroke() }
    })
    // the question and its nearest passages
    if (qa > 0) {
      const qx = X(HIT.xy[0]), qy = Y(HIT.xy[1])
      top3.forEach((i) => { ctx.strokeStyle = rgba(C.ink, 0.5 * qa); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(X(D[i].xy[0]), Y(D[i].xy[1])); ctx.stroke(); ctx.setLineDash([]) })
      ctx.fillStyle = rgba(C.tok[0], qa); ctx.beginPath(); ctx.arc(qx, qy, 6, 0, 7); ctx.fill()
      ctx.strokeStyle = rgba(C.bg, qa); ctx.lineWidth = 2; ctx.stroke()
      caption('question', qx + 10, qy - 8, qa, C.ink, 'left')
      raw(() => top3.forEach((i, r) => caption(`${r + 1}. ${D[i].title}`, X(D[i].xy[0]) + 8, Y(D[i].xy[1]) + 14, qa, C.ink2, 'left')))
    }
    // the hovered passage, or a legend of the dots
    const xr = x1 + 50, wr = W - pad - xr
    if (hover >= 0) {
      const d = D[hover], x = X(d.xy[0]), y = Y(d.xy[1])
      ctx.strokeStyle = rgba(C.ink, 1); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 7, 0, 7); ctx.stroke()
      title(d.title, xr, y0 - 16, 1)
      block([{ text: d.text, who: 'prompt' }], xr, y0, wr, 1, 14)
    } else {
      ctx.fillStyle = rgba(C.ink, 0.8); ctx.beginPath(); ctx.arc(xr + 5, y0 + 6, 3.5, 0, 7); ctx.fill(); caption('legend line', xr + 16, y0 + 10, 1, C.mute, 'left')
      ctx.strokeStyle = rgba(C.ink2, 0.7); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(xr + 5, y0 + 28, 3.5, 0, 7); ctx.stroke(); caption('glossary definition', xr + 16, y0 + 32, 1, C.mute, 'left')
      caption('distances here only roughly follow the', xr, y0 + 70, 1, C.mute, 'left')
      caption(`cosine similarity in all ${rag.dims} dimensions`, xr, y0 + 86, 1, C.mute, 'left')
    }
    mk.formula = { segs: [['x, y', C.ink2], [' = ', C.mute], ['the vector’s components along the two directions of greatest spread', C.ink]], note: 'The legend lines cluster apart from the glossary: they share a topic (how to read the pictures) that no definition has.' }
  }

  /* ---------- 4: ranking ---------- */
  function sceneRank(p: number) {
    const { W } = stage, colW = (W - 2 * pad - 60) / 2
    const list = (q: Query, x: number, a: number, mark: string) => {
      if (a <= 0) return
      title('question', x, top + 14, a)
      block([{ text: q.question, who: 'prompt' }], x, top + 26, colW, a, 2)
      const y0 = top + 90, bw = colW - 240, max = q.top[0].score
      q.top.slice(0, 8).forEach((t, i) => {
        const g = eout(clamp((p - 0.1 - i * 0.03) / 0.15)), y = y0 + i * 26 + (i >= K ? 20 : 0), d = D[t.doc], on = d.title === mark
        ctx.font = F.mono(11, on ? 600 : 400); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(on ? C.ink : C.ink2, a)
        raw(() => ctx.fillText(`${i + 1}. ${d.title.length > 24 ? d.title.slice(0, 23) + '…' : d.title}`, x, y))
        ctx.fillStyle = rgba(C.ink, 0.07 * a); ctx.fillRect(x + 200, y - 3, bw, 6)
        ctx.fillStyle = rgba(C.ink, (on ? 0.9 : 0.45) * a); ctx.fillRect(x + 200, y - 3, (t.score / max) * bw * g, 6)
        ctx.font = F.small; ctx.textAlign = 'right'; ctx.fillStyle = rgba(on ? C.ink : C.mute, a); ctx.fillText(fmt3(t.score), x + colW, y)
      })
      const cy = y0 + K * 26 - 3
      ctx.strokeStyle = rgba(C.ink, 0.6 * a); ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(x + 110, cy); ctx.lineTo(x + colW, cy); ctx.stroke(); ctx.setLineDash([])
      caption(`top ${K} kept`, x, cy + 4, a, C.mute, 'left')
    }
    list(HIT, pad, 1, HIT_DOC.title)
    list(NEAR, pad + colW + 60, eout(clamp((p - 0.45) / 0.12)), 'Speculative decoding')
    mk.formula = { segs: [['score(d)', C.ink2], [' = ', C.mute], ['q · d', C.ink], ['   for all ', C.mute], [String(D.length), C.ink], [' passages', C.mute]], note: `With millions of passages, an approximate nearest-neighbour index (HNSW, IVF) finds the top scores without comparing against every one. Here all ${D.length} are compared.` }
  }

  /* ---------- 5: the prompt ---------- */
  function scenePrompt(p: number) {
    const { W } = stage, x = pad, w = W - 2 * pad, y0 = top + 26
    title(`the user message · ${HIT.tokens} tokens with the chat template`, x, y0 - 14, 1)
    const parts = HIT.user.split('\n').filter((l) => l.trim())
    const items: { text: string; who: Who; strong?: boolean }[] = parts.map((l) => ({ text: l, who: /^\[\d\]/.test(l) ? 'tool' : 'prompt', strong: /^Question:/.test(l) }))
    const n = Math.floor(clamp((p - 0.04) / 0.5) * items.length + 0.999)
    block(items.slice(0, n), x, y0, w, 1, 20)
    const ka = eout(clamp((p - 0.6) / 0.1)), ky = y0 - 18
    whoBar('prompt', W - pad - 390, ky - 6, ky + 6, ka); caption('written by the program’s author', W - pad - 378, ky + 4, ka, C.mute, 'left')
    whoBar('tool', W - pad - 150, ky - 6, ky + 6, ka); caption('retrieved', W - pad - 138, ky + 4, ka, C.mute, 'left')
    mk.formula = { segs: [['instruction', C.ink2], [' + ', C.mute], [`[1] … [${K}]`, C.ink], [' + ', C.mute], ['question', C.ink2]], note: 'Numbering the passages lets the answer cite them. The instruction to say so when the context lacks the answer is a request, not a guarantee.' }
  }

  /* ---------- 6: with and without ---------- */
  function sceneAnswer(p: number) {
    const { W } = stage, colW = (W - 2 * pad - 50) / 2, y0 = top + 60
    title('question', pad, top + 14, 1)
    runs([[HIT.question, C.ink2]], pad, top + 32, 1, f11)
    const aa = eout(clamp((p - 0.05) / 0.15)), ba = eout(clamp((p - 0.45) / 0.15))
    title(`${rag.model}, no retrieval`, pad, y0, aa)
    block([{ text: cut(plainText(HIT.bare)), who: 'model' }], pad, y0 + 14, colW, aa, 13)
    title(`with the top ${K} passages`, pad + colW + 50, y0, ba)
    block([{ text: plainText(HIT.grounded), who: 'model', strong: true }], pad + colW + 50, y0 + 14, colW, ba, 13)
    mk.formula = { segs: [['no retrieval: ', C.mute], ['a generic guess', C.ink2], ['   ·   with retrieval: ', C.mute], ['this site’s own meaning', C.ink]], note: `The same holds for the second question: without passages it answers “${plainText(NEAR.bare).split('\n')[0].slice(0, 70)}…”; with them, “${plainText(NEAR.grounded).slice(0, 80)}”.` }
  }

  /* ---------- 7: misses ---------- */
  function sceneMiss(p: number) {
    const { W } = stage, colW = (W - 2 * pad - 50) / 2
    const col = (q: Query, x: number, a: number, label: string) => {
      if (a <= 0) return
      title(label, x, top + 14, a)
      block([{ text: q.question, who: 'prompt' }], x, top + 26, colW, a, 2)
      const y0 = top + 76
      caption('retrieved', x, y0, a, C.mute, 'left')
      q.top.slice(0, K).forEach((t, i) => runs([[`${i + 1}. ${D[t.doc].title}`, C.ink2], [`  ${fmt3(t.score)}`, C.mute]], x + 14, y0 + 16 + i * rowH, a, f11))
      caption('answer with them', x, y0 + 16 + K * rowH + 14, a, C.mute, 'left')
      const ye = block([{ text: plainText(q.grounded), who: 'model' }], x, y0 + 16 + K * rowH + 22, colW, a, 8)
      return ye
    }
    col(MISS, pad, eout(clamp(p / 0.12)), 'nothing relevant comes back')
    const ga = eout(clamp((p - 0.45) / 0.12)), ye = col(GAP, pad + colW + 50, ga, 'the right passage, without the answer')
    const guess = plainText(GAP.bare).replace(/\n/g, ' ').match(/[^.:]*proposed by[^.]*\.?/i)?.[0].trim()
    const wa = eout(clamp((p - 0.7) / 0.12))
    if (guess && ye) {
      caption('answer without any passages', pad + colW + 50, ye + 22, wa, C.mute, 'left')
      block([{ text: guess, who: 'model' }], pad + colW + 50, ye + 30, colW, wa, 4)
    }
    mk.formula = { segs: [['retrieval quality', C.ink2], [' bounds ', C.mute], ['answer quality', C.ink]], note: 'Fixes used in practice: hybrid search (keywords plus vectors), rewriting the question before searching, a reranker model over the top results, and better chunks.' }
  }

  return { corpus: sceneCorpus, embed: sceneEmbed, map: sceneMap, rank: sceneRank, prompt: scenePrompt, answer: sceneAnswer, miss: sceneMiss }
}
