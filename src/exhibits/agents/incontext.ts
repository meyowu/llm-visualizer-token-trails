import { F, chipW, ctx, drawChip } from '../../core/draw'
import { C, rgba } from '../../core/theme'
import { clamp, eout } from '../../core/util'
import { symbolText } from '../../lib/gpt2/bpe'
import { icl, type ShotRow } from '../../lib/icl/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'
import { bars, chipLines, distRows, modelGlyph, pct, runs, type Bar } from './common'

/*
 * In-context learning: the same frozen GPT-2 small does a different task depending on the examples in its
 * prompt, a copying (induction) head is part of how, the examples also bias it, and an instruction-tuned
 * model follows a plain instruction instead. Every probability is from a real run (scripts/icl-export.ts).
 */

const PHASES = [
  { id: 'zero', name: 'No examples', short: '0 examples', dur: 8 },
  { id: 'shots', name: 'Examples set the task', short: 'Few-shot', dur: 12 },
  { id: 'program', name: 'The prompt is the program', short: 'Programs', dur: 10 },
  { id: 'copy', name: 'A head that copies', short: 'Induction', dur: 10 },
  { id: 'bias', name: 'Examples also mislead', short: 'Bias', dur: 10 },
  { id: 'instruct', name: 'Instructions instead of examples', short: 'Instructions', dur: 11 },
]

const T = icl.tasks, Q = icl.query, CAP = T.capital, HEAD = icl.head
const layerHead = `layer ${HEAD.layer + 1}, head ${HEAD.head + 1}`
const extra = icl.curve[3].tokens - icl.curve[0].tokens

const COMPARE: Record<string, [string, string]> = {
  zero: ['GPT-2’s next-token distribution', 'anatomy/unembed?phase=softmax'],
  shots: ['a training step, which does change weights', 'training/loss?phase=step'],
  program: ['GPT-2’s forward pass', 'anatomy?phase=pick'],
  copy: ['attention weights', 'anatomy/attention?phase=softmax'],
  bias: ['the weighted sum of values', 'anatomy/attention?phase=av'],
  instruct: ['special tokens and ids', 'anatomy/tokenizer?phase=ids'],
}

const CAPS: Record<string, [string, string]> = {
  zero: [`GPT-2 small is given “${Q}:” and nothing else. Nothing says what should follow, so its guesses are generic: a newline, “The”. “ Cairo” gets ${pct(CAP.rows[0].p)}.`, `p( Cairo | “${Q}:” ) = ${pct(CAP.rows[0].p)}`],
  shots: [`The same model, with examples of country: capital lines before the question. One example is enough: “ Cairo” jumps to ${pct(CAP.rows[1].p)}. No weight changes; the examples only sit in the context. Over ${icl.countries} countries, one example takes the right answer to first place ${icl.curve[1].top1} times out of ${icl.countries}.`, 'few-shot prompting: examples in the context, weights fixed'],
  program: [`Keep the question, change the examples, and the model computes a different function: capital, language or continent. Each is GPT-2’s real top guess after two examples. This is in-context learning: the prompt acts as the program.`, 'same weights · same question · three tasks'],
  copy: [`Part of how: in ${layerHead}, the last “:” puts ${pct(HEAD.share)} of its attention on the words that followed the earlier colons, the answers. It looks for what came after this token before and copies from there. Heads that do this are called induction heads.`, `${layerHead}: attention from the last “:”`],
  bias: [`Copying also brings bias. With two “Europe” labels among three examples, GPT-2 says “ Europe” for ${Q}; “ Africa” drops to ${pct(T.continent.rows[3].p)}. With capitals, the last example (Rome) pulls in “ Rome” and “ Milan”. More examples do not help here: ${icl.curve[1].top1}, ${icl.curve[2].top1}, ${icl.curve[3].top1}, ${icl.curve[4].top1} of ${icl.countries} right with 1 to 4.`, 'majority and recency bias'],
  instruct: [`Given only the instruction, GPT-2 keeps writing text like its training data. ${icl.tuned.model} was instruction-tuned: trained further on instructions and answers in a chat template. It answers “${icl.tuned.answer}” with no examples. Examples still help to pin down a format, at ${extra} tokens for three here, paid on every call.`, 'instruction tuning: the instruction is the program'],
}

export function mountInContext(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'Real runs of GPT-2 small and Qwen3-1.7B, exported offline: every probability shown comes from the models.',
      eyebrow: 'Agents · Prompting',
      title: 'In-context learning',
      subtitle: 'the prompt is the program; the weights stay fixed',
      specs: [
        { label: 'model', value: 'GPT-2 small', real: 'weights fixed', realLabel: '' },
        { label: 'task', value: 'country → capital' },
        { label: 'examples', value: '0 to 3' },
        { label: 'copying head', value: layerHead },
        { label: 'instruction-tuned', value: icl.tuned.model },
      ],
    },
    size: [1040, 470],
    aria: 'In-context learning: GPT-2 small, given examples of country: capital lines in its prompt, answers Cairo for Egypt; changing the examples changes the task; an induction head copies from earlier answers; examples bias the answer; an instruction-tuned model answers from an instruction alone.',
    phases: PHASES, learn: 'incontext', tokens: words(CAP.rows[3].prompt), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['shots', 11],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption, arrow } = k
  const pad = 36, top = 56
  const isAns = (ans: string) => (r: Bar) => r.text === 'Ġ' + ans
  /** Top 5, plus the answer's row when it is not among them. */
  const withAnswer = (r: ShotRow, ans: string): Bar[] => {
    const rows = bars(r.top)
    return rows.some(isAns(ans)) ? rows : [...rows, { text: 'Ġ' + ans, p: r.p }]
  }
  /** Prompt on the left, the model in the middle, the next-token bars on the right. */
  function pipeline(r: ShotRow, ans: string, a: number, grow: number, lineA: (i: number) => number, sub: string) {
    const { W } = stage, xM = 520, xD = 600, wD = W - pad - xD, y0 = top + 20
    title(`prompt · ${r.prompt.length} tokens`, pad, y0, a)
    chipLines(r.prompt, pad, y0 + 30, a, { alpha: lineA })
    modelGlyph(k, xM, y0 + 80, 'GPT-2 small', sub, a)
    arrow([[xM - 60, y0 + 80], [xM - 34, y0 + 80]], a)
    arrow([[xM + 34, y0 + 80], [xD - 14, y0 + 80]], a)
    title('next token', xD, y0, a)
    const rows = withAnswer(r, ans)
    distRows(rows, xD, y0 + 30, wD, a, { grow, mark: isAns(ans), tag: (b, i) => (i > 0 && isAns(ans)(b) ? 'the answer' : undefined) })
  }

  /* ---------- 1: no examples ---------- */
  function sceneZero(p: number) {
    const r = CAP.rows[0]
    pipeline(r, CAP.answer, 1, eout(clamp((p - 0.1) / 0.3)), () => 1, 'weights fixed')
    mk.formula = { segs: [[`p( Cairo | ${Q}: )`, C.ink2], ['  =  ', C.mute], [pct(r.p), C.ink]], note: `GPT-2 small reads the ${r.prompt.length} tokens and gives a probability to each of its 50,257 tokens. Top guess: a newline, ${pct(r.top[0][1])}.` }
  }

  /* ---------- 2: add examples, one at a time ---------- */
  function sceneShots(p: number) {
    const { H } = stage
    const n = Math.min(3, Math.floor(clamp((p - 0.04) / 0.6) * 4)), t0 = 0.04 + n * 0.15, r = CAP.rows[n]
    const g = eout(clamp((p - t0) / 0.1)), qStart = r.prompt.length - 2
    // the newest example line fades in: it is the 4 tokens before the query
    pipeline(r, CAP.answer, 1, g, (i) => (n > 0 && i >= qStart - 4 && i < qStart ? g : 1), `${n} example${n === 1 ? '' : 's'}`)
    // the average over countries
    const ca = eout(clamp((p - 0.66) / 0.1))
    if (ca > 0) {
      const x0 = pad, yB = H - 70, bw = 40, gap = 34, hMax = 90
      title(`average over ${icl.countries} countries`, x0, yB - hMax - 34, ca)
      icl.curve.forEach((c, i) => {
        const x = x0 + i * (bw + gap), h = (c.p / 0.4) * hMax * ca
        ctx.fillStyle = rgba(C.ink, (i === n ? 0.85 : 0.4) * ca); ctx.fillRect(x, yB - h, bw, h)
        ctx.font = F.small; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(i === n ? C.ink : C.mute, ca)
        ctx.fillText(pct(c.p), x + bw / 2, yB - h - 6)
        ctx.fillText(`${c.n}`, x + bw / 2, yB + 16)
        ctx.fillText(`${c.top1}/${icl.countries}`, x + bw / 2, yB + 32)
      })
      ctx.strokeStyle = rgba(C.faint, ca); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 - 4, yB + 0.5); ctx.lineTo(x0 + 5 * (bw + gap) - gap + 4, yB + 0.5); ctx.stroke()
      caption('examples', x0 + 5 * (bw + gap) - gap + 12, yB + 16, ca, C.mute, 'left')
      caption('right answer first', x0 + 5 * (bw + gap) - gap + 12, yB + 32, ca, C.mute, 'left')
      caption('mean p(answer)', x0 + 5 * (bw + gap) - gap + 12, yB - 40, ca, C.mute, 'left')
    }
    mk.formula = { segs: [[`${n} example${n === 1 ? '' : 's'}`, C.ink2], ['  →  ', C.mute], [`p( Cairo ) = ${pct(r.p)}`, C.ink]], note: n === 0 ? 'Examples are added as plain lines of text before the question.' : `The prompt grew to ${r.prompt.length} tokens. Nothing was trained: the same weights read a longer context.` }
  }

  /* ---------- 3: three programs ---------- */
  function sceneProgram(p: number) {
    const { W } = stage, names = ['capital', 'language', 'continent'] as const, colW = (W - 2 * pad) / 3, y0 = top + 20
    names.forEach((name, c) => {
      const a = eout(clamp((p - 0.05 - c * 0.18) / 0.12)), x = pad + c * colW, r = T[name].rows[2]
      if (a <= 0) return
      title(`${name} · 2 examples`, x, y0, a)
      chipLines(r.prompt, x, y0 + 30, a, { c0: c * 3 })
      const yA = y0 + 30 + 2 * 30 + 36
      arrow([[x + 30, yA - 20], [x + 30, yA]], a)
      caption('GPT-2 small', x + 40, yA - 6, a, C.mute, 'left')
      distRows(bars(r.top.slice(0, 3)), x, yA + 24, colW - 30, a, { grow: eout(clamp((p - 0.12 - c * 0.18) / 0.15)), mark: (_r, i) => i === 0, labelW: 110 })
    })
    const fa = eout(clamp((p - 0.7) / 0.1))
    caption(`the same weights and the same last line, “${Q}:”`, pad, top + 20 + 30 + 60 + 36 + 24 + 3 * 24 + 30, fa, C.ink, 'left')
    mk.formula = { segs: [['capital → “ Cairo”', C.ink], ['   ·   ', C.mute], ['language → “ Egyptian”', C.ink], ['   ·   ', C.mute], ['continent → “ Africa”', C.ink]], note: 'GPT-2 small says “ Egyptian” rather than “ Arabic” (4%): a plausible reading of the pattern. The examples define what the question means.' }
  }

  /* ---------- 4: the copying head ---------- */
  function sceneCopy(p: number) {
    const { W } = stage, toks = HEAD.prompt, N = toks.length, y = top + 260
    // lay the prompt out on one line, centred
    const ws = toks.map((t) => chipW(t)), total = ws.reduce((s, w) => s + w + 6, -6)
    const x0 = Math.max(pad, (W - total - 170) / 2), xs: number[] = []
    let x = x0
    ws.forEach((w) => { xs.push(x); x += w + 6 })
    const cx = (i: number) => xs[i] + ws[i] / 2, last = N - 1
    toks.forEach((t, i) => drawChip(xs[i], y, { text: t, c: i }, 1, 22, i === last || HEAD.answers.includes(i)))
    const g = eout(clamp((p - 0.12) / 0.4))
    title(`${layerHead} · attention from the last “:”`, x0, top + 30, 1)
    HEAD.row.forEach((w, j) => {
      if (j === last || w < 0.004) return
      const x1 = cx(last), x2 = cx(j), hgt = 40 + (x1 - x2) * 0.32
      ctx.strokeStyle = rgba(C.tok[j % 7], (0.15 + 0.85 * Math.sqrt(w)) * g); ctx.lineWidth = 0.6 + 9 * w * g
      ctx.beginPath(); ctx.moveTo(x1, y - 13); ctx.bezierCurveTo(x1, y - 13 - hgt, x2, y - 13 - hgt, x2, y - 13); ctx.stroke()
      if (w >= 0.05) caption(pct(w), (x1 + x2) / 2, y - 13 - hgt * 0.75 - 8, g, C.ink)
    })
    // the prediction
    const pa = eout(clamp((p - 0.6) / 0.12)), xe = xs[last] + ws[last] + 18
    if (pa > 0) {
      arrow([[xe, y], [xe + 30, y]], pa)
      drawChip(xe + 38, y, { text: CAP.rows[3].top[0][0], c: N }, pa, 22, true)
      caption(`${pct(CAP.rows[3].top[0][1])}, GPT-2’s top guess`, xe + 38, y + 30, pa, C.ink2, 'left')
    }
    const ba = eout(clamp((p - 0.4) / 0.12))
    HEAD.answers.forEach((j) => caption('after “:”', cx(j), y + 30, ba, C.mute))
    const segs = HEAD.answers.map((j) => [`${symbolText(toks[j]).trim()} ${pct(HEAD.row[j])}`, C.ink] as [string, typeof C.ink])
    mk.formula = { segs: [...segs.flatMap((s, i) => (i ? [[' + ', C.mute] as [string, typeof C.ink], s] : [s])), ['  =  ', C.mute], [pct(HEAD.share), C.ink2]], note: 'Real attention weights. The latest example gets the most. An induction head matches the current token (“:”) with its earlier copies and attends to the token after each one; later layers turn that into “ Cairo”.' }
  }

  /* ---------- 5: bias ---------- */
  function sceneBias(p: number) {
    const { W } = stage, r = T.continent.rows[3], y0 = top + 20, xD = 600, wD = W - pad - xD
    title(`continent · 3 examples`, pad, y0, 1)
    const labels = [2, 6, 10]
    chipLines(r.prompt, pad, y0 + 30, 1, { hl: (i) => labels.includes(i) })
    const la = eout(clamp((p - 0.1) / 0.12))
    caption('labels in the examples: Europe ×2, Asia ×1', pad, y0 + 30 + 4 * 30 + 6, la, C.ink2, 'left')
    const g = eout(clamp((p - 0.15) / 0.25))
    title('next token', xD, y0, 1)
    distRows(bars(r.top), xD, y0 + 30, wD, 1, { grow: g, mark: isAns(T.continent.answer), tag: (b) => (isAns(T.continent.answer)(b) ? 'the answer' : undefined) })
    // recency: the capital prompt ends with Italy: Rome
    const ra = eout(clamp((p - 0.5) / 0.12)), yR = y0 + 200
    title('capital · 3 examples, the last is Italy: Rome', xD, yR, ra)
    distRows(bars(CAP.rows[3].top), xD, yR + 30, wD, ra, { grow: eout(clamp((p - 0.55) / 0.2)), mark: (b) => b.text === 'ĠRome' || b.text === 'ĠMilan' })
    mk.formula = { segs: [['p( Europe )', C.ink], [` ${pct(r.top[0][1])}`, C.ink], ['   ·   ', C.mute], ['p( Africa )', C.ink2], [` ${pct(r.p)}`, C.ink2]], note: 'Few-shot answers lean toward labels that are frequent in the examples and toward the last example (Zhao et al., 2021). Balancing and shuffling examples, or calibrating the probabilities, reduces it.' }
  }

  /* ---------- 6: instruction tuning ---------- */
  function sceneInstruct(p: number) {
    const { W } = stage, xT = pad + 170, xD = W - pad - 270, wD = W - pad - xD
    const yA = top + 40, yB = top + 210
    // GPT-2, base
    const aa = eout(clamp(p / 0.08))
    k.rowName('GPT-2 small', 'base model', pad, yA + 6, aa)
    title('prompt → what it writes', xT, yA - 20, aa)
    runs([[icl.instruction + '↵', C.ink2]], xT, yA + 8, aa, F.mono(11))
    const ga = eout(clamp((p - 0.12) / 0.12))
    runs([['writes  ', C.mute], [icl.base.continuation.replace(/\n/g, '↵') + ' …', C.ink]], xT, yA + 32, ga, F.mono(11))
    title('first token', xD, yA - 20, aa)
    distRows(bars(icl.base.top.slice(0, 3)), xD, yA + 8, wD, ga, { mark: (_r, i) => i === 0, labelW: 80 })
    // Qwen3, instruction-tuned
    const ba = eout(clamp((p - 0.35) / 0.1))
    k.rowName(icl.tuned.model, 'instruction-tuned', pad, yB + 6, ba)
    title(`chat template · ${icl.tuned.tokens} tokens`, xT, yB - 20, ba)
    const sp = C.mute, f = F.mono(11)
    runs([['<|im_start|>', sp], ['user↵', C.ink2]], xT, yB + 8, ba, f)
    runs([[icl.instruction, C.ink2], ['<|im_end|>', sp], ['↵', C.ink2]], xT, yB + 30, ba, f)
    const qa = eout(clamp((p - 0.5) / 0.12))
    const xa = runs([['<|im_start|>', sp], ['assistant↵', C.ink2], ['<think>↵↵</think>↵↵', sp]], xT, yB + 52, ba, f)
    runs([[icl.tuned.answer, C.ink]], xa + 4, yB + 52, qa, F.mono(12, 600))
    title('first token', xD, yB - 20, ba)
    distRows(bars(icl.tuned.top.slice(0, 3)), xD, yB + 8, wD, qa, { mark: (_r, i) => i === 0, labelW: 80 })
    const na = eout(clamp((p - 0.7) / 0.1))
    caption('special tokens mark who is speaking; the empty think block switches Qwen3’s reasoning off', xT, yB + 84, na, C.mute, 'left')
    mk.formula = { segs: [['base model: continue the text', C.ink2], ['   ·   ', C.mute], ['instruction-tuned: answer it', C.ink]], note: `${icl.tuned.model} puts ${pct(icl.tuned.top[0][1])} on “${icl.tuned.top[0][0]}”, the first token of “${icl.tuned.answer}”. GPT-2 puts ${pct(icl.base.top[0][1])} on a newline and then writes on about Egypt.` }
  }

  return { zero: sceneZero, shots: sceneShots, program: sceneProgram, copy: sceneCopy, bias: sceneBias, instruct: sceneInstruct }
}
