import { F, chipW, ctx, drawChip, rr } from '../../core/draw'
import { raw, t, tf } from '../../core/i18n'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp } from '../../core/util'
import { sft as S } from '../../lib/sft/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'
import { noLigatures, pct, runs, wrap } from '../agents/common'

/*
 * Supervised fine-tuning: the pretraining loss, applied to conversations in the chat template, graded only on the
 * answer's tokens. Real numbers from Qwen3-1.7B-Base (before) and Qwen3-1.7B (after Qwen's post-training) on one
 * example (scripts/sft-export.ts). One picture throughout: token chips, with the probability a model gave each
 * token as a bar under it.
 */

const PHASES = [
  { id: 'chat', name: 'What the model sees', short: 'Template', dur: 10 },
  { id: 'grade', name: 'Graded on the answer only', short: 'Loss', dur: 13 },
  { id: 'before', name: 'What it learns to do', short: 'Behaviour', dur: 10 },
  { id: 'sharp', name: 'Sure of its own words', short: 'Sharpness', dur: 10 },
  { id: 'untrained', name: 'Tokens it never learned', short: 'Masked', dur: 9 },
]

const T = S.tokens, N = T.length, ANS = S.promptTokens
const ALL = T.map((_, i) => i)
const answer = T.filter((t) => t.trained)
/** Where the assistant's turn starts (its <|im_start|>), and the user's words: after "<|im_start|>user\n", up to <|im_end|>. */
const ASSIST = T.findIndex((t, i) => i > 0 && t.text === '<|im_start|>')
const USER0 = 3, USER_END = T.findIndex((t) => t.text === '<|im_end|>')
const show = (s: string) => s.replace(/\n/g, '↵')
const one = (s: string) => s.replace(/\s+/g, ' ').trim()
const prob = (nll: number | null) => (nll === null ? 0 : Math.exp(-nll))
const hue = (i: number): RGB => C.tok[i % 7]
/** The answer's summed −log p under the base model: the loss before dividing by its length. */
const TOTAL = answer.reduce((s, t) => s + t.base!, 0)
/** Where Qwen3-1.7B's own answer leaves the page's: the first word that differs (one token per word up to there). */
const DIV_WORD = (() => { const a = S.answer.split(' '), b = one(S.wrote.tuned).split(' '); let d = 0; while (a[d] === b[d]) d++; return d })()
const DIV = ANS + DIV_WORD
/** Template tokens the tuned model now finds far less likely than the base did: never trained, so free to drift. */
const DRIFT = ALL.filter((i) => !T[i].trained && T[i].base !== null && T[i].tuned! - T[i].base! > 4).sort((a, b) => (T[b].tuned! - T[b].base!) - (T[a].tuned! - T[a].base!)).slice(0, 3).sort((a, b) => a - b)
/** A small probability as odds: 6, 530,000, 7.6 × 10¹⁵ (for "1 in …"). */
function odds(nll: number): string {
  const n = Math.exp(nll)
  if (n < 1e6) return Number(n < 10 ? n.toFixed(1) : n.toPrecision(2)).toLocaleString('en-US')
  const e = Math.floor(Math.log10(n))
  return `${(n / 10 ** e).toFixed(1)} × 10${[...String(e)].map((d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join('')}`
}

const COMPARE: Record<string, [string, string]> = {
  chat: ['the chat template in in-context learning', 'agents/in-context?phase=instruct'],
  grade: ['the same loss in pretraining', 'training/loss?phase=loss'],
  before: ['what a base model does with an instruction', 'agents/in-context?phase=instruct'],
  sharp: ['temperature and sharp distributions', 'anatomy/unembed?phase=temp'],
  untrained: ['the causal mask, which is different', 'anatomy/attention?phase=mask'],
}

const CAPS: Record<string, [string, string]> = {
  chat: [`A model never sees chat bubbles. The chat template turns the conversation into one token sequence, with special tokens around each turn: <|im_start|>, the role, the text, <|im_end|>. SFT trains on many such sequences; this one has ${N} tokens, ${answer.length} of them the answer.`, `${N} tokens · ${answer.length} in the answer`],
  grade: [`Training reads the whole sequence and, at every position, asks the model for the next token, exactly as in pretraining. Only the answer’s guesses are graded: their −log p is added up into the loss, while the prompt and template are masked. Before any tuning, Qwen3-1.7B-Base pays ${S.answerLoss.base.toFixed(2)} nats per answer token.`, `loss = mean −log p over the ${answer.length} answer tokens`],
  before: [`What that training does, over many thousands of such examples: the same prompt, before and after. The base model treats the chat as a transcript to continue and writes the speaker label itself; the tuned model just answers. (Qwen’s real post-training used far more than SFT.)`, 'base: continue the text · tuned: answer it'],
  sharp: [`Training on answers also makes a model very sure of its own phrasing. On the answer written for this page, the tuned model gives most tokens a probability near 1, but where the page’s wording leaves its own it drops to nearly 0. Per token it pays ${S.cross.tuned[1].toFixed(2)} nats on its own answer and ${S.answerLoss.tuned.toFixed(2)} on this one.`, 'post-training sharpens the distribution'],
  untrained: [`Masked tokens get no gradient, so nothing holds the model’s guesses there in place. After post-training most of these tokens became more likely, but a few fell off a cliff: Qwen3 now gives “assistant” after <|im_start|> a probability below 10⁻¹⁵. It never has to predict it: the program writes the template.`, 'no loss, no training signal'],
}

export function mountSft(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: `Real probabilities from ${S.base} and ${S.model}, exported offline.`,
      eyebrow: 'Training · After pretraining',
      title: 'SFT',
      subtitle: 'instruction tuning: the loss, on answers only',
      specs: [
        { label: 'before', value: S.base },
        { label: 'after', value: S.model, real: 'Qwen’s post-training', realLabel: '' },
        { label: 'example', value: `${N} tokens`, real: `${answer.length} trained`, realLabel: '' },
      ],
    },
    size: [1040, 480],
    aria: 'Supervised fine-tuning: a chat becomes one token sequence in the chat template; the model predicts every next token and only the answer’s predictions are added into the loss; the same prompt answered before and after tuning; the tuned model’s probabilities on another wording and on the template it was never trained on.',
    phases: PHASES, learn: 'sft', tokens: words(['user', 'assistant']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['grade', 12],
    scenes,
  })
}

type Box = { x: number; y: number; w: number; h: number }

function scenes({ stage, mk, k }: Env) {
  const { title, caption, arrow, rowName } = k
  const pad = 36, top = 56, lh = 19, font = F.mono(12)

  /** Diagonal hatching over a chip: this token is masked. */
  function hatchRect(x: number, y: number, w: number, h: number, a: number) {
    if (a <= 0) return
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip()
    ctx.strokeStyle = rgba(C.mute, 0.55 * a); ctx.lineWidth = 1; ctx.beginPath()
    for (let d = -h; d < w; d += 6) { ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y) }
    ctx.stroke(); ctx.restore()
  }
  const chip = (i: number, x: number, y: number, a: number, hl = false) => {
    if (a <= 0) return
    noLigatures(true); drawChip(x, y, { text: show(T[i].text), c: i }, a, 22, hl); noLigatures(false)
  }
  /** Chip places for tokens idx, wrapping at the right edge; `breakAt` starts a new row after a gap. */
  function slots(idx: number[], y0: number, pitch: number, breakAt = -1, breakGap = 0) {
    const { W } = stage, out: Record<number, { x: number; y: number; w: number }> = {}
    let x = pad, y = y0
    for (const i of idx) {
      const w = chipW(show(T[i].text))
      if (i === breakAt && x > pad) { x = pad; y += pitch + breakGap }
      else if (x + w > W - pad) { x = pad; y += pitch }
      out[i] = { x, y, w }; x += w + 5
    }
    return out
  }
  /** A probability as a bar under a chip, filled from the bottom: full height is p = 1. */
  function pbar(cx: number, y: number, p: number, h: number, col: RGB, a: number, w = 10) {
    if (a <= 0) return
    ctx.fillStyle = rgba(C.ink, 0.07 * a); ctx.fillRect(cx - w / 2, y, w, h)
    const f = p * h
    if (f > 0.1) { ctx.fillStyle = rgba(col, a); ctx.fillRect(cx - w / 2, y + h - Math.max(1, f), w, Math.max(1, f)) }
  }
  /** A chat bubble: the user's filled, the assistant's outlined; a small label above. */
  function bubble(b: Box, lines: string[], a: number, label: string, side: 'left' | 'right', n = Infinity) {
    if (a <= 0) return
    rr(b.x, b.y, b.w, b.h, 12)
    if (side === 'right') { ctx.fillStyle = rgba(C.ink, 0.07 * a); ctx.fill() }
    ctx.strokeStyle = rgba(C.ink, (side === 'right' ? 0.2 : 0.45) * a); ctx.lineWidth = 1; ctx.stroke()
    if (label) caption(label, side === 'right' ? b.x + b.w : b.x, b.y - 8, a, C.mute, side === 'right' ? 'right' : 'left')
    let left = n
    lines.forEach((l, i) => { runs([[l.slice(0, Math.max(0, left)), C.ink]], b.x + 14, b.y + 11 + lh * i + lh / 2, a, font); left -= l.length + 1 })
  }
  /** Where character `c` of the text wrapped into `lines` sits in bubble b. */
  function textPos(b: Box, lines: string[], c: number): [number, number] {
    let at = 0
    for (let i = 0; i < lines.length; i++) {
      if (c <= at + lines[i].length || i === lines.length - 1) {
        ctx.font = font
        const x = b.x + 14 + raw(() => ctx.measureText(lines[i].slice(0, Math.max(0, c - at))).width)
        return [x, b.y + 11 + lh * i + lh / 2]
      }
      at += lines[i].length + 1
    }
    return [b.x, b.y]
  }
  /** Offset of token i's first visible character in the text that starts at token `from`. */
  const charAt = (from: number, i: number) => T.slice(from, i).reduce((s, t) => s + t.text.length, 0) + (i > from && T[i].text.startsWith(' ') ? 1 : 0)
  /** Model names with a key: a thin bar for before, a bar in the token's colour for after. */
  function modelKey(x: number, y: number, a: number) {
    ctx.fillStyle = rgba(C.ink2, 0.45 * a); ctx.fillRect(x, y - 5, 8, 10)
    caption(tf('{}, before', S.base), x + 14, y + 4, a, C.mute, 'left')
    ctx.font = F.small
    const x2 = x + ctx.measureText(tf('{}, before', S.base)).width + 40
    for (let j = 0; j < 3; j++) { ctx.fillStyle = rgba(hue(j * 2), 0.95 * a); ctx.fillRect(x2 + j * 3, y - 5, 3, 10) }
    caption(tf('{}, after', S.model), x2 + 15, y + 4, a, C.mute, 'left')
  }

  /* ---------- 1: from chat bubbles to one token sequence ---------- */
  function sceneChat(p: number) {
    const { W } = stage, L = slots(ALL, top + 44, 74, ASSIST, 30)
    const bw = Math.min(460, W * 0.44)
    const uLines = wrap(S.user, bw - 28, font), aLines = wrap(S.answer, bw - 28, font)
    const ub: Box = { x: W - pad - bw, y: top + 30, w: bw, h: uLines.length * lh + 22 }
    const ab: Box = { x: pad, y: ub.y + ub.h + 34, w: bw, h: aLines.length * lh + 22 }
    // the bubbles fade as their words leave them
    const out = 1 - eout(clamp((p - 0.4) / 0.16))
    bubble(ub, uLines, eout(clamp(p / 0.08)) * out, 'the user asks', 'right')
    bubble(ab, aLines, eout(clamp((p - 0.1) / 0.08)) * out, 'the assistant answers', 'left')
    const src = (i: number): [number, number] | null => {
      if (i >= USER0 && i < USER_END) return textPos(ub, uLines, charAt(USER0, i))
      if (i >= ANS && i < N - 1) return textPos(ab, aLines, charAt(ANS, i))
      return null
    }
    // words fly to their places in the sequence; the template's tokens appear around them
    ALL.forEach((i) => {
      const s = 0.3 + 0.34 * (i / N), f = eio(clamp((p - s) / 0.14)), l = L[i], from = src(i)
      if (f <= 0) return
      if (from) chip(i, lerp(from[0], l.x, f), lerp(from[1], l.y, f), 1)
      else chip(i, l.x, l.y, f)
    })
    const la = eout(clamp((p - 0.74) / 0.1))
    title('the user’s turn', L[0].x, L[0].y - 22, la)
    title('the assistant’s turn', L[ASSIST].x, L[ASSIST].y - 22, la)
    title('the answer', L[ANS].x, L[ANS].y - 22, la)
    // a rule under the answer, row by row
    ctx.strokeStyle = rgba(C.ink, 0.6 * la); ctx.lineWidth = 1.5
    for (let i = ANS; i < N; i++) {
      const l = L[i], nx = L[i + 1], end = nx && nx.y === l.y ? nx.x : l.x + l.w
      ctx.beginPath(); ctx.moveTo(l.x, l.y + 16); ctx.lineTo(end, l.y + 16); ctx.stroke()
    }
    noLigatures(true); caption('the template adds <|im_start|>, the role, <|im_end|> and Qwen3’s empty <think> block', pad, L[N - 1].y + 52, la, C.mute, 'left'); noLigatures(false)
    mk.formula = { segs: [['<|im_start|>user↵', C.mute], [' question ', C.ink2], ['<|im_end|>↵<|im_start|>assistant↵', C.mute], [' answer ', C.ink], ['<|im_end|>', C.mute]], note: 'The template is part of the data. The model learns that an answer follows <|im_start|>assistant and ends with <|im_end|>, which is how it knows when to stop.' }
  }

  /* ---------- 2: predict every token, grade only the answer ---------- */
  function sceneGrade(p: number) {
    const { W } = stage, L = slots(ALL, top + 44, 74, ASSIST, 30), bh = 30
    const sweep = clamp((p - 0.04) / 0.74), pos = 1 + sweep * (N - 1), done = sweep >= 1
    const cur = Math.min(N - 1, Math.floor(pos)), sub = done ? 1 : pos - cur
    const ha = eout(clamp(p / 0.05))
    title(`bar: the probability ${S.base} gave the real next token`, pad, top + 6, 1)
    ALL.forEach((i) => {
      const l = L[i], a = done || i <= cur ? 1 : 0.3
      chip(i, l.x, l.y, a, !done && i === cur)
      if (!T[i].trained) hatchRect(l.x, l.y - 11, l.w, 22, ha * a)
      if (i === 0 || (!done && i > cur)) return
      const g = done || i < cur ? 1 : eout(clamp(sub * 2))
      pbar(l.x + l.w / 2, l.y + 16, prob(T[i].base) * g, bh, T[i].trained ? hue(i) : C.mute, T[i].trained ? 0.9 : 0.45)
    })
    // the cursor: everything before is read, this token is guessed
    if (!done) {
      const a0 = L[cur - 1], a1 = L[cur], c1 = a1.x + a1.w / 2
      if (a0.y === a1.y) { const c0 = a0.x + a0.w / 2; arrow([[c0, a0.y - 14], [(c0 + c1) / 2, a1.y - 26], [c1, a1.y - 14]], 1) }
      else arrow([[a1.x - 18, a1.y - 24], [c1, a1.y - 14]], 1)
      const t = T[cur], ly = a1.y + 16 + bh + 12
      caption(t.trained ? pct(prob(t.base)) : 'not graded', c1, ly, eout(clamp(sub * 3)), t.trained ? C.ink : C.mute)
    }
    // the loss: each graded token's −log p, added up
    const my = L[N - 1].y + 16 + bh + 48, mx0 = pad, mx1 = W - pad - 250, sc = (mx1 - mx0) / TOTAL
    title('the loss: −log p of the answer’s tokens, added up', mx0, my - 12, 1)
    ctx.fillStyle = rgba(C.ink, 0.07); ctx.fillRect(mx0, my, mx1 - mx0, 12)
    let x = mx0, sum = 0, n = 0
    for (let i = ANS; i < N && (done || i <= cur); i++) {
      const g = done || i < cur ? 1 : clamp(sub * 2 - 0.5), w = T[i].base! * sc * g
      ctx.fillStyle = rgba(hue(i), 0.85); ctx.fillRect(x, my, w, 12)
      x += w; sum += T[i].base! * g; if (g >= 1) n++
    }
    ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, 1)
    ctx.fillText(`${sum.toFixed(1)} nats`, mx1 + 14, my + 6)
    caption(done ? `÷ ${answer.length} tokens = ${S.answerLoss.base.toFixed(2)} nats per token` : `${n} of ${answer.length} answer tokens so far`, mx1 + 14, my + 28, 1, C.mute, 'left')
    if (done || cur < 2) mk.formula = { segs: [['L', C.ink], [' = ', C.mute], [`mean over ${answer.length} answer tokens of −log p`, C.ink2], [' = ', C.mute], [S.answerLoss.base.toFixed(3), C.ink]], note: 'The hatched positions are predicted too, in the same forward pass, but their guesses are dropped from the loss. SFT lowers the graded ones, over many thousands of such examples.' }
    else {
      const t = T[cur], pv = prob(t.base)
      mk.formula = { segs: [[`p(“${show(t.text)}” | the ${cur} tokens before)`, C.ink2], [' = ', C.mute], [pct(pv), C.ink], ...(t.trained ? [['   −log p = ', C.mute], [t.base!.toFixed(2), C.ink]] as [string, RGB][] : [['   masked: not in the loss', C.mute]] as [string, RGB][])], note: 'Every position is predicted in one forward pass, as in pretraining; the causal mask keeps each guess from seeing its own token. The loss mask is a different thing: it only decides which guesses are graded.' }
    }
  }

  /* ---------- 3: the same prompt, before and after ---------- */
  function sceneBefore(p: number) {
    const { W } = stage, gap = 48, colW = (W - 2 * pad - gap) / 2
    const cols: [number, string, string, string, number][] = [[pad, S.base, 'before post-training', one(S.wrote.base), 0.06], [pad + colW + gap, S.model, 'after post-training', one(S.wrote.tuned), 0.42]]
    cols.forEach(([x, name, sub, text, t0], c) => {
      const a = eout(clamp((p - t0 + 0.06) / 0.08))
      if (a <= 0) return
      rowName(name, sub, x, top + 12, a)
      const uw = colW * 0.82, ul = wrap(S.user, uw - 28, font), ub: Box = { x: x + colW - uw, y: top + 58, w: uw, h: ul.length * lh + 22 }
      bubble(ub, ul, a, '', 'right')
      const rw = colW * 0.9, rl = wrap(text, rw - 28, font), rb: Box = { x, y: ub.y + ub.h + 20, w: rw, h: rl.length * lh + 22 }
      const nch = Math.round(text.length * clamp((p - t0) / 0.26))
      bubble(rb, [], a, '', 'left')
      // the reply, written out; the base model's speaker label in the colour of the template's "assistant"
      const label = c === 0 && text.startsWith('Assistant:') ? 'Assistant:'.length : 0
      let left = nch, yy = rb.y + 11 + lh / 2
      rl.forEach((line, li) => {
        const sh = line.slice(0, Math.max(0, left)), cut = li === 0 ? Math.min(label, sh.length) : 0
        const x2 = runs([[sh.slice(0, cut), hue(ASSIST + 1)]], rb.x + 14, yy, a, F.mono(12, 600))
        runs([[sh.slice(cut), C.ink]], x2, yy, a, font)
        left -= line.length + 1; yy += lh
      })
      const na = eout(clamp((p - t0 - 0.28) / 0.1)) * a
      if (na <= 0) return
      let ny = rb.y + rb.h + 26, nx = rb.x + 14
      if (label) nx = runs([['Assistant:', hue(ASSIST + 1)]], nx, ny, na, F.mono(11, 600)) + 8
      const note = t(label ? 'it writes the speaker label itself: to a base model, a chat is a transcript to continue' : 'it answers the question, in one sentence as asked')
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, na)
      for (const l of wrap(note, rw - (nx - rb.x) - 10, F.small)) { raw(() => ctx.fillText(l, nx, ny)); ny += 17 }
    })
    mk.formula = { segs: [['same prompt, same template', C.ink2], ['  ·  ', C.mute], ['different weights', C.ink]], note: 'Base models were trained to continue documents, so a chat is just more text to continue. SFT shows the model thousands of turns that end where the answer ends.' }
  }

  /* ---------- 4: sure of its own words ---------- */
  function sceneSharp(p: number) {
    const idx = ALL.filter((i) => T[i].trained), L = slots(idx, top + 50, 100), bh = 44
    title('the page’s answer · bars: the probability of each token', pad, top + 6, 1)
    const { W } = stage
    modelKey(W - pad - 430, top + 2, 1)
    idx.forEach((i, n) => {
      const l = L[i], a = eout(clamp((p - 0.03 - n * 0.012) / 0.08)), g = eout(clamp((p - 0.12 - n * 0.012) / 0.2)), cx = l.x + l.w / 2
      chip(i, l.x, l.y, a, i === DIV && p > 0.45)
      pbar(cx - 6, l.y + 16, prob(T[i].base) * g, bh, C.ink2, 0.45 * a, 8)
      pbar(cx + 6, l.y + 16, prob(T[i].tuned) * g, bh, hue(i), 0.95 * a, 8)
    })
    // what the tuned model wrote itself, and where the page's wording leaves it
    const oa = eout(clamp((p - 0.45) / 0.12)), d = L[DIV], last = L[idx[idx.length - 1]], oy = last.y + 16 + bh + 64
    title(`what ${S.model} wrote itself`, pad, oy - 22, oa)
    const w = one(S.wrote.tuned).split(' ')
    let x = runs([[w.slice(0, DIV_WORD).join(' ') + ' ', C.ink2]], pad, oy, oa, font)
    const wx = x
    x = runs([[w[DIV_WORD], hue(DIV)]], x, oy, oa, F.mono(12, 600))
    runs([[' ' + w.slice(DIV_WORD + 1).join(' '), C.mute]], x, oy, oa, font)
    const by = d.y + 16 + bh + 6
    arrow([[d.x + d.w / 2, by], [d.x + d.w / 2, by + 12], [(wx + x) / 2, oy - 14]], oa)
    ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, oa)
    raw(() => {
      const cx = Math.max(d.x + d.w / 2, (wx + x) / 2) + 14
      ctx.fillText(tf('before: 1 in {}', odds(T[DIV].base!)), cx, by + 8)
      ctx.fillText(tf('after: 1 in {}', odds(T[DIV].tuned!)), cx, by + 24)
    })
    mk.formula = { segs: [[`${S.model}, per token:`, C.ink2], ['  on its own answer ', C.mute], [S.cross.tuned[1].toFixed(2), C.ink], ['   on the base model’s ', C.mute], [S.cross.tuned[0].toFixed(2), C.ink], ['   on this page’s ', C.mute], [S.answerLoss.tuned.toFixed(2), C.ink]], note: 'A sharper distribution is what makes tuned models consistent, and also why they repeat stock phrases: probability has moved from many acceptable wordings onto a few.' }
  }

  /* ---------- 5: the masked tokens drift ---------- */
  function sceneUntrained(p: number) {
    const idx = ALL.filter((i) => i > 0 && !T[i].trained), L = slots(idx, top + 50, 100, ASSIST, 0), bh = 44
    const { W } = stage
    title('the prompt and template · masked, so never trained', pad, top + 6, 1)
    modelKey(W - pad - 430, top + 2, 1)
    idx.forEach((i, n) => {
      const l = L[i], a = eout(clamp((p - 0.03 - n * 0.01) / 0.08)), g = eout(clamp((p - 0.1 - n * 0.01) / 0.2)), cx = l.x + l.w / 2
      chip(i, l.x, l.y, a, DRIFT.includes(i) && p > 0.4)
      hatchRect(l.x, l.y - 11, l.w, 22, a * 0.8)
      pbar(cx - 6, l.y + 16, prob(T[i].base) * g, bh, C.ink2, 0.45 * a, 8)
      pbar(cx + 6, l.y + 16, prob(T[i].tuned) * g, bh, hue(i), 0.95 * a, 8)
    })
    // the tokens that fell furthest, as odds
    const y0 = Math.max(...idx.map((i) => L[i].y)) + 16 + bh + 44
    DRIFT.forEach((i, r) => {
      const a = eout(clamp((p - 0.42 - r * 0.08) / 0.1)), y = y0 + r * 34
      if (a <= 0) return
      caption('after', pad, y + 4, a, C.mute, 'left')
      noLigatures(true)
      ctx.font = F.small
      let x = pad + ctx.measureText('after').width + 10
      x += drawChip(x, y, { text: show(T[i - 1].text), c: i - 1 }, a * 0.7, 20) + 5
      x += drawChip(x, y, { text: show(T[i].text), c: i }, a, 20, true) + 16
      noLigatures(false)
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a)
      raw(() => ctx.fillText(`${tf('before: 1 in {}', odds(T[i].base!))}   →   ${tf('after: 1 in {}', odds(T[i].tuned!))}`, x, y + 0.5))
    })
    mk.formula = { segs: [['masked', C.ink2], [' → ', C.mute], ['no gradient', C.ink], [' → ', C.mute], ['no constraint on those predictions', C.ink2]], note: 'Harmless here, since the program writes the template. It is also why a tuned model cannot be judged by its loss on ordinary text: it was trained to be good at answers, not at everything.' }
  }

  return { chat: sceneChat, grade: sceneGrade, before: sceneBefore, sharp: sceneSharp, untrained: sceneUntrained }
}
