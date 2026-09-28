import { F, chipW, ctx, drawChip, mathName } from '../../core/draw'
import { C, rgba } from '../../core/theme'
import { clamp, eout } from '../../core/util'
import { sft as S, type SftToken } from '../../lib/sft/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'
import { noLigatures, runs, whoBar, wrap } from '../agents/common'

/*
 * Supervised fine-tuning: the pretraining loss, applied to conversations in the chat template, counted only on the
 * answer's tokens. Real numbers from Qwen3-1.7B-Base (before) and Qwen3-1.7B (after Qwen's post-training) on one
 * example (scripts/sft-export.ts).
 */

const PHASES = [
  { id: 'format', name: 'A conversation, in the template', short: 'Format', dur: 9 },
  { id: 'mask', name: 'Loss only on the answer', short: 'Mask', dur: 11 },
  { id: 'before', name: 'Before and after', short: 'Behaviour', dur: 10 },
  { id: 'sharp', name: 'Sure of its own words', short: 'Sharpness', dur: 10 },
  { id: 'untrained', name: 'Tokens it never learned', short: 'Masked', dur: 10 },
]

const T = S.tokens
const answer = T.filter((t) => t.trained)
const show = (t: string) => t.replace(/\n/g, '↵')
const one = (s: string) => s.replace(/\s+/g, ' ').trim()
/** Template tokens where the tuned model does much worse than the base: never trained, so free to drift. */
const DRIFT = T.map((t, i) => ({ ...t, i })).filter((t) => !t.trained && t.base !== null && t.tuned !== null && t.tuned - t.base > 4).sort((a, b) => (b.tuned! - b.base!) - (a.tuned! - a.base!))

const COMPARE: Record<string, [string, string]> = {
  format: ['the chat template in in-context learning', 'agents/in-context?phase=instruct'],
  mask: ['the same loss in pretraining', 'training/loss?phase=loss'],
  before: ['what a base model does with an instruction', 'agents/in-context?phase=instruct'],
  sharp: ['temperature and sharp distributions', 'anatomy/unembed?phase=temp'],
  untrained: ['the causal mask, which is different', 'anatomy/attention?phase=mask'],
}

const CAPS: Record<string, [string, string]> = {
  format: [`Supervised fine-tuning trains a pretrained model on conversations written in its chat template: special tokens mark whose turn it is, the user asks, the assistant answers. This example is ${T.length} tokens, ${answer.length} of them the answer.`, `${T.length} tokens · ${answer.length} in the answer`],
  mask: [`The loss is the same next-token loss as in pretraining, but only the answer’s tokens count; the prompt and template are masked. Under the base model, before any SFT, this answer costs ${S.answerLoss.base.toFixed(2)} nats per token.`, `loss = mean −log p over the ${answer.length} answer tokens`],
  before: [`What each model writes for the prompt. The base model has no notion of turns and invents a speaker label; the tuned one answers in one sentence and stops. (Qwen’s real post-training used far more than SFT, and far more than one example.)`, 'base: continue the text · tuned: answer it'],
  sharp: [`Training on its answers makes a model very sure of its own phrasing. Each model scored both answers: the tuned model finds its own nearly certain and the base model’s unlikely. On the answer written for this page it does worse than the base, ${S.answerLoss.tuned.toFixed(2)} against ${S.answerLoss.base.toFixed(2)}.`, 'post-training sharpens the distribution'],
  untrained: [`Masked tokens get no gradient, so nothing keeps the model’s guesses there sensible. After post-training, Qwen3 finds even the template’s own “assistant” very unlikely where it appears. It never has to predict it: the program writes the template.`, 'no loss, no training signal'],
}

export function mountSft(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: `Real per-token losses from ${S.base} and ${S.model}, exported offline.`,
      eyebrow: 'Training · After pretraining',
      title: 'SFT',
      subtitle: 'instruction tuning: the loss, on answers only',
      specs: [
        { label: 'before', value: S.base },
        { label: 'after', value: S.model, real: 'Qwen’s post-training', realLabel: '' },
        { label: 'example', value: `${T.length} tokens`, real: `${answer.length} trained`, realLabel: '' },
      ],
    },
    size: [1040, 480],
    aria: 'Supervised fine-tuning: a conversation in the chat template, the loss counted only on the answer’s tokens, what a base and an instruction-tuned model write, how the tuned model becomes sure of its own phrasing, and how template tokens that are never trained drift.',
    phases: PHASES, learn: 'sft', tokens: words(['user', 'assistant']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['mask', 10.5],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption } = k
  const pad = 36, top = 56
  /** Diagonal hatching over a chip: this token is masked. */
  function hatchRect(x: number, y: number, w: number, h: number, a: number) {
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip()
    ctx.strokeStyle = rgba(C.mute, 0.55 * a); ctx.lineWidth = 1; ctx.beginPath()
    for (let d = -h; d < w; d += 6) { ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y) }
    ctx.stroke(); ctx.restore()
  }
  /** The example as chips, wrapped; masked tokens hatched. Returns each chip's box. */
  function tokens(a: number, o: { hatchMasked?: number; bars?: 'base' | 'tuned' | null; grow?: number; only?: (t: SftToken) => boolean } = {}) {
    const { W } = stage, boxes: { x: number; y: number; w: number }[] = []
    let x = pad, y = top + 34
    noLigatures(true)
    T.forEach((t, i) => {
      const w = chipW(show(t.text))
      if (x + w > W - pad) { x = pad; y += o.bars ? 74 : 34 }
      const on = o.only ? o.only(t) : true
      drawChip(x, y, { text: show(t.text), c: i }, a * (on ? 1 : 0.35), 22, t.trained && (o.hatchMasked ?? 0) > 0)
      if (!t.trained && (o.hatchMasked ?? 0) > 0) hatchRect(x, y - 11, w, 22, (o.hatchMasked ?? 0) * a)
      if (o.bars && on) {
        const v = t[o.bars]
        if (v !== null) {
          const h = Math.min(40, v * 3) * (o.grow ?? 1)
          ctx.fillStyle = rgba(t.trained ? C.ink : C.mute, (t.trained ? 0.8 : 0.35) * a); ctx.fillRect(x + 4, y + 16, Math.max(2, (w - 10) * 0.4), Math.max(1, h))
          if (t.trained) { ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(v.toFixed(1), x + 4 + (w - 10) * 0.4 + 3, y + 16) }
        }
      }
      boxes.push({ x, y, w })
      x += w + 5
    })
    noLigatures(false)
    return boxes
  }

  /* ---------- 1: the format ---------- */
  function sceneFormat(p: number) {
    title('one training example, as Qwen3’s tokens', pad, top + 12, 1)
    tokens(1, { hatchMasked: eout(clamp((p - 0.5) / 0.15)) })
    const ka = eout(clamp((p - 0.6) / 0.12)), ky = stage.H - 110
    hatchRect(pad, ky - 8, 16, 16, ka); caption('prompt and template: masked', pad + 24, ky + 4, ka, C.mute, 'left')
    ctx.strokeStyle = rgba(C.ink, ka); ctx.lineWidth = 1.5; ctx.strokeRect(pad + 260, ky - 8, 16, 16); caption('the answer: trained', pad + 284, ky + 4, ka, C.mute, 'left')
    mk.formula = { segs: [['<|im_start|>user', C.mute], [' … ', C.ink2], ['<|im_end|>', C.mute], ['  ', C.mute], ['<|im_start|>assistant', C.mute], [' answer ', C.ink], ['<|im_end|>', C.mute]], note: 'The template is part of the data: the model learns to write the answer and then <|im_end|>, which is how it knows to stop.' }
  }

  /* ---------- 2: the loss, on the answer only ---------- */
  function sceneMask(p: number) {
    title(`−log p under ${S.base}, per token · the answer counts, the rest is masked`, pad, top + 12, 1)
    tokens(1, { hatchMasked: 1, bars: 'base', grow: eout(clamp((p - 0.1) / 0.3)) })
    const worst = answer.filter((t) => t.base !== null).sort((a, b) => b.base! - a.base!)[0]
    mk.formula = { segs: [['L', C.ink], [' = ', C.mute], [`mean over ${answer.length} answer tokens of −log p`, C.ink2], [' = ', C.mute], [S.answerLoss.base.toFixed(3), C.ink]], note: `The hardest answer token for the base model is “${worst.text.trim()}” (${worst.base!.toFixed(1)} nats). SFT lowers exactly these losses, over many thousands of such examples.` }
  }

  /* ---------- 3: before and after ---------- */
  function sceneBefore(p: number) {
    const { W } = stage, colW = (W - 2 * pad - 50) / 2, y0 = top + 60
    title('prompt', pad, top + 12, 1)
    runs([[S.user, C.ink2]], pad, top + 32, 1, F.mono(12))
    const col = (x: number, name: string, text: string, a: number) => {
      if (a <= 0) return
      mathName(name, x, y0 + 18, a, 20)
      let y = y0 + 40
      for (const l of wrap(one(text), colW - 20, F.mono(11))) { runs([[l, C.ink]], x + 14, y + 9, a, F.mono(11)); y += 19 }
      whoBar('model', x, y0 + 43, y - 3, a)
    }
    col(pad, `${S.base}, before`, S.wrote.base, eout(clamp(p / 0.15)))
    col(pad + colW + 50, `${S.model}, after`, S.wrote.tuned, eout(clamp((p - 0.35) / 0.15)))
    mk.formula = { segs: [['same prompt, same template', C.ink2], ['  ·  ', C.mute], ['different weights', C.ink]], note: 'Base models were trained to continue documents, so a chat transcript is just more text to continue; tuned models were trained to end their turn.' }
  }

  /* ---------- 4: sharpness ---------- */
  function sceneSharp(p: number) {
    const x0 = pad + 250, y0 = top + 70, cw = 230, rh = 60
    title('mean −log p per token: rows score, columns wrote', pad, top + 12, 1)
    const cols = [`${S.base}’s answer`, `${S.model}’s answer`, 'the answer on this page']
    cols.forEach((c, j) => { ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink2, 1); ctx.fillText(c, x0 + j * cw, y0 - 16) })
    const rows: [string, number[]][] = [[S.base, [...S.cross.base, S.answerLoss.base]], [S.model, [...S.cross.tuned, S.answerLoss.tuned]]]
    const mx = Math.max(...rows.flatMap((r) => r[1]))
    rows.forEach(([name, vals], i) => {
      const a = eout(clamp((p - 0.05 - i * 0.25) / 0.15)), y = y0 + 10 + i * rh
      mathName(name, pad, y + 14, a, 18)
      vals.forEach((v, j) => {
        const x = x0 + j * cw
        ctx.fillStyle = rgba(C.ink, 0.07 * a); ctx.fillRect(x, y, cw - 60, 12)
        ctx.fillStyle = rgba(C.ink, 0.75 * a); ctx.fillRect(x, y, (v / mx) * (cw - 60), 12)
        ctx.font = F.mono(12, 600); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(v.toFixed(2), x + (v / mx) * (cw - 60) + 8, y + 6)
      })
    })
    mk.formula = { segs: [[`${S.model} on its own answer`, C.ink2], [' ', C.mute], [S.cross.tuned[1].toFixed(2), C.ink], ['   ·   on the base model’s', C.ink2], [' ', C.mute], [S.cross.tuned[0].toFixed(2), C.ink]], note: 'A sharper distribution is what makes tuned models consistent, and also why they repeat stock phrases: probability has moved from many acceptable wordings onto a few.' }
  }

  /* ---------- 5: the masked tokens drift ---------- */
  function sceneUntrained(p: number) {
    const x0 = pad + 200, y0 = top + 60, bw = 520, mx = Math.max(...DRIFT.slice(0, 6).map((t) => t.tuned!))
    title('masked tokens where the tuned model got much worse · −log p', pad, top + 12, 1)
    DRIFT.slice(0, 6).forEach((t, i) => {
      const a = eout(clamp((p - 0.05 - i * 0.08) / 0.12)), y = y0 + i * 48, prev = T[t.i - 1]?.text ?? ''
      noLigatures(true)
      ctx.font = F.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(`after ${show(prev)}`, x0 - 110, y + 8)
      drawChip(x0 - 100, y + 8, { text: show(t.text), c: t.i }, a, 22)
      noLigatures(false)
      ctx.fillStyle = rgba(C.ink, 0.35 * a); ctx.fillRect(x0, y, (t.base! / mx) * bw, 7)
      ctx.fillStyle = rgba(C.ink, 0.85 * a); ctx.fillRect(x0, y + 10, (t.tuned! / mx) * bw, 7)
      ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(`base ${t.base!.toFixed(1)}`, x0 + (t.base! / mx) * bw + 6, y + 4)
      ctx.fillStyle = rgba(C.ink, a); ctx.fillText(`tuned ${t.tuned!.toFixed(1)}`, x0 + (t.tuned! / mx) * bw + 6, y + 14)
    })
    mk.formula = { segs: [['masked', C.ink2], [' → ', C.mute], ['no gradient', C.ink], [' → ', C.mute], ['no constraint on those predictions', C.ink2]], note: 'Harmless here, since the program writes the template. It is also why a tuned model cannot be judged by its loss on ordinary text: it was trained to be good at answers, not at everything.' }
  }

  return { format: sceneFormat, mask: sceneMask, before: sceneBefore, sharp: sceneSharp, untrained: sceneUntrained }
}
