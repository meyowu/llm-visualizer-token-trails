import { F, drawChip, serifAt, spaced, tokLabel, tokText, useCtx } from '../../core/draw'
import { createFrame } from '../../core/frame'
import { fmtF } from '../../core/matrix'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, rgba } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import { nextDist, presets, trainingRun } from '../../lib/gpt2/data'
import { teach } from '../learn'
import type { Nav } from '../registry'

/*
 * Next-token loss on one sentence, the way training scores it: every position predicts the token
 * after it at once, the loss is −log of the probability given to the right token, and its
 * gradient on the logits is p − onehot(target). The probabilities are GPT-2 small's; the last
 * step changes the logits directly, a toy stand-in for changing 124M weights.
 */

const PHASES = [
  { id: 'shift', name: 'Inputs and targets', short: 'Shift', dur: 5 },
  { id: 'predict', name: 'Probability of the right token', short: 'p(target)', dur: 6 },
  { id: 'loss', name: 'Cross-entropy loss', short: '−log p', dur: 6 },
  { id: 'grad', name: 'Gradient on the logits', short: 'p − y', dur: 6 },
  { id: 'step', name: 'A step downhill', short: 'Step', dur: 8 },
]
/** Toy learning rate for stepping the logits directly, and how many steps the curve shows. */
const LR = 0.6, STEPS = 30
const pct = (p: number) => (p < 0.001 ? '<0.1%' : (p * 100).toFixed(p < 0.1 ? 1 : 0) + '%')

export function mountLoss(root: HTMLElement, _nav: Nav): () => void {
  const reduced = reducedMotion()
  const run = trainingRun(), N = run.positions.length
  const toks = run.texts.map((text, i) => ({ text, id: run.ids[i], c: i % 7 }))
  const losses = run.positions.map((q) => -Math.log(q.p)), mean = losses.reduce((s, v) => s + v, 0) / N
  // the last position (Ġthe → Ġfloor): GPT-2's six likeliest tokens plus everything else as one row
  const last = nextDist(presets()[0].passes[0].next, 1, 6)
  const target = last.rows.findIndex((r) => r.id === run.ids[N])
  const z0 = [...last.rows.map((r) => Math.log(r.p)), Math.log(last.rest)]
  const soft = (z: number[]) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s) }
  // toy gradient descent on the logits themselves: z ← z − η (p − y)
  const traj: number[][] = [z0]
  for (let t = 0; t < STEPS; t++) { const z = traj[t], p = soft(z); traj.push(z.map((v, k) => v - LR * (p[k] - (k === target ? 1 : 0)))) }
  const curve = traj.map((z) => -Math.log(soft(z)[target]))

  const frame = createFrame(root, {
    formula: true,
    formulaHint: '',
    eyebrow: 'Training',
    title: 'Next-token loss',
    subtitle: 'cross-entropy · the signal every weight learns from',
    specs: [
      { label: 'shown', value: 'real GPT-2', real: 'toy last step', realLabel: '' },
      { label: 'examples', value: `${N} per sentence` },
      { label: 'mean loss', value: fmtF(mean) },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 470, 'Next-token loss: each position of "The cat sat on the floor" predicts the token after it; the loss is minus the log of the probability GPT-2 gives the right token, and the gradient on the logits pushes the right token up and the others down.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  teach(player, 'loss')
  const prog = (id: string) => player.prog(id)
  const pad = 36

  const title = (t: string, x: number, y: number, a: number) => {
    ctx.font = F.label; spaced(true); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(t, x, y); spaced(false)
  }
  const colX = (i: number) => pad + 150 + i * ((stage.W - pad * 2 - 170) / N)

  /* ---------- the sentence as five examples ---------- */
  function examples(a: number, pr: number, pl: number) {
    const yIn = 130, yOut = 200, base = stage.H - 60, h = stage.H - 60 - 250
    ctx.font = serifAt(28); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(run.text, pad, 62)
    ctx.font = F.small; ctx.fillStyle = rgba(C.mute, a); ctx.textBaseline = 'middle'
    ctx.fillText('at position', pad, yIn); ctx.fillText('must predict', pad, yOut)
    run.positions.forEach((q, i) => {
      const x = colX(i), e = eout(clamp(a * 1.4 - i * 0.08))
      // context: everything up to and including position i
      drawChip(x, yIn, toks[i], e, 22)
      ctx.strokeStyle = rgba(C.ink, 0.35 * e); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x + 14, yIn + 14); ctx.lineTo(x + 14, yOut - 14); ctx.stroke()
      drawChip(x, yOut, toks[i + 1], e, 22, true)
      if (pr > 0) {
        const g = eout(clamp(pr * 1.3 - i * 0.06)), v = pl > 0 ? lerp(q.p, losses[i] / 10, eio(pl)) : q.p, yv = base - v * h * g
        ctx.fillStyle = rgba(C.tok[toks[i + 1].c], 0.85); ctx.fillRect(x, Math.min(yv, base), 34, Math.max(1, base - yv))
        ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.ink, g)
        ctx.fillText(pl > 0.5 ? fmtF(losses[i]) : pct(q.p), x, Math.min(yv, base) - 4)
        ctx.font = F.small; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, g * (1 - pl))
        const top = q.top[0]
        if (top.id !== q.target) { ctx.fillText('top guess', x, base + 8); tokText(x, base + 30, top.text, g * (1 - pl) * 0.9, null, F.mono(11)) }
      }
    })
    if (pr > 0) {
      ctx.strokeStyle = rgba(C.ink, 0.25); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(colX(0) - 10, base); ctx.lineTo(colX(N - 1) + 60, base); ctx.stroke()
      title(pl > 0.5 ? 'LOSS  −ln p' : 'GPT-2’S p(TARGET)', pad, base - h - 12, 1)
    }
    if (pl > 0.5) {
      const my = base - (mean / 10) * h
      ctx.setLineDash([3, 4]); ctx.strokeStyle = rgba(C.ink, 0.6 * (pl - 0.5) * 2); ctx.beginPath(); ctx.moveTo(colX(0) - 10, my); ctx.lineTo(colX(N - 1) + 60, my); ctx.stroke(); ctx.setLineDash([])
      ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.ink, (pl - 0.5) * 2); ctx.fillText(`mean ${fmtF(mean)}`, colX(N - 1) + 64, my + 5)
    }
  }

  /* ---------- the last example: gradient and a step ---------- */
  function lastExample(pg: number, ps: number) {
    const x0 = pad + 40, base = stage.H - 70, h = stage.H - 70 - 150, slot = 86
    const k = ps > 0 ? Math.min(STEPS, Math.floor(eio(clamp(ps / 0.8)) * STEPS)) : 0, z = traj[k], p = soft(z), p0 = soft(z0)
    title(`“${run.text.split(' ').slice(0, -1).join(' ')}” → ${tokLabel(toks[N].text)}`, x0, 120, 1)
    ctx.strokeStyle = rgba(C.ink, 0.25); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 - 10, base); ctx.lineTo(x0 + 7 * slot, base); ctx.stroke()
    p.forEach((v, i) => {
      const x = x0 + i * slot, yv = base - v * h, right = i === target, col = right ? C.tok[toks[N].c] : C.ink
      ctx.fillStyle = rgba(col, right ? 0.9 : 0.45); ctx.fillRect(x, yv, 36, base - yv)
      if (k > 0 && right) { const y0 = base - p0[i] * h; ctx.strokeStyle = rgba(C.ink, 0.5); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x - 4, y0); ctx.lineTo(x + 40, y0); ctx.stroke(); ctx.setLineDash([]) }
      ctx.font = F.mono(11.5, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.ink2)
      ctx.fillText(pct(v), x, yv - 4)
      if (i < 6) tokText(x, base + 16, last.rows[i].text, 1, right ? col : null, F.mono(11.5))
      else { ctx.font = F.small; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute); ctx.fillText(`${last.restCount.toLocaleString('en-US')} others`, x, base + 16) }
      // the gradient p − y: right token up, the others down by their probability
      const gA = eout(clamp(pg / 0.4)) * (ps > 0 ? 1 - clamp(ps / 0.2) : 1)
      if (gA > 0) {
        // both kinds sit above the percentage: up arrows start there, down arrows end there
        const gval = (right ? 1 : 0) - p0[i], len = gval * 90, ax = x + 18, ay = yv - 26, dir = Math.sign(len) || 1
        const tail = len > 0 ? ay : ay + len, head = len > 0 ? ay - len : ay
        ctx.strokeStyle = rgba(right ? col : C.ink, gA); ctx.lineWidth = 2
        ctx.beginPath(); ctx.moveTo(ax, tail); ctx.lineTo(ax, head); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(ax - 4, head + 5 * dir); ctx.lineTo(ax, head); ctx.lineTo(ax + 4, head + 5 * dir); ctx.stroke()
      }
    })
    // loss curve over the toy steps
    if (ps > 0) {
      const cx = x0 + 7 * slot + 40, cw = stage.W - pad - cx, cy = 150, ch = base - cy, lmax = curve[0]
      title('LOSS OVER STEPS · TOY', cx, cy - 12, 1)
      ctx.strokeStyle = rgba(C.ink, 0.25); ctx.lineWidth = 1; ctx.strokeRect(cx + 0.5, cy + 0.5, cw - 1, ch - 1)
      ctx.strokeStyle = rgba(C.tok[toks[N].c]); ctx.lineWidth = 1.8; ctx.beginPath()
      for (let t = 0; t <= k; t++) { const x = cx + (t / STEPS) * cw, y = cy + ch - (curve[t] / lmax) * (ch - 10); if (t) ctx.lineTo(x, y); else ctx.moveTo(x, y) }
      ctx.stroke()
      ctx.font = F.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute)
      ctx.fillText(`step ${k} · loss ${fmtF(curve[k])}`, cx + cw - 6, cy + 6)
    }
  }

  function draw() {
    useCtx(ctx)
    stage.begin()
    const pg = prog('grad'), ps = prog('step')
    if (pg <= 0) examples(prog('shift'), prog('predict'), prog('loss'))
    else lastExample(pg, ps)
    const fs = (segs: [string, typeof C.ink][], note: string) => frame.setFormula(segs, note)
    if (pg > 0 && ps <= 0) fs([['∂ loss / ∂ zₖ  =  pₖ − [k = target]', C.ink]], `For ${tokLabel(toks[N].text)} that is ${fmtF(soft(z0)[target] - 1)}: raise its logit. For every other token it is its own probability: lower it a little.`)
    else if (ps > 0) fs([['z  ←  z − η (p − y)', C.ink], ['   ·   toy: stepping the logits directly', C.mute]], 'Real training does not change the logits; backpropagation carries this gradient back through every layer and nudges all 124M weights, which moves the logits in much the same direction.')
    else frame.setFormula(null)
  }

  const CAPS: Record<string, [string, string]> = {
    shift: ['Training reads real text. Every position predicts the token after it, so one sentence is five examples scored at once; the causal mask keeps each position from seeing its answer.', 'inputs = tokens[:-1] · targets = tokens[1:]'],
    predict: ['For each example, how much probability does GPT-2 small give the right next token? It knows little from “The” alone and much more by “on the”.', 'p(target | context)'],
    loss: ['The loss for one example is −log p: 0 for a certain right answer, large for a surprised one. Training minimises the average over billions of tokens, the cross-entropy.', 'loss = −(1/N) Σ log p(target)'],
    grad: ['The gradient of the loss with respect to the logits is simple: the probabilities minus a one-hot of the target. It says: raise the right token, lower the rest by as much as they took.', '∂L/∂z = p − y'],
    step: ['Stepping against the gradient raises the right token’s probability and lowers the loss. Here the logits are stepped directly as a toy; real training reaches them through the weights, one batch at a time.', 'z ← z − η (p − y)'],
  }
  if (reduced && player.t === 0) player.t = player.start('loss') + 5
  player.describe = (i) => CAPS[PHASES[i].id]
  const stop = runLoop((dt) => {
    player.tick(dt)
    draw()
    player.updateUI()
    const cur = player.cur(), [t, s] = CAPS[cur.id]
    frame.setCaption(cur.name, cur.short ?? cur.name, t, s)
  }, () => !player.playing)
  return () => { stop(); player.destroy(); stage.destroy() }
}
