import { F, chipW, ctx, drawChip } from '../../core/draw'
import { toggle } from '../../core/frame'
import { raw, tf } from '../../core/i18n'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp, rng } from '../../core/util'
import { ft } from '../../lib/finetune/data'
import { nextDist, presets } from '../../lib/gpt2/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'
import { modelGlyph, pct, runs } from '../agents/common'

/*
 * Learning from preferences. Pairs of answers where one is preferred; RLHF's reward model and its reinforcement
 * learning loop; DPO, which gets there from the pairs alone; then a real DPO run on GPT-2 small with rank-4
 * adapters and the frozen model as reference (scripts/finetune-export.ts).
 *
 * The loop and DPO's steps (phases 3 and 4) are computed in the page on GPT-2's real next-token probabilities
 * after "The cat sat on the", over six candidate tokens; the reward model's scores there are made up.
 */

const PHASES = [
  { id: 'pairs', name: 'Preferences, not answers', short: 'Pairs', dur: 9 },
  { id: 'reward', name: 'RLHF: a reward model', short: 'Reward', dur: 9 },
  { id: 'rl', name: 'RLHF: sample, score, shift', short: 'RL', dur: 14 },
  { id: 'dpo', name: 'DPO: no reward model', short: 'DPO', dur: 11 },
  { id: 'run', name: 'A real DPO run', short: 'Train', dur: 11 },
  { id: 'effect', name: 'Where the probability went', short: 'Effect', dur: 9 },
]

const Dp = ft.dpo, P = Dp.pairs, St = Dp.steps, N = St.length - 1, BETA = Dp.beta
const last = St[N]
const sig = (x: number) => 1 / (1 + Math.exp(-x))

/* ---------- the toy: six next tokens after "The cat sat on the" ---------- */
const CAT = presets().find((p) => p.key === 'cat')!.passes[0]
const PROMPT = CAT.texts.join('')
const DIST = nextDist(CAT.next, 1, 256).rows
const CANDS = [' floor', ' bed', ' couch', ' ground', ' edge', ' mat']
/** GPT-2's real probabilities for the six, and the same renormalized over them: the reference π_ref. */
const GPT = CANDS.map((s) => DIST.find((r) => r.text === s)!.p)
const REF = GPT.map((p) => p / GPT.reduce((a, b) => a + b, 0))
/** The reward model's scores: made up, in line with the pair “mat.” ≻ “floor.”. */
const REWARD = [-1, 0.5, 1, -1.2, -1.5, 2.5]
/** DPO's toy pairs, chosen ≻ rejected, by those scores. */
const TOY_PAIRS: [number, number][] = [[5, 0], [2, 3], [1, 4]]
const BETAS = [0.25, 1, 4]
const STEPS = 16
/** Chosen and rejected answers' colours (read when drawn: the palette comes from the page's theme). */
const chosen = (): RGB => C.tok[4], rejected = (): RGB => C.tok[2]
const candHue = (j: number): RGB => C.tok[(CAT.texts.length + j) % 7]

const norm = (lp: number[]) => { const m = Math.max(...lp), z = Math.log(lp.reduce((s, v) => s + Math.exp(v - m), 0)) + m; return lp.map((v) => v - z) }
/**
 * The RL loop, one row of probabilities per step. Each step moves log π a fraction of the way toward
 * log π_ref + r/β (the exact direction of the KL-regularised objective, as with a very large batch), so it
 * ends at the optimum π_ref · exp(r/β). The sampled token of each step only shows what the loop sees.
 */
function rlRun(beta: number) {
  const target = norm(REF.map((p, j) => Math.log(p) + REWARD[j] / beta)), eta = 0.3, r = rng(7)
  let lp = REF.map(Math.log)
  const probs: number[][] = [], samples: number[] = []
  for (let k = 0; k <= STEPS; k++) {
    const pr = lp.map(Math.exp)
    probs.push(pr)
    let u = r(), j = 0
    while (j < 5 && u > pr[j]) { u -= pr[j]; j++ }
    samples.push(j)
    lp = norm(lp.map((v, i) => (1 - eta) * v + eta * target[i]))
  }
  return { probs, samples }
}
/** DPO on the toy pairs: logits z, gradient steps on the mean pair loss (step size scaled by 1/β). */
function dpoRun(beta: number) {
  const z = REF.map(Math.log), eta = 0.35, lr = REF.map(Math.log)
  const probs: number[][] = [], losses: number[] = [], margins: number[][] = []
  for (let k = 0; k <= STEPS; k++) {
    const pz = norm(z)
    probs.push(pz.map(Math.exp))
    const d = TOY_PAIRS.map(([w, l]) => beta * ((z[w] - lr[w]) - (z[l] - lr[l])))
    margins.push(d); losses.push(d.reduce((s, m) => s + Math.log1p(Math.exp(-m)), 0) / d.length)
    TOY_PAIRS.forEach(([w, l], i) => { const g = eta * sig(-d[i]); z[w] += g; z[l] -= g })
  }
  return { probs, losses, margins, implicit: (k: number) => probs[k].map((p, j) => beta * Math.log(p / REF[j])) }
}

const COMPARE: Record<string, [string, string]> = {
  pairs: ['instruction tuning, the step before', 'training/sft?phase=before'],
  reward: ['the log-probabilities in the loss', 'training/loss?phase=loss'],
  rl: ['sampling from the policy', 'anatomy/unembed?phase=sample'],
  dpo: ['the log-probabilities in the loss', 'training/loss?phase=loss'],
  run: ['the adapters it trains', 'training/lora?phase=idea'],
  effect: ['GPT-2’s next-token distribution', 'anatomy/unembed?phase=softmax'],
}

const ratio = Math.exp(P[0].ref.rejected - P[0].ref.chosen)
const CAPS: Record<string, [string, string]> = {
  pairs: [`After instruction tuning, models are tuned on preferences: for one prompt, two answers and which one people preferred. Judging is easier than writing. In these three pairs GPT-2 small leans the other way: after “${P[0].prompt}” it finds “${P[0].rejected.trim()}” ${ratio.toFixed(0)} times as likely as “${P[0].chosen.trim()}”.`, 'prompt · chosen ≻ rejected'],
  reward: [`Classic RLHF first trains a reward model on such pairs: a network that reads a prompt and an answer and outputs one score, trained so that the preferred answer scores higher. Here it scores six possible next tokens after “${PROMPT}”; these scores are made up for the example.`, 'P(chosen ≻ rejected) = σ(r_chosen − r_rejected)'],
  rl: [`Then reinforcement learning: the model samples an answer, the reward model scores it, and probability moves toward what scores well. A penalty on drifting from the start (the outlined bars) holds it back, so it settles at π_ref · exp(r / β) rather than on the single best token. Change β to loosen or tighten that pull.`, 'max E[r] − β · KL(π ‖ π_ref)'],
  dpo: [`DPO drops the reward model and the sampling. Its loss looks at each pair directly: how much more likely the model makes the chosen answer than the reference does, against the same for the rejected one, and it pushes that gap open. β again sets how far it may go.`, '−log σ(β [Δ log π(chosen) − Δ log π(rejected)])'],
  run: [`A real run on GPT-2 small: rank-4 adapters, the frozen model as reference, β = ${BETA}, ${N} steps on the three pairs. Each answer sits at its log-probability; the gap opens until the loss falls from ${St[0].loss.toFixed(3)} to ${last.loss.toFixed(3)}. Most of the gap comes from the rejected answers falling.`, `loss ${St[0].loss.toFixed(3)} → ${last.loss.toFixed(3)}`],
  effect: [`Where the probability went, for the next token after “${P[0].prompt}”: GPT-2 spread it over floor, bed, couch and more; after ${N} steps nearly all of it is on “mat”. With three pairs the model simply memorized them. Real runs use many thousands of pairs, and β and the reference keep them from drifting this far.`, 'rejected pushed down more than chosen pulled up'],
}

export function mountDpo(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'A real DPO run on GPT-2 small, exported offline; the preference pairs are made up for it.',
      eyebrow: 'Training · After pretraining',
      title: 'RLHF & DPO',
      subtitle: 'learning which of two answers is better',
      specs: [
        { label: 'policy', value: 'GPT-2 small + LoRA', real: 'rank 4', realLabel: '' },
        { label: 'reference', value: 'GPT-2 small, frozen' },
        { label: 'β', value: String(BETA), real: `${N} steps`, realLabel: '' },
        { label: 'rewards', value: 'toy', real: 'RLHF steps', realLabel: '' },
      ],
    },
    size: [1040, 480],
    aria: 'Learning from preferences: pairs of chosen and rejected answers; a reward model scoring six next tokens; the RLHF loop moving GPT-2’s next-token probabilities toward high reward while a penalty holds them near the start; DPO doing the same from the pairs alone; and a real DPO run on GPT-2 small in which the rejected answers fall far.',
    phases: PHASES, learn: 'dpo', tokens: words(['chosen', 'rejected']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['rl', 13],
    hints: { rl: 'β in the controls: how hard the start pulls back.', dpo: 'β in the controls: how far DPO may move.' },
    scenes,
  })
}

function scenes({ stage, mk, k, player, onFrame }: Env) {
  const { title, caption, arrow } = k
  const pad = 36, top = 56

  // β for the toy loop and toy DPO, shown in those two phases only
  let bi = 1, RL = rlRun(BETAS[bi]), DPO = dpoRun(BETAS[bi])
  toggle(player.meta, 'β', BETAS.map((b) => `β ${b}`), bi, (i) => { bi = i; RL = rlRun(BETAS[i]); DPO = dpoRun(BETAS[i]) })
  const betaEl = player.meta.querySelector('.toggle') as HTMLElement
  onFrame(() => { const on = ['rl', 'dpo'].includes(player.cur().id); if (betaEl.hidden === on) betaEl.hidden = !on })

  const tokChip = (s: string, x: number, y: number, col: RGB | number, a: number, hl = false) =>
    drawChip(x, y, { text: s, c: typeof col === 'number' ? col : 0 }, a, 22, hl)
  /** A tick or a cross after an answer: chosen or rejected. */
  function mark(ok: boolean, x: number, y: number, a: number) {
    ctx.strokeStyle = rgba(ok ? chosen() : rejected(), a); ctx.lineWidth = 2; ctx.beginPath()
    if (ok) { ctx.moveTo(x, y); ctx.lineTo(x + 4, y + 4); ctx.lineTo(x + 11, y - 5) }
    else { ctx.moveTo(x, y - 5); ctx.lineTo(x + 9, y + 4); ctx.moveTo(x + 9, y - 5); ctx.lineTo(x, y + 4) }
    ctx.stroke()
  }
  /** The prompt as plain text, then the six candidates as rows: chip, bar of π with π_ref outlined, percentage. */
  function distBars(x: number, y: number, w: number, pr: number[], a: number, o: { note?: (j: number) => string; hl?: number } = {}) {
    const rowH = 36, lw = 92, bx = x + lw, bw = w - lw - 64
    runs([[PROMPT + ' …', C.ink2]], x, y - 30, a, F.mono(12))
    CANDS.forEach((s, j) => {
      const yy = y + j * rowH, on = o.hl === j
      drawChip(x, yy, { text: s, c: CAT.texts.length + j }, a, 22, on)
      ctx.fillStyle = rgba(C.ink, 0.06 * a); ctx.fillRect(bx, yy - 5, bw, 10)
      ctx.fillStyle = rgba(candHue(j), 0.9 * a); ctx.fillRect(bx, yy - 5, Math.max(1, pr[j] * bw), 10)
      ctx.strokeStyle = rgba(C.ink, 0.7 * a); ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, yy - 7.5, Math.max(1, REF[j] * bw), 15)
      ctx.font = F.mono(11.5, 500); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a)
      ctx.fillText(pct(pr[j]), x + w, yy)
      const n = o.note?.(j)
      if (n) { ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, a); raw(() => ctx.fillText(n, x + w + 14, yy)) }
    })
    ctx.strokeStyle = rgba(C.ink, 0.7 * a); ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, y + 6 * rowH - 12, 14, 9)
    caption('outlined: the start, π_ref (GPT-2)', bx + 22, y + 6 * rowH - 4, a, C.mute, 'left')
    return { rowH, bx, bw }
  }
  const signed = (v: number) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1)

  /* ---------- 1: pairs ---------- */
  function scenePairs(p: number) {
    const { W } = stage, y0 = top + 60, x1 = pad + 330, sw = Math.min(360, W - x1 - 300)
    title('prompt · two answers · which one a person preferred', pad, top + 6, 1)
    title('GPT-2 small between the two', x1 + 250, top + 6, eout(clamp((p - 0.3) / 0.1)))
    P.forEach((q, i) => {
      const a = eout(clamp((p - 0.04 - i * 0.12) / 0.12)), y = y0 + i * 104
      if (a <= 0) return
      runs([[q.prompt, C.ink2]], pad, y + 16, a, F.mono(12.5))
      tokChip(q.chosen, x1, y, 4, a, true); mark(true, x1 + chipW(q.chosen) + 10, y, a)
      tokChip(q.rejected, x1, y + 32, 2, a); mark(false, x1 + chipW(q.rejected) + 10, y + 32, a)
      caption('chosen', x1 + 150, y + 4, a, chosen(), 'left')
      caption('rejected', x1 + 150, y + 36, a, rejected(), 'left')
      // GPT-2's own split between the two answers: the chosen share and the rejected share
      const ga = eout(clamp((p - 0.3 - i * 0.1) / 0.12)), sc = Math.exp(q.ref.chosen), sr = Math.exp(q.ref.rejected), fc = sc / (sc + sr)
      if (ga <= 0) return
      const bx = x1 + 250, by = y + 10
      ctx.fillStyle = rgba(chosen(), 0.85 * ga); ctx.fillRect(bx, by, sw * fc, 12)
      ctx.fillStyle = rgba(rejected(), 0.85 * ga); ctx.fillRect(bx + sw * fc, by, sw * (1 - fc), 12)
      ctx.font = F.small; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, ga)
      ctx.textAlign = 'left'; ctx.fillText(pct(fc), bx, by + 26)
      ctx.textAlign = 'right'; ctx.fillText(pct(1 - fc), bx + sw, by + 26)
    })
    mk.formula = { segs: [['GPT-2', C.ink2], [': ', C.mute], [`p(“${P[0].rejected.trim()}”) / p(“${P[0].chosen.trim()}”)`, C.ink2], [' = ', C.mute], [ratio.toFixed(1), C.ink]], note: 'In every pair GPT-2 leans toward the rejected answer. Preference tuning has to turn that around. Real preference data comes from people, or from a stronger model, comparing two samples.' }
  }

  /* ---------- 2: the reward model ---------- */
  function sceneReward(p: number) {
    const { W } = stage, y0 = top + 64, rowH = 40, xc = pad, xm = pad + 300, xs = xm + 90, sw = Math.min(300, W - xs - 140), zero = xs + sw / 2
    title('six possible next tokens', xc, top + 6, 1)
    title('reward model r(x, y)', xm - 6, top + 6, 1, 'center')
    title('score · toy', xs, top + 6, 1)
    ctx.strokeStyle = rgba(C.faint, 1); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(zero + 0.5, y0 - 16); ctx.lineTo(zero + 0.5, y0 + 5 * rowH + 16); ctx.stroke()
    CANDS.forEach((s, j) => {
      const s0 = 0.05 + j * 0.1, f = eio(clamp((p - s0) / 0.1)), g = eout(clamp((p - s0 - 0.08) / 0.08)), y = y0 + j * rowH
      drawChip(xc, y, { text: s, c: CAT.texts.length + j }, 1, 22)
      // the answer goes through the reward model and comes out as one number
      k.glass(xm - 14, y - 14, y + 14, f * (1 - g) + 0.15, { w: 5, d: 6 }, 1)
      k.glass(xm + 2, y - 14, y + 14, f * (1 - g) + 0.15, { w: 5, d: 6 }, 1)
      if (f > 0 && g < 1) { const x = lerp(xc + chipW(s) + 8, xm - 24, f); ctx.fillStyle = rgba(candHue(j), 0.9 * (1 - g)); ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 7); ctx.fill() }
      if (g <= 0) return
      // filled: positive, outlined: negative
      const r = REWARD[j], w = (r / 3) * (sw / 2) * g
      ctx.fillStyle = rgba(candHue(j), 0.85); ctx.strokeStyle = rgba(candHue(j), 0.95); ctx.lineWidth = 1.2
      if (r >= 0) ctx.fillRect(zero, y - 6, w, 12)
      else ctx.strokeRect(zero + w + 0.5, y - 5.5, -w - 1, 11)
      ctx.font = F.mono(11.5, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, g)
      ctx.fillText(signed(r), xs + sw + 12, y)
    })
    const pa = eout(clamp((p - 0.72) / 0.1)), py = y0 + 6 * rowH + 20
    if (pa > 0) {
      runs([['P(', C.ink2]], pad, py, pa, F.mono(12))
      let x = pad + 16
      x += drawChip(x, py, { text: CANDS[5], c: CAT.texts.length + 5 }, pa, 22, true) + 6
      runs([['≻', C.ink2]], x, py, pa, F.mono(12)); x += 16
      x += drawChip(x, py, { text: CANDS[0], c: CAT.texts.length }, pa, 22) + 6
      runs([[`) = σ(${signed(REWARD[5])} − (${signed(REWARD[0])})) = ${pct(sig(REWARD[5] - REWARD[0]))}`, C.ink]], x, py, pa, F.mono(12))
      caption('the reward model agrees with the person, as it was trained to', pad, py + 26, pa, C.mute, 'left')
    }
    mk.formula = { segs: [['loss', C.ink2], [' = ', C.mute], ['−log σ(r(x, y_chosen) − r(x, y_rejected))', C.ink]], note: 'Trained on many pairs, the reward model gives every answer a score, including answers no one compared. InstructGPT’s was a 6B-parameter copy of the model with its output replaced by one number.' }
  }

  /* ---------- 3: the RL loop ---------- */
  function sceneRl(p: number) {
    const { W } = stage, x0 = pad + 150, y0 = top + 80, w = Math.min(470, W - x0 - 330), xr = W - pad - 110, yc = y0 + 90
    const pos = clamp((p - 0.06) / 0.86) * STEPS, kk = Math.min(STEPS - 1, Math.floor(pos)), u = pos >= STEPS ? 1 : pos - kk, done = pos >= STEPS
    const tw = done ? 1 : eio(clamp((u - 0.55) / 0.4)), pr = RL.probs[kk].map((v, j) => lerp(v, RL.probs[kk + 1][j], tw))
    const smp = RL.samples[kk]
    modelGlyph(k, pad + 40, yc, 'GPT-2', 'the policy π', 1, done ? 0.2 : 0.5)
    const g = distBars(x0, y0, w, pr, 1, { hl: done ? undefined : smp, note: (j) => `r ${signed(REWARD[j])}` })
    modelGlyph(k, xr, yc, 'reward model', 'toy scores', 1, !done && u > 0.3 && u < 0.6 ? 0.8 : 0.2)
    // this step's sample: out of the policy, into the reward model, back as a score
    if (!done) {
      const f = eio(clamp(u / 0.35)), sy = y0 + smp * g.rowH, sx = g.bx + pr[smp] * g.bw + 10
      const x = lerp(sx, xr - 44, f), y = lerp(sy, yc, f) - Math.sin(Math.PI * f) * 30
      if (u < 0.4) { ctx.fillStyle = rgba(candHue(smp), 0.95); ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fill() }
      const sa = eout(clamp((u - 0.35) / 0.1)) * (1 - eout(clamp((u - 0.85) / 0.15)))
      if (sa > 0) {
        ctx.font = F.mono(13, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(candHue(smp), sa)
        raw(() => ctx.fillText(`${CANDS[smp].trim()} → ${signed(REWARD[smp])}`, xr, yc - 60))
        arrow([[xr - 50, yc - 40], [g.bx + g.bw + 60, y0 + smp * g.rowH - 12]], sa * 0.8, true)
      }
    }
    const er = pr.reduce((s, v, j) => s + v * REWARD[j], 0), kl = pr.reduce((s, v, j) => s + (v > 0 ? v * Math.log(v / REF[j]) : 0), 0)
    title(tf('step {} of {}', done ? STEPS : kk + 1, STEPS), x0, top + 6, 1)
    mk.formula = { segs: [['E[r]', C.ink2], [' = ', C.mute], [er.toFixed(2), C.ink], ['   KL(π ‖ π_ref)', C.ink2], [' = ', C.mute], [kl.toFixed(2), C.ink], ['   β', C.ink2], [' = ', C.mute], [String(BETAS[bi]), C.ink]], note: 'Each step here moves the policy along the exact expected direction, as with a very large batch; PPO estimates that direction from the answers it samples. It needs the policy, a frozen reference, the reward model and a value model in memory at once.' }
  }

  /* ---------- 4: DPO, from the pairs alone ---------- */
  function sceneDpo(p: number) {
    const { W } = stage, x0 = pad + 20, y0 = top + 80, w = Math.min(470, W - x0 - 420)
    const pos = clamp((p - 0.08) / 0.8) * STEPS, kk = Math.min(STEPS - 1, Math.floor(pos)), f = pos >= STEPS ? 1 : eio(pos - kk)
    const pr = DPO.probs[kk].map((v, j) => lerp(v, DPO.probs[kk + 1][j], f)), step = Math.round(pos)
    const imp = DPO.implicit(step)
    distBars(x0, y0, w, pr, 1, { note: (j) => `β log π/π_ref ${signed(imp[j])}` })
    // the pairs, and how sure the model now is of each
    const xp = W - pad - 270
    title('the pairs · σ(margin)', xp, top + 6, 1)
    TOY_PAIRS.forEach(([wi, li], i) => {
      const y = y0 + i * 60, m = DPO.margins[step][i]
      let x = xp
      x += drawChip(x, y, { text: CANDS[wi], c: CAT.texts.length + wi }, 1, 22, true) + 6
      runs([['≻', C.ink2]], x, y, 1, F.mono(12)); x += 16
      drawChip(x, y, { text: CANDS[li], c: CAT.texts.length + li }, 1, 22)
      ctx.fillStyle = rgba(C.ink, 0.06); ctx.fillRect(xp, y + 20, 200, 6)
      ctx.fillStyle = rgba(chosen(), 0.9); ctx.fillRect(xp, y + 20, 200 * sig(m), 6)
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, 1); ctx.fillText(pct(sig(m)), xp + 208, y + 23)
    })
    const ly = y0 + 3 * 60 + 10
    ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, 1)
    ctx.fillText(`loss ${DPO.losses[step].toFixed(3)}`, xp, ly)
    caption(tf('step {} of {} · no reward model, no sampling', step, STEPS), xp, ly + 22, 1, C.mute, 'left')
    mk.formula = { segs: [['m', C.ink], [' = ', C.mute], ['β [log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)]', C.ink2], ['   L = −log σ(m)', C.ink]], note: 'β log π/π_ref acts as the reward the model implies: DPO trains the policy so that this implicit reward orders each pair the way people did. At step 0 every margin is 0 and the loss is ln 2 ≈ 0.693. Rafailov et al. (2023).' }
  }

  /* ---------- 5: the real run ---------- */
  function sceneRun(p: number) {
    const { W } = stage, x0 = pad + 40, x1 = W - pad - 250, lo = -70, X = (v: number) => lerp(x0, x1, (v - lo) / (0 - lo))
    const pos = clamp((p - 0.06) / 0.8) * N, s0 = Math.min(N - 1, Math.floor(pos)), f = pos >= N ? 1 : eio(pos - s0), step = Math.round(pos)
    const at = (i: number, key: 'chosen' | 'rejected') => lerp(St[s0].pairs[i][key], St[s0 + 1].pairs[i][key], f)
    title('log π(answer | prompt) · each −2.3 is 10 times less likely', x0, top + 6, 1)
    P.forEach((q, i) => {
      const y = top + 58 + i * 108, ya = y + 38
      runs([[q.prompt + ' …', C.ink2]], pad, y - 6, 1, F.mono(12))
      ctx.strokeStyle = rgba(C.faint, 1); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, ya + 0.5); ctx.lineTo(x1, ya + 0.5); ctx.stroke()
      for (let v = lo; v <= 0; v += 10) { ctx.beginPath(); ctx.moveTo(X(v) + 0.5, ya - 3); ctx.lineTo(X(v) + 0.5, ya + 3); ctx.stroke() }
      if (i === 2) { ctx.font = F.small; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, 1); for (let v = lo; v <= 0; v += 10) ctx.fillText(String(v), X(v), ya + 36) }
      const c = at(i, 'chosen'), r = at(i, 'rejected')
      // the answers ride on their log-probabilities; the start is a hollow tick
      for (const [v0, v, ans, col, up] of [[q.ref.chosen, c, q.chosen, 4, true], [q.ref.rejected, r, q.rejected, 2, false]] as const) {
        ctx.strokeStyle = rgba(C.tok[col], 0.8); ctx.beginPath(); ctx.arc(X(v0), ya, 3, 0, 7); ctx.stroke()
        const cx = X(v), w = chipW(ans), cy = up ? ya - 16 : ya + 16
        ctx.strokeStyle = rgba(C.tok[col], 0.6); ctx.beginPath(); ctx.moveTo(cx + 0.5, ya); ctx.lineTo(cx + 0.5, cy + (up ? 11 : -11)); ctx.stroke()
        drawChip(Math.max(x0 - 30, cx - w / 2), cy, { text: ans, c: col }, 1, 20, up)
      }
      const m = St[step].pairs[i].margin
      ctx.font = F.mono(11.5, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, 1)
      ctx.fillText(`margin ${m.toFixed(2)}`, x1 + 30, ya - 8)
      caption(`P(chosen ≻ rejected) ${pct(sig(m))}`, x1 + 30, ya + 14, 1, C.mute, 'left')
    })
    mk.formula = { segs: [['step', C.ink2], [` ${step}`, C.ink], ['   loss ', C.ink2], [St[step].loss.toFixed(4), C.ink], ['   margin = β × (gap now − gap at the start)', C.mute]], note: 'The chosen answers barely move; the rejected ones fall by tens of nats. Lowering the rejected answer is the cheapest way to widen the gap, a known DPO behaviour. Adam, learning rate 0.002, rank-4 adapters on W_qkv.' }
  }

  /* ---------- 6: the next token, before and after ---------- */
  function sceneEffect(p: number) {
    const { W } = stage, y0 = top + 86, rowH = 34, colW = Math.min(360, (W - 2 * pad - 180) / 2), xa = pad + 110, xb = xa + colW + 70
    const after = (s: string) => { const r = Dp.after.find(([tk]) => tk === 'Ġ' + s.trim()); return r ? r[1] : null }
    runs([[P[0].prompt + ' …', C.ink2]], pad, top + 10, 1, F.mono(12.5))
    title('GPT-2, before', xa, y0 - 30, 1)
    const ba = eout(clamp((p - 0.3) / 0.12))
    title(tf('after {} steps of DPO', N), xb, y0 - 30, ba)
    CANDS.forEach((s, j) => {
      const y = y0 + j * rowH, on = s === ' mat'
      drawChip(pad, y, { text: s, c: CAT.texts.length + j }, 1, 22, on)
      const bar = (x: number, v: number | null, a: number) => {
        ctx.fillStyle = rgba(C.ink, 0.06 * a); ctx.fillRect(x, y - 5, colW - 60, 10)
        if (v !== null) { ctx.fillStyle = rgba(candHue(j), 0.9 * a); ctx.fillRect(x, y - 5, Math.max(1, v * (colW - 60)), 10) }
        ctx.font = F.mono(11.5, 500); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a)
        ctx.fillText(v === null ? '<0.1%' : pct(v), x + colW, y)
      }
      bar(xa, GPT[j], 1)
      bar(xb, after(s) ?? null, ba * eout(clamp((p - 0.36 - j * 0.03) / 0.12)))
    })
    const na = eout(clamp((p - 0.6) / 0.12))
    caption(tf('“mat” was number {} for GPT-2; the others are no longer in the top {}', DIST.findIndex((r) => r.text === ' mat') + 1, Dp.after.length), pad, y0 + 6 * rowH + 10, na, C.mute, 'left')
    mk.formula = { segs: [['p(“mat”)', C.ink2], [' ', C.mute], [pct(GPT[5]), C.ink], [' → ', C.mute], [pct(after(' mat') ?? 0), C.ink], ['   p(“floor”)', C.ink2], [' ', C.mute], [pct(GPT[0]), C.ink], [' → ', C.mute], ['<0.1%', C.ink]], note: 'Three pairs and 25 steps: the model has memorized its preferences. With thousands of varied pairs and the reference’s pull, the same loss makes a model prefer better answers without forgetting everything else.' }
  }

  return { pairs: scenePairs, reward: sceneReward, rl: sceneRl, dpo: sceneDpo, run: sceneRun, effect: sceneEffect }
}
