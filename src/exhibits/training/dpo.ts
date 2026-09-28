import { F, ctx, drawChip, mathName, rr } from '../../core/draw'
import { C, rgba } from '../../core/theme'
import { clamp, eout, lerp } from '../../core/util'
import { ft } from '../../lib/finetune/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * Learning from preferences: pairs of answers where one is preferred, RLHF's reward model and PPO, and DPO, which gets
 * the same objective from one loss. A real DPO run on GPT-2 small with rank-4 adapters and the frozen model as
 * reference (scripts/finetune-export.ts).
 */

const PHASES = [
  { id: 'pairs', name: 'Preferences, not answers', short: 'Pairs', dur: 9 },
  { id: 'rlhf', name: 'RLHF: a reward model, then PPO', short: 'RLHF', dur: 11 },
  { id: 'dpo', name: 'DPO: one loss instead', short: 'DPO', dur: 11 },
  { id: 'run', name: 'A real DPO run', short: 'Train', dur: 11 },
  { id: 'where', name: 'Where the probability went', short: 'Effect', dur: 11 },
]

const Dp = ft.dpo, P = Dp.pairs, St = Dp.steps, N = St.length - 1, BETA = Dp.beta
const last = St[N]
const pct = (v: number) => (v < 0.001 ? '<0.1%' : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`)
const sig = (x: number) => 1 / (1 + Math.exp(-x))

const COMPARE: Record<string, [string, string]> = {
  pairs: ['instruction tuning, the step before', 'training/sft?phase=before'],
  rlhf: ['sampling from the policy', 'anatomy/unembed?phase=sample'],
  dpo: ['the log-probabilities in the loss', 'training/loss?phase=loss'],
  run: ['the adapters it trains', 'training/lora?phase=idea'],
  where: ['GPT-2’s next-token distribution', 'anatomy/unembed?phase=softmax'],
}

const CAPS: Record<string, [string, string]> = {
  pairs: [`After instruction tuning, models are tuned on preferences: for one prompt, two answers and which one people preferred. It is easier to judge than to write. These three toy pairs ask GPT-2 small to prefer the answer it finds less likely.`, 'prompt · chosen ≻ rejected'],
  rlhf: [`Classic RLHF first trains a reward model to score answers so that preferred ones score higher, then tunes the model with reinforcement learning (PPO) to write high-scoring answers, with a penalty for drifting too far from where it started.`, 'max E[r(x, y)] − β KL(π ‖ π_ref)'],
  dpo: [`DPO reaches the same optimum without a reward model or sampling: it raises the chosen answer’s probability relative to the reference model and lowers the rejected one’s, through one loss on log-probabilities. At the start the two models agree, so every margin is 0 and the loss is ln 2.`, '−log σ(β [Δ log π(chosen) − Δ log π(rejected)])'],
  run: [`A real run: rank-4 adapters on GPT-2 small, the frozen model as reference, β = ${BETA}, ${N} steps. The loss falls from ${St[0].loss.toFixed(3)} to ${last.loss.toFixed(3)} as each pair’s margin grows.`, `loss ${St[0].loss.toFixed(3)} → ${last.loss.toFixed(3)}`],
  where: [`The margins grew mostly by pushing the rejected answers down: “${P[0].rejected.trim()}” fell from ${P[0].ref.rejected.toFixed(1)} to ${last.pairs[0].rejected.toFixed(1)} in log-probability, while “${P[0].chosen.trim()}” moved from ${P[0].ref.chosen.toFixed(1)} to ${last.pairs[0].chosen.toFixed(1)}. With three pairs it simply memorized them.`, 'rejected pushed down more than chosen pulled up'],
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
      ],
    },
    size: [1040, 480],
    aria: 'Learning from preferences: pairs of chosen and rejected answers, RLHF with a reward model and PPO, DPO’s single loss, and a real DPO run on GPT-2 small in which the rejected answers’ probabilities fall far while the chosen ones rise a little.',
    phases: PHASES, learn: 'dpo', tokens: words(['chosen', 'rejected']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['run', 10.5],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption, arrow } = k
  const pad = 36, top = 56

  /* ---------- 1: pairs ---------- */
  function scenePairs(p: number) {
    const y0 = top + 50
    title('prompt · chosen · rejected · log p under GPT-2 small', pad, y0 - 22, 1)
    P.forEach((q, i) => {
      const a = eout(clamp((p - 0.05 - i * 0.15) / 0.15)), y = y0 + 34 + i * 100
      ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(q.prompt, pad, y)
      const x = pad + 300
      drawChip(x, y - 14, { text: q.chosen, c: 3 }, a, 22, true)
      ctx.font = F.mono(11); ctx.fillStyle = rgba(C.ink, a); ctx.fillText(`chosen  ${q.ref.chosen.toFixed(2)}`, x + 110, y - 14)
      drawChip(x, y + 16, { text: q.rejected, c: 1 }, a, 22)
      ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(`rejected  ${q.ref.rejected.toFixed(2)}`, x + 110, y + 16)
    })
    mk.formula = { segs: [['log p(answer | prompt)', C.ink], [' = ', C.mute], ['Σ log p of its tokens', C.ink2]], note: 'In every pair GPT-2 prefers the rejected answer to start with (a higher, less negative log-probability). Real preference data comes from people or from a stronger model comparing samples.' }
  }

  /* ---------- 2: RLHF ---------- */
  function sceneRlhf(p: number) {
    const { W } = stage, y0 = top + 60, bw = 190, gap = (W - 2 * pad - 4 * bw) / 3
    const boxes: [string, string][] = [['preference pairs', 'x, y_w ≻ y_l'], ['reward model r(x, y)', 'σ(r_w − r_l) → 1'], ['policy π samples y', 'from the tuned model'], ['PPO update', 'raise r, keep near π_ref']]
    boxes.forEach(([a, b], i) => {
      const al = eout(clamp((p - 0.05 - i * 0.15) / 0.12)), x = pad + i * (bw + gap)
      rr(x, y0, bw, 64, 8); ctx.fillStyle = rgba(C.ink, 0.05 * al); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.5 * al); ctx.lineWidth = 1; ctx.stroke()
      ctx.font = F.mono(12, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, al); ctx.fillText(a, x + bw / 2, y0 + 26)
      ctx.font = F.small; ctx.fillStyle = rgba(C.mute, al); ctx.fillText(b, x + bw / 2, y0 + 46)
      if (i < 3) arrow([[x + bw + 4, y0 + 32], [x + bw + gap - 4, y0 + 32]], al)
    })
    const la = eout(clamp((p - 0.7) / 0.1)), xl = pad + 3 * (bw + gap) + bw / 2, xr = pad + 2 * (bw + gap) + bw / 2
    arrow([[xl, y0 + 68], [xl, y0 + 100], [xr, y0 + 100], [xr, y0 + 70]], la, true)
    caption('repeat: sample, score, update', (xl + xr) / 2, y0 + 118, la, C.mute)
    const ta = eout(clamp((p - 0.45) / 0.12))
    mathName('P(y_w ≻ y_l) = σ(r(x, y_w) − r(x, y_l))', pad, y0 + 190, ta, 20)
    caption('Bradley–Terry: the reward model is trained so this is high on the preference data', pad, y0 + 216, ta, C.ink2, 'left')
    mk.formula = { segs: [['objective', C.ink2], ['  ', C.mute], ['E[r(x, y)] − β · KL(π(·|x) ‖ π_ref(·|x))', C.ink]], note: 'InstructGPT (Ouyang et al. 2022) used this pipeline. It needs four models in memory (policy, reference, reward, value) and sampling in the loop, which makes it slow and finicky.' }
  }

  /* ---------- 3: the DPO loss ---------- */
  function sceneDpo(p: number) {
    const y0 = top + 70, a1 = eout(clamp(p / 0.15)), a2 = eout(clamp((p - 0.3) / 0.15)), a3 = eout(clamp((p - 0.6) / 0.15))
    mathName('r(x, y) = β log π(y|x) / π_ref(y|x)', pad, y0, a1, 22)
    caption('the reward DPO implies: how much more likely the tuned model makes y than the reference does', pad, y0 + 26, a1, C.ink2, 'left')
    mathName('L = −log σ(r(x, y_w) − r(x, y_l))', pad, y0 + 90, a2, 22)
    caption('the reward model’s Bradley–Terry loss, with that implicit reward plugged in', pad, y0 + 116, a2, C.ink2, 'left')
    // step 0, for real
    const x0 = pad, y1 = y0 + 190
    title('step 0 of the run', x0, y1 - 14, a3)
    P.forEach((q, i) => {
      const m = St[0].pairs[i].margin
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, a3)
      ctx.fillText(`${q.prompt}${q.chosen} ≻ …${q.rejected}   margin ${m.toFixed(2)}   loss ${Math.log1p(Math.exp(-m)).toFixed(3)}`, x0, y1 + 12 + i * 22)
    })
    mk.formula = { segs: [['∂L/∂θ', C.ink], [' = ', C.mute], ['−β σ(−m) [∇ log π(y_w) − ∇ log π(y_l)]', C.ink2]], note: 'The gradient is two backward passes (one per answer) weighted by σ(−m): pairs the model already gets right (large margin m) stop contributing. Rafailov et al. (2023).' }
  }

  /* ---------- 4: the run ---------- */
  function sceneRun(p: number) {
    const { H } = stage, x0 = pad + 60, x1 = pad + 520, y0 = top + 40, y1 = H - 90
    const g = Math.floor(clamp((p - 0.05) / 0.7) * N), X = (s: number) => lerp(x0, x1, s / N)
    const Yl = (v: number) => lerp(y1, y0, v / 0.7)
    title('DPO loss', x0, y0 - 16, 1)
    ctx.strokeStyle = rgba(C.faint, 1); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke()
    ctx.strokeStyle = rgba(C.ink, 1); ctx.lineWidth = 2; ctx.beginPath()
    for (let s = 0; s <= g; s++) { if (s) ctx.lineTo(X(s), Yl(St[s].loss)); else ctx.moveTo(X(s), Yl(St[s].loss)) }
    ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText('ln 2', x0 - 6, Yl(Math.LN2)); ctx.fillText('0', x0 - 6, y1)
    ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(`${N} steps`, x1, y1 + 6)
    // margins per pair
    const xm = x1 + 70
    title('margin per pair · P(chosen ≻ rejected)', xm, y0 - 16, 1)
    P.forEach((q, i) => {
      const m = St[g].pairs[i].margin, y = y0 + 20 + i * 62
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.tok[i], 1); ctx.fillText(`${q.prompt} …`, xm, y)
      ctx.fillStyle = rgba(C.ink, 0.07); ctx.fillRect(xm, y + 12, 260, 8)
      ctx.fillStyle = rgba(C.tok[i], 0.9); ctx.fillRect(xm, y + 12, (m / 7) * 260, 8)
      ctx.fillStyle = rgba(C.ink, 1); ctx.fillText(`${m.toFixed(2)} · ${pct(sig(m))}`, xm + 270, y + 16)
    })
    mk.formula = { segs: [['step', C.ink2], [` ${g}`, C.ink], ['   loss ', C.ink2], [St[g].loss.toFixed(4), C.ink]], note: 'The margin m is β times the gap in log-ratio; σ(m) is the preference probability the implicit reward model assigns. Adam, learning rate 0.002, rank-4 adapters on W_qkv.' }
  }

  /* ---------- 5: where the probability went ---------- */
  function sceneWhere(p: number) {
    const { W } = stage, x0 = pad + 20, y0 = top + 50, colW = (W - 2 * pad) / 2 - 20
    title('log π(answer) per step · chosen and rejected', x0, y0 - 18, 1)
    const g = Math.floor(clamp((p - 0.05) / 0.6) * N), lo = -70, X = (s: number) => lerp(x0 + 40, x0 + colW, s / N), Y = (v: number) => lerp(y0 + 250, y0, (v - lo) / (0 - lo))
    ctx.strokeStyle = rgba(C.faint, 1); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 + 40, y0); ctx.lineTo(x0 + 40, y0 + 250); ctx.lineTo(x0 + colW, y0 + 250); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; for (const v of [0, -35, -70]) ctx.fillText(String(v), x0 + 34, Y(v))
    P.forEach((_, i) => {
      for (const [key, dash] of [['chosen', false], ['rejected', true]] as const) {
        ctx.strokeStyle = rgba(C.tok[i], 1); ctx.lineWidth = 1.8; if (dash) ctx.setLineDash([4, 3]); ctx.beginPath()
        for (let s = 0; s <= g; s++) { const v = St[s].pairs[i][key]; if (s) ctx.lineTo(X(s), Y(v)); else ctx.moveTo(X(s), Y(v)) }
        ctx.stroke(); ctx.setLineDash([])
      }
    })
    caption('solid: chosen · dashed: rejected', x0 + 40, y0 + 276, 1, C.mute, 'left')
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(`step ${N}`, x0 + colW, y0 + 256)
    P.forEach((q, i) => { ctx.fillStyle = rgba(C.tok[i], 1); ctx.fillRect(x0 + 40, y0 + 296 + i * 18, 14, 3); caption(`${q.prompt} … ${q.chosen.trim()} / ${q.rejected.trim()}`, x0 + 62, y0 + 301 + i * 18, 1, C.tok[i], 'left') })
    // the next token after the first prompt
    const ta = eout(clamp((p - 0.6) / 0.12)), xr = x0 + colW + 60
    title(`next token after “${P[0].prompt}”`, xr, y0 - 18, ta)
    const col = (items: [string, number][], x: number, name: string) => {
      caption(name, x, y0 + 4, ta, C.ink2, 'left')
      items.slice(0, 4).forEach(([t, v], j) => { ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, ta); ctx.fillText(`${t.replace(/^Ġ/, ' ')}  ${pct(v)}`, x, y0 + 30 + j * 22) })
    }
    col(Dp.before, xr, 'before')
    col(Dp.after, xr + 170, 'after')
    mk.formula = { segs: [['chosen', C.ink2], [` ${P[0].ref.chosen.toFixed(1)} → ${last.pairs[0].chosen.toFixed(1)}`, C.ink], ['   rejected', C.ink2], [` ${P[0].ref.rejected.toFixed(1)} → ${last.pairs[0].rejected.toFixed(1)}`, C.ink]], note: 'Lowering the rejected answer is the cheapest way to widen the gap, a known DPO behaviour. β and the reference keep real runs from drifting this far; three pairs and 25 steps do not.' }
  }

  return { pairs: scenePairs, rlhf: sceneRlhf, dpo: sceneDpo, run: sceneRun, where: sceneWhere }
}
