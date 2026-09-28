import { F, ctx, mathName } from '../../core/draw'
import { gemm, type M, type Rect } from '../../core/matrix'
import { C, rgba } from '../../core/theme'
import { clamp, eout, lerp } from '../../core/util'
import { ft, qwenDelta as QD } from '../../lib/finetune/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * LoRA: keep the pretrained weights frozen and learn a low-rank change ΔW = A·B beside them. A real rank-4 run on
 * GPT-2 small against full fine-tuning (scripts/finetune-export.ts), and how low-rank a real fine-tuning change is:
 * that full fine-tune's, and Qwen3-1.7B's post-training (scripts/lora-export.ts).
 */

const PHASES = [
  { id: 'idea', name: 'Freeze W, learn A · B', short: 'W + AB', dur: 11 },
  { id: 'count', name: 'How few weights train', short: 'Params', dur: 9 },
  { id: 'train', name: 'A real run', short: 'Train', dur: 11 },
  { id: 'rank', name: 'Is a real change low-rank?', short: 'Rank', dur: 11 },
  { id: 'merge', name: 'Merge it, or swap it', short: 'Serve', dur: 8 },
]

const Lo = ft.lora, R = Lo.rank
/* the idea, drawn at toy size with made-up numbers: W 6 × 8, A 6 × 2, B 2 × 8 */
const rnd = (() => { let s = 7; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5 } })()
const TW: M = Array.from({ length: 6 }, () => Array.from({ length: 8 }, () => +(rnd() * 1.6).toFixed(2)))
const TA: M = Array.from({ length: 6 }, () => Array.from({ length: 2 }, () => +(rnd() * 1.4).toFixed(2)))
const TB: M = Array.from({ length: 2 }, () => Array.from({ length: 8 }, () => +(rnd() * 1.4).toFixed(2)))
const TD: M = TA.map((a) => TB[0].map((_, j) => +(a[0] * TB[0][j] + a[1] * TB[1][j]).toFixed(3)))
const pct = (v: number) => (v === 0 ? '0%' : `${(v * 100).toFixed(v < 0.01 ? 2 : v < 0.1 ? 1 : 0)}%`)
/** LLaMA 3 8B with rank-16 adapters on W_q, W_k, W_v and W_o of all 32 layers. */
const LLAMA_LORA = 32 * 16 * (4096 + 4096 + (4096 + 1024) * 2 + 4096 + 4096), LLAMA = 8.03e9

const COMPARE: Record<string, [string, string]> = {
  idea: ['the matrix product itself', 'foundations'],
  count: ['what full training keeps per weight', 'training/optimizer?phase=memory'],
  train: ['the backward pass that trains it', 'training/backprop?phase=linear'],
  rank: ['instruction tuning, the big change', 'training/sft?phase=before'],
  merge: ['batching many requests', 'serving/continuous-batching?phase=why'],
}

const CAPS: Record<string, [string, string]> = {
  idea: [`Fine-tuning changes a weight matrix by some ΔW. LoRA keeps W frozen and learns ΔW as a product A · B of two thin matrices, with r columns between them. B starts at zero, so training starts from the unchanged model.`, `W′ = W + A · B · rank r`],
  count: [`Here: rank ${R} on the attention input matrix of each of GPT-2’s 12 blocks, ${Lo.run.trainable.toLocaleString('en-US')} trainable weights, ${pct(Lo.run.trainable / Lo.full.trainable)} of the model. Only those need gradients and optimizer state.`, `${Lo.run.trainable.toLocaleString('en-US')} of ${(Lo.full.trainable / 1e6).toFixed(0)}M weights`],
  train: [`A real run: teach GPT-2 small that things sit “on the moon”, from two sentences. Both LoRA and full fine-tuning drive the loss near zero; LoRA takes more steps. On an unseen sentence, “${Lo.eval} … moon” goes from ${pct(Lo.run.before.eval)} to ${pct(Lo.run.after.eval)}: it learned the pattern, and overdid it.`, `loss ${Lo.run.curve[0].toFixed(2)} → ${Lo.run.curve[Lo.steps].toFixed(2)} (LoRA) · ${Lo.full.curve[Lo.steps].toFixed(2)} (full)`],
  rank: [`LoRA bets that the change a fine-tune needs is low-rank. For this narrow task it is: rank 4 holds ${pct(Lo.full.rank.captured[2])} of the full fine-tune’s change to one matrix. Qwen3’s broad post-training is not: rank 16 holds only ${pct(QD.mats[0].delta.captured[4])} of its change to a query matrix.`, 'narrow tasks: low rank · broad training: not'],
  merge: [`After training, A · B can be added into W once, so the model runs exactly as fast as before. Or the adapters stay separate: one frozen base can serve many small adapters, one per customer or task, swapped per request.`, 'W′ = W + A·B, or many A·B on one W'],
}

export function mountLora(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'The runs are real (GPT-2 small fine-tuned offline; Qwen3’s weights compared directly); the first picture uses toy numbers.',
      eyebrow: 'Training · After pretraining',
      title: 'LoRA',
      subtitle: 'low-rank adapters beside frozen weights',
      specs: [
        { label: 'here', value: `rank ${R}`, real: 'W_qkv × 12 blocks', realLabel: '' },
        { label: 'trainable', value: Lo.run.trainable.toLocaleString('en-US'), real: pct(Lo.run.trainable / Lo.full.trainable), realLabel: '' },
        { label: 'model', value: ft.model },
      ],
    },
    size: [1040, 480],
    aria: 'LoRA: a frozen weight matrix plus a low-rank product of two thin matrices; a real rank-4 run on GPT-2 small against full fine-tuning; how low-rank a narrow fine-tune’s change is compared with Qwen3’s post-training; merging or swapping adapters.',
    phases: PHASES, learn: 'lora', tokens: words(['a', 'b', 'c', 'd', 'e', 'f']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    hints: { idea: 'Hover or tap a cell of A · B for its sum.' },
    still: ['train', 10.5],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption } = k
  const pad = 36, top = 56

  /* ---------- 1: W + A·B ---------- */
  function sceneIdea(p: number) {
    const c = 26, fin = eout(clamp(p / 0.08))
    const rW: Rect = { x: pad + 40, y: top + 150, c }, rA: Rect = { x: rW.x + 8 * c + 80, y: rW.y, c }, rB: Rect = { x: rA.x + 2 * c + 30, y: rW.y - 2 * c - 34, c }, rD: Rect = { x: rB.x, y: rW.y, c }
    mk.drawMat({ r: rW, vals: TW, kind: 'w', alpha: fin * 0.55, name: 'W (frozen)', shape: '6 × 8', real: '768 × 2,304', label: 'bottom' })
    ctx.font = F.mono(22); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, fin); ctx.fillText('+', rW.x + 8 * c + 40, rW.y + 3 * c)
    mk.drawMat({ r: rA, vals: TA, kind: 'w', alpha: fin, name: 'A', shape: '6 × 2', real: '768 × 4', label: 'bottom' })
    mk.drawMat({ r: rB, vals: TB, kind: 'w', alpha: fin, name: 'B', shape: '2 × 8', real: '4 × 2,304' })
    const g = gemm(clamp((p - 0.15) / 0.6), 6, 8, 2, 'fast')
    mk.drawMat({ r: rD, vals: TD, kind: 'score', alpha: fin, name: 'A · B = ΔW', shape: '6 × 8, rank 2', label: 'bottom', reveal: g.rev })
    mk.hit('d', rD, 6, 8)
    const fs = mk.resolve({ d: { g, K: 2 } })
    if (fs) mk.gemmOverlay({ A: rA, Av: TA, B: rB, Bv: TB, C: rD, f: fs, names: ['ΔW', 'A', 'B'], note: 'Each cell of ΔW is a sum of just r products. Every row of ΔW is a mix of the same r rows of B: that is what rank r means.' })
    else mk.formula = { segs: [['ΔW = A · B', C.ink], ['   ', C.mute], ['d·r + r·k numbers instead of d·k', C.ink2]], note: 'Toy numbers. In the real run below, W is 768 × 2,304 and r = 4: 12,288 trainable numbers per matrix instead of 1,769,472.' }
    caption('toy numbers', rW.x, rW.y - 20, fin, C.mute, 'left')
  }

  /* ---------- 2: how few ---------- */
  function sceneCount(p: number) {
    const { W } = stage, x0 = pad + 250, y0 = top + 60, bw = W - pad - x0 - 170
    const rows: [string, string, number, number][] = [['GPT-2 small', `here: r = ${R} on W_qkv`, Lo.run.trainable, Lo.full.trainable], ['LLaMA 3 8B', 'r = 16 on W_q, W_k, W_v, W_o', LLAMA_LORA, LLAMA]]
    rows.forEach(([name, sub, t, total], i) => {
      const a = eout(clamp((p - 0.05 - i * 0.3) / 0.15)), y = y0 + i * 110
      ctx.font = F.mono(12, 600); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(name, x0 - 16, y + 8)
      ctx.font = F.small; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(sub, x0 - 16, y + 26)
      ctx.fillStyle = rgba(C.ink, 0.12 * a); ctx.fillRect(x0, y, bw, 16)
      ctx.fillStyle = rgba(C.ink, 0.9 * a); ctx.fillRect(x0, y, Math.max(2, (t / total) * bw), 16)
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(`${(t / 1e6).toFixed(2)}M trainable · ${pct(t / total)}`, x0 + bw + 10, y + 8)
    })
    const ma = eout(clamp((p - 0.65) / 0.12))
    caption(`optimizer state for LLaMA 3 8B: about ${Math.round((LLAMA_LORA * 8) / 1e6)} MB for the adapters, against about ${Math.round((LLAMA * 8) / 1e9)} GB for every weight`, pad, y0 + 250, ma, C.ink2, 'left')
    mk.formula = { segs: [['per matrix', C.ink2], ['  r (d + k)', C.ink], ['  vs  ', C.mute], ['d k', C.ink2]], note: 'The frozen weights still need memory and a forward and backward pass, but no gradients and no Adam state; that is where most of the saving comes from.' }
  }

  /* ---------- 3: the real run ---------- */
  function sceneTrain(p: number) {
    const { H } = stage, x0 = pad + 60, x1 = pad + 520, y0 = top + 40, y1 = H - 90, n = Lo.steps
    const X = (s: number) => lerp(x0, x1, s / n), Y = (v: number) => lerp(y1, y0, Math.log10(v / 0.05) / Math.log10(10 / 0.05))
    title('training loss, log scale', x0, y0 - 16, 1)
    ctx.strokeStyle = rgba(C.faint, 1); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke()
    const g = Math.floor(clamp((p - 0.05) / 0.55) * n)
    const line = (cv: number[], col: typeof C.ink, name: string, i: number) => {
      ctx.strokeStyle = rgba(col, 1); ctx.lineWidth = 2; ctx.beginPath()
      for (let s = 0; s <= g; s++) { if (s) ctx.lineTo(X(s), Y(cv[s])); else ctx.moveTo(X(s), Y(cv[s])) }
      ctx.stroke()
      caption(`${name} ${cv[g].toFixed(3)}`, x1 + 10, y0 + 14 + i * 18, 1, col, 'left')
    }
    line(Lo.full.curve, C.tok[3], `all ${(Lo.full.trainable / 1e6).toFixed(0)}M weights`, 0)
    line(Lo.run.curve, C.tok[0], `LoRA, ${(Lo.run.trainable / 1e3).toFixed(0)}k weights`, 1)
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    for (const v of [0.1, 1, 10]) ctx.fillText(String(v), x0 - 6, Y(v))
    ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(`${n} steps`, x1, y1 + 6)
    // what it learned
    const xa = x1 + 60, ta = eout(clamp((p - 0.6) / 0.12))
    title('p( moon ) after …', xa, y0 + 70, ta)
    const rows: [string, number, number][] = [...Lo.train.map((t, i) => [t.replace(' moon.', ''), Lo.run.before.train[i], Lo.run.after.train[i]] as [string, number, number]), [`${Lo.eval} (unseen)`, Lo.run.before.eval, Lo.run.after.eval]]
    rows.forEach(([t, b, af], i) => {
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, ta); ctx.fillText(t, xa, y0 + 96 + i * 42)
      ctx.fillStyle = rgba(C.ink, ta); ctx.fillText(`${pct(b)} → ${pct(af)}`, xa, y0 + 114 + i * 42)
    })
    mk.formula = { segs: [['learning rate', C.ink2], [` LoRA ${Lo.lr.lora}`, C.ink], ['  ·  ', C.mute], [`full ${Lo.lr.full}`, C.ink]], note: 'Two sentences are far too few for a real fine-tune; the unseen sentence shows why: the model now puts the moon under everything. Real fine-tunes use thousands of examples and watch a held-out set.' }
  }

  /* ---------- 4: how low-rank is a real change ---------- */
  function sceneRank(p: number) {
    const { H } = stage, x0 = pad + 60, x1 = pad + 560, y0 = top + 40, y1 = H - 90
    const ranks = [1, 2, 4, 8, 16, 32], X = (r: number) => lerp(x0, x1, Math.log2(r) / 5), Y = (v: number) => lerp(y1, y0, v)
    title('share of the change held by the top r directions', x0, y0 - 16, 1)
    ctx.strokeStyle = rgba(C.faint, 1); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'center'; ctx.textBaseline = 'top'
    ranks.forEach((r) => ctx.fillText(String(r), X(r), y1 + 6))
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; for (const v of [0, 0.5, 1]) ctx.fillText(pct(v), x0 - 6, Y(v))
    const curve = (vals: number[], rs: number[], col: typeof C.ink, name: string, a: number, i: number) => {
      if (a <= 0) return
      ctx.strokeStyle = rgba(col, a); ctx.lineWidth = 2; ctx.beginPath()
      rs.forEach((r, j) => { if (j) ctx.lineTo(X(r), Y(vals[j])); else ctx.moveTo(X(r), Y(vals[j])) })
      ctx.stroke()
      rs.forEach((r, j) => { ctx.fillStyle = rgba(col, a); ctx.beginPath(); ctx.arc(X(r), Y(vals[j]), 3, 0, 7); ctx.fill() })
      caption(name, x1 + 14, y0 + 20 + i * 36, a, col, 'left')
    }
    curve(Lo.full.rank.captured, Lo.full.rank.ranks, C.tok[3], `GPT-2, full fine-tune on 2 sentences: ${Lo.full.rank.matrix}`, eout(clamp(p / 0.2)), 0)
    const q = QD.mats[0], qr = QD.ranks.filter((r) => r <= 32)
    curve(q.delta.captured.slice(0, qr.length), qr, C.tok[0], `Qwen3-1.7B minus its base: layer ${QD.layer + 1} W_q`, eout(clamp((p - 0.35) / 0.2)), 1)
    curve(q.weight.captured.slice(0, qr.length), qr, C.ink2, 'and the base weight W_q itself', eout(clamp((p - 0.6) / 0.2)), 2)
    mk.formula = { segs: [['captured(r)', C.ink], [' = ', C.mute], ['Σ top-r σᵢ² / ‖ΔW‖²', C.ink2]], note: `Singular values by randomized SVD. Qwen’s post-training changed its weights by about ${pct(q.relChange)} of their size, in every direction at once: as full-rank as the weights themselves.` }
  }

  /* ---------- 5: serving ---------- */
  function sceneMerge(p: number) {
    const { W } = stage, y0 = top + 80, a1 = eout(clamp(p / 0.2)), a2 = eout(clamp((p - 0.4) / 0.2))
    title('merged: one matrix, no extra cost', pad, y0 - 30, a1)
    mathName('W′ = W + A · B', pad + 20, y0 + 20, a1, 24)
    caption('computed once after training; the model runs exactly as before', pad + 20, y0 + 50, a1, C.ink2, 'left')
    const xr = W / 2 + 20
    title('separate: many adapters, one base', xr, y0 - 30, a2)
    ;['customer A', 'customer B', 'SQL', 'legal'].forEach((n, i) => {
      const x = xr + i * 110
      ctx.strokeStyle = rgba(C.tok[i], a2); ctx.lineWidth = 1.2; ctx.strokeRect(x, y0, 90, 28)
      ctx.font = F.mono(11); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.tok[i], a2); ctx.fillText(n, x + 45, y0 + 14)
    })
    ctx.fillStyle = rgba(C.ink, 0.1 * a2); ctx.fillRect(xr, y0 + 60, 430, 40)
    ctx.font = F.mono(12, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a2); ctx.fillText('one frozen base model', xr + 215, y0 + 80)
    caption('each request picks its adapter; batches can mix them', xr, y0 + 130, a2, C.ink2, 'left')
    mk.formula = { segs: [['merged', C.ink2], [': zero extra latency  ·  ', C.mute], ['separate', C.ink2], [': a few MB per adapter', C.mute]], note: 'Serving systems such as S-LoRA and vLLM keep thousands of adapters beside one base model. QLoRA trains the adapters against a 4-bit quantized base to save more memory.' }
  }

  return { idea: sceneIdea, count: sceneCount, train: sceneTrain, rank: sceneRank, merge: sceneMerge }
}
