import { F, chipW, drawChip, rr, type TokLike } from '../../core/draw'
import { fmt, fmtF } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp } from '../../core/util'
import { plain, spec } from '../../lib/spec/data'
import { mountExhibit, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * Speculative decoding: a small draft model guesses several tokens, the large model checks them all in
 * one pass and keeps the longest agreeing prefix. The rounds, tokens and probabilities are a real run
 * of distilgpt2 drafting for GPT-2 small (exported offline); the output equals GPT-2's own.
 */

const PHASES = [
  { id: 'idea', name: 'Guess cheaply, check in one pass', short: 'Idea', dur: 9 },
  { id: 'round', name: 'One round', short: 'A round', dur: 12 },
  { id: 'run', name: 'A whole run', short: 'Run', dur: 11 },
  { id: 'sampling', name: 'When sampling', short: 'min(1, p/q)', dur: 11 },
  { id: 'speedup', name: 'How much faster', short: 'Speed-up', dur: 9 },
]

const K = spec.k, R = spec.rounds
const tok = (s: string, c: number): TokLike => ({ text: plain(s), c })
const PROMPT_T = spec.prompt.map((s, i) => tok(s, i))
const compared = R.reduce((a, r) => a + Math.min(K, r.accepted + 1), 0), accepted = R.reduce((a, r) => a + r.accepted, 0)
const ALPHA = accepted / compared
const expected = (a: number, k: number) => (1 - Math.pow(a, k + 1)) / (1 - a)
/** A toy 3-token distribution to show that the acceptance rule keeps the target's distribution exactly. */
const TP = [0.5, 0.3, 0.2], TQ = [0.2, 0.6, 0.2]

const COMPARE: Record<string, [string, string]> = {
  idea: ['decode’s memory limit', 'serving/kv-cache?phase=bound'], round: ['GPT-2’s greedy pick', 'anatomy?phase=pick'], run: ['GPT-2’s generation', 'anatomy/unembed?phase=sample'],
  sampling: ['GPT-2’s sampling', 'anatomy/unembed?phase=sample'], speedup: ['batching', 'serving/continuous-batching?phase=why'],
}

const CAPS: Record<string, [string, string]> = {
  idea: ['A large model’s decode step is limited by reading its weights, so checking five tokens in one pass costs about as much as producing one. Speculative decoding uses that: a small draft model guesses the next few tokens cheaply, and the large model verifies all the guesses at once.', 'draft k tokens · verify in 1 pass'],
  round: [`A real round: distilgpt2 (6 layers) guesses ${K} tokens one after another; GPT-2 small (12 layers) runs once over all of them and writes down its own choice at each position. Guesses are kept up to the first disagreement, and GPT-2’s choice there is added, so every round gains at least one token.`, 'keep the agreeing prefix + 1 token from the target'],
  run: [`The whole run: ${spec.output.length} new tokens from ${spec.targetPasses} passes of GPT-2 instead of ${spec.output.length}, and the text is exactly what GPT-2 alone would have written greedily, because every token was checked by GPT-2.`, `${spec.output.length} tokens · ${spec.targetPasses} target passes · same text`],
  sampling: ['When sampling instead of picking the top token, a drafted token x is accepted with probability min(1, p(x)/q(x)), where p is the target’s probability and q the draft’s. If rejected, the replacement is drawn from what p has in excess of q. The tokens that come out follow the target’s distribution exactly.', 'accept x with probability min(1, p(x) / q(x))'],
  speedup: ['If each guess is accepted with probability α, one target pass yields (1 − α^(k+1)) / (1 − α) tokens on average. The draft must also be cheap: distilgpt2 is half of GPT-2 small, so this run is actually slower; real systems use drafts 10 to 100 times smaller and get 2 to 3 times faster.', 'tokens per pass = (1 − α^(k+1)) / (1 − α)'],
}

export function mountSpeculative(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'A real run: distilgpt2 drafts for GPT-2 small, and every number shown comes from the two models.',
      eyebrow: 'Serving · Decoding',
      title: 'Speculative Decoding',
      subtitle: 'draft with a small model, verify with the large one',
      specs: [
        { label: 'target', value: 'GPT-2 small', real: '12 layers', realLabel: '' },
        { label: 'draft', value: 'distilgpt2', real: '6 layers', realLabel: '' },
        { label: 'guesses per round', value: String(K) },
        { label: 'this run', value: `${spec.output.length} tokens`, real: `${spec.targetPasses} target passes`, realLabel: '·' },
        { label: 'output', value: spec.identical ? 'identical to GPT-2' : 'differs' },
      ],
    },
    size: [1040, 480],
    aria: 'Speculative decoding: a small draft model proposes several tokens, the large model checks them all in one pass and keeps the longest agreeing prefix plus one token of its own; a real run of distilgpt2 drafting for GPT-2 small produces exactly GPT-2’s text in fewer passes.',
    phases: PHASES, learn: 'speculative', tokens: PROMPT_T, compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['round', 11],
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { caption, title, arrow } = k
  const pad = 36, top = 56, bot = 46
  /** A row of token chips from x; returns the x after each chip. */
  function chips(ts: TokLike[], x: number, y: number, a: number, hl: (i: number) => boolean = () => false) {
    const xs: number[] = []
    ts.forEach((t, i) => { xs.push(x); x += drawChip(x, y, t, a, 22, hl(i)) + 6 })
    xs.push(x)
    return xs
  }
  function strike(x0: number, x1: number, y: number, a: number) { ctx.strokeStyle = rgba(C.ink, 0.9 * a); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x0 - 2, y); ctx.lineTo(x1 + 2, y); ctx.stroke() }
  function mark(x: number, y: number, ok: boolean, a: number) {
    ctx.strokeStyle = rgba(ok ? C.tok[4] : C.ink, a); ctx.lineWidth = 2; ctx.beginPath()
    if (ok) { ctx.moveTo(x - 5, y); ctx.lineTo(x - 1, y + 4); ctx.lineTo(x + 6, y - 5) } else { ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4) }
    ctx.stroke()
  }

  /* ---------- scene 1: the idea ---------- */
  function sceneIdea(p: number) {
    const { W } = stage
    const x0 = pad + 130, unit = (W - x0 - pad - 40) / 14, y1 = top + 90, y2 = top + 250
    title('plain decoding: one large-model pass per token', pad, y1 - 44, 1)
    const n1 = Math.min(5, Math.floor(eio(clamp((p - 0.05) / 0.4)) * 6))
    for (let i = 0; i < 5; i++) {
      const a = i < n1 ? 1 : 0.15, x = x0 + i * 2.2 * unit
      rr(x, y1 - 16, 2 * unit, 32, 5); ctx.fillStyle = rgba(C.ink, 0.6 * a); ctx.fill()
      ctx.font = F.mono(11); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.bg, a); ctx.fillText('GPT-2', x + unit, y1 + 0.5)
      caption('1 token', x + unit, y1 + 34, a, C.mute)
    }
    ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, 1); ctx.fillText('large model', pad, y1 + 4)
    title('speculative: small guesses, one large check', pad, y2 - 44, 1)
    const g = eio(clamp((p - 0.4) / 0.45))
    for (let i = 0; i < K; i++) {
      const a = g * 5 > i ? 1 : 0.15, x = x0 + i * 0.9 * unit
      rr(x, y2 - 12, 0.8 * unit, 24, 4); ctx.fillStyle = rgba(C.ink2, 0.45 * a); ctx.fill()
      ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.bg, a); ctx.fillText('draft', x + 0.4 * unit, y2 + 0.5)
    }
    const va = g * 5 > K ? 1 : 0.15, vx = x0 + K * 0.9 * unit + 10
    rr(vx, y2 - 16, 2 * unit, 32, 5); ctx.fillStyle = rgba(C.ink, 0.6 * va); ctx.fill()
    ctx.font = F.mono(11); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.bg, va); ctx.fillText('GPT-2', vx + unit, y2 + 0.5)
    caption(`checks ${K} guesses at once`, vx + unit, y2 + 34, va, C.ink2)
    caption(`${K} small passes`, x0 + (K * 0.9 * unit) / 2, y2 + 34, g > 0.1 ? 1 : 0, C.mute)
    ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, 1); ctx.fillText('small + large', pad, y2 + 4)
    const na = eout(clamp((p - 0.85) / 0.1))
    caption('up to 5 tokens for one large pass, if the guesses are right', vx + 2 * unit + 20, y2 + 4, na, C.ink, 'left')
    mk.formula = { segs: [['one GPT-2 pass over 5 tokens ≈ one pass over 1 token', C.ink]], note: 'Both read every weight once; the extra arithmetic for four more tokens is small. This is why checking is cheap and why a good guesser helps.' }
  }

  /* ---------- scene 2: one real round ---------- */
  const RI = 1
  function sceneRound(p: number) {
    const { W } = stage
    const r = R[RI], before = spec.output.slice(0, r.ctx - spec.prompt.length)
    const ctxT = [...spec.prompt, ...before].map((s, i) => tok(s, i)), y0 = top + 40
    title('fixed so far', pad, y0 - 14, 1)
    chips(ctxT, pad, y0 + 14, 1)
    const n = ctxT.length, x0 = pad + 150, cw = (W - x0 - pad) / (K + 1)
    // drafted, one at a time
    const yD = y0 + 90, yT = y0 + 170, yO = y0 + 260
    ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, 1); ctx.fillText('distilgpt2 guesses', pad, yD)
    ctx.fillStyle = rgba(C.ink, 1); ctx.fillText('GPT-2 would pick', pad, yT)
    ctx.fillStyle = rgba(C.ink, 1); ctx.fillText('result', pad, yO)
    const dShow = Math.floor(eio(clamp((p - 0.05) / 0.3)) * (K + 0.99)), tA = eout(clamp((p - 0.42) / 0.12)), cmp = Math.floor(clamp((p - 0.58) / 0.25) * (K + 1))
    for (let i = 0; i < K; i++) {
      const cx = x0 + (i + 0.5) * cw
      if (i < dShow) { const t = tok(r.drafted[i], n + i); drawChip(cx - chipW(t.text) / 2, yD, t, 1, 22, false) }
      if (tA > 0) { const t = tok(r.choice[i], n + i); drawChip(cx - chipW(t.text) / 2, yT, t, tA, 22, false) }
      if (i < cmp) {
        const ok = i < r.accepted, stop = i === r.accepted
        if (i <= r.accepted) mark(cx, (yD + yT) / 2, ok, 1)
        if (ok) { const t = tok(r.drafted[i], n + i); drawChip(cx - chipW(t.text) / 2, yO, t, 1, 22, true) }
        if (stop) { const t = tok(r.next, n + i); drawChip(cx - chipW(t.text) / 2, yO, t, 1, 22, true); caption('GPT-2’s choice', cx, yO + 30, 1, C.ink2) }
        if (i > r.accepted) { const t = tok(r.drafted[i], n + i), w = chipW(t.text); drawChip(cx - w / 2, yD, t, 0.35, 22); strike(cx - w / 2, cx + w / 2, yD, 1) }
      }
    }
    if (tA > 0) { arrow([[x0 - 30, yD + 16], [x0 - 30, yT - 16]], tA); caption('1 pass', x0 - 36, (yD + yT) / 2 + 4, tA, C.ink2, 'right') }
    const fa = eout(clamp((p - 0.86) / 0.08))
    ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, fa)
    ctx.fillText(`${r.accepted} accepted + 1 from GPT-2 = ${r.accepted + 1} new tokens from one GPT-2 pass`, x0, yO + 70)
    mk.formula = { segs: [['keep drafted tokens while they equal GPT-2’s pick', C.ink2], ['   ·   ', C.mute], ['at the first mismatch, take GPT-2’s token and stop', C.ink]], note: `Round ${RI + 1} of the real run. GPT-2 computes its choice at every position in the same pass (its causal mask means position i only sees the guesses before it), so one pass checks all ${K}.` }
  }

  /* ---------- scene 3: the whole run ---------- */
  function sceneRun(p: number) {
    const x0 = pad + 110, y0 = top + 36, rh = 50, shown = Math.floor(eio(clamp((p - 0.04) / 0.7)) * R.length + 0.999)
    title('round · drafted (kept, struck) → gpt-2’s token', x0, y0 - 12, 1)
    let n = spec.prompt.length
    R.forEach((r, i) => {
      const a = i < shown ? 1 : 0.1, y = y0 + 20 + i * rh
      ctx.font = F.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(`round ${i + 1}`, x0 - 12, y)
      let x = x0
      r.drafted.forEach((s, j) => {
        const t = tok(s, n + j), kept = j < r.accepted, w = drawChip(x, y, t, a * (kept ? 1 : 0.35), 20, kept)
        if (!kept) strike(x, x + w, y, a)
        x += w + 6
      })
      x += 14
      arrow([[x - 12, y], [x - 2, y]], a)
      const t = tok(r.next, n + r.accepted)
      x += drawChip(x + 4, y, t, a, 20, true) + 10
      caption(`+${r.accepted + 1}`, x + 6, y + 4, a, C.ink2, 'left')
      n += r.accepted + 1
    })
    // the text so far
    const done = R.slice(0, shown).reduce((s, r) => s + r.accepted + 1, 0), text = plain([...spec.prompt, ...spec.output.slice(0, done)].join('')).replace(/\n/g, ' ↵ ')
    const ty = y0 + 20 + R.length * rh + 24
    ctx.font = F.mono(13); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, 1); ctx.fillText(text, x0, ty)
    const ra = eout(clamp((p - 0.8) / 0.1))
    caption(`${spec.output.length} tokens in ${spec.targetPasses} GPT-2 passes (${spec.draftPasses} distilgpt2 passes) · plain greedy GPT-2 needs ${spec.output.length} passes and writes the identical text`, x0, ty + 24, ra, C.ink2, 'left')
    mk.formula = { segs: [['tokens per GPT-2 pass', C.ink2], ['  =  ', C.mute], [`${spec.output.length} / ${spec.targetPasses} = ${(spec.output.length / spec.targetPasses).toFixed(1)}`, C.ink]], note: 'Rounds where the draft goes wrong on its first guess (3 and 4) still yield one token, GPT-2’s own, so speculative decoding is never worse in passes of the large model.' }
  }

  /* ---------- scene 4: the sampling rule ---------- */
  function sceneSampling(p: number) {
    const { W } = stage
    // left: real p and q for round 1's drafted tokens
    const r = R[0], x0 = pad + 20, y0 = top + 50, bw = 150, rows = K
    title('round 1 · p (gpt-2) and q (distilgpt2) of each guess', x0, y0 - 18, 1)
    for (let i = 0; i < rows; i++) {
      const a = eout(clamp((p - 0.04 - i * 0.06) / 0.1)), y = y0 + 12 + i * 58, t = tok(r.drafted[i], i)
      if (a <= 0) continue
      drawChip(x0, y + 12, t, a, 20)
      const bx = x0 + 110
      ctx.fillStyle = rgba(C.ink, 0.8 * a); ctx.fillRect(bx, y, Math.max(1.5, r.p[i] * bw), 10)
      ctx.fillStyle = rgba(C.ink2, 0.45 * a); ctx.fillRect(bx, y + 14, Math.max(1.5, r.q[i] * bw), 10)
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(`p ${fmt(r.p[i])}`, bx + bw + 10, y + 5)
      ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(`q ${fmt(r.q[i])}`, bx + bw + 10, y + 19)
      const acc = Math.min(1, r.p[i] / r.q[i])
      ctx.font = F.mono(12, 500); ctx.fillStyle = rgba(C.ink, a); ctx.fillText(`accept ${Math.round(acc * 100)}%`, bx + bw + 80, y + 12)
    }
    // right: the rule keeps the target's distribution (toy, 3 tokens)
    const ra = eout(clamp((p - 0.45) / 0.1)), rx = W / 2 + 60, ry = y0, cw = 56, hMax = 120
    if (ra > 0) {
      title('why it is exact · a 3-token example', rx, ry - 18, ra)
      const acc = TP.map((pp, i) => Math.min(pp, TQ[i])), rej = 1 - acc.reduce((a, b) => a + b, 0), res = TP.map((pp, i) => Math.max(0, pp - TQ[i])), rz = res.reduce((a, b) => a + b, 0)
      const out = TP.map((_, i) => acc[i] + rej * (res[i] / rz))
      const col = (vals: number[], x: number, label: string, t0: number, colr: RGB) => {
        const a = ra * eout(clamp((p - t0) / 0.1))
        vals.forEach((v, i) => { const h = v * hMax / 0.6; ctx.fillStyle = rgba(colr, 0.8 * a); ctx.fillRect(x + i * 18, ry + 150 - h, 14, h) })
        caption(label, x + 24, ry + 170, a, C.ink2)
      }
      col(TQ, rx, 'q, the draft', 0.5, C.ink2)
      col(acc, rx + cw + 30, 'accepted', 0.58, C.tok[4])
      col(res.map((v) => (v / rz) * rej), rx + 2 * (cw + 30), 'from p − q', 0.66, C.tok[3])
      col(out, rx + 3 * (cw + 30), '= output', 0.74, C.ink)
      const ea = ra * eout(clamp((p - 0.82) / 0.1))
      caption(`output ${out.map(fmt).join(' ')}  =  p ${TP.map(fmt).join(' ')}`, rx, ry + 200, ea, C.ink, 'left')
    }
    mk.formula = { segs: [['P(x) = q(x) · min(1, p(x)/q(x))', C.ink2], ['  +  ', C.mute], ['P(reject) · max(0, p(x) − q(x)) / Σ', C.ink2], ['  =  ', C.mute], ['p(x)', C.ink]], note: `Real probabilities from round 1: GPT-2 disagreed with “${plain(r.drafted[1]).trim()}”, giving it p = ${fmtF(r.p[1])} against the draft’s q = ${fmtF(r.q[1])}. When the draft is overconfident the guess is often rejected; when it is underconfident (p ≥ q) the guess is always kept.` }
  }

  /* ---------- scene 5: expected speed-up ---------- */
  function sceneSpeedup(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const cx0 = pad + 70, cx1 = W / 2 + 40, cy0 = top + 40, cy1 = top + avail - 40, fin = eout(clamp(p / 0.06))
    const X = (a: number) => lerp(cx0, cx1, a), Y = (v: number) => lerp(cy1, cy0, (v - 1) / 8)
    title('tokens per target pass', cx0, cy0 - 18, fin)
    ctx.strokeStyle = rgba(C.ink, 0.25 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy1); ctx.lineTo(cx1, cy1); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, fin)
    for (const v of [1, 3, 5, 7, 9]) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(v), cx0 - 6, Y(v)) }
    for (const a of [0, 0.25, 0.5, 0.75, 1]) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(String(a), X(a), cy1 + 6) }
    caption('α, the chance a guess is accepted', (cx0 + cx1) / 2, cy1 + 32, fin)
    const g = eio(clamp((p - 0.08) / 0.4))
    ;[2, 4, 8].forEach((kk, i) => {
      ctx.strokeStyle = rgba(i === 1 ? C.ink : C.ink2, fin); ctx.lineWidth = i === 1 ? 2 : 1.2; ctx.beginPath()
      for (let s = 0; s <= 60 * g; s++) { const a = (s / 60) * 0.98, y = Y(expected(a, kk)); if (s) ctx.lineTo(X(a), y); else ctx.moveTo(X(a), y) }
      ctx.stroke()
      if (g >= 1) caption(`k = ${kk}`, X(0.98) + 6, Y(expected(0.98, kk)) + 4, fin, i === 1 ? C.ink : C.ink2, 'left')
    })
    const ma = eout(clamp((p - 0.5) / 0.1))
    if (ma > 0) {
      const ex = expected(ALPHA, K)
      ctx.fillStyle = rgba(C.tok[4], ma); ctx.beginPath(); ctx.arc(X(ALPHA), Y(ex), 5, 0, 7); ctx.fill()
      caption(`this run: α ≈ ${fmt(ALPHA)} (${accepted} of ${compared} checks)`, X(ALPHA) + 12, Y(ex) + 34, ma, C.ink, 'left')
      caption(`${ex.toFixed(1)} tokens per pass expected, ${(spec.output.length / spec.targetPasses).toFixed(1)} observed`, X(ALPHA) + 12, Y(ex) + 50, ma, C.ink2, 'left')
    }
    // the cost of the draft
    const ca = eout(clamp((p - 0.65) / 0.12)), rx = cx1 + 90
    if (ca > 0) {
      title('speed-up, counting the draft’s cost c', rx, cy0 - 18, ca)
      const ex = expected(ALPHA, K), rows: [string, number][] = [['c = 0.5 (distilgpt2 vs GPT-2)', 0.5], ['c = 0.1', 0.1], ['c = 0.02', 0.02]]
      rows.forEach(([name, c], i) => {
        const sp = ex / (1 + K * c), y = cy0 + 20 + i * 52
        caption(name, rx, y, ca, C.ink2, 'left')
        ctx.fillStyle = rgba(sp >= 1 ? C.tok[4] : C.neg, 0.85 * ca); ctx.fillRect(rx, y + 10, sp * 80, 16)
        ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, ca); ctx.fillText(`${sp.toFixed(2)} ×`, rx + sp * 80 + 8, y + 18.5)
      })
      caption('below 1 ×: slower than plain decoding', rx, cy0 + 20 + 3 * 52 + 4, ca, C.mute, 'left')
    }
    mk.formula = { segs: [['speed-up = (1 − α^(k+1)) / ((1 − α)(k c + 1))', C.ink]], note: 'c is the cost of one draft pass relative to one target pass. This formula (from Leviathan et al.) assumes each guess is accepted independently; real acceptance comes in runs, as in this example. Drafts can also be the model itself with a few layers skipped, or extra heads trained to guess ahead (Medusa, EAGLE).' }
  }

  return { idea: sceneIdea, round: sceneRound, run: sceneRun, sampling: sceneSampling, speedup: sceneSpeedup }
}
