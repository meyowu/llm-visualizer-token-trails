import { F, mathName, spaced, useCtx } from '../core/draw'
import { createFrame } from '../core/frame'
import { MatrixKit, fmt, fmtF, gemm, type M, type Rect } from '../core/matrix'
import { Player } from '../core/player'
import { Stage, runLoop } from '../core/stage'
import { C, rgba } from '../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../core/util'
import { teach } from './learn'
import type { Nav } from './registry'

/*
 * The four pieces of math the Transformer pages lean on, in the same visual language: the dot
 * product as similarity, the matrix product laid out A-left, B-above, softmax, and one-hot
 * times a matrix as a row lookup. The numbers are small made-up examples.
 */

const PHASES = [
  { id: 'dot', name: 'Dot product', short: 'a · b', dur: 8 },
  { id: 'matmul', name: 'Matrix product', short: 'A · B', dur: 10 },
  { id: 'softmax', name: 'Softmax', dur: 7 },
  { id: 'onehot', name: 'One-hot lookup', short: 'One-hot', dur: 6 },
]
const a = [0.8, -0.3, 0.5, 0.1], b = [0.6, 0.2, 0.9, -0.4]
const A: M = [[0.8, -0.3, 0.5, 0.1], [-0.2, 0.9, 0.1, 0.4], [0.5, 0.5, -0.6, 0.2]]
const B: M = [[0.6, -0.1, 0.3], [0.2, 0.7, -0.5], [0.9, 0.1, 0.2], [-0.4, 0.3, 0.8]]
const Cm: M = A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)))
const SCORES = [2.0, 1.0, 0.2, -1.0]
const ONEHOT: M = [[0, 0, 1, 0]]
const Wm: M = [[0.3, -0.7, 0.2], [0.9, 0.1, -0.4], [-0.5, 0.6, 0.8], [0.2, 0.4, -0.1]]

export function mountFoundations(root: HTMLElement, _nav: Nav): () => void {
  const reduced = reducedMotion()
  const frame = createFrame(root, {
    formula: true,
    eyebrow: 'Warm-up',
    title: 'Foundations',
    subtitle: 'the four pieces of math the rest uses',
    specs: [{ label: 'shown', value: 'small examples' }, { label: 'takes', value: '≈ 30 s' }],
  })
  const stage = new Stage(frame.stageHost, 1040, 470, 'Foundations: a dot product as a similarity score, a matrix product as many dot products laid out row against column, softmax turning scores into probabilities, and a one-hot row times a matrix picking out one row.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  teach(player, 'foundations')
  const prog = (id: string) => player.prog(id)
  const mk = new MatrixKit(stage, [], frame.setFormula)
  const pad = 36, top = 70

  const title = (t: string, x: number, y: number, al: number) => {
    ctx.font = F.label; spaced(true); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, al); ctx.fillText(t, x, y); spaced(false)
  }

  /* ---------- dot product ---------- */
  function sceneDot(p: number) {
    const c = 40, x0 = pad + 70, y0 = top + 30, fin = eout(clamp(p / 0.1))
    const ra: Rect = { x: x0, y: y0, c }, rb: Rect = { x: x0, y: y0 + c * 1.6, c }, rp: Rect = { x: x0, y: y0 + c * 3.4, c }
    title('TWO VECTORS', x0, y0 - 18, fin)
    mk.drawMat({ r: ra, vals: [a], kind: 'w', alpha: fin, name: 'a', shape: '4 numbers', label: 'none' })
    mk.drawMat({ r: rb, vals: [b], kind: 'w', alpha: fin, name: 'b', shape: '4 numbers', label: 'none' })
    mathName('a', x0 - 30, y0 + c * 0.7, fin, 20); mathName('b', x0 - 30, rb.y + c * 0.7, fin, 20)
    const k = clamp((p - 0.12) / 0.4) * 4, prods = a.map((v, i) => v * b[i])
    mk.drawMat({ r: rp, vals: [prods], kind: 'score', alpha: clamp(k), name: 'a ⊙ b', shape: '', label: 'none', reveal: (_, j) => (k > j ? 1 : 0) })
    ctx.font = F.small; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, clamp(k)); ctx.fillText('aₖ × bₖ', x0 - 8, rp.y + c / 2)
    const sum = prods.reduce((s, v) => s + v, 0), sa = clamp((p - 0.55) / 0.1)
    ctx.font = F.mono(14, 500); ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink, sa); ctx.fillText(`Σ = ${fmtF(sum)}`, x0 + 4 * c + 18, rp.y + c / 2)
    // the geometric view: the same score is large when two arrows point the same way
    const gx = stage.W * 0.62, gy = top + 150, R = 58, ga = eout(clamp((p - 0.5) / 0.2))
    title('A SIMILARITY SCORE', gx - R, top + 12, ga)
    const pairs: [number, string][] = [[0.35, 'same way'], [Math.PI / 2, 'at right angles'], [Math.PI * 0.9, 'opposite']]
    pairs.forEach(([ang, label], i) => {
      const cx = gx + i * (R * 2 + 44), base = -0.4
      ctx.strokeStyle = rgba(C.ink, 0.18 * ga); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, gy, R, 0, 7); ctx.stroke()
      ;[[base, C.tok[0]], [base + ang, C.tok[2]]].forEach(([t, col]) => {
        const tt = t as number, cc = col as [number, number, number]
        ctx.strokeStyle = rgba(cc, ga); ctx.lineWidth = 2.2
        ctx.beginPath(); ctx.moveTo(cx, gy); ctx.lineTo(cx + Math.cos(tt) * R, gy - Math.sin(tt) * R); ctx.stroke()
      })
      ctx.font = F.mono(13, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.ink, ga)
      ctx.fillText(`a · b = ${fmt(Math.cos(ang))}`, cx, gy + R + 14)
      ctx.font = F.small; ctx.fillStyle = rgba(C.mute, ga); ctx.fillText(label, cx, gy + R + 34)
    })
    mk.formula = { segs: [['a · b  =  Σₖ aₖ bₖ  =  ', C.mute], [a.map((v, i) => `${fmtF(v)}×${fmtF(b[i])}`).join(' + '), C.ink2], ['  =  ', C.mute], [fmtF(sum), C.ink]], note: 'Multiply matching numbers and add them up. For vectors of the same length it is large when they point the same way, 0 at right angles and negative when opposite; attention scores and logits are dot products.' }
  }

  /* ---------- matrix product ---------- */
  function sceneMatmul(p: number) {
    const c = 40, K = 4, bx = stage.W / 2 - 40, by = top + 10
    const RB: Rect = { x: bx, y: by, c }, RA: Rect = { x: bx - K * c - 24, y: by + K * c + 24, c }, RC: Rect = { x: bx, y: RA.y, c }
    const fin = eout(clamp(p / 0.08)), g = gemm((p - 0.08) / 0.86, 3, 3, K, 'slow')
    mk.drawMat({ r: RA, vals: A, kind: 'w', alpha: fin, name: 'A', shape: '3 × 4', label: 'bottom' })
    mk.drawMat({ r: RB, vals: B, kind: 'w', alpha: fin, name: 'B', shape: '4 × 3', labelW: 200 })
    mk.drawMat({ r: RC, vals: Cm, kind: 'score', alpha: fin, name: 'C = A · B', shape: '3 × 3', label: 'bottom', reveal: g.rev })
    mk.hit('C', RC, 3, 3)
    ctx.font = F.small; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, fin)
    ctx.fillText('row i of A →', RA.x - 10, RA.y + c / 2)
    ctx.textAlign = 'center'; ctx.fillText('column j of B ↓', RB.x + c / 2, RB.y - 26)
    const f = mk.resolve({ C: { g, K } })
    if (f) mk.gemmOverlay({ A: RA, Av: A, B: RB, Bv: B, C: RC, f, names: ['C', 'A', 'B'], note: `Cell (${f.i}, ${f.j}) is the dot product of row ${f.i} of A and column ${f.j} of B. On every page A sits on the left, B above, and C where they meet.` })
  }

  /* ---------- softmax ---------- */
  function sceneSoftmax(p: number) {
    const e = SCORES.map(Math.exp), z = e.reduce((s, v) => s + v, 0), pr = e.map((v) => v / z)
    const s1 = eio(clamp((p - 0.25) / 0.25)), s2 = eio(clamp((p - 0.6) / 0.25))
    const bw = 64, gap = 40, x0 = stage.W / 2 - (SCORES.length * (bw + gap)) / 2, base = top + 250, h = 150
    const fin = eout(clamp(p / 0.1))
    title(s2 > 0.5 ? 'PROBABILITIES · SUM 1' : s1 > 0.5 ? 'exp(score)'.toUpperCase() : 'SCORES', x0, top + 20, fin)
    ctx.strokeStyle = rgba(C.ink, 0.25 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 - 10, base); ctx.lineTo(x0 + SCORES.length * (bw + gap), base); ctx.stroke()
    SCORES.forEach((sv, i) => {
      const v = lerp(lerp(sv / 8, e[i] / 8, s1), pr[i], s2), x = x0 + i * (bw + gap)
      const yv = base - v * h
      ctx.fillStyle = rgba(C.tok[i], 0.85 * fin); ctx.fillRect(x, Math.min(yv, base), bw, Math.abs(base - yv))
      ctx.font = F.mono(12, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.ink, fin)
      ctx.fillText(s2 > 0.5 ? (pr[i] * 100).toFixed(1) + '%' : s1 > 0.5 ? fmtF(e[i]) : fmtF(sv), x + bw / 2, Math.min(yv, base) - 4)
      ctx.font = F.small; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, fin); ctx.fillText(`score ${fmtF(sv)}`, x + bw / 2, base + 8)
    })
    mk.formula = { segs: [['softmax(s)ₖ  =  exp(sₖ) / Σ exp(s)', C.ink], ['   ·   Σ exp = ', C.mute], [fmtF(z), C.ink2]], note: 'exp makes every score positive and widens the gaps; dividing by the sum makes the results add up to 1. The largest score gets most of the weight, but never all of it.' }
  }

  /* ---------- one-hot ---------- */
  function sceneOnehot(p: number) {
    const c = 40, bx = stage.W / 2 - 20, by = top + 10
    const RB: Rect = { x: bx, y: by, c }, RA: Rect = { x: bx - 4 * c - 24, y: by + 4 * c + 24, c }, RC: Rect = { x: bx, y: RA.y, c }
    const fin = eout(clamp(p / 0.1)), g = gemm((p - 0.1) / 0.6, 1, 3, 4, 'fast')
    mk.drawMat({ r: RA, vals: ONEHOT, kind: 'w', alpha: fin, name: 'onehot(2)', shape: '1 × 4', label: 'bottom' })
    mk.drawMat({ r: RB, vals: Wm, kind: 'w', alpha: fin, name: 'M', shape: '4 × 3' })
    const lit = eout(clamp((p - 0.3) / 0.3))
    ctx.strokeStyle = rgba(C.ink, lit); ctx.lineWidth = 2; ctx.strokeRect(RB.x - 1, RB.y + 2 * c - 1, 3 * c + 2, c + 2)
    ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, lit); ctx.fillText('← row 2', RB.x + 3 * c + 10, RB.y + 2.5 * c)
    mk.drawMat({ r: RC, vals: [Wm[2]], kind: 'score', alpha: fin, name: 'onehot · M', shape: '1 × 3', label: 'bottom', reveal: g.rev })
    mk.formula = { segs: [['[0, 0, 1, 0] · M  =  row 2 of M  =  ', C.mute], [`[${Wm[2].map(fmtF).join(', ')}]`, C.ink]], note: 'Every term with a 0 vanishes, so the product just picks out one row. That is all an embedding lookup is: onehot(id) · W_E = row id of W_E, done as an index in practice.' }
  }

  function draw() {
    useCtx(ctx)
    stage.begin()
    mk.begin()
    const pm = prog('matmul'), ps = prog('softmax'), po = prog('onehot')
    if (pm <= 0) sceneDot(prog('dot'))
    else if (ps <= 0) sceneMatmul(pm)
    else if (po <= 0) sceneSoftmax(ps)
    else sceneOnehot(po)
    mk.drawFormula()
  }

  const CAPS: Record<string, [string, string]> = {
    dot: ['A dot product multiplies two vectors number by number and adds the results. It works as a similarity score: large when the vectors point the same way.', 'a · b = Σ aₖ bₖ'],
    matmul: ['A matrix product is a grid of dot products: cell (i, j) takes row i of A and column j of B. Every page draws it this way, with A on the left and B above.', '[3 × 4] · [4 × 3] → [3 × 3]'],
    softmax: ['Softmax turns any list of scores into positive weights that sum to 1, keeping their order. Attention and next-token prediction both use it.', 'exp(sₖ) / Σ exp(s)'],
    onehot: ['A one-hot row has a single 1. Multiplied by a matrix, it simply picks out one row: the idea behind the embedding lookup.', 'onehot · M = one row'],
  }
  if (reduced && player.t === 0) player.t = player.start('matmul') + 5
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
