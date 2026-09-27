import { createFrame, toggle } from '../../core/frame'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, blend, rgba } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import { F, chipW, drawChip, fillRich, mathName, mathRun, plate, subLabel, useCtx } from '../../core/draw'
import { MatrixKit, fmt, gemm, lr, type M, type Rect } from '../../core/matrix'
import { teach } from '../learn'
import type { Nav } from '../registry'
import { TOY, attention, laneMix, matmul, promptTokens, transpose, type Head } from './model'

/*
 * Attention, opened up. One head of block 1 at toy scale (d_model 8, d_head 4), with every
 * matrix product drawn as a GEMM: A on the left, B above, C where A's row meets B's column.
 * Everything drawn is a pure function of the player's time, so the timeline scrubs freely.
 */

const PHASES = [
  { id: 'qkv', name: 'Projections', short: 'Q · K · V', dur: 8 },
  { id: 'scores', name: 'Scores', short: 'QKᵀ', dur: 6.5 },
  { id: 'scale', name: 'Scale', short: '÷ √d', dur: 5 },
  { id: 'mask', name: 'Causal mask', short: 'Mask', dur: 5.5 },
  { id: 'softmax', name: 'Softmax', dur: 5 },
  { id: 'av', name: 'Weighted sum', short: 'A · V', dur: 7.5 },
  { id: 'out', name: 'Output projection', short: 'W_O', dur: 6 },
]

const SUBS = '₀₁₂₃₄₅₆₇₈₉'

export function mountAttention(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const seq = promptTokens(), N = seq.length, R = attention(seq)
  const S = { head: 0, brk: 0 }

  const frame = createFrame(root, {
    formula: true,
    eyebrow: 'Anatomy · Attention',
    title: 'Attention',
    subtitle: 'causal self-attention · block 1 · head 1',
    back: { label: 'Forward pass', onClick: () => nav('anatomy') },
    specs: [
      { label: 'shown', value: 'toy scale' },
      { label: 'tokens', value: String(N) },
      { label: 'd_model', value: String(TOY.d), real: '768' },
      { label: 'd_head', value: String(TOY.dh), real: '64' },
      { label: 'heads', value: String(TOY.heads), real: '12' },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 470, 'Step-by-step attention: X times W_Q, W_K and W_V gives Q, K and V; Q times K transposed gives scores, which are scaled, masked and softmaxed into attention weights A; A times V is projected by W_O and added back to the residual stream.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  teach(player, 'attention')
  toggle(player.meta, 'Attention head', ['head 1', 'head 2'], 0, (i) => {
    S.head = i
    frame.setSubtitle(`causal self-attention · block 1 · head ${i + 1}`)
  })
  const prog = (id: string) => player.prog(id)
  // "break it": recompute this head without the ÷√d scale or without the causal mask
  const BREAKS = ['as trained', 'no ÷√d', 'no mask']
  toggle(player.meta, 'Break it', BREAKS, 0, (i) => { S.brk = i })
  function broken(h: Head): Head {
    if (!S.brk) return h
    const noMask = S.brk === 2, Ss = S.brk === 1 ? h.S : h.Ss
    const A = Ss.map((r, i) => {
      const vis = r.map((_, j) => noMask || j <= i), m = Math.max(...r.filter((_, j) => vis[j]))
      const e = r.map((v, j) => (vis[j] ? Math.exp(v - m) : 0)), z = e.reduce((a, b) => a + b, 0)
      return e.map((v) => v / z)
    })
    return { ...h, Ss, A, O: matmul(A, h.V) }
  }
  const hd = () => broken(R.heads[S.head])
  /** What the break changes, for the notes. */
  const breakNote = () => {
    if (S.brk === 1) {
      const peak = (A: M) => A.reduce((s, r) => s + Math.max(...r), 0) / A.length
      return ` Broken: without ÷√d the rows get sharper (largest weight per row ${fmt(peak(hd().A))} instead of ${fmt(peak(R.heads[S.head].A))}). At GPT-2's d_head 64 the scores would be 8× too large and each row would collapse onto one token.`
    }
    if (S.brk === 2) return ` Broken: without the mask ${tl(0)} attends to ${tl(1)} and later tokens. In training that is reading the answer: position i could copy token i + 1.`
    return ''
  }

  /* ---------- layouts, one per scene ---------- */
  const pad = 36, tokW = 76, top = 60, bot = 46
  const G = {
    qkv: { lx: 0, lnX: 0, X: r0(), Wq: r0(), Wk: r0(), Wv: r0(), Q: r0(), K: r0(), V: r0() },
    scores: { Q: r0(), KT: r0(), S: r0(), Vp: r0(), opsX: 0 },
    av: { A: r0(), V: r0(), O: r0(), px: 0 },
    out: { CC: r0(), Wo: r0(), Out: r0() },
  }
  function r0(): Rect { return { x: 0, y: 0, c: 30 } }
  function geom() {
    const { W, H } = stage
    const fit = (cols: number, rows: number, exW: number, exH: number) =>
      Math.floor(clamp(Math.min((W - 2 * pad - tokW - exW) / cols, (H - top - bot - exH) / rows), 18, 40))
    const cx = (tw: number) => Math.max(pad, (W - tw) / 2)
    const cy = (th: number, exTop: number) => top + exTop + Math.max(0, (H - top - bot - th) / 2)
    {
      const lw = 84, c = fit(22.4, 14, lw, 56), lx = cx(lw + tokW + 22.4 * c), x0 = lx + lw + tokW, y0 = cy(14 * c + 56, 24)
      const col = (k: number) => x0 + (9 + k * 4.7) * c
      Object.assign(G.qkv, {
        lx, lnX: lx + 42, X: { x: x0, y: y0 + 9 * c, c },
        Wq: { x: col(0), y: y0, c }, Wk: { x: col(1), y: y0, c }, Wv: { x: col(2), y: y0, c },
        Q: { x: col(0), y: y0 + 9 * c, c }, K: { x: col(1), y: y0 + 9 * c, c }, V: { x: col(2), y: y0 + 9 * c, c },
      })
    }
    {
      const c = fit(14, 10, 214, 70), x0 = cx(tokW + 14 * c + 214) + tokW, y0 = cy(10 * c + 40, 36)
      Object.assign(G.scores, {
        Q: { x: x0, y: y0 + 5 * c, c }, KT: { x: x0 + 5 * c, y: y0, c }, S: { x: x0 + 5 * c, y: y0 + 5 * c, c },
        Vp: { x: x0 + 10 * c + 214, y: y0 + 5 * c, c }, opsX: x0 + 10 * c + 64,
      })
    }
    {
      const c = fit(10, 11, 300, 70), x0 = cx(tokW + 10 * c + 300) + tokW, y0 = cy(11 * c + 40, 24)
      Object.assign(G.av, { V: { x: x0 + 6 * c, y: y0, c }, A: { x: x0, y: y0 + 6 * c, c }, O: { x: x0 + 6 * c, y: y0 + 6 * c, c }, px: x0 + 10 * c + 52 })
    }
    {
      const c = fit(17, 14, 250, 56), x0 = cx(tokW + 17 * c + 250) + tokW, y0 = cy(14 * c + 56, 24)
      Object.assign(G.out, { CC: { x: x0, y: y0 + 9 * c, c }, Wo: { x: x0 + 9 * c, y: y0, c }, Out: { x: x0 + 9 * c, y: y0 + 9 * c, c } })
    }
  }
  stage.onResize = geom
  geom()

  /* ---------- matrix drawing ---------- */
  const mk = new MatrixKit(stage, seq, frame.setFormula)
  const { tokRGB, tl } = mk
  const drawMat = mk.drawMat.bind(mk), rowChips = mk.rowChips.bind(mk), paintCell = mk.paintCell.bind(mk)
  const paintAttn = mk.paintAttn.bind(mk), hatch = mk.hatch.bind(mk), cellText = mk.cellText.bind(mk)
  const resolve = mk.resolve.bind(mk), gemmOverlay = mk.gemmOverlay.bind(mk)

  /* ---------- colour of a token after attention (matches the overview's lanes) ---------- */
  const hues = () => seq.map((_, i) => tokRGB(i))
  const oCols = (A: M) => A.map((row) => blend(hues(), row))
  const laneCols = () => laneMix(R).map((w) => blend(hues(), w))

  /* ---------- scene 1: X · W_Q, W_K, W_V ---------- */
  function sceneQKV(p: number) {
    const L = G.qkv, h = hd(), c = L.X.c, pin = eout(clamp(p / 0.06)), pw = eout(clamp((p - 0.02) / 0.07))
    seq.forEach((t, i) => {
      const y = L.X.y + (i + 0.5) * c
      ctx.strokeStyle = rgba(tokRGB(i), 0.85 * pin); ctx.lineWidth = 1.6
      ctx.beginPath(); ctx.moveTo(L.lx, y); ctx.lineTo(L.lnX - 9, y); ctx.moveTo(L.lnX + 9, y); ctx.lineTo(L.X.x - 18 - chipW(t.text), y); ctx.stroke()
    })
    plate(L.lnX, L.X.y - 2, L.X.y + N * c + 2, p > 0 && p < 0.08 ? 1 : 0, { w: 7, d: 8 })
    subLabel('ln_1', L.lnX, L.X.y - 16, false)
    ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, pin); ctx.fillText('h', L.lx, L.X.y - 16)
    rowChips(L.X, pin)
    drawMat({ r: L.X, vals: R.X, kind: 'row', alpha: pin, name: 'X', shape: '5 × 8', real: 'N × 768' })
    const st = {
      Q: { g: gemm((p - 0.08) / 0.52, N, 4, 8, 'slow'), K: 8 },
      K: { g: gemm((p - 0.6) / 0.18, N, 4, 8, 'fast'), K: 8 },
      V: { g: gemm((p - 0.78) / 0.18, N, 4, 8, 'fast'), K: 8 },
    }
    const W = { Q: h.Wq, K: h.Wk, V: h.Wv }, Rm = { Q: h.Q, K: h.K, V: h.V }
    const Wr = { Q: L.Wq, K: L.Wk, V: L.Wv }, Rr = { Q: L.Q, K: L.K, V: L.V }
    ;(['Q', 'K', 'V'] as const).forEach((k) => {
      drawMat({ r: Wr[k], vals: W[k], kind: 'w', alpha: pw, name: 'W_' + k, shape: '8 × 4', real: '768 × 64' })
      drawMat({ r: Rr[k], vals: Rm[k], kind: 'row', alpha: pw, name: k, shape: '5 × 4', real: 'N × 64', label: 'bottom', reveal: st[k].g.rev })
      mk.hit(k, Rr[k], N, 4)
    })
    const f = resolve(st)
    if (f) {
      const k = f.key as 'Q' | 'K' | 'V', role = { Q: 'query', K: 'key', V: 'value' }[k]
      gemmOverlay({ A: L.X, Av: R.X, B: Wr[k], Bv: W[k], C: Rr[k], f, names: [k, 'X', 'W_' + k], note: `${tl(f.i)}'s ${role} vector, dim ${f.j}: row ${f.i} of X times column ${f.j} of W_${k}, term by term, summed.` })
    }
  }

  /* ---------- scene 2: Q · Kᵀ, then scale, mask, softmax ---------- */
  function transposeCells(K: M, from: Rect, to: Rect, t: number) {
    const e = eio(t), vmax = Math.max(...K.flat().map(Math.abs))
    K.forEach((row, i) => row.forEach((v, j) => {
      const c = lerp(from.c, to.c, e)
      const x = lerp(from.x + j * from.c, to.x + i * to.c, e), y = lerp(from.y + i * from.c, to.y + j * to.c, e)
      const fa = paintCell(x, y, c, v, vmax, tokRGB(i), 0, 1)
      cellText(fmt(v), x, y, c, fa, 1)
    }))
  }
  function drawOps(x: number, y0: number, c: number, steps: [string, number, string][], a: number) {
    steps.forEach(([name, p, note], k) => {
      const y = y0 + 10 + k * Math.max(34, c * 1.1), done = p >= 1, on = p > 0 && p < 1
      ctx.beginPath(); ctx.arc(x + 4, y - 4, 3.5, 0, 7)
      if (done) { ctx.fillStyle = rgba(C.ink2, a); ctx.fill() } else { ctx.strokeStyle = rgba(on ? C.ink : C.faint, a); ctx.lineWidth = 1.2; ctx.stroke() }
      ctx.font = F.mono(12, on ? 500 : 400); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
      ctx.fillStyle = rgba(on ? C.ink : done ? C.ink2 : C.mute, a); fillRich(name, x + 16, y)
      if (on || done) { ctx.font = F.body; ctx.fillStyle = rgba(C.mute, a * (on ? 1 : 0.7)); fillRich(note, x + 16, y + 16) }
    })
  }
  function sceneScores(ps: number, pc: number, pm: number, pso: number) {
    const Lq = G.qkv, L = G.scores, h = hd(), tr = eio(clamp(ps / 0.2)), c = L.S.c
    if (tr < 1) {
      const fa = 1 - tr
      drawMat({ r: Lq.X, vals: R.X, kind: 'row', alpha: fa, name: 'X', shape: '5 × 8' })
      drawMat({ r: Lq.Wq, vals: h.Wq, kind: 'w', alpha: fa, name: 'W_Q', shape: '8 × 4' })
      drawMat({ r: Lq.Wk, vals: h.Wk, kind: 'w', alpha: fa, name: 'W_K', shape: '8 × 4' })
      drawMat({ r: Lq.Wv, vals: h.Wv, kind: 'w', alpha: fa, name: 'W_V', shape: '8 × 4' })
      transposeCells(h.K, Lq.K, L.KT, tr)
    } else drawMat({ r: L.KT, vals: transpose(h.K), kind: 'col', alpha: 1, name: 'Kᵀ', shape: '4 × 5', real: '64 × N', colToks: true })
    rowChips(lr(Lq.X, L.Q, tr), 1)
    drawMat({ r: lr(Lq.Q, L.Q, tr), vals: h.Q, kind: 'row', alpha: 1, name: 'Q', shape: '5 × 4', real: 'N × 64', labelAlpha: tr })
    drawMat({ r: lr(Lq.V, L.Vp, tr), vals: h.V, kind: 'row', alpha: lerp(1, 0.35, tr), name: 'V', shape: '5 × 4 · used later', labelAlpha: tr })

    const g = gemm((ps - 0.22) / 0.75, N, N, 4, 'slow')
    const scaleT = eio(clamp(pc / 0.6))
    const vals = h.S.map((r, i) => r.map((v, j) => lerp(v, h.Ss[i][j], scaleT)))
    const vmax = Math.max(...vals.flat().map(Math.abs))
    const maskP = (i: number, j: number) => (j > i && S.brk !== 2 ? clamp((pm - (j - i - 1) * 0.12) / 0.4) : 0)
    const softR = (i: number) => clamp((pso - i * 0.13) / 0.35)
    const named = pso > 0.5 ? 'A' : scaleT > 0.5 ? 'S′' : 'S'
    drawMat({
      r: L.S, vals, kind: 'score', alpha: tr, name: named, shape: '5 × 5', real: 'N × N', reveal: g.rev,
      paint: (x, y, cc, i, j, a) => {
        const m = maskP(i, j), sr = softR(i)
        let fa = 0
        if (m < 1) fa = paintCell(x, y, cc, vals[i][j], vmax, null, 0.45, a * (1 - sr) * (1 - m))
        if (sr > 0 && (j <= i || S.brk === 2)) fa = Math.max(fa, paintAttn(x, y, cc, h.A[i][j], tokRGB(j), a * sr))
        if (m > 0) hatch(x, y, cc, a * m)
        return m > 0.5 ? 0 : sr > 0.5 ? 0.05 + 0.9 * Math.sqrt(h.A[i][j]) : fa
      },
      text: (i, j) => (maskP(i, j) > 0.5 ? (softR(i) > 0.5 ? '0' : '−∞') : softR(i) > 0.5 ? fmt(h.A[i][j]) : fmt(vals[i][j])),
    })
    mk.hit('S', L.S, N, N)
    // row sums once a row has been normalised
    for (let i = 0; i < N; i++) {
      const a = clamp((softR(i) - 0.8) / 0.2)
      if (a <= 0) continue
      ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
      ctx.fillStyle = rgba(C.mute, a); ctx.fillText('Σ 1.00', L.S.x + N * c + 12, L.S.y + (i + 0.5) * c)
    }
    drawOps(L.opsX, L.S.y, c, [
      ['Q · Kᵀ', clamp((ps - 0.22) / 0.75), 'one dot product per cell'],
      ['÷ √d_head', pc, S.brk === 1 ? 'skipped (broken)' : `÷ ${Math.sqrt(TOY.dh)}  (GPT-2: ÷ 8)`],
      ['mask', pm, S.brk === 2 ? 'skipped (broken)' : 'set j > i to −∞'],
      ['softmax', pso, 'each row sums to 1'],
    ], tr)

    const f = resolve({ S: { g, K: 4 } })
    if (f && pc === 0) {
      gemmOverlay({ A: L.Q, Av: h.Q, B: L.KT, Bv: transpose(h.K), C: L.S, f, names: ['S', 'Q', 'Kᵀ'], note: `${tl(f.i)}'s query · ${tl(f.j)}'s key: the higher the score, the more ${tl(f.i)} attends to ${tl(f.j)}.` })
    } else if (pso > 0) {
      const i = mk.focus?.key === 'S' ? mk.focus.i : Math.min(N - 1, Math.floor(clamp(pso / 0.87) * N))
      const n = S.brk === 2 ? N : i + 1, row = h.Ss[i].slice(0, n), arow = h.A[i].slice(0, n)
      mk.formula = {
        segs: [[`A[${i}]`, C.ink], ['  =  softmax( ', C.mute], [row.map(fmt).join(', '), C.ink2], [' )  =  ', C.mute], [arow.map((v) => v.toFixed(2)).join(', '), C.ink]],
        note: `The ${n} tokens ${tl(i)} can see: exponentiate, divide by the sum. The largest score takes most of the weight.${breakNote()}`,
      }
      ctx.strokeStyle = rgba(C.ink, 0.9); ctx.lineWidth = 1.5; ctx.strokeRect(L.S.x - 1, L.S.y + i * c - 1, N * c + 2, c + 2)
    } else if (pm > 0) {
      mk.formula = S.brk === 2
        ? { segs: [['no mask', C.ink], ['    (broken)', C.mute]], note: 'Every token now sees every other one, later ones included.' + breakNote() }
        : { segs: [['S′[i,j]  =  −∞', C.ink], ['    when j > i', C.mute]], note: 'Token i sees only itself and earlier tokens. exp(−∞) = 0, so these weights become 0.' }
    } else if (pc > 0) {
      mk.formula = { segs: [['S′  =  S / √d_head  =  S / ', C.ink], [String(Math.sqrt(TOY.dh)), C.ink]], note: `After scaling, the ${tl(N - 1)} row goes from [${h.S[N - 1].map(fmt).join(', ')}] to [${h.Ss[N - 1].map(fmt).join(', ')}].` }
    }
  }

  /* ---------- scene 3: A · V ---------- */
  function attnPaint(A: M) {
    return (x: number, y: number, c: number, i: number, j: number, a: number) => {
      if (j > i && S.brk !== 2) { hatch(x, y, c, a); return 0 }
      return paintAttn(x, y, c, A[i][j], tokRGB(j), a)
    }
  }
  const attnText = (A: M) => (i: number, j: number) => (j > i && S.brk !== 2 ? '' : fmt(A[i][j]))
  function drawPanel(i: number, a: number) {
    const L = G.av, h = hd(), c = L.O.c, rh = clamp(c * 0.82, 22, 30), s = Math.min(rh * 0.68, 16)
    const x = L.px, y0 = L.A.y, vmax = Math.max(...h.V.flat().map(Math.abs))
    mathRun([['o', false], [String(i), true], [' = Σ', false], ['j', true], [' a', false], [`${i}j`, true], [' · v', false], ['j', true]], x, y0 - 12, a)
    const amax = Math.max(...h.A[i]), n = S.brk === 2 ? N : i + 1
    for (let j = 0; j < n; j++) {
      const y = y0 + j * rh + rh / 2, w = h.A[i][j], wa = a * (0.25 + 0.75 * (w / amax))
      drawChip(x, y, seq[j], wa, Math.min(18, rh * 0.75), false, F.mono(11, 500))
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
      ctx.fillStyle = rgba(C.ink2, wa); ctx.fillText('× ' + w.toFixed(2), x + 60, y)
      for (let k = 0; k < TOY.dh; k++) paintCell(x + 118 + k * (s + 2), y - s / 2, s, h.V[j][k] * w / amax, vmax, tokRGB(j), 0, a)
    }
    const ys = y0 + n * rh + 6, xr = x + 118 + TOY.dh * (s + 2)
    ctx.strokeStyle = rgba(C.ink, 0.5 * a); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x + 118, ys); ctx.lineTo(xr - 2, ys); ctx.stroke()
    const oc = oCols(h.A)[i], omax = Math.max(...h.O.flat().map(Math.abs))
    ctx.font = F.mono(11); ctx.fillStyle = rgba(C.mute, a); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    ctx.fillText(`o${SUBS[i]} =`, x + 110, ys + 6 + s / 2)
    for (let k = 0; k < TOY.dh; k++) paintCell(x + 118 + k * (s + 2), ys + 6, s, h.O[i][k], omax, oc, 0, a)
  }
  function sceneAV(pav: number) {
    const Ls = G.scores, L = G.av, h = hd(), tr = eio(clamp(pav / 0.15))
    if (tr < 1) {
      drawMat({ r: Ls.Q, vals: h.Q, kind: 'row', alpha: 1 - tr, name: 'Q', shape: '5 × 4' })
      drawMat({ r: Ls.KT, vals: transpose(h.K), kind: 'col', alpha: 1 - tr, name: 'Kᵀ', shape: '4 × 5', colToks: true })
    }
    rowChips(lr(Ls.Q, L.A, tr), 1, mk.hovered('A', 'O')?.i ?? -1)
    drawMat({ r: lr(Ls.S, L.A, tr), vals: h.A, kind: 'attn', alpha: 1, name: 'A', shape: '5 × 5', real: 'N × N', colToks: tr > 0.5, paint: attnPaint(h.A), text: attnText(h.A) })
    const vr = lr(Ls.Vp, L.V, tr)
    drawMat({ r: vr, vals: h.V, kind: 'row', alpha: lerp(0.35, 1, tr), name: 'V', shape: '5 × 4', real: 'N × 64' })
    ctx.font = F.mono(10); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    for (let j = 0; j < N; j++) { ctx.fillStyle = rgba(tokRGB(j), tr); ctx.fillText(tl(j), vr.x - 8, vr.y + (j + 0.5) * vr.c) }
    const g = gemm((pav - 0.17) / 0.78, N, TOY.dh, N, 'slow')
    drawMat({ r: L.O, vals: h.O, kind: 'row', alpha: tr, name: 'O', shape: '5 × 4', real: 'N × 64', reveal: g.rev, rowCols: oCols(h.A) })
    mk.hit('O', L.O, N, TOY.dh); mk.hit('A', L.A, N, N)
    const f = resolve({ O: { g, K: N } })
    if (f) gemmOverlay({ A: L.A, Av: h.A, B: L.V, Bv: h.V, C: L.O, f, names: ['O', 'A', 'V'], note: `${tl(f.i)}'s output, dim ${f.j}: dim ${f.j} of every value, weighted by ${tl(f.i)}'s attention.` })
    const row = mk.hovered('A', 'O')?.i ?? (f ? f.i : pav >= 0.95 ? N - 1 : -1)
    if (row >= 0) drawPanel(row, tr)
  }

  /* ---------- scene 4: concat heads · W_O, + residual ---------- */
  function sceneOut(po: number) {
    const La = G.av, L = G.out, h = hd(), tr = eio(clamp(po / 0.15)), c = L.CC.c
    if (tr < 1) {
      drawMat({ r: La.A, vals: h.A, kind: 'attn', alpha: 1 - tr, name: 'A', shape: '5 × 5', paint: attnPaint(h.A), text: attnText(h.A), colToks: true })
      drawMat({ r: La.V, vals: h.V, kind: 'row', alpha: 1 - tr, name: 'V', shape: '5 × 4' })
    }
    rowChips(lr(La.A, L.CC, tr), 1)
    const slot = (k: number): Rect => ({ x: L.CC.x + k * TOY.dh * c, y: L.CC.y, c })
    const other = 1 - S.head
    drawMat({ r: slot(other), vals: R.heads[other].O, kind: 'row', alpha: tr, name: 'O', shape: '', label: 'none', rowCols: oCols(R.heads[other].A) })
    drawMat({ r: lr(La.O, slot(S.head), tr), vals: h.O, kind: 'row', alpha: 1, name: 'O', shape: '', label: 'none', rowCols: oCols(h.A) })
    // head brackets + concat label
    ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'
    for (let k = 0; k < TOY.heads; k++) {
      const s = slot(k)
      ctx.fillStyle = rgba(k === S.head ? C.ink2 : C.mute, tr); ctx.fillText(`head ${k + 1}`, s.x + (TOY.dh * c) / 2, s.y - 8)
    }
    const la = clamp((tr - 0.6) / 0.4)
    const w1 = mathName('concat', L.CC.x, L.CC.y - 26, la, 18)
    ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, la); ctx.fillText('5 × 8', L.CC.x + w1 + 8, L.CC.y - 26)
    ctx.fillStyle = rgba(C.mute, 0.55 * la); ctx.fillText('· N × 768', L.CC.x + w1 + 8 + ctx.measureText('5 × 8').width + 6, L.CC.y - 26)

    const pw = clamp((po - 0.1) / 0.1)
    drawMat({ r: L.Wo, vals: R.Wo, kind: 'w', alpha: pw, name: 'W_O', shape: '8 × 8', real: '768 × 768' })
    const g = gemm((po - 0.18) / 0.5, N, TOY.d, TOY.d, 'fast')
    const mixed = R.heads[0].A.map((_, i) => blend(hues(), hues().map((__, k) => (R.heads[0].A[i][k] + R.heads[1].A[i][k]) / 2)))
    drawMat({ r: L.Out, vals: R.out, kind: 'row', alpha: pw, name: 'attn_out', shape: '5 × 8', real: 'N × 768', reveal: g.rev, rowCols: mixed })
    mk.hit('Out', L.Out, N, TOY.d)

    // residual add, then back into the stream
    const pr = clamp((po - 0.72) / 0.22)
    if (pr > 0) {
      const xp = L.Out.x + TOY.d * c + 24, xEnd = Math.min(stage.W - pad, xp + 230), lc = laneCols()
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, pr)
      ctx.fillText('+ h', xp, L.Out.y - 10)
      for (let i = 0; i < N; i++) {
        const y = L.Out.y + (i + 0.5) * c, e = eio(clamp(pr * 1.3 - i * 0.06))
        ctx.strokeStyle = rgba(C.ink, 0.35 * pr); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(L.Out.x + TOY.d * c + 6, y); ctx.lineTo(xp - 7, y); ctx.stroke()
        ctx.beginPath(); ctx.arc(xp, y, 7, 0, 7); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(xp - 3.5, y); ctx.lineTo(xp + 3.5, y); ctx.moveTo(xp, y - 3.5); ctx.lineTo(xp, y + 3.5); ctx.stroke()
        ctx.strokeStyle = rgba(lc[i], 0.9 * e); ctx.lineWidth = 1.8
        ctx.beginPath(); ctx.moveTo(xp + 8, y); ctx.lineTo(lerp(xp + 8, xEnd, e), y); ctx.stroke()
      }
      ctx.font = F.small; ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.mute, clamp(pr * 2 - 1))
      fillRich('→ ln_2 · MLP', xEnd, L.Out.y - 10)
    }

    const f = resolve({ Out: { g, K: TOY.d } })
    if (f) gemmOverlay({ A: L.CC, Av: R.concat, B: L.Wo, Bv: R.Wo, C: L.Out, f, names: ['out', 'concat', 'W_O'], note: `${tl(f.i)}'s attention output, dim ${f.j}: W_O mixes what the two heads found.` })
    else if (pr > 0) mk.formula = { segs: [['h′  =  h + attn_out', C.ink]], note: "The result is added to the residual stream, not substituted for it. Each row's color is what that token gathered from the others." }
  }

  /* ---------- frame ---------- */
  function draw() {
    useCtx(ctx)
    stage.begin()
    mk.begin()
    const ps = prog('scores'), pav = prog('av'), po = prog('out')
    if (ps <= 0) sceneQKV(prog('qkv'))
    else if (pav <= 0) sceneScores(ps, prog('scale'), prog('mask'), prog('softmax'))
    else if (po <= 0) sceneAV(pav)
    else sceneOut(po)
    mk.drawFormula()
  }

  const CAPS: Record<string, [string, string]> = {
    qkv: ['X (after ln_1) is multiplied by W_Q, W_K and W_V. A query is what a token is looking for, a key is what it offers, and a value is what it hands over when chosen. For example, Ġthe might look for the noun it belongs to. GPT-2 does all three in one GEMM, X · W_qkv.', 'GPT-2 [N×768]·[768×2304] · 17.7 MFLOPs'],
    scores: ["Q times K transposed. Row i, column j is the dot product of token i's query with token j's key: how much i should attend to j. Each head works in its own slice of d_model / heads numbers (64 in GPT-2, 4 here).", 'GPT-2 12 × [N×64]·[64×N]'],
    scale: ['Divide by √d_head. Dot products grow with dimension; scaling keeps softmax from saturating from the start.', 'toy ÷ 2 · GPT-2 ÷ 8'],
    mask: ['Position i predicts token i + 1, so it may only look at positions ≤ i: the upper triangle is set to −∞. In training every position is predicted at once, and without the mask each could just read the next word.', 'causal: j > i → −∞'],
    softmax: ["Softmax each row, turning scores into weights that sum to 1; −∞ becomes 0 after exp. Each row is one token's attention distribution. These toy weights are random, so the pattern means nothing; the Forward pass shows GPT-2's real heads.", 'A = softmax(S / √d + mask)'],
    av: ['Weight the rows of V by attention and sum them. Output row i blends the values of every visible token, and its color blends with them. The panel on the right breaks down the current row.', 'GPT-2 12 × [N×N]·[N×64]'],
    out: ["Concatenate the heads' outputs, multiply by W_O to mix them, and add the result back to the residual stream. Several small heads can each follow a different relation for the cost of one big one. In code: (B, T, 768) is split into (B, 12, T, 64) for attention and merged back here.", 'GPT-2 [N×768]·[768×768] · 5.9 MFLOPs'],
  }

  if (reduced && player.t === 0) player.t = player.start('av') + 4
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
