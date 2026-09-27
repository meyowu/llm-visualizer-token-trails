import { F, chipW, drawChip, fillRich, plate, rr, serifAt, spaced, upper, useCtx } from '../../core/draw'
import { createFrame } from '../../core/frame'
import { MatrixKit, fmt } from '../../core/matrix'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, gauss, lerp, reducedMotion, rng } from '../../core/util'
import type { Nav } from '../registry'
import { embedRow, gelu, promptTokens } from '../transformer/model'

/*
 * LLaMA as a diff against GPT-2. The block keeps its shape (pre-norm, residual adds, causal
 * attention); four parts change: position (learned W_P → RoPE inside attention), normalisation
 * (LayerNorm → RMSNorm), MLP (GELU → SwiGLU) and attention heads (MHA → GQA).
 * Real numbers are LLaMA 3 8B; the small drawings use toy vectors.
 */

const PHASES = [
  { id: 'blocks', name: 'GPT-2 vs LLaMA', short: 'Blocks', dur: 7 },
  { id: 'rope', name: 'Rotary positions', short: 'RoPE', dur: 11 },
  { id: 'rms', name: 'RMSNorm', dur: 7 },
  { id: 'swiglu', name: 'SwiGLU MLP', short: 'SwiGLU', dur: 9 },
  { id: 'gqa', name: 'Grouped-query attention', short: 'GQA', dur: 8 },
]
const DH = 8, PAIRS = DH / 2, BASE = 10000 // toy RoPE: θ_j = 10000^(−2j/8) = 1, 0.1, 0.01, 0.001
const theta = (j: number) => Math.pow(BASE, (-2 * j) / DH)
const XR = 4
const silu = (x: number) => x / (1 + Math.exp(-x))

export function mountLlama(root: HTMLElement, _nav: Nav): () => void {
  const reduced = reducedMotion()
  const seq = promptTokens(), N = seq.length, D = 8
  const frame = createFrame(root, {
    formula: true,
    eyebrow: 'Lineage · Decoder-only',
    title: 'LLaMA',
    subtitle: 'LLaMA 3 8B · what changed since GPT-2',
    specs: [
      { label: 'compared', value: 'LLaMA 3 8B', real: 'GPT-2 small' },
      { label: 'layers', value: '32', real: '12' },
      { label: 'd_model', value: '4,096', real: '768' },
      { label: 'heads', value: '32 q · 8 kv', real: '12' },
      { label: 'd_ff', value: '14,336', real: '3,072' },
      { label: 'vocab', value: '128,256', real: '50,257' },
      { label: 'context', value: '8,192', real: '1,024' },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 470, 'LLaMA compared with GPT-2: rotary position embeddings inside attention, RMSNorm instead of LayerNorm, a gated SwiGLU MLP, and grouped-query attention that shares keys and values across query heads.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  const prog = (id: string) => player.prog(id)
  const mk = new MatrixKit(stage, seq, frame.setFormula)
  const hue = (i: number): RGB => C.tok[seq[i].c % 7]
  const pad = 36, top = 56, bot = 46

  // toy vectors: per-token query/key (8 = 4 rotation pairs) and the residual rows used for the norms
  const qv = seq.map((t) => { const r = rng(t.id * 131 + 7); return Array.from({ length: DH }, () => gauss(r)) })
  const kv = seq.map((t) => { const r = rng(t.id * 137 + 11); return Array.from({ length: DH }, () => gauss(r)) })
  const h = seq.map((t, i) => embedRow(t, i).map((v) => v + 0.6)) // a shifted stream, so centring visibly matters
  const rotate = (v: number[], pos: number) => v.map((_, k) => {
    const j = Math.floor(k / 2), a = pos * theta(j), x = v[2 * j], y = v[2 * j + 1]
    return k % 2 === 0 ? x * Math.cos(a) - y * Math.sin(a) : x * Math.sin(a) + y * Math.cos(a)
  })
  const dot = (a: number[], b: number[]) => a.reduce((s, v, k) => s + v * b[k], 0)

  /* ---------- helpers ---------- */
  let pills: { x: number; y: number; w: number; h: number; phase: string }[] = []
  let hoverPill = ''
  function pill(t: string, x: number, y: number, phase: string, a: number) {
    ctx.font = F.mono(11, 500)
    const w = ctx.measureText(t).width + 16, hv = hoverPill === phase
    rr(x - w / 2, y - 10, w, 20, 10)
    ctx.fillStyle = rgba(C.ink, (hv ? 1 : 0.9) * a); ctx.fill()
    ctx.fillStyle = rgba(C.bg, a); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    ctx.fillText(t, x, y + 0.5)
    pills.push({ x: x - w / 2, y: y - 10, w, h: 20, phase })
  }
  function addNode(x: number, y: number, a: number) {
    ctx.fillStyle = rgba(C.bg, a); ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.fill()
    ctx.strokeStyle = rgba(C.ink, 0.8 * a); ctx.lineWidth = 1.2; ctx.stroke()
    ctx.beginPath(); ctx.moveTo(x - 4.5, y); ctx.lineTo(x + 4.5, y); ctx.moveTo(x, y - 4.5); ctx.lineTo(x, y + 4.5); ctx.stroke()
  }
  function caption(t: string, x: number, y: number, a: number, col: RGB = C.mute, align: CanvasTextAlign = 'center') {
    ctx.font = F.small; ctx.textAlign = align; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(col, a); fillRich(t, x, y)
  }
  function title(t: string, x: number, y: number, a: number) {
    ctx.font = F.label; spaced(true); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
    fillRich(upper(t), x, y); spaced(false)
  }

  /* ---------- scene 1: the two blocks ---------- */
  function sceneBlocks(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const sx0 = pad + 130, sx1 = W - pad - 16, fx = (f: number) => lerp(sx0, sx1, f)
    const X = { pos: fx(0.05), n1: fx(0.2), at: fx(0.37), a1: fx(0.52), n2: fx(0.66), ml: fx(0.81), a2: fx(0.95) }
    const rows: [number, string, string, string[], number][] = [
      [top + avail * 0.26, 'GPT-2', 'small · 124M', ['LayerNorm', 'MHA', 'LayerNorm', 'MLP · GELU'], eout(clamp(p / 0.3))],
      [top + avail * 0.72, 'LLaMA 3', '8B', ['RMSNorm', 'GQA + RoPE', 'RMSNorm', 'MLP · SwiGLU'], eout(clamp((p - 0.25) / 0.3))],
    ]
    rows.forEach(([y, name, sub, labs, a], r) => {
      if (a <= 0) return
      ctx.font = serifAt(24); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(name, pad, y + 2)
      ctx.font = F.small; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(sub, pad, y + 20)
      for (let i = 0; i < N; i++) {
        const yy = y + (i - 2) * 5
        ctx.strokeStyle = rgba(hue(i), 0.8 * a); ctx.lineWidth = 1.3
        ctx.beginPath(); ctx.moveTo(sx0, yy); ctx.lineTo(lerp(sx0, sx1, a), yy); ctx.stroke()
      }
      // position: added at the input in GPT-2, absent in LLaMA
      if (r === 0) {
        rr(X.pos - 26, y - 13, 52, 26, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.7 * a); ctx.lineWidth = 1; ctx.stroke()
        ctx.font = F.small; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a); fillRich('+ W_P', X.pos, y + 0.5)
        caption('learned positions', X.pos, y + 44, a)
      } else caption('no position vector', X.pos, y + 44, a, C.faint)
      const at = [X.n1, X.at, X.n2, X.ml]
      at.forEach((x, k) => { plate(x, y - 26, y + 26, r === 1 ? 0.55 : 0.2, { w: 8, d: 9 }); caption(labs[k], x, y + 44, a, r === 1 ? C.ink2 : C.mute) })
      addNode(X.a1, y, a); addNode(X.a2, y, a)
    })
    // what changed: dashed links between the rows, then pills on the LLaMA row
    const la = eout(clamp((p - 0.55) / 0.25)), yA = rows[0][0], yB = rows[1][0]
    if (la > 0) {
      ctx.setLineDash([2, 4]); ctx.strokeStyle = rgba(C.ink, 0.35 * la); ctx.lineWidth = 1
      for (const x of [X.n1, X.at, X.n2, X.ml]) { ctx.beginPath(); ctx.moveTo(x, yA + 52); ctx.lineTo(x, yB - 56); ctx.stroke() }
      ctx.beginPath(); ctx.moveTo(X.pos, yA + 52); ctx.bezierCurveTo(X.pos, (yA + yB) / 2, X.at - 30, (yA + yB) / 2, X.at - 12, yB - 58); ctx.stroke()
      ctx.setLineDash([])
      caption('moves into attention', (X.pos + X.at) / 2 - 10, (yA + yB) / 2 + 4, la, C.ink2)
      const py = yB - 44
      pill('RMSNorm', X.n1, py, 'rms', la); pill('RMSNorm', X.n2, py, 'rms', la)
      pill('RoPE', X.at - 28, py, 'rope', la); pill('GQA', X.at + 26, py, 'gqa', la)
      pill('SwiGLU', X.ml, py, 'swiglu', la)
    }
    mk.formula = { segs: [['h′ = h + attn(norm(h))   ·   h″ = h′ + mlp(norm(h′))', C.ink]], note: 'The block keeps its shape. Four parts change; click a label to open that change.' }
  }

  /* ---------- scene 2: RoPE ---------- */
  function sceneRope(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.08)), e = eio(clamp((p - 0.12) / 0.38))
    const R = Math.floor(clamp(Math.min(avail / (N * 2.5 + 1.5), (W * 0.55 - 150) / (PAIRS * 2.6)), 13, 24))
    const gx0 = pad + 120, gy0 = top + 40 + R, dx = R * 2.6, dy = R * 2.5
    title('each pair of q turns with its position', pad, top + 12, fin)
    for (let j = 0; j < PAIRS; j++) {
      const t = theta(j)
      caption(`θ ${t >= 1 ? '1' : t.toString().replace(/^0/, '')}`, gx0 + j * dx, gy0 - R - 10, fin)
    }
    caption('angle = position × θ   ·   fast pairs spin, slow pairs barely move', gx0 - R, gy0 + (N - 1) * dy + R + 22, fin, C.mute, 'left')
    for (let i = 0; i < N; i++) {
      const y = gy0 + i * dy
      drawChip(pad, y, seq[i], fin, 20)
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, fin)
      ctx.fillText(`pos ${i}`, pad + chipW(seq[i].text) + 8, y)
      for (let j = 0; j < PAIRS; j++) {
        const cx = gx0 + j * dx, a0 = Math.atan2(qv[i][2 * j + 1], qv[i][2 * j]), ang = a0 + e * i * theta(j)
        ctx.strokeStyle = rgba(C.ink, 0.18 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, y, R, 0, 7); ctx.stroke()
        if (i > 0 && e > 0) { ctx.strokeStyle = rgba(hue(i), 0.45 * fin); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, y, R * 0.55, -a0, -ang, true); ctx.stroke() }
        ctx.strokeStyle = rgba(hue(i), fin); ctx.lineWidth = 1.8
        ctx.beginPath(); ctx.moveTo(cx, y); ctx.lineTo(cx + Math.cos(ang) * R * 0.9, y - Math.sin(ang) * R * 0.9); ctx.stroke()
        ctx.fillStyle = rgba(hue(i), fin); ctx.beginPath(); ctx.arc(cx + Math.cos(ang) * R * 0.9, y - Math.sin(ang) * R * 0.9, 2.4, 0, 7); ctx.fill()
      }
    }
    // relative property: q at m = 4 (Ġthe) and k at n = 1 (Ġcat), both shifted by s
    const ra = eout(clamp((p - 0.5) / 0.12)), m = 4, n = 1
    if (ra > 0) {
      const s = eio(clamp((p - 0.66) / 0.28)) * 10
      const px = gx0 + PAIRS * dx + 40, BR = Math.min(avail * 0.28, 76), cx = px + BR + 10, cy = top + 70 + BR
      title('relative, not absolute', px, top + 12, ra)
      ctx.strokeStyle = rgba(C.ink, 0.2 * ra); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, BR, 0, 7); ctx.stroke()
      const aq = Math.atan2(qv[m][1], qv[m][0]) + (m + s) * theta(0), ak = Math.atan2(kv[n][1], kv[n][0]) + (n + s) * theta(0)
      ;[[aq, m, 'q'], [ak, n, 'k']].forEach(([ang, i, l]) => {
        const a = ang as number, ii = i as number
        ctx.strokeStyle = rgba(hue(ii), ra); ctx.lineWidth = 2.2
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a) * BR, cy - Math.sin(a) * BR); ctx.stroke()
        ctx.font = serifAt(16); ctx.fillStyle = rgba(hue(ii), ra); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
        ctx.fillText(l as string, cx + Math.cos(a) * (BR + 14), cy - Math.sin(a) * (BR + 14))
      })
      ctx.strokeStyle = rgba(C.ink, 0.7 * ra); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(cx, cy, BR * 0.35, -aq, -ak, aq < ak); ctx.stroke()
      const score = dot(rotate(qv[m], m + s), rotate(kv[n], n + s))
      const ly = cy + BR + 30
      caption(`q at position ${m + Math.round(s)} (${mk.tl(m)}) · k at ${n + Math.round(s)} (${mk.tl(n)})`, cx, ly, ra, C.ink2)
      caption(`offset m − n = ${m - n}   ·   q·k = ${fmt(score)}`, cx, ly + 18, ra, C.ink)
      caption(s > 0.05 ? `both shifted by +${s.toFixed(1)}: angle and score unchanged` : 'shift both positions…', cx, ly + 36, ra)
    }
    mk.formula = { segs: [["q′ = R(m·θ) q,   k′ = R(n·θ) k   ⇒   q′ · k′ = f(q, k, m − n)", C.ink]], note: 'GPT-2 adds a learned position vector once, at the input. LLaMA rotates q and k in every attention layer, so scores depend only on how far apart two tokens are. LLaMA 3: 64 pairs per head, base 500,000.' }
  }

  /* ---------- scene 3: RMSNorm vs LayerNorm ---------- */
  function sceneRms(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.08)), e1 = eio(clamp((p - 0.12) / 0.33)), e2 = eio(clamp((p - 0.5) / 0.35))
    const rh = Math.min(40, (avail - 60) / N), y0 = top + 46, lx0 = pad + 110, colW = (W - lx0 - pad - 40) / 2
    const panels: [string, number][] = [['LayerNorm · GPT-2', lx0], ['RMSNorm · LLaMA', lx0 + colW + 40]]
    panels.forEach(([name, x0], k) => {
      const x1 = x0 + colW - 90, nx = (v: number) => x0 + ((clamp(v, -XR, XR) + XR) / (2 * XR)) * (x1 - x0)
      title(name, x0, y0 - 18, fin)
      ctx.setLineDash([2, 3]); ctx.strokeStyle = rgba(C.ink, 0.22 * fin); ctx.beginPath(); ctx.moveTo(nx(0), y0 - 6); ctx.lineTo(nx(0), y0 + N * rh); ctx.stroke(); ctx.setLineDash([])
      for (let i = 0; i < N; i++) {
        const y = y0 + (i + 0.5) * rh, v = h[i], mu = v.reduce((a, b) => a + b, 0) / D
        const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / D + 1e-5), rms = Math.sqrt(v.reduce((a, b) => a + b * b, 0) / D + 1e-5)
        ctx.strokeStyle = rgba(C.ink, 0.16 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke()
        const shift = k === 0 ? e1 * mu : 0, scale = lerp(1, k === 0 ? sd : rms, e2)
        const mean = (mu - shift) / scale
        ctx.strokeStyle = rgba(C.ink, 0.85 * fin); ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(nx(mean), y - rh * 0.34); ctx.lineTo(nx(mean), y + rh * 0.34); ctx.stroke()
        v.forEach((x) => { ctx.fillStyle = rgba(hue(i), 0.9 * fin); ctx.beginPath(); ctx.arc(nx((x - shift) / scale), y, 3.3, 0, 7); ctx.fill() })
        ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, fin)
        ctx.fillText(k === 0 ? `μ ${fmt(mu)}  σ ${fmt(sd)}` : `rms ${fmt(rms)}`, x1 + 10, y)
        if (k === 0) drawChip(pad, y, seq[i], fin, Math.min(20, rh * 0.7))
      }
      ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, fin)
      for (const t of [-4, -2, 0, 2, 4]) ctx.fillText(String(t).replace('-', '−'), nx(t), y0 + N * rh + 6)
      caption(k === 0 ? 'subtract μ, divide by σ, then γ and β' : 'divide by rms only, then γ; no β', x0, y0 + N * rh + 34, fin, C.ink2, 'left')
    })
    mk.formula = { segs: [['LayerNorm  γ ⊙ (x − μ) / σ + β', C.ink2], ['     ·     ', C.mute], ['RMSNorm  γ ⊙ x / √(mean(x²) + ε)', C.ink]], note: 'RMSNorm skips the mean: one statistic instead of two and no β. The bar on each row is the token\'s mean; RMSNorm leaves it off-centre, and it works as well in practice.' }
  }

  /* ---------- scene 4: SwiGLU ---------- */
  function strip(vals: number[], x: number, y: number, cs: number, col: RGB, a: number, reveal = 1) {
    const vmax = Math.max(1e-6, ...vals.map(Math.abs))
    vals.forEach((v, k) => {
      if (k / vals.length >= reveal) { ctx.strokeStyle = rgba(C.ink, 0.1 * a); ctx.lineWidth = 1; ctx.strokeRect(x + k * cs + 0.5, y + 0.5, cs - 1, cs - 1); return }
      mk.paintCell(x + k * cs, y, cs, v, vmax, col, 0, a)
    })
    return x + vals.length * cs
  }
  function arrowLabel(x0: number, x1: number, y: number, t: string, a: number) {
    ctx.strokeStyle = rgba(C.ink, 0.45 * a); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(x0 + 4, y); ctx.lineTo(x1 - 6, y); ctx.lineTo(x1 - 10, y - 3); ctx.moveTo(x1 - 6, y); ctx.lineTo(x1 - 10, y + 3); ctx.stroke()
    ctx.font = F.small; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.ink2, a); fillRich(t, (x0 + x1) / 2, y - 5)
  }
  function miniCurve(x: number, y: number, w: number, hh: number, f: (v: number) => number, g: ((v: number) => number) | null, a: number) {
    const X = (v: number) => x + ((v + 3) / 6) * w, Y = (v: number) => y + hh - ((v + 0.5) / 3.5) * hh
    ctx.strokeStyle = rgba(C.ink, 0.2 * a); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, Y(0)); ctx.lineTo(x + w, Y(0)); ctx.stroke()
    if (g) { ctx.setLineDash([2, 3]); ctx.strokeStyle = rgba(C.ink, 0.35 * a); ctx.beginPath(); for (let s = 0; s <= 40; s++) { const v = -3 + (6 * s) / 40; if (s) ctx.lineTo(X(v), Y(g(v))); else ctx.moveTo(X(v), Y(g(v))) } ctx.stroke(); ctx.setLineDash([]) }
    ctx.strokeStyle = rgba(C.ink, 0.9 * a); ctx.lineWidth = 1.4; ctx.beginPath()
    for (let s = 0; s <= 40; s++) { const v = -3 + (6 * s) / 40; if (s) ctx.lineTo(X(v), Y(f(v))); else ctx.moveTo(X(v), Y(f(v))) }
    ctx.stroke()
  }
  const sw = (() => {
    const r = rng(4242), x = h[1].map((v) => v / Math.sqrt(h[1].reduce((s, u) => s + u * u, 0) / D))
    const mat = (a: number, b: number, s: number) => Array.from({ length: a }, () => Array.from({ length: b }, () => gauss(r) * s))
    const mv = (v: number[], M: number[][]) => M[0].map((_, j) => v.reduce((s, u, k) => s + u * M[k][j], 0))
    const Wfc = mat(D, 32, 0.45), Wproj = mat(32, D, 0.25), Wg = mat(D, 21, 0.45), Wu = mat(D, 21, 0.45), Wd = mat(21, D, 0.3)
    const hA = mv(x, Wfc), gA = hA.map(gelu), yA = mv(gA, Wproj)
    const a = mv(x, Wg), s = a.map(silu), u = mv(x, Wu), m = s.map((v, k) => v * u[k]), yB = mv(m, Wd)
    return { x, hA, gA, yA, a, s, u, m, yB }
  })()
  function sceneSwiglu(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.08)), col = hue(1)
    const cs = Math.floor(clamp((W - 2 * pad - 110 - 4 * 70) / (8 + 32 + 32 + 8), 7, 13)), gap = 70
    const yA = top + avail * 0.2, yB = top + avail * 0.58, x0 = pad + 110
    const rev = (t0: number, t1: number) => clamp((p - t0) / (t1 - t0))
    // GPT-2
    title('GPT-2 · 2 matrices, 4 × width', pad, yA - 26, fin)
    drawChip(pad, yA + cs / 2, seq[1], fin, 18)
    let x = strip(sw.x, x0, yA, cs, col, fin)
    let nx = x + gap; arrowLabel(x, nx, yA + cs / 2, '· W_fc', fin * rev(0.05, 0.1)); x = strip(sw.hA, nx, yA, cs, col, fin * rev(0.05, 0.1), rev(0.08, 0.2))
    nx = x + gap; arrowLabel(x, nx, yA + cs / 2, 'GELU', fin * rev(0.2, 0.25)); x = strip(sw.gA, nx, yA, cs, col, fin * rev(0.2, 0.25), rev(0.22, 0.32))
    miniCurve(x - gap + 8, yA - 44, gap - 16, 30, gelu, null, fin * rev(0.2, 0.25))
    nx = x + gap; arrowLabel(x, nx, yA + cs / 2, '· W_proj', fin * rev(0.32, 0.36)); strip(sw.yA, nx, yA, cs, col, fin * rev(0.32, 0.36), rev(0.34, 0.42))
    // LLaMA: gate and up side by side, multiplied, then down
    const bA = fin * rev(0.42, 0.47)
    title('LLaMA · 3 matrices, gated', pad, yB - 26, bA)
    drawChip(pad, yB + cs * 1.4, seq[1], bA, 18)
    const xs = strip(sw.x, x0, yB + cs * 0.9, cs, col, bA), yG = yB, yU = yB + cs * 2.6
    nx = xs + gap
    ctx.strokeStyle = rgba(C.ink, 0.45 * bA); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(xs + 4, yB + cs * 1.4); ctx.lineTo(xs + 18, yG + cs / 2); ctx.moveTo(xs + 4, yB + cs * 1.4); ctx.lineTo(xs + 18, yU + cs / 2); ctx.stroke()
    arrowLabel(xs + 14, nx, yG + cs / 2, '· W_gate', bA); arrowLabel(xs + 14, nx, yU + cs / 2, '· W_up', bA)
    let xg = strip(sw.a, nx, yG, cs, col, bA, rev(0.47, 0.58)); strip(sw.u, nx, yU, cs, col, bA, rev(0.47, 0.58))
    const sA = fin * rev(0.58, 0.63)
    nx = xg + gap; arrowLabel(xg, nx, yG + cs / 2, 'SiLU', sA)
    miniCurve(xg + 8, yG - 44, gap - 16, 30, silu, gelu, sA)
    xg = strip(sw.s, nx, yG, cs, col, sA, rev(0.6, 0.7))
    // ⊙: gate times up
    const mA = fin * rev(0.7, 0.75), mx = nx, my = yU
    ctx.strokeStyle = rgba(C.ink, 0.45 * mA); ctx.setLineDash([2, 3])
    ctx.beginPath(); ctx.moveTo(xg + 6, yG + cs / 2); ctx.lineTo(xg + 26, yG + cs / 2); ctx.lineTo(xg + 26, my + cs * 2.2); ctx.stroke(); ctx.setLineDash([])
    const pm = strip(sw.m, mx, my + cs * 1.8, cs, col, mA, rev(0.72, 0.82))
    ctx.font = serifAt(16); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, mA); ctx.fillText('⊙', xg + 26, my + cs * 2.3)
    nx = pm + gap; arrowLabel(pm, nx, my + cs * 2.3, '· W_down', fin * rev(0.82, 0.86)); strip(sw.yB, nx, my + cs * 1.8, cs, col, fin * rev(0.82, 0.86), rev(0.84, 0.93))
    caption('SiLU (solid) and GELU (dashed) are almost the same curve; the new part is the gate', x0, yU + cs * 6, sA, C.mute, 'left')
    mk.formula = { segs: [['MLP(x)  =  W_down ( SiLU(W_gate x) ⊙ W_up x )', C.ink]], note: 'The gate decides, per hidden unit, how much of W_up x passes. Three matrices at 8/3 × width keep the parameter count of two at 4 ×; LLaMA 3 uses 14,336 = 3.5 × 4,096.' }
  }

  /* ---------- scene 5: MHA, GQA, MQA ---------- */
  function sceneGqa(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.08)), le = eio(clamp((p - 0.1) / 0.4)), be = eout(clamp((p - 0.5) / 0.4))
    const variants: [string, number, string][] = [['MHA', 32, 'every q head has its own k, v'], ['GQA · LLaMA 3', 8, '4 q heads share one k, v'], ['MQA', 1, 'all q heads share one k, v']]
    const colW = (W - 2 * pad) / 3, qy = top + avail * 0.18, ky = top + avail * 0.52
    const kib = (nkv: number) => (2 * nkv * 128 * 32 * 2) / 1024 // K and V × kv heads × d_head 128 × 32 layers × 2 bytes
    variants.forEach(([name, nkv, desc], v) => {
      const x0 = pad + v * colW + 12, w = colW - 36, qs = w / 32
      title(name, x0, top + 14, fin)
      caption(desc, x0, top + 32, fin, C.ink2, 'left')
      const kvW = w / nkv
      for (let q = 0; q < 32; q++) {
        const g = Math.floor(q / (32 / nkv)), qx = x0 + (q + 0.5) * qs, kx = x0 + (g + 0.5) * kvW, col = C.tok[g % 7]
        ctx.strokeStyle = rgba(col, 0.5 * le * fin); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(qx, qy + 8); ctx.bezierCurveTo(qx, lerp(qy, ky, 0.5), kx, lerp(qy, ky, 0.5), kx, lerp(qy + 8, ky - 10, le)); ctx.stroke()
        ctx.fillStyle = rgba(C.ink, 0.7 * fin); ctx.fillRect(qx - qs * 0.35, qy - 8, qs * 0.7, 16)
      }
      for (let g = 0; g < nkv; g++) {
        const kx = x0 + g * kvW + 1, col = C.tok[g % 7]
        ctx.fillStyle = rgba(col, 0.85 * fin); ctx.fillRect(kx, ky - 10, Math.max(2, kvW - 2), 20)
      }
      caption('32 query heads', x0, qy - 14, fin, C.mute, 'left')
      caption(`${nkv} key/value head${nkv > 1 ? 's' : ''}`, x0, ky + 26, fin, C.mute, 'left')
      // KV cache per token and at 8k context
      const by = ky + 64, bw = w * (kib(nkv) / kib(32)) * be
      ctx.fillStyle = rgba(C.ink, 0.08 * fin); ctx.fillRect(x0, by, w, 14)
      ctx.fillStyle = rgba(v === 1 ? C.tok[4] : C.ink, 0.85 * fin); ctx.fillRect(x0, by, Math.max(2, bw), 14)
      const perTok = kib(nkv), at8k = (perTok * 8192) / 1024 / 1024
      caption(`KV cache ${perTok >= 1 ? perTok.toLocaleString('en-US') : perTok} KiB / token`, x0, by + 34, fin * be, C.ink, 'left')
      caption(`${at8k >= 1 ? at8k + ' GiB' : at8k * 1024 + ' MiB'} at 8,192 tokens`, x0, by + 52, fin * be, C.mute, 'left')
    })
    mk.formula = { segs: [['KV cache / token  =  2 × n_kv × d_head × layers × 2 bytes  =  2 × 8 × 128 × 32 × 2  =  128 KiB', C.ink]], note: 'Queries keep all 32 heads; keys and values are shared by groups of 4. Quality stays close to MHA while the cache is 4× smaller. GPT-2 small (MHA, 12 layers) needs 36 KiB per token.' }
  }

  /* ---------- frame ---------- */
  function draw() {
    useCtx(ctx)
    stage.begin()
    mk.begin()
    pills = []
    const pr = prog('rope'), pm = prog('rms'), ps = prog('swiglu'), pg = prog('gqa')
    if (pr <= 0) sceneBlocks(prog('blocks'))
    else if (pm <= 0) sceneRope(pr)
    else if (ps <= 0) sceneRms(pm)
    else if (pg <= 0) sceneSwiglu(ps)
    else sceneGqa(pg)
    mk.drawFormula()
  }

  stage.canvas.addEventListener('pointermove', (e) => {
    const [x, y] = stage.local(e), hit = pills.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h)
    hoverPill = hit?.phase ?? ''
    stage.canvas.style.cursor = hit ? 'pointer' : 'default'
  })
  stage.canvas.addEventListener('click', (e) => {
    const [x, y] = stage.local(e), hit = pills.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h)
    if (hit) player.t = player.start(hit.phase) + 0.001
  })

  const CAPS: Record<string, [string, string]> = {
    blocks: ['LLaMA keeps GPT-2\'s block: pre-norm, residual adds, causal attention. Four parts change, marked in the lower row. Click one to jump to it.', '4 changes · same block'],
    rope: ['GPT-2 adds a learned position vector once at the input. LLaMA instead rotates each pair of query and key numbers by an angle that grows with position, inside every attention layer.', 'θⱼ = base^(−2j / d_head)'],
    rms: ['LayerNorm centres each token and scales it to unit spread. RMSNorm only rescales by the root mean square: simpler, slightly faster, and just as stable.', 'x / √(mean(x²) + ε)'],
    swiglu: ['GPT-2\'s MLP widens, applies GELU and narrows. LLaMA\'s runs two projections side by side and lets one gate the other, a pattern called SwiGLU.', 'W_down(SiLU(W_gate x) ⊙ W_up x)'],
    gqa: ['In GPT-2 every attention head has its own keys and values. LLaMA 3 shares each key/value head between 4 query heads, cutting the KV cache that limits context length and batch size.', '32 q heads · 8 kv heads'],
  }

  if (reduced && player.t === 0) player.t = player.start('rope') + 6
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
