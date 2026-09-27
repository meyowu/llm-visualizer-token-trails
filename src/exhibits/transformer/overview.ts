import { createFrame } from '../../core/frame'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, blend, mixc, pop, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import { F, bracketLabel, chipW, drawChip, plate, rr, spaced, subLabel, tokDisp, tokLabel, tokText, useCtx } from '../../core/draw'
import type { Nav } from '../registry'
import { PROMPT, TAIL, buildPass, dist, promptTokens, tokId, type DistRow, type Pass, type Tok } from './model'

const PHASES = [
  { id: 'tokenize', name: 'Tokenize', dur: 2.4 },
  { id: 'embed', name: 'Embed', dur: 1.8 },
  { id: 'attn', name: 'Attention', dur: 4.8 },
  { id: 'mlp', name: 'MLP', dur: 2.2 },
  { id: 'stack', name: 'Blocks 2–12', short: 'Blocks', dur: 1.8 },
  { id: 'unembed', name: 'Unembed', dur: 2.0 },
  { id: 'sample', name: 'Sample', dur: 2.8 },
]
const MAXPASS = 3
const U = [0.22, 0.35, 0.3]

interface Pick { idx: number; text: string; d: DistRow[]; u: number }

/** The whole forward pass of GPT-2 small, from text to the sampled next token. */
export function mountOverview(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const frame = createFrame(root, {
    eyebrow: 'Transformer',
    title: 'Forward pass',
    subtitle: 'decoder-only · GPT-2 small',
    specs: [
      { label: 'layers', value: '12' }, { label: 'd_model', value: '768' }, { label: 'heads', value: '12' },
      { label: 'vocab', value: '50,257' }, { label: 'context', value: '1,024' }, { label: 'params', value: '124M' },
    ],
  })
  const stage = new Stage(frame.stageHost, 980, 460, 'Animation of a Transformer forward pass: text is tokenized and embedded, flows through attention and MLP layers, and the LM head produces next-token probabilities that are sampled.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })

  // temperature control
  const temp = document.createElement('label')
  temp.className = 'temp'
  temp.htmlFor = 'ov-temp'
  temp.innerHTML = '<span>T</span><input id="ov-temp" type="range" min="0.2" max="2" step="0.05" value="0.8"><output>0.80</output>'
  player.meta.append(temp)
  const passEl = document.createElement('span')
  passEl.className = 'pass'
  player.meta.prepend(passEl)

  const S = {
    temp: 0.8, head: 0, passIdx: 0, seq: [] as Tok[], prevYs: null as number[] | null, prevSp: 0,
    hover: -1, hoverPlate: '' as string, pick: null as unknown as Pick,
  }
  const tempIn = temp.querySelector('input')!, tempOut = temp.querySelector('output')!
  tempIn.addEventListener('input', () => { S.temp = +tempIn.value; tempOut.textContent = S.temp.toFixed(2) })

  let P: Pass
  function reset() {
    S.seq = promptTokens()
    S.passIdx = 0; S.prevYs = null; player.t = 0; P = buildPass(S.seq)
  }
  function pick(): Pick {
    const d = dist(P.seq[P.N - 1].text, S.temp), u = U[S.passIdx % U.length]
    let c = 0
    for (let k = 0; k < d.length; k++) {
      c += d[k].p
      if (u < c || k === d.length - 1) return { idx: k, text: d[k].other ? TAIL[S.passIdx % TAIL.length] : d[k].text!, d, u }
    }
    throw new Error('unreachable')
  }
  function advance() {
    const ch = pick()
    if (S.passIdx >= MAXPASS - 1) { reset(); return }
    const a = ysFor(P.N), b = ysFor(P.N + 1)
    S.prevYs = [...a.ys, b.ys[P.N]]; S.prevSp = a.sp
    S.seq = [...S.seq, { text: ch.text, id: tokId(ch.text), c: S.seq.length % 7 }]
    S.passIdx++; P = buildPass(S.seq); player.t = 0
  }
  player.onEnd = advance
  const prog = (id: string) => player.prog(id)

  /* ---------- geometry ---------- */
  const G = {
    padL: 28, padR: 28, sentY: 50, labY: 96, top: 156, bot: 0, mid: 0, xTok: 28, xEmb0: 0, cell: 6, xEmb1: 0,
    distW: 0, xDist0: 0, xWU: 0, xLn: 0, xAttn: 0, xMlp: 0, xS0: 0, xS1: 0, stack: [] as number[], rowH: 0,
    chipEnd: [] as number[], mat: null as null | { x: number; y: number; w: number; h: number },
    plateY: [0, 0] as [number, number],
  }
  function geom() {
    const { W, H } = stage
    G.bot = H - 120; G.mid = (G.top + G.bot) / 2
    G.xEmb0 = G.padL + 152; G.xEmb1 = G.xEmb0 + 16 * G.cell
    G.distW = clamp(W * 0.19, 196, 264); G.xDist0 = W - G.padR - G.distW
    G.xWU = G.xDist0 - 78; G.xLn = G.xWU - 54
    const a = G.xEmb1 + 44, b = G.xLn - 46, s = b - a
    G.xAttn = a + s * 0.24; G.xMlp = a + s * 0.46; G.xS0 = a + s * 0.64; G.xS1 = b
    G.stack = Array.from({ length: 11 }, (_, k) => lerp(G.xS0, G.xS1, k / 10))
    G.rowH = clamp((G.bot - G.top + 40) / 7.4, 22, 32)
  }
  stage.onResize = geom
  geom()

  function ysFor(N: number) {
    const sp = Math.min(56, (G.bot - G.top) / Math.max(N, 4))
    return { sp, ys: Array.from({ length: N }, (_, i) => G.mid + (i - (N - 1) / 2) * sp) }
  }
  function curYs() {
    const r = ysFor(P.N)
    if (S.passIdx > 0 && S.prevYs && S.prevYs.length === P.N) {
      const e = eio(clamp(prog('tokenize') / 0.3)), prev = S.prevYs
      return { sp: lerp(S.prevSp, r.sp, e), ys: r.ys.map((y, i) => lerp(prev[i], y, e)) }
    }
    return r
  }

  /* ---------- colour along a lane ---------- */
  const tokC = (i: number) => C.tok[P.seq[i].c % 7]
  const mixCol = (Mx: number[][], i: number) => blend(P.seq.map((_, k) => tokC(k)), Mx[i])
  function lanePts(i: number): [number, RGB][] {
    const base = tokC(i), m = P.mix, gap = (G.xS1 - G.xS0) / 10
    const pts: [number, RGB][] = [[G.xEmb1, base], [G.xAttn, base], [G.xAttn + 18, mixCol(m[0], i)]]
    for (let k = 0; k < 11; k++) { pts.push([G.stack[k], mixCol(m[k], i)]); pts.push([G.stack[k] + gap * 0.7, mixCol(m[k + 1], i)]) }
    pts.push([G.xLn, mixCol(m[11], i)], [G.xWU + 16, mixCol(m[11], i)])
    return pts
  }
  function colorAt(pts: [number, RGB][], x: number): RGB {
    if (x <= pts[0][0]) return pts[0][1]
    for (let k = 0; k < pts.length - 1; k++) if (x <= pts[k + 1][0]) return mixc(pts[k][1], pts[k + 1][1], (x - pts[k][0]) / (pts[k + 1][0] - pts[k][0]))
    return pts[pts.length - 1][1]
  }
  const laneDim = (i: number) => (S.hover >= 0 && S.hover !== i ? 0.35 : 1)

  /* ---------- scene parts ---------- */
  function drawTop(ci: number) {
    const groups: [string, number, number][] = [
      ['tokenizer', G.xTok, G.xTok + 120], ['embedding', G.xEmb0, G.xEmb1],
      ['block 1', G.xAttn - 16, G.xMlp + 26], ['blocks 2–12', G.xS0 - 10, G.xS1 + 10],
      ['unembed', G.xLn - 12, G.xWU + 16], ['next token', G.xDist0, stage.W - G.padR]]
    const gi = [0, 1, 2, 2, 3, 4, 5][ci]
    groups.forEach(([t, a, b], k) => bracketLabel(t, a, b, G.labY, k === gi))
  }
  function drawSentence(pT: number, pS: number, endFade: number, isLast: boolean, now: number) {
    ctx.font = F.label; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; spaced(true)
    ctx.fillStyle = rgba(C.mute, 1); ctx.fillText('CONTEXT', G.padL, G.sentY - 30); spaced(false)
    ctx.font = F.serif
    const baseA = S.passIdx === 0 ? clamp(player.t / 0.4) : 1, fa = isLast ? 1 - endFade : 1
    let x = G.padL
    const xs: number[] = [], cs: number[] = []
    P.seq.forEach((t, i) => {
      const s = tokDisp(t.text), w = ctx.measureText(s).width, lead = s.startsWith(' ') ? ctx.measureText(' ').width : 0
      ctx.fillStyle = rgba(i >= PROMPT.length ? tokC(i) : C.ink, baseA * fa)
      ctx.fillText(s, x, G.sentY)
      xs.push(x + lead); cs.push(x + lead + (w - lead) / 2); x += w
    })
    if (pS > 0.9) {
      const s = tokDisp(S.pick.text)
      ctx.fillStyle = rgba(C.tok[P.N % 7], clamp((pS - 0.9) / 0.05) * fa)
      ctx.fillText(s, x, G.sentY); x += ctx.measureText(s).width
    }
    const blink = 0.5 + 0.5 * Math.sin(now / 180)
    ctx.fillStyle = rgba(C.ink, blink * 0.7 * baseA * fa); ctx.fillRect(x + 4, G.sentY - 19, 1.5, 23)
    if (S.passIdx === 0 && pT > 0 && pT < 0.75) {
      for (let i = 1; i < P.N; i++) {
        const a = clamp((pT - 0.05 - i * 0.02) / 0.1) * (1 - clamp((pT - 0.5) / 0.2))
        ctx.strokeStyle = rgba(C.ink, a * 0.6); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(xs[i] - 1.5, G.sentY - 22); ctx.lineTo(xs[i] - 1.5, G.sentY + 8); ctx.stroke()
      }
    }
    return { cs }
  }
  function drawChips(ys: number[], sp: number, pT: number, sent: { cs: number[] }, alphaAll: number) {
    const h = Math.min(22, sp * 0.8)
    G.chipEnd = []
    P.seq.forEach((t, i) => {
      let x = G.xTok, y = ys[i], a = alphaAll, idA = 1, glow = 0
      if (S.passIdx === 0) {
        const ap = clamp((pT - 0.14 - i * 0.025) / 0.1)
        if (ap <= 0) return
        const f = eio(clamp((pT - 0.3 - i * 0.05) / 0.36)), w0 = chipW(t.text)
        x = lerp(sent.cs[i] - w0 / 2, G.xTok, f); y = lerp(G.sentY + 30, ys[i], f)
        a *= ap; idA = clamp((f - 0.8) / 0.2)
      } else if (i === P.N - 1) glow = 1 - clamp(pT / 0.8)
      const hl = S.hover === i
      a *= S.hover >= 0 && !hl ? 0.45 : 1
      const w = drawChip(x, y, t, a, h, hl || glow > 0.4)
      if (glow > 0) { rr(x - 4, y - h / 2 - 4, w + 8, h + 8, 8); ctx.strokeStyle = rgba(tokC(i), glow * 0.6); ctx.stroke() }
      ctx.font = F.small; ctx.textBaseline = 'middle'; ctx.textAlign = 'left'
      ctx.fillStyle = rgba(C.mute, a * idA); ctx.fillText(String(t.id), x + w + 8, y + 0.5)
      G.chipEnd[i] = x + w + 8 + ctx.measureText(String(t.id)).width
    })
  }
  function drawEmb(ys: number[], sp: number, pe: number, a: number) {
    const ch = Math.min(13, sp * 0.42), cw = G.cell, pos = clamp(pe * 2 - 1)
    P.seq.forEach((_, i) => {
      const y = ys[i], la = a * laneDim(i), fi = clamp(pe * 1.6 - i * 0.1)
      if (pe > 0 && G.chipEnd[i]) {
        ctx.setLineDash([1.5, 3]); ctx.strokeStyle = rgba(C.ink, 0.3 * clamp(pe * 4) * la); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(G.chipEnd[i] + 6, y); ctx.lineTo(G.xEmb0 - 6, y); ctx.stroke(); ctx.setLineDash([])
      }
      for (let k = 0; k < 16; k++) {
        const x = G.xEmb0 + k * cw
        if (k / 16 < fi) {
          const v = clamp(P.emb[i][k] + 0.35 * Math.sin(i * 1.3 + k * 0.8) * pos, -1, 1)
          ctx.fillStyle = v >= 0 ? rgba(tokC(i), (0.2 + 0.8 * v) * la) : rgba(C.neg, (0.1 + 0.55 * -v) * la)
          ctx.fillRect(x, y - ch / 2, cw - 1.5, ch)
        } else {
          ctx.strokeStyle = rgba(C.ink, 0.09 * la); ctx.lineWidth = 1
          ctx.strokeRect(x + 0.5, y - ch / 2 + 0.5, cw - 2.5, ch - 1)
        }
      }
    })
  }
  function frontX(pa: number, pm: number, ps: number, pu: number) {
    if (pu > 0) return lerp(G.xS1, G.xWU + 16, eio(clamp(pu / 0.45)))
    if (ps > 0) return lerp(G.xMlp, G.xS1, eio(ps))
    if (pm > 0) return lerp(G.xAttn, G.xMlp, eio(clamp(pm / 0.3)))
    if (pa > 0) return lerp(G.xEmb1, G.xAttn, eio(clamp(pa / 0.12)))
    return G.xEmb1
  }
  function drawLanes(ys: number[], pts: [number, RGB][][], front: number, pu: number, a: number) {
    const N = P.N, dimO = 1 - 0.6 * eio(pu)
    for (let i = 0; i < N; i++) {
      const y = ys[i], end = i === N - 1 ? G.xWU + 16 : G.xLn + 10
      const la = a * laneDim(i) * (i === N - 1 ? 1 : dimO)
      const from = Math.max(front, G.xEmb1)
      if (from < end) {
        ctx.setLineDash([1.5, 5]); ctx.strokeStyle = rgba(C.ink, 0.2 * laneDim(i)); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(from, y); ctx.lineTo(end, y); ctx.stroke(); ctx.setLineDash([])
      }
      ctx.lineWidth = S.hover === i ? 2.4 : 1.6; ctx.lineCap = 'butt'
      const p = pts[i]
      for (let k = 0; k < p.length - 1; k++) {
        const [xa, ca] = p[k], [xb, cb] = p[k + 1]
        if (xa >= front || xa >= end) break
        const x2 = Math.min(xb, front, end)
        if (x2 <= xa) continue
        const g = ctx.createLinearGradient(xa, 0, xb, 0)
        g.addColorStop(0, rgba(ca, la)); g.addColorStop(1, rgba(cb, la))
        ctx.strokeStyle = g; ctx.beginPath(); ctx.moveTo(xa, y); ctx.lineTo(x2, y); ctx.stroke()
      }
      if (i < N - 1 && front >= G.xLn + 10) { ctx.fillStyle = rgba(colorAt(p, G.xLn), la); ctx.beginPath(); ctx.arc(G.xLn + 10, y, 1.8, 0, 7); ctx.fill() }
    }
  }
  function drawParticles(ys: number[], pts: [number, RGB][][], front: number, pu: number, a: number) {
    if (reduced) return
    const N = P.N, dimO = 1 - 0.6 * eio(pu)
    for (let i = 0; i < N; i++) {
      const end = Math.min(front, i === N - 1 ? G.xWU + 16 : G.xLn + 10), span = end - G.xEmb1
      if (span < 8) continue
      const la = a * laneDim(i) * (i === N - 1 ? 1 : dimO)
      for (let k = 0; k * 110 < span + 110; k++) {
        const x = G.xEmb1 + ((player.t * 70 + k * 110 + i * 41) % Math.max(span, 1))
        ctx.fillStyle = rgba(pop(colorAt(pts[i], x)), 0.9 * la)
        ctx.beginPath(); ctx.arc(x, ys[i], 1.6, 0, 7); ctx.fill()
      }
      const moving = player.t > player.start('attn') && prog('unembed') < 1
      if (moving && front > G.xEmb1 + 2 && front < (i === N - 1 ? G.xWU + 16 : G.xLn + 10)) {
        const c = pop(colorAt(pts[i], front))
        ctx.fillStyle = rgba(c, 0.25 * la); ctx.beginPath(); ctx.arc(front, ys[i], 6, 0, 7); ctx.fill()
        ctx.fillStyle = rgba(c, la); ctx.beginPath(); ctx.arc(front, ys[i], 2.4, 0, 7); ctx.fill()
      }
    }
  }
  function drawArcs(ys: number[], row: number[], q: number, grow: number, a: number) {
    const x = G.xAttn - 10
    row.forEach((w, j) => {
      if (j > q) return
      const col = tokC(j)
      if (j === q) {
        ctx.beginPath(); ctx.arc(x - 6, ys[q], 2 + 9 * w * grow, 0, 7)
        ctx.lineWidth = 1 + 2.5 * w; ctx.strokeStyle = rgba(col, (0.4 + 0.6 * Math.sqrt(w)) * a); ctx.stroke()
      } else {
        const y0 = ys[j], y1 = ys[q], cx = Math.max(G.xEmb1 + 8, x - (16 + Math.abs(y1 - y0) * 0.62)), cy = (y0 + y1) / 2
        ctx.beginPath()
        for (let s = 0; s <= 32; s++) {
          const t = (s / 32) * grow, u = 1 - t
          const px = u * u * x + 2 * u * t * cx + t * t * x, py = u * u * y0 + 2 * u * t * cy + t * t * y1
          if (s) ctx.lineTo(px, py); else ctx.moveTo(px, py)
        }
        ctx.lineWidth = 0.75 + 7 * w; ctx.lineCap = 'round'
        ctx.strokeStyle = rgba(col, (0.28 + 0.66 * Math.sqrt(w)) * a); ctx.stroke(); ctx.lineCap = 'butt'
      }
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'
      ctx.fillStyle = rgba(w > 0.12 ? C.ink2 : C.mute, a * clamp(grow * 1.4))
      ctx.fillText(w.toFixed(2), G.xAttn + 13, ys[j] - 3)
    })
    ctx.beginPath(); ctx.arc(G.xAttn, ys[q], 3, 0, 7); ctx.fillStyle = rgba(C.ink, a); ctx.fill()
  }
  function drawMatrix(ys: number[], sp: number, A: number[][], rowsDone: number, cur: number, curGrow: number, a: number) {
    const N = P.N, c = clamp(Math.floor(84 / N), 7, 12), w = N * c
    const x0 = Math.round(G.xAttn - w / 2), y0 = Math.round(ys[N - 1] + sp * 0.5 + 26)
    G.mat = { x: x0 - 6, y: y0 - 6, w: w + 12, h: w + 12 }
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = x0 + j * c, y = y0 + i * c
      if (j > i) { ctx.fillStyle = rgba(C.ink, 0.12 * a); ctx.fillRect(x + c / 2 - 0.5, y + c / 2 - 0.5, 1, 1); continue }
      const v = i < rowsDone ? A[i][j] : i === cur ? A[i][j] * curGrow : -1
      if (v >= 0) { ctx.fillStyle = rgba(tokC(j), (0.1 + 0.9 * Math.sqrt(v)) * a); ctx.fillRect(x + 0.5, y + 0.5, c - 1, c - 1) }
      else { ctx.strokeStyle = rgba(C.ink, 0.12 * a); ctx.lineWidth = 1; ctx.strokeRect(x + 1, y + 1, c - 2, c - 2) }
    }
    if (cur >= 0) { ctx.strokeStyle = rgba(C.ink, 0.9 * a); ctx.lineWidth = 1; ctx.strokeRect(x0 - 0.5, y0 + cur * c - 0.5, (cur + 1) * c + 1, c + 1) }
    ctx.font = F.small; ctx.textAlign = 'right'; ctx.textBaseline = 'top'
    ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(`head ${String(S.head + 1).padStart(2, '0')} / 12`, x0 - 12, y0)
    ctx.fillStyle = rgba(C.mute, a); ctx.fillText('causal mask', x0 - 12, y0 + 15)
    ctx.fillStyle = rgba(C.faint, a); ctx.fillText('click ↻', x0 - 12, y0 + 30)
  }
  function drawAttention(ys: number[], sp: number, pa: number, a: number) {
    if (pa <= 0) return
    const N = P.N, A = P.att[S.head], qf = ((pa - 0.12) / 0.84) * N
    let qi = -1, lp = 0
    if (pa > 0.12 && pa < 0.96) { qi = Math.min(N - 1, Math.floor(qf)); lp = qf - qi }
    const rowsDone = pa >= 0.96 ? N : Math.max(0, Math.min(N, Math.floor(qf)))
    drawMatrix(ys, sp, A, rowsDone, qi, eout(clamp(lp * 2.5)), a)
    let show = -1, grow = 0, fa = 1
    if (S.hover >= 0 && S.hover < rowsDone) { show = S.hover; grow = 1 }
    else if (qi >= 0) { show = qi; grow = eout(clamp(lp * 2.2)); fa = 1 - clamp((lp - 0.86) / 0.14) }
    if (show >= 0) drawArcs(ys, A[show], show, grow, fa * a)
  }
  function drawMLP(ys: number[], sp: number, pts: [number, RGB][][], pm: number, py1: number, a: number) {
    if (pm <= 0 || pm >= 1) return
    const e = pm < 0.3 ? eout(pm / 0.3) : pm > 0.78 ? 1 - eio((pm - 0.78) / 0.22) : 1
    const hw = 24, hmax = Math.min(sp * 0.46, 22), X = G.xMlp
    P.seq.forEach((_, i) => {
      const y = ys[i], hh = hmax * e, col = colorAt(pts[i], X), la = a * laneDim(i)
      ctx.beginPath(); ctx.moveTo(X - hw, y); ctx.quadraticCurveTo(X, y - 2 * hh, X + hw, y); ctx.quadraticCurveTo(X, y + 2 * hh, X - hw, y)
      ctx.fillStyle = rgba(col, 0.12 * la); ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = rgba(col, 0.6 * la); ctx.stroke()
      ;[-12, -4, 4, 12].forEach((dx, c) => {
        const lim = hh * (1 - (dx / hw) ** 2) * 0.85
        for (let r = 0; r < 6; r++) {
          const dy = ((r - 2.5) / 2.5) * hh * 0.85
          if (Math.abs(dy) > lim) continue
          const act = P.act[i][c * 6 + r], fl = 0.6 + 0.4 * Math.sin(player.t * 9 + c * 1.7 + r * 2.3 + i)
          ctx.fillStyle = act > 0 ? rgba(pop(col), act * fl * e * la) : rgba(C.ink, 0.14 * e * la)
          ctx.beginPath(); ctx.arc(X + dx, y + dy, act > 0 ? 1.8 : 1, 0, 7); ctx.fill()
        }
      })
    })
    ctx.font = F.small; ctx.textAlign = 'center'; ctx.textBaseline = 'top'
    ctx.fillStyle = rgba(C.ink2, e * a); ctx.fillText('768 → 3072 → 768', X, py1 + 16)
    ctx.fillStyle = rgba(C.mute, e * a); ctx.fillText('GELU', X, py1 + 31)
  }
  function drawDist(ys: number[], pts: [number, RGB][][], pu: number, pS: number, a: number) {
    const g = eout(clamp((pu - 0.42) / 0.5)), W = stage.W
    const N = P.N, d = S.pick.d, yl = ys[N - 1], ry = (k: number) => G.mid + (k - 3) * G.rowH
    const bx0 = G.xDist0 + 92, bx1 = W - G.padR - 52, lastCol = colorAt(pts[N - 1], G.xWU), nextCol = C.tok[N % 7]
    ctx.font = F.label; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; spaced(true)
    ctx.fillStyle = rgba(C.mute, 0.5 + 0.5 * g); ctx.fillText('P( NEXT | CONTEXT )', G.xDist0, ry(0) - G.rowH * 0.75); spaced(false)
    ctx.font = F.small; ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.ink2, 0.5 + 0.5 * g)
    ctx.fillText(`T = ${S.temp.toFixed(2)}`, W - G.padR, ry(0) - G.rowH * 0.75)
    if (g <= 0) {
      for (let k = 0; k < 7; k++) { ctx.fillStyle = rgba(C.ink, 0.06); ctx.fillRect(G.xDist0, ry(k), G.distW, 1) }
      return
    }
    let hl = -1, settled = false
    if (pS > 0) { if (pS < 0.45) hl = Math.floor(eout(pS / 0.45) * (14 + S.pick.idx)) % 7; else { hl = S.pick.idx; settled = true } }
    d.forEach((r, k) => {
      const gk = eout(clamp((pu - 0.42 - k * 0.035) / 0.45)) * a, y = ry(k)
      ctx.beginPath(); ctx.moveTo(G.xWU + 16, yl)
      ctx.bezierCurveTo(G.xWU + 50, yl, G.xDist0 - 40, y, G.xDist0 - 10, y)
      ctx.lineWidth = 0.5 + 5 * r.p; ctx.strokeStyle = rgba(lastCol, (0.1 + 0.6 * Math.sqrt(r.p)) * gk); ctx.stroke()
      const isHl = k === hl, isPick = settled && k === S.pick.idx
      if (r.other) {
        ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
        ctx.fillStyle = rgba(isHl ? C.ink : C.mute, gk); ctx.fillText('50,251 others', G.xDist0, y)
      } else tokText(G.xDist0, y, r.text!, gk * (isHl ? 1 : 0.8), isPick ? nextCol : null)
      ctx.fillStyle = rgba(C.ink, 0.07 * gk); ctx.fillRect(bx0, y - 3, bx1 - bx0, 6)
      ctx.fillStyle = isPick ? rgba(nextCol, gk) : rgba(C.ink, (isHl ? 0.95 : 0.5) * gk)
      ctx.fillRect(bx0, y - 3, Math.max(1, r.p * (bx1 - bx0) * gk), 6)
      ctx.font = F.small; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
      ctx.fillStyle = rgba(isHl ? C.ink : C.mute, gk)
      ctx.fillText(r.p < 0.001 ? '<0.1%' : (r.p * 100).toFixed(1) + '%', W - G.padR, y)
    })
    if (settled) {
      const sa = clamp((pS - 0.45) / 0.08) * a
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
      ctx.fillStyle = rgba(C.mute, sa); ctx.fillText(`u = ${S.pick.u.toFixed(2)}  →`, G.xDist0, ry(6) + G.rowH)
      tokText(G.xDist0 + 92, ry(6) + G.rowH, S.pick.text, sa, nextCol)
    }
  }
  function drawFlight(pS: number, isLast: boolean) {
    if (pS <= 0.5) return
    const N = P.N, tok = { text: S.pick.text, c: N % 7, id: tokId(S.pick.text) }
    const f = eio(clamp((pS - 0.5) / 0.4)), sx = G.xDist0, sy = G.mid + (S.pick.idx - 3) * G.rowH
    const nxt = ysFor(N + 1), tx = G.xTok, ty = nxt.ys[N], cx = (sx + tx) / 2, cy = G.labY - 150
    const a = isLast ? 1 - clamp((pS - 0.9) / 0.1) : 1
    ctx.setLineDash([2, 5]); ctx.strokeStyle = rgba(C.tok[tok.c], 0.35 * a * (1 - clamp((pS - 0.9) / 0.1))); ctx.lineWidth = 1
    ctx.beginPath()
    for (let s = 0; s <= 40; s++) {
      const t = (s / 40) * f, u = 1 - t
      const px = u * u * sx + 2 * u * t * cx + t * t * tx, py = u * u * sy + 2 * u * t * cy + t * t * ty
      if (s) ctx.lineTo(px, py); else ctx.moveTo(px, py)
    }
    ctx.stroke(); ctx.setLineDash([])
    const u = 1 - f, x = u * u * sx + 2 * u * f * cx + f * f * tx, y = u * u * sy + 2 * u * f * cy + f * f * ty
    const h = Math.min(22, nxt.sp * 0.8), w = drawChip(x, y, tok, a, h, true)
    if (f >= 1) { ctx.font = F.small; ctx.textBaseline = 'middle'; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(String(tok.id), x + w + 8, y + 0.5) }
  }
  /** Hover hint on a plate that opens a detail view. */
  function drawOpenHint(x: number, y: number) {
    const t = 'Open details ↗'
    ctx.font = F.body
    const w = ctx.measureText(t).width + 16
    rr(x - w / 2, y - 26, w, 20, 10)
    ctx.fillStyle = rgba(C.ink, 0.92); ctx.fill()
    ctx.fillStyle = rgba(C.bg); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    ctx.fillText(t, x, y - 15.5)
  }

  /* ---------- frame ---------- */
  function draw(now: number) {
    useCtx(ctx)
    stage.begin()
    S.pick = pick()
    const ci = player.curIndex(), N = P.N, { sp, ys } = curYs()
    const pT = prog('tokenize'), pe = prog('embed'), pa = prog('attn'), pm = prog('mlp'), ps = prog('stack'), pu = prog('unembed'), pS = prog('sample')
    const isLast = S.passIdx === MAXPASS - 1, endFade = clamp((pS - 0.86) / 0.14), compA = 1 - 0.75 * endFade
    const py0 = ys[0] - sp * 0.5 - 12, py1 = ys[N - 1] + sp * 0.5 + 12
    G.plateY = [py0, py1]
    const wu0 = Math.max(G.labY + 50, Math.min(py0, G.mid - 3.5 * G.rowH)), wu1 = Math.max(py1, G.mid + 3.5 * G.rowH)
    const front = frontX(pa, pm, ps, pu), pts = P.seq.map((_, i) => lanePts(i))

    drawTop(ci)
    const sent = drawSentence(pT, pS, endFade, isLast, now)
    drawLanes(ys, pts, front, pu, compA)

    const hovA = S.hoverPlate === 'attn', hovM = S.hoverPlate === 'mlp'
    const aAct = Math.max(pa > 0 && pa < 1 ? 1 : 0, hovA ? 0.8 : 0), mAct = Math.max(pm > 0 && pm < 1 ? 1 : 0, hovM ? 0.8 : 0)
    plate(G.xAttn, py0, py1, aAct); subLabel(hovA ? 'attn ↗' : 'attn', G.xAttn, G.labY + 32, aAct > 0.5)
    plate(G.xMlp, py0, py1, mAct); subLabel(hovM ? 'mlp ↗' : 'mlp', G.xMlp, G.labY + 32, mAct > 0.5)
    G.stack.forEach((x, k) => plate(x, py0, py1, ps > 0 && ps < 1 ? clamp(1 - Math.abs(ps * 11 - k - 0.5) / 1.3) : 0, { w: 6, d: 8 }))
    const lnAct = pu > 0 && pu < 1 ? clamp(1 - Math.abs(pu - 0.2) / 0.2) : 0, wuAct = pu > 0 && pu < 1 ? clamp(1 - Math.abs(pu - 0.55) / 0.3) : 0
    plate(G.xLn, py0, py1, lnAct); subLabel('ln_f', G.xLn, G.labY + 32, lnAct > 0.1)
    plate(G.xWU, wu0, wu1, wuAct, { w: 11, d: 11, hatch: 34 }); subLabel('W_U', G.xWU, G.labY + 32, wuAct > 0.1)

    drawEmb(ys, sp, pe, isLast ? 1 - endFade : 1)
    drawAttention(ys, sp, pa, compA)
    drawMLP(ys, sp, pts, pm, py1, compA)
    drawParticles(ys, pts, front, pu, compA)
    drawChips(ys, sp, pT, sent, isLast ? 1 - endFade : 1)
    drawDist(ys, pts, pu, pS, 1 - endFade)
    drawFlight(pS, isLast)
    if (hovA) drawOpenHint(G.xAttn, py0 - 4)
    if (hovM) drawOpenHint(G.xMlp, py0 - 4)
  }

  /* ---------- captions ---------- */
  function caption(ci: number) {
    const N = P.N
    switch (PHASES[ci].id) {
      case 'tokenize': return S.passIdx === 0
        ? { t: "GPT-2's byte-level BPE splits the text into subwords, each an id in a 50,257-entry vocabulary. Ġ marks a token that begins with a space.", s: `ids [${P.seq.map((t) => t.id).join(', ')}]` }
        : { t: 'The token sampled in the last pass is appended to the sequence as is, with no re-tokenizing. There is no KV cache here, so the whole sequence is recomputed from scratch.', s: `${N} tokens · +${P.seq[N - 1].id}` }
      case 'embed': return { t: 'Each id selects one row of the embedding matrix W_E, and the position vector for slot i is added. From here on, every token is a 768-wide residual stream.', s: `[${N} × 768]` }
      case 'attn': return { t: 'Each position compares its query with the keys of every earlier position and pulls in their values, weighted by similarity. The causal mask hides the future. Click the attn plate to open up every matrix product.', s: `12 heads × [${N} × ${N}]` }
      case 'mlp': return { t: 'Each position is expanded to 3,072 dimensions, passed through GELU, and projected back to 768. Positions exchange no information in this step. Click the mlp plate to see the matrix products and GELU.', s: `[${N} × 768] → [${N} × 3072]` }
      case 'stack': return { t: 'The same block repeats 11 more times. Each block adds its result to the residual stream instead of replacing it; the blending colors trace information moving between positions.', s: '12 blocks · ≈85M params' }
      case 'unembed': return { t: 'Only the last position is used: after ln_f it is dotted with every vocabulary vector, giving 50,257 logits. GPT-2 ties W_U to W_E.', s: '[1 × 768] · W_Eᵀ → [1 × 50,257]' }
      default: {
        const p = S.pick, pp = p.d[p.idx].p
        return { t: 'softmax(logit / T) turns scores into probabilities; one token is drawn, appended, and the next pass begins. Drag T: lower is more decisive, higher leaks probability into the long tail.', s: `u ${p.u.toFixed(2)} → ${tokLabel(p.text)} · ${(pp * 100).toFixed(1)}%` }
      }
    }
  }

  /* ---------- pointer ---------- */
  const inMat = (x: number, y: number) => { const m = G.mat; return !!m && prog('attn') > 0 && x >= m.x && x <= m.x + m.w && y >= m.y && y <= m.y + m.h }
  /** Plates that open a detail view, keyed by the route they open. */
  const plateAt = (x: number, y: number) => {
    if (y < G.plateY[0] - 14 || y > G.plateY[1] + 14) return ''
    if (Math.abs(x - G.xAttn) < 14) return 'attn'
    if (Math.abs(x - G.xMlp) < 14) return 'mlp'
    return ''
  }
  const PLATE_ROUTES: Record<string, string> = { attn: 'transformer/attention', mlp: 'transformer/mlp' }
  const cv = stage.canvas
  cv.addEventListener('pointermove', (e) => {
    const [x, y] = stage.local(e), { sp, ys } = curYs()
    S.hoverPlate = plateAt(x, y)
    let h = -1
    if (!S.hoverPlate && x > G.xTok - 8 && x < G.xWU + 20) ys.forEach((yy, i) => { if (Math.abs(y - yy) < sp / 2) h = i })
    S.hover = h
    cv.style.cursor = inMat(x, y) || S.hoverPlate ? 'pointer' : 'default'
  })
  cv.addEventListener('pointerleave', () => { S.hover = -1; S.hoverPlate = '' })
  cv.addEventListener('click', (e) => {
    const [x, y] = stage.local(e)
    const pl = plateAt(x, y)
    if (pl) nav(PLATE_ROUTES[pl], { x: e.clientX, y: e.clientY })
    else if (inMat(x, y)) S.head = (S.head + 1) % 12
  })

  reset()
  if (reduced) player.t = player.start('attn') + PHASES[2].dur * 0.7
  const stop = runLoop((dt, now) => {
    player.tick(dt)
    draw(now)
    player.updateUI()
    const ci = player.curIndex(), c = caption(ci)
    frame.setCaption(PHASES[ci].name, PHASES[ci].short ?? PHASES[ci].name, c.t, c.s)
    const pt = `pass ${S.passIdx + 1} / ${MAXPASS} · ${P.N} tokens`
    if (passEl.textContent !== pt) passEl.textContent = pt
  })
  return () => { stop(); player.destroy(); stage.destroy() }
}
