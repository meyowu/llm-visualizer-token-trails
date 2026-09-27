import { createFrame, stepper, toggle } from '../../core/frame'
import { getParams, setParams } from '../../core/link'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, blend, mixc, pop, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import { F, bracketLabel, chipW, drawChip, plate, rr, spaced, subLabel, tokDisp, tokLabel, tokText, useCtx } from '../../core/draw'
import { PROMPT_LABELS, headKind, mixing, nextDist, presets, type Dist, type GptPass, type Preset } from '../../lib/gpt2/data'
import { teach } from '../learn'
import type { Nav } from '../registry'

const PHASES = [
  { id: 'tokenize', name: 'Tokenize', dur: 2.4 },
  { id: 'embed', name: 'Embed', dur: 1.8 },
  { id: 'attn', name: 'Attention', dur: 4.8 },
  { id: 'mlp', name: 'MLP', dur: 2.2 },
  { id: 'stack', name: 'Blocks 2–12', short: 'Blocks', dur: 3.2 },
  { id: 'unembed', name: 'Unembed', dur: 2.0 },
  { id: 'pick', name: 'Pick the next token', short: 'Pick', dur: 2.4 },
]
/** The head each prompt opens on, [block, head]; the counting prompt shows an induction head. */
const START_HEAD: Record<string, [number, number]> = { cat: [0, 0], france: [0, 0], count: [5, 5] }
/** Candidates listed by name; the rest of the vocabulary is one row. */
const TOP = 6
/** |value| drawn at full strength in the embedding strip. */
const EMB_SCALE = 0.25

interface Tok { text: string; id: number; c: number }
/** Where the reader left the overview, restored (paused) when they come back from a detail view. */
let saved: { preset: number; passIdx: number; t: number; layer: number; head: number; lock: number } | null = null
interface View {
  N: number
  seq: Tok[]
  pass: GptPass
  mix: number[][][]
  dist: Dist
  /** The greedy pick, appended by the next pass. */
  next: Tok
  /** Activation drawn at full strength in the MLP lens. */
  actMax: number
}

function makeView(pr: Preset, k: number): View {
  const pass = pr.passes[k], N = pass.ids.length
  const seq = pass.texts.map((text, i) => ({ text, id: pass.ids[i], c: i % 7 }))
  const dist = nextDist(pass.next, 1, TOP), top = dist.rows[0]
  // position 0 gets a few huge activations whatever the token, so it does not set the scale
  const actMax = Math.max(0.5, ...pass.neurons.act.slice(1).flat())
  return { N, seq, pass, mix: mixing(pass.att), dist, next: { text: top.text, id: top.id, c: N % 7 }, actMax }
}

/** The whole forward pass of GPT-2 small, from text to the next token, with the numbers of a real run. */
export function mountOverview(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const frame = createFrame(root, {
    eyebrow: 'Anatomy',
    title: 'Forward pass',
    subtitle: 'decoder-only Transformer · GPT-2 small, real run',
    specs: [
      { label: 'shown', value: 'real GPT-2' }, { label: 'layers', value: '12' }, { label: 'd_model', value: '768' },
      { label: 'heads', value: '12' }, { label: 'vocab', value: '50,257' }, { label: 'params', value: '124M' },
    ],
  })
  const stage = new Stage(frame.stageHost, 980, 460, 'Animation of a GPT-2 small forward pass with real numbers: the prompt is tokenized and embedded, flows through attention and MLP blocks, and the last position is turned into next-token probabilities; the most likely token is appended and the pass repeats.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  teach(player, 'overview')
  const PRESETS = presets()

  const S = { preset: 0, passIdx: 0, layer: 0, head: 0, prevYs: null as number[] | null, prevSp: 0, hover: -1, lock: -1, hoverPlate: '' }
  const pr = () => PRESETS[S.preset]
  let V = makeView(pr(), 0)

  // controls, prepended right to left: prompt, block, head, pass readout, then the player's speed toggle
  // go back to an earlier pass (or on) without waiting for the loop
  const passStep = stepper(player.meta, 'pass', 3, 0, (k) => { S.prevYs = null; setPass(Math.min(k, pr().passes.length - 1)) })
  const headStep = stepper(player.meta, 'head', 12, S.head, (h) => { S.head = h; linkHead() })
  const layerStep = stepper(player.meta, 'block', 12, S.layer, (l) => { S.layer = l; linkHead() })
  /** The address keeps the prompt, pass and head, so a copied link shows the same view. */
  const linkHead = () => setParams({ head: `${S.layer + 1}.${S.head + 1}` })
  const presetToggle = toggle(player.meta, 'Prompt', PRESETS.map((p) => PROMPT_LABELS[p.key] ?? p.text), 0, (i) => { S.preset = i; startPreset() })

  function setPass(k: number) {
    S.passIdx = k
    V = makeView(pr(), k)
    player.t = 0
    if (k === 0) S.prevYs = null
    setParams({ prompt: pr().key, pass: String(k + 1) })
  }
  function startPreset() {
    const [l, h] = START_HEAD[pr().key] ?? [0, 0]
    S.layer = l; S.head = h; S.lock = -1
    layerStep.set(l); headStep.set(h); linkHead()
    setPass(0)
  }
  function advance() {
    if (S.passIdx >= pr().passes.length - 1) { setPass(0); return }
    const a = ysFor(V.N), b = ysFor(V.N + 1)
    S.prevYs = [...a.ys, b.ys[V.N]]; S.prevSp = a.sp
    setPass(S.passIdx + 1)
  }
  player.onEnd = advance
  const prog = (id: string) => player.prog(id)

  /* ---------- geometry ---------- */
  const G = {
    padL: 28, padR: 28, sentY: 50, labY: 96, top: 156, bot: 0, mid: 0, xTok: 28, xEmb0: 0, cell: 6, xEmb1: 0,
    distW: 0, xDist0: 0, xWU: 0, xLn: 0, xLn1: 0, xAttn: 0, xLn2: 0, xMlp: 0, xS0: 0, xS1: 0, stack: [] as number[], rowH: 0,
    chipEnd: [] as number[], mat: null as null | { x: number; y: number; w: number; h: number },
    thumbs: [] as { x: number; y: number; w: number; h: number; head: number }[],
    plateY: [0, 0] as [number, number],
    wuY: [0, 0] as [number, number],
  }
  function geom() {
    const { W, H } = stage
    G.bot = H - 120; G.mid = (G.top + G.bot) / 2
    G.xEmb0 = G.padL + 152; G.xEmb1 = G.xEmb0 + 16 * G.cell
    G.distW = clamp(W * 0.19, 196, 264); G.xDist0 = W - G.padR - G.distW
    G.xWU = G.xDist0 - 78; G.xLn = G.xWU - 54
    const a = G.xEmb1 + 44, b = G.xLn - 58, s = b - a
    G.xAttn = a + s * 0.18; G.xMlp = a + s * 0.46; G.xS0 = a + s * 0.66; G.xS1 = b
    G.xLn1 = G.xAttn - 22; G.xLn2 = G.xMlp - 22
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
    const r = ysFor(V.N)
    if (S.passIdx > 0 && S.prevYs && S.prevYs.length === V.N) {
      const e = eio(clamp(prog('tokenize') / 0.3)), prev = S.prevYs
      return { sp: lerp(S.prevSp, r.sp, e), ys: r.ys.map((y, i) => lerp(prev[i], y, e)) }
    }
    return r
  }

  /* ---------- colour along a lane ---------- */
  const tokC = (i: number) => C.tok[V.seq[i].c % 7]
  const mixCol = (Mx: number[][], i: number) => blend(V.seq.map((_, k) => tokC(k)), Mx[i])
  function lanePts(i: number): [number, RGB][] {
    const base = tokC(i), m = V.mix, gap = (G.xS1 - G.xS0) / 10
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
  /** The lane in focus: hovered, else clicked. */
  const focus = () => (S.hover >= 0 ? S.hover : S.lock)
  const laneDim = (i: number) => (focus() >= 0 && focus() !== i ? 0.35 : 1)

  /* ---------- scene parts ---------- */
  function drawTop(ci: number, lensMode: boolean, dim = 1) {
    ctx.globalAlpha = dim
    const groups: [string, number, number][] = [
      ['tokenizer ↗', G.xTok, G.xTok + 120], ['embedding ↗', G.xEmb0, G.xEmb1],
      ['block 1', G.xLn1 - 10, G.xMlp + 34], ['blocks 2–12', G.xS0 - 10, G.xS1 + 10],
      ['unembed', G.xLn - 12, G.xWU + 16], ['next token ↗', G.xDist0, stage.W - G.padR]]
    const gi = [0, 1, 2, 2, 3, 4, 5][ci]
    groups.forEach(([t, a, b], k) => bracketLabel(t, a, b, G.labY, k === gi))
    // column notes on the sub-label row
    ctx.font = F.small; ctx.textBaseline = 'alphabetic'
    ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, 0.85)
    ctx.fillText('Ġ space · Ċ newline', G.xTok, G.labY + 32)
    ctx.textAlign = 'center'; ctx.fillText('dims 1–16 of 768', (G.xEmb0 + G.xEmb1) / 2, G.labY + 32)
    ctx.textAlign = 'left'
    ctx.fillText(lensMode ? 'last position, after block n' : 'GPT-2 small, real run · T = 1', G.xDist0, G.labY + 32)
    ctx.globalAlpha = 1
  }
  function drawSentence(pT: number, pS: number, endFade: number, isLast: boolean, now: number) {
    ctx.font = F.label; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; spaced(true)
    ctx.fillStyle = rgba(C.mute, 1); ctx.fillText('CONTEXT', G.padL, G.sentY - 30); spaced(false)
    ctx.font = F.serif
    const baseA = S.passIdx === 0 ? clamp(player.t / 0.4) : 1, fa = isLast ? 1 - endFade : 1, nPrompt = pr().passes[0].ids.length
    let x = G.padL
    const starts: number[] = [], cs: number[] = []
    V.seq.forEach((t, i) => {
      const s = tokDisp(t.text), w = ctx.measureText(s).width, lead = s.startsWith(' ') ? ctx.measureText(' ').width : 0
      ctx.fillStyle = rgba(i >= nPrompt ? tokC(i) : C.ink, baseA * fa)
      ctx.fillText(s, x, G.sentY)
      starts.push(x); cs.push(x + lead + (w - lead) / 2); x += w
    })
    if (pS > 0.9) {
      const s = tokDisp(V.next.text)
      ctx.fillStyle = rgba(C.tok[V.next.c], clamp((pS - 0.9) / 0.05) * fa)
      ctx.fillText(s, x, G.sentY); x += ctx.measureText(s).width
    }
    const blink = reduced || !player.playing ? 1 : 0.5 + 0.5 * Math.sin(now / 180)
    ctx.fillStyle = rgba(C.ink, blink * 0.7 * baseA * fa); ctx.fillRect(x + 4, G.sentY - 19, 1.5, 23)
    // split marks go before the space, which belongs to the token that follows (Ġ)
    if (S.passIdx === 0 && pT > 0 && pT < 0.75) {
      for (let i = 1; i < V.N; i++) {
        const a = clamp((pT - 0.05 - i * 0.02) / 0.1) * (1 - clamp((pT - 0.5) / 0.2))
        ctx.strokeStyle = rgba(C.ink, a * 0.6); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(starts[i] + 0.5, G.sentY - 22); ctx.lineTo(starts[i] + 0.5, G.sentY + 8); ctx.stroke()
      }
    }
    return { cs }
  }
  function drawChips(ys: number[], sp: number, pT: number, sent: { cs: number[] }, alphaAll: number) {
    const h = Math.min(22, sp * 0.8)
    G.chipEnd = []
    V.seq.forEach((t, i) => {
      let x = G.xTok, y = ys[i], a = alphaAll, idA = 1, glow = 0
      if (S.passIdx === 0) {
        const ap = clamp((pT - 0.14 - i * 0.025) / 0.1)
        if (ap <= 0) return
        const f = eio(clamp((pT - 0.3 - i * 0.05) / 0.36)), w0 = chipW(t.text)
        x = lerp(sent.cs[i] - w0 / 2, G.xTok, f); y = lerp(G.sentY + 30 + (i % 2) * 24, ys[i], f)
        a *= ap; idA = clamp((f - 0.8) / 0.2)
      } else if (i === V.N - 1) glow = 1 - clamp(pT / 0.8)
      const hl = focus() === i
      a *= focus() >= 0 && !hl ? 0.45 : 1
      const w = drawChip(x, y, t, a, h, hl || glow > 0.4)
      if (glow > 0) { rr(x - 4, y - h / 2 - 4, w + 8, h + 8, 8); ctx.strokeStyle = rgba(tokC(i), glow * 0.6); ctx.stroke() }
      ctx.font = F.small; ctx.textBaseline = 'middle'; ctx.textAlign = 'left'
      ctx.fillStyle = rgba(C.mute, a * idA); ctx.fillText(String(t.id), x + w + 8, y + 0.5)
      G.chipEnd[i] = x + w + 8 + ctx.measureText(String(t.id)).width
    })
  }
  /** The first 16 of 768 real numbers: the token's W_E row, then its position's W_P row added. */
  function drawEmb(ys: number[], sp: number, pe: number, a: number) {
    const ch = Math.min(13, sp * 0.42), cw = G.cell, pos = eio(clamp(pe * 2 - 1))
    V.seq.forEach((_, i) => {
      const y = ys[i], la = a * laneDim(i), fi = clamp(pe * 1.6 - i * 0.1)
      if (pe > 0 && G.chipEnd[i]) {
        ctx.setLineDash([1.5, 3]); ctx.strokeStyle = rgba(C.ink, 0.3 * clamp(pe * 4) * la); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(G.chipEnd[i] + 6, y); ctx.lineTo(G.xEmb0 - 6, y); ctx.stroke(); ctx.setLineDash([])
      }
      for (let k = 0; k < 16; k++) {
        const x = G.xEmb0 + k * cw
        if (k / 16 < fi) {
          const v = clamp((V.pass.wte[i][k] + pos * V.pass.wpe[i][k]) / EMB_SCALE, -1, 1)
          if (v >= 0) { ctx.fillStyle = rgba(tokC(i), (0.12 + 0.88 * v) * la); ctx.fillRect(x, y - ch / 2, cw - 1.5, ch) }
          else { ctx.strokeStyle = rgba(tokC(i), (0.15 + 0.85 * -v) * la); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y - ch / 2 + 0.5, cw - 2.5, ch - 1) }
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
    const N = V.N, dimO = 1 - 0.6 * eio(pu)
    for (let i = 0; i < N; i++) {
      const y = ys[i], end = i === N - 1 ? G.xWU + 16 : G.xLn + 10
      const la = a * laneDim(i) * (i === N - 1 ? 1 : dimO)
      const from = Math.max(front, G.xEmb1)
      if (from < end) {
        ctx.setLineDash([1.5, 5]); ctx.strokeStyle = rgba(C.ink, 0.2 * laneDim(i)); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(from, y); ctx.lineTo(end, y); ctx.stroke(); ctx.setLineDash([])
      }
      ctx.lineWidth = focus() === i ? 2.4 : 1.6; ctx.lineCap = 'butt'
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
  /** After the last block: each lane's sources as a stacked bar, one segment per token hue. */
  function drawSources(ys: number[], front: number, pu: number, a: number) {
    const x0 = G.xS1 + 12, w = G.xLn - 16 - x0
    if (front < x0 + 4 || w < 12) return
    const g = clamp((front - x0) / 30), dimO = 1 - 0.6 * eio(pu), M = V.mix[11]
    V.seq.forEach((_, i) => {
      const la = g * a * laneDim(i) * (i === V.N - 1 ? 1 : dimO) * (focus() === i ? 1 : 0.75)
      let x = x0
      M[i].forEach((share, k) => {
        if (share < 0.005) return
        ctx.fillStyle = rgba(tokC(k), la); ctx.fillRect(x, ys[i] - 8, Math.max(0.5, share * w - 0.5), 3)
        x += share * w
      })
    })
    ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.mute, g * a)
    ctx.fillText('from', x0, ys[0] - 14)
  }
  function drawParticles(ys: number[], pts: [number, RGB][][], front: number, pu: number, a: number) {
    if (reduced) return
    const N = V.N, dimO = 1 - 0.6 * eio(pu)
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
      const col = tokC(j), tiny = w < 0.02
      if (tiny) { /* nothing to draw: the label says how little */ } else if (j === q) {
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
      ctx.fillStyle = rgba(w > 0.12 ? C.ink2 : C.mute, a * clamp(grow * 1.4) * (tiny ? 0.6 : 1))
      ctx.fillText(tiny ? '<.02' : w.toFixed(2), G.xAttn + 24, ys[j] - 4)
    })
    ctx.beginPath(); ctx.arc(G.xAttn, ys[q], 3, 0, 7); ctx.fillStyle = rgba(C.ink, a); ctx.fill()
  }
  function drawMatrix(ys: number[], sp: number, A: number[][], rowsDone: number, cur: number, curGrow: number, a: number) {
    const N = V.N, c = clamp(Math.floor(84 / N), 7, 12), w = N * c
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
    const kind = headKind(A, V.pass.ids)
    ctx.font = F.small; ctx.textAlign = 'right'; ctx.textBaseline = 'top'
    ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(`block ${S.layer + 1} · head ${S.head + 1}`, x0 - 12, y0)
    ctx.fillStyle = rgba(C.mute, a); ctx.fillText(kind === 'mixed' ? 'mixed head' : `${kind} head`, x0 - 12, y0 + 15)
    ctx.fillStyle = rgba(C.mute, a); ctx.fillText('click for next head', x0 - 12, y0 + 30)
  }
  function drawAttention(ys: number[], sp: number, pa: number, a: number) {
    if (pa <= 0) return
    const N = V.N, A = V.pass.att[S.layer][S.head], qf = ((pa - 0.12) / 0.84) * N
    let qi = -1, lp = 0
    if (pa > 0.12 && pa < 0.96) { qi = Math.min(N - 1, Math.floor(qf)); lp = qf - qi }
    const rowsDone = pa >= 0.96 ? N : Math.max(0, Math.min(N, Math.floor(qf)))
    drawMatrix(ys, sp, A, rowsDone, qi, eout(clamp(lp * 2.5)), a)
    let show = -1, grow = 0, fa = 1
    if (focus() >= 0 && focus() < rowsDone) { show = focus(); grow = 1 }
    else if (qi >= 0) { show = qi; grow = eout(clamp(lp * 2.2)); fa = 1 - clamp((lp - 0.86) / 0.14) }
    if (show >= 0) drawArcs(ys, A[show], show, grow, fa * a)
  }
  /**
   * The residual connection in block 1: the stream passes in front of the attn and mlp plates
   * (the sub-layer only reads a copy, through ln_1 / ln_2), and a ⊕ on each lane marks where its
   * output is added back.
   */
  function drawResidual(ys: number[], sp: number, pts: [number, RGB][][], front: number, a: number) {
    const N = V.N, r = clamp(sp * 0.16, 3.2, 5)
    for (const x of [G.xAttn, G.xMlp]) {
      const xa = x + (x === G.xMlp ? 28 : 16), done = front > xa // clear of the MLP lens
      for (let i = 0; i < N; i++) {
        const y = ys[i], la = a * laneDim(i)
        if (front > x + 10) {
          ctx.strokeStyle = rgba(colorAt(pts[i], x), la); ctx.lineWidth = 1.6
          ctx.beginPath(); ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y); ctx.stroke()
        }
        ctx.fillStyle = rgba(C.bg, la); ctx.beginPath(); ctx.arc(xa, y, r, 0, 7); ctx.fill()
        ctx.strokeStyle = rgba(C.ink, (done ? 0.85 : 0.3) * la); ctx.lineWidth = 1
        ctx.stroke()
        ctx.beginPath(); ctx.moveTo(xa - r * 0.55, y); ctx.lineTo(xa + r * 0.55, y); ctx.moveTo(xa, y - r * 0.55); ctx.lineTo(xa, y + r * 0.55); ctx.stroke()
      }
    }
    if (front > G.xEmb1 + 8) {
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, a)
      ctx.fillText('residual stream →', G.xEmb0, ys[N - 1] + Math.max(8, sp * 0.3))
    }
  }
  /** Block 1's MLP as a lens per position; the dots are 24 real neurons after GELU. */
  function drawMLP(ys: number[], sp: number, pts: [number, RGB][][], pm: number, a: number) {
    if (pm <= 0 || pm >= 1) return
    const e = pm < 0.3 ? eout(pm / 0.3) : pm > 0.78 ? 1 - eio((pm - 0.78) / 0.22) : 1
    const hw = 18, hmax = Math.min(sp * 0.46, 22), X = G.xMlp
    V.seq.forEach((_, i) => {
      const y = ys[i], hh = hmax * e, col = colorAt(pts[i], X), la = a * laneDim(i)
      ctx.beginPath(); ctx.moveTo(X - hw, y); ctx.quadraticCurveTo(X, y - 2 * hh, X + hw, y); ctx.quadraticCurveTo(X, y + 2 * hh, X - hw, y)
      ctx.fillStyle = rgba(col, 0.12 * la); ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = rgba(col, 0.6 * la); ctx.stroke()
      ;[-9, -3, 3, 9].forEach((dx, c) => {
        const lim = hh * (1 - (dx / hw) ** 2) * 0.85
        for (let r = 0; r < 6; r++) {
          const dy = ((r - 2.5) / 2.5) * hh * 0.85
          if (Math.abs(dy) > lim) continue
          const act = clamp(V.pass.neurons.act[i][c * 6 + r] / V.actMax)
          ctx.fillStyle = act > 0.02 ? rgba(pop(col), (0.25 + 0.75 * act) * e * la) : rgba(C.ink, 0.14 * e * la)
          ctx.beginPath(); ctx.arc(X + dx, y + dy, act > 0.02 ? 1 + act : 1, 0, 7); ctx.fill()
        }
      })
    })
    const m = G.mat, lx = m ? m.x + m.w + 14 : X - 40, ly = m ? m.y + 6 : ys[V.N - 1] + sp
    ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'top'
    ctx.fillStyle = rgba(C.ink2, e * a); ctx.fillText('mlp: 768 → 3072 → 768', lx, ly)
    ctx.fillStyle = rgba(C.mute, e * a); ctx.fillText('GELU · 24 neurons shown', lx, ly + 15)
  }
  /** The 12 heads of the shown block side by side, while attention plays; click one to select it. */
  function drawThumbs(a: number) {
    G.thumbs = []
    if (a <= 0) return
    const N = V.N, gap = 10, tw = (G.distW - 3 * gap) / 4, cs = Math.max(3, Math.floor(tw / N)), side = cs * N, y0 = G.labY + 68
    ctx.font = F.label; spaced(true); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
    ctx.fillText(`BLOCK ${S.layer + 1} · 12 HEADS`, G.xDist0, y0 - 14); spaced(false)
    for (let h = 0; h < 12; h++) {
      const A = V.pass.att[S.layer][h], x = G.xDist0 + (h % 4) * (tw + gap), y = y0 + Math.floor(h / 4) * (side + 26), on = h === S.head
      for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
        ctx.fillStyle = rgba(tokC(j), (0.08 + 0.92 * Math.sqrt(A[i][j])) * a); ctx.fillRect(x + j * cs, y + i * cs, cs - 0.5, cs - 0.5)
      }
      if (on) { ctx.strokeStyle = rgba(C.ink, a); ctx.lineWidth = 1.5; ctx.strokeRect(x - 2.5, y - 2.5, side + 5, side + 5) }
      const kind = headKind(A, V.pass.ids), tag = kind === 'mixed' ? '' : kind[0]
      ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(on ? C.ink : C.mute, a)
      ctx.fillText(`${h + 1}${tag ? ' ' + tag : ''}`, x, y + side + 4)
      G.thumbs.push({ x: x - 3, y: y - 3, w: side + 6, h: side + 20, head: h })
    }
    ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, a)
    ctx.fillText('p previous token · f first token', G.xDist0, y0 + 3 * (side + 26))
    ctx.fillText('s self · i induction', G.xDist0, y0 + 3 * (side + 26) + 15)
  }

  /** The logit lens: the last position's best guess after each block, as rows in the next-token column. */
  function drawLens(front: number, a: number) {
    if (a <= 0) return
    const y0 = G.labY + 72, lh = Math.min(24, (G.bot + 20 - y0) / 11), W = stage.W
    const ly = (k: number) => y0 + k * lh
    const xs = [G.xMlp + 28, ...G.stack.map((x) => x + 3)]
    const shown = xs.filter((x) => front >= x).length
    const bx0 = G.xDist0 + 112, bx1 = W - G.padR - 44, nextCol = C.tok[V.next.c]
    for (let k = 0; k < 12; k++) {
      const y = ly(k)
      if (k >= shown) { ctx.fillStyle = rgba(C.ink, 0.06 * a); ctx.fillRect(G.xDist0, y, G.distW, 1); continue }
      const g = V.pass.lens[k][0], cur = k === shown - 1, final = g.id === V.next.id
      ctx.font = F.small; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
      ctx.fillStyle = rgba(cur ? C.ink : C.mute, a); ctx.fillText(String(k + 1), G.xDist0 + 14, y)
      tokText(G.xDist0 + 24, y, g.text, a * (cur ? 1 : 0.75), final ? nextCol : null, F.small)
      ctx.fillStyle = rgba(C.ink, 0.07 * a); ctx.fillRect(bx0, y - 2, bx1 - bx0, 4)
      ctx.fillStyle = final ? rgba(nextCol, a) : rgba(C.ink, (cur ? 0.9 : 0.45) * a)
      ctx.fillRect(bx0, y - 2, Math.max(1, g.p * (bx1 - bx0)), 4)
      ctx.textAlign = 'right'; ctx.fillStyle = rgba(cur ? C.ink : C.mute, a)
      ctx.fillText((g.p * 100).toFixed(g.p < 0.1 ? 1 : 0) + '%', W - G.padR, y)
    }
    ctx.font = F.label; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; spaced(true)
    ctx.fillStyle = rgba(C.mute, a); ctx.fillText('LOGIT LENS', G.xDist0, y0 - 22); spaced(false)
  }
  function drawDist(ys: number[], pts: [number, RGB][][], pu: number, pS: number, a: number) {
    const g = eout(clamp((pu - 0.42) / 0.5)), W = stage.W
    const N = V.N, d = V.dist, yl = ys[N - 1], ry = (k: number) => G.mid + (k - 3) * G.rowH
    const bx0 = G.xDist0 + 92, bx1 = W - G.padR - 52, lastCol = colorAt(pts[N - 1], G.xWU), nextCol = C.tok[V.next.c]
    if (g <= 0) return
    ctx.font = F.label; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; spaced(true)
    ctx.fillStyle = rgba(C.mute, g * a); ctx.fillText('P( NEXT | CONTEXT )', G.xDist0, ry(0) - G.rowH * 0.75); spaced(false)
    const picked = pS > 0 ? clamp(pS / 0.3) : 0
    const rows = [...d.rows.map((r) => ({ text: r.text, p: r.p, other: false })), { text: '', p: d.rest, other: true }]
    rows.forEach((r, k) => {
      const gk = eout(clamp((pu - 0.42 - k * 0.035) / 0.45)) * a, y = ry(k)
      ctx.beginPath(); ctx.moveTo(G.xWU + 16, yl)
      ctx.bezierCurveTo(G.xWU + 50, yl, G.xDist0 - 40, y, G.xDist0 - 10, y)
      ctx.lineWidth = 0.5 + 5 * r.p; ctx.strokeStyle = rgba(lastCol, (0.1 + 0.6 * Math.sqrt(r.p)) * gk); ctx.stroke()
      const isPick = k === 0 && picked > 0
      if (r.other) {
        ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
        ctx.fillStyle = rgba(C.mute, gk); ctx.fillText(`${d.restCount.toLocaleString('en-US')} others`, G.xDist0, y)
      } else tokText(G.xDist0, y, r.text, gk * (isPick ? 1 : 0.8), isPick ? mixc(C.ink, nextCol, picked) : null)
      ctx.fillStyle = rgba(C.ink, 0.07 * gk); ctx.fillRect(bx0, y - 3, bx1 - bx0, 6)
      ctx.fillStyle = isPick ? rgba(mixc(C.ink, nextCol, picked), gk) : rgba(C.ink, (r.other ? 0.3 : 0.5) * gk)
      ctx.fillRect(bx0, y - 3, Math.max(1, r.p * (bx1 - bx0) * gk), 6)
      ctx.font = F.small; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
      ctx.fillStyle = rgba(isPick ? C.ink : C.mute, gk)
      ctx.fillText(r.p < 0.001 ? '<0.1%' : (r.p * 100).toFixed(1) + '%', W - G.padR, y)
    })
    if (picked > 0) {
      const sa = picked * a, y = ry(6) + G.rowH
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
      ctx.fillStyle = rgba(C.mute, sa); ctx.fillText('greedy: top token →', G.xDist0, y)
      tokText(G.xDist0 + ctx.measureText('greedy: top token →').width + 8, y, V.next.text, sa, nextCol)
    }
  }
  function drawFlight(pS: number, isLast: boolean) {
    if (pS <= 0.5) return
    const N = V.N, tok = V.next
    const f = eio(clamp((pS - 0.5) / 0.4)), sx = G.xDist0, sy = G.mid - 3 * G.rowH
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
    x = clamp(x, w / 2 + 4, stage.W - w / 2 - 4)
    rr(x - w / 2, y - 26, w, 20, 10)
    ctx.fillStyle = rgba(C.ink, 0.92); ctx.fill()
    ctx.fillStyle = rgba(C.bg); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    ctx.fillText(t, x, y - 15.5)
  }

  /* ---------- frame ---------- */
  function draw(now: number) {
    useCtx(ctx)
    stage.begin()
    const ci = player.curIndex(), N = V.N, { sp, ys } = curYs()
    const pT = prog('tokenize'), pe = prog('embed'), pa = prog('attn'), pm = prog('mlp'), ps = prog('stack'), pu = prog('unembed'), pS = prog('pick')
    const isLast = S.passIdx === pr().passes.length - 1, endFade = clamp((pS - 0.86) / 0.14), compA = 1 - 0.75 * endFade
    const py0 = ys[0] - sp * 0.5 - 12, py1 = ys[N - 1] + sp * 0.5 + 12
    G.plateY = [py0, py1]
    const wu0 = Math.max(G.labY + 50, Math.min(py0, G.mid - 3.5 * G.rowH)), wu1 = Math.max(py1, G.mid + 3.5 * G.rowH)
    const front = frontX(pa, pm, ps, pu), pts = V.seq.map((_, i) => lanePts(i))
    const lensA = pu > 0 ? 1 - clamp(pu / 0.35) : pm > 0 ? clamp(pm / 0.3) : 0

    const flying = pS > 0.5 ? Math.sin(Math.PI * clamp((pS - 0.5) / 0.4)) : 0
    drawTop(ci, lensA > 0.5 && front >= G.xMlp + 28, 1 - 0.75 * flying)
    const sent = drawSentence(pT, pS, endFade, isLast, now)
    drawLanes(ys, pts, front, pu, compA)

    const hp = S.hoverPlate, act = (on: boolean, hov: boolean) => Math.max(on ? 1 : 0, hov ? 0.8 : 0)
    const aAct = act(pa > 0 && pa < 1, hp === 'attn'), mAct = act(pm > 0 && pm < 1, hp === 'mlp')
    plate(G.xLn1, py0, py1, act(pa > 0 && pa < 0.12, hp === 'ln1') * 0.8, { w: 2.5, d: 8 })
    plate(G.xAttn, py0, py1, aAct)
    plate(G.xLn2, py0, py1, act(pm > 0 && pm < 0.2, hp === 'ln2') * 0.8, { w: 2.5, d: 8 })
    plate(G.xMlp, py0, py1, mAct)
    if (hp === 'ln1') subLabel('ln_1 ↗', G.xLn1, G.labY + 32, true)
    else subLabel('attn ↗', G.xAttn, G.labY + 32, aAct > 0.5 || hp === 'attn')
    if (hp === 'ln2') subLabel('ln_2 ↗', G.xLn2, G.labY + 32, true)
    else subLabel('mlp ↗', G.xMlp, G.labY + 32, mAct > 0.5 || hp === 'mlp')
    // blocks 2–12, one plate each; the block whose head is shown stays lit
    G.stack.forEach((x, k) => {
      const run = ps > 0 && ps < 1 ? clamp(1 - Math.abs(ps * 11 - k - 0.5) / 1.3) : 0, sel = S.layer === k + 1 && pa > 0 ? 0.55 : 0
      plate(x, py0, py1, Math.max(run, sel), { w: 4, d: 8 })
    })
    subLabel('each: attn + mlp', (G.xS0 + G.xS1) / 2, G.labY + 32, false)
    const lnAct = pu > 0 && pu < 1 ? clamp(1 - Math.abs(pu - 0.2) / 0.2) : 0, wuAct = pu > 0 && pu < 1 ? clamp(1 - Math.abs(pu - 0.55) / 0.3) : 0
    const hovL = hp === 'ln', hovW = hp === 'wu'
    plate(G.xLn, py0, py1, Math.max(lnAct, hovL ? 0.8 : 0)); subLabel('ln_f ↗', G.xLn, G.labY + 32, lnAct > 0.1 || hovL)
    plate(G.xWU, wu0, wu1, Math.max(wuAct, hovW ? 0.8 : 0), { w: 11, d: 11, hatch: 34 }); subLabel('W_U ↗', G.xWU, G.labY + 32, wuAct > 0.1 || hovW)
    G.wuY = [wu0, wu1]

    drawEmb(ys, sp, pe, isLast ? 1 - endFade : 1)
    drawAttention(ys, sp, pa, compA)
    drawMLP(ys, sp, pts, pm, compA)
    drawResidual(ys, sp, pts, front, compA)
    drawSources(ys, front, pu, compA)
    drawParticles(ys, pts, front, pu, compA)
    drawChips(ys, sp, pT, sent, isLast ? 1 - endFade : 1)
    drawThumbs(pa > 0 && pu <= 0 ? 1 - lensA : 0)
    drawLens(front, lensA)
    drawDist(ys, pts, pu, pS, 1 - endFade)
    drawFlight(pS, isLast)
    const hint: Record<string, [number, number]> = {
      attn: [G.xAttn, py0 - 4], mlp: [G.xMlp, py0 - 4], ln1: [G.xLn1, py0 - 4], ln2: [G.xLn2, py0 - 4], tok: [G.xTok + 56, py0 - 4],
      emb: [(G.xEmb0 + G.xEmb1) / 2, py0 - 4], ln: [G.xLn, py0 - 4], wu: [G.xWU, wu0 - 4], bars: [G.xDist0 + G.distW / 2, G.mid - 3.75 * G.rowH - 8],
    }
    if (hp === 'bars') { rr(G.xDist0 - 8, G.mid - 3.6 * G.rowH, G.distW + 12, 7.2 * G.rowH, 8); ctx.strokeStyle = rgba(C.ink, 0.35); ctx.lineWidth = 1; ctx.stroke() }
    if (hint[hp]) drawOpenHint(...hint[hp])
  }

  /* ---------- captions ---------- */
  function caption(ci: number) {
    const N = V.N, pct = (p: number) => (p * 100).toFixed(1) + '%'
    switch (PHASES[ci].id) {
      case 'tokenize': return S.passIdx === 0
        ? { t: "GPT-2's byte-level BPE splits the text into tokens, each an id in a 50,257-entry vocabulary; Ġ marks a token that starts with a space. Click the tokens to see the merges.", s: `ids [${V.seq.map((t) => t.id).join(', ')}]` }
        : { t: 'The token picked in the last pass is appended and the whole sequence runs again. Real systems keep a KV cache (the keys and values of earlier positions) so those are not recomputed.', s: `${N} tokens · +${V.seq[N - 1].id}` }
      case 'embed': return { t: "Each id selects its row of the embedding matrix W_E, and the row of W_P for its position is added; the strips show the first 16 of the 768 real numbers. Click a strip to see the lookup.", s: `[${N} × 768]` }
      case 'attn': return { t: 'Each position pulls in information from itself and earlier positions: its query (what it looks for) is scored against their keys (what they offer), and it takes a weighted mix of their values; later tokens are masked. Hover a lane to see its weights, and step through all 144 real heads below. Click attn for every product.', s: `12 heads × [${N} × ${N}]` }
      case 'mlp': return { t: 'The MLP works on each position alone: it expands the 768 numbers to 3,072 neurons, applies GELU, and projects back. ln_1 and ln_2, the thin panes, normalise the stream before attn and mlp read it. Click mlp for the products.', s: `[${N} × 768] → [${N} × 3072]` }
      case 'stack': return { t: 'Eleven more blocks, same shape but each with its own weights, add their results to the stream. Right: the logit lens, what the last position would predict if the model stopped after each block. Lane colour sketches where information came from; the bars show the shares.', s: '12 blocks ≈85M params · embeddings ≈39M (W_E, reused as W_U)' }
      case 'unembed': return { t: 'Only the last position predicts: ln_f normalises it and the unembedding W_U (the LM head, which GPT-2 ties to W_Eᵀ) scores all 50,257 tokens with one dot product each. softmax turns the scores into these probabilities, the real ones from GPT-2 small. Click ln_f or W_U to see it step by step.', s: '[1 × 768] · W_Eᵀ → [1 × 50,257]' }
      default: {
        const p = V.dist.rows[0].p
        return { t: 'This page always takes the most likely token (greedy decoding) so every number stays real; chat models sample instead, see Unembed & Sampling.' + note(pct), s: `greedy → ${tokLabel(V.next.text)} · ${pct(p)}` }
      }
    }
  }
  /** What the real numbers show for this prompt, on its first pass. */
  function note(pct: (p: number) => string) {
    if (S.passIdx > 0) return ''
    if (pr().key === 'france') {
      const d = nextDist(V.pass.next, 1, 256), r = d.rows.findIndex((c) => c.text === ' Paris')
      return r < 0 ? '' : ` GPT-2 small is weak at facts: Paris is only number ${r + 1} (${pct(d.rows[r].p)}).`
    }
    if (pr().key === 'count') return ' Block 6, head 6 is an induction head: at the second “three” it attends to “four”, the token that followed the first one.'
    return ''
  }

  /* ---------- pointer ---------- */
  const inMat = (x: number, y: number) => { const m = G.mat; return !!m && prog('attn') > 0 && x >= m.x && x <= m.x + m.w && y >= m.y && y <= m.y + m.h }
  /** Plates that open a detail view, keyed by the route they open. */
  const plateAt = (x: number, y: number) => {
    if (Math.abs(x - G.xWU) < 16 && y > G.wuY[0] - 14 && y < G.wuY[1] + 14) return 'wu'
    if (x >= G.xDist0 - 10 && prog('unembed') > 0.4 && y > G.wuY[0] - 20 && y < G.wuY[1] + 30) return 'bars'
    if (y < G.plateY[0] - 14 || y > G.plateY[1] + 14) return ''
    if (Math.abs(x - G.xLn1) < 7) return 'ln1'
    if (Math.abs(x - G.xLn2) < 7) return 'ln2'
    if (Math.abs(x - G.xAttn) < 14) return 'attn'
    if (Math.abs(x - G.xMlp) < 14) return 'mlp'
    if (x >= G.xTok - 4 && x < G.xEmb0 - 12) return 'tok'
    if (x >= G.xEmb0 - 4 && x <= G.xEmb1 + 4) return 'emb'
    if (Math.abs(x - G.xLn) < 14) return 'ln'
    return ''
  }
  const PLATE_ROUTES: Record<string, string> = {
    tok: 'anatomy/tokenizer', emb: 'anatomy/embedding', ln1: 'anatomy/layernorm', attn: 'anatomy/attention',
    ln2: 'anatomy/layernorm', mlp: 'anatomy/mlp', ln: 'anatomy/unembed', wu: 'anatomy/unembed', bars: 'anatomy/unembed',
  }
  /** The detail view opens at the step that matches what the overview is showing. */
  const plateLink = (pl: string) => {
    const at = (id: string) => `${PLATE_ROUTES[pl]}?phase=${id}`
    if (pl === 'attn' && prog('attn') > 0.1) return at('softmax')
    if (pl === 'mlp' && prog('mlp') > 0) return at('gelu')
    if (pl === 'ln') return at('lnf')
    if (pl === 'wu' || pl === 'bars') return prog('pick') > 0 ? at('sample') : prog('unembed') > 0.4 ? at('softmax') : at('logits')
    if (pl === 'emb' && prog('embed') > 0.5) return at('pos')
    return PLATE_ROUTES[pl]
  }
  const laneAt = (x: number, y: number) => {
    const { sp, ys } = curYs()
    let h = -1
    if (x > G.xTok - 8 && x < G.xWU + 20) ys.forEach((yy, i) => { if (Math.abs(y - yy) < sp / 2) h = i })
    return h
  }
  const cv = stage.canvas
  cv.addEventListener('pointermove', (e) => {
    const [x, y] = stage.local(e)
    S.hoverPlate = plateAt(x, y)
    S.hover = S.hoverPlate ? -1 : laneAt(x, y)
    cv.style.cursor = inMat(x, y) || S.hoverPlate || S.hover >= 0 || thumbAt(x, y) ? 'pointer' : 'default'
  })
  cv.addEventListener('pointerleave', () => { S.hover = -1; S.hoverPlate = '' })
  const thumbAt = (x: number, y: number) => G.thumbs.find((t) => x >= t.x && x <= t.x + t.w && y >= t.y && y <= t.y + t.h)
  cv.addEventListener('click', (e) => {
    const [x, y] = stage.local(e)
    const th = thumbAt(x, y)
    if (th) { S.head = th.head; headStep.set(th.head); linkHead(); return }
    const pl = plateAt(x, y)
    if (pl) { nav(plateLink(pl), { x: e.clientX, y: e.clientY }); return }
    if (inMat(x, y)) { S.head = (S.head + 1) % 12; headStep.set(S.head); return }
    const lane = laneAt(x, y)
    S.lock = lane === S.lock ? -1 : lane
  })

  // the same detail views for keyboard and screen-reader users
  const links = document.createElement('nav')
  links.className = 'sr-only'
  links.setAttribute('aria-label', 'Open a step')
  ;([['Tokenizer', 'tok'], ['Embedding', 'emb'], ['Attention', 'attn'], ['MLP', 'mlp'], ['LayerNorm', 'ln1'], ['Unembed and sampling', 'wu']] as const).forEach(([t, k]) => {
    const b = document.createElement('button')
    b.type = 'button'; b.textContent = `Open ${t}`
    b.addEventListener('click', () => nav(PLATE_ROUTES[k]))
    links.append(b)
  })
  frame.stageHost.append(links)

  const linked = player.t // a link like #/anatomy?phase=mlp already placed the player
  if (saved) {
    // coming back from a detail view: the same prompt, pass, frame and head, paused
    S.preset = saved.preset; presetToggle.set(saved.preset)
    startPreset()
    setPass(saved.passIdx); player.t = saved.t
    S.layer = saved.layer; S.head = saved.head; S.lock = saved.lock; layerStep.set(S.layer); headStep.set(S.head); linkHead()
    player.setPlaying(false)
  } else {
    // a shared link's prompt, pass and head
    const q = getParams(), qk = PRESETS.findIndex((p) => p.key === q.get('prompt'))
    if (qk >= 0) { S.preset = qk; presetToggle.set(qk) }
    startPreset()
    const [l, h] = (q.get('head') ?? '').split('.').map((v) => clamp(Math.round(+v) - 1, 0, 11))
    if (q.has('head') && !isNaN(l) && !isNaN(h)) { S.layer = l; S.head = h; layerStep.set(l); headStep.set(h); linkHead() }
    const k = clamp(Math.round(+(q.get('pass') ?? 1)) - 1, 0, pr().passes.length - 1)
    if (k) setPass(k)
    if (linked) player.t = linked
    else if (reduced) player.t = player.start('attn') + PHASES[2].dur * 0.7
  }
  player.describe = (i) => { const c = caption(i); return [c.t, c.s] }
  let shownPass = -1
  const stop = runLoop((dt, now) => {
    player.tick(dt)
    draw(now)
    player.updateUI()
    const ci = player.curIndex(), c = caption(ci)
    frame.setCaption(PHASES[ci].name, PHASES[ci].short ?? PHASES[ci].name, c.t, c.s)
    if (shownPass !== S.passIdx) { shownPass = S.passIdx; passStep.set(S.passIdx) }
  }, () => !player.playing)
  return () => {
    saved = { preset: S.preset, passIdx: S.passIdx, t: player.t, layer: S.layer, head: S.head, lock: S.lock }
    stop(); player.destroy(); stage.destroy()
  }
}
