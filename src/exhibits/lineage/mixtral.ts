import { F, chipW, drawChip, fillRich, mathName, rr, serifAt } from '../../core/draw'
import { fmt, fmtF, gemm, type Rect } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, gauss, lerp, rng } from '../../core/util'
import type { Nav } from '../registry'
import { mountExhibit, words, type Env } from '../kit'

/*
 * Mixtral 8x7B as a diff against GPT-2. The block is LLaMA's (RMSNorm, RoPE, grouped-query attention)
 * except the MLP, which becomes a sparse mixture of experts: a router scores 8 SwiGLU experts per
 * token and runs the top 2. Real sizes are Mixtral 8x7B's; the routing uses toy vectors (d_model 4).
 */

const PHASES = [
  { id: 'blocks', name: 'GPT-2 vs Mixtral', short: 'Blocks', dur: 7 },
  { id: 'route', name: 'The router', short: 'Router', dur: 11 },
  { id: 'dispatch', name: 'Two experts per token', short: 'Top-2', dur: 10 },
  { id: 'params', name: 'Stored vs used', short: 'Params', dur: 8 },
  { id: 'balance', name: 'Load balancing', short: 'Balance', dur: 9 },
]

const TOKS = words(['The', 'cat', 'sat', 'on', 'the', 'mat'])
const N = TOKS.length, D = 4, E = 8, TOP = 2
/** Seed for readable toy routing: uneven load (one expert idle, one with 3 tokens), both ‘the’s agree. */
const ROUTE_SEED = 21

const softmax = (z: number[]) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s) }
const R = (() => {
  const r = rng(ROUTE_SEED)
  const X = TOKS.map(() => Array.from({ length: D }, () => gauss(r)))
  const Wg = Array.from({ length: D }, () => Array.from({ length: E }, () => gauss(r) * 0.9))
  const Z = X.map((x) => Wg[0].map((_, e) => x.reduce((s, v, k) => s + v * Wg[k][e], 0)))
  const top = Z.map((z) => [...z.keys()].sort((a, b) => z[b] - z[a]).slice(0, TOP))
  const G = Z.map((z, i) => { const w = softmax(top[i].map((e) => z[e])); return z.map((_, e) => { const k = top[i].indexOf(e); return k < 0 ? 0 : w[k] }) })
  const P = Z.map(softmax)
  const load = Array.from({ length: E }, (_, e) => top.filter((t) => t.includes(e)).length)
  const f = load.map((n) => n / (N * TOP)), Pm = Array.from({ length: E }, (_, e) => P.reduce((s, row) => s + row[e], 0) / N)
  const aux = E * f.reduce((s, v, e) => s + v * Pm[e], 0)
  return { X, Wg, Z, top, G, P, load, f, Pm, aux }
})()
const eName = (e: number) => `E${e + 1}`

/** Mixtral 8x7B parameter counts, in billions. */
const PB = (() => {
  const d = 4096, ff = 14336, L = 32, V = 32000
  const emb = (2 * V * d) / 1e9, attn = (L * (2 * d * d + 2 * d * 1024)) / 1e9, expert = (L * 3 * d * ff) / 1e9
  return { emb, attn, expert, total: emb + attn + E * expert, active: emb + attn + TOP * expert }
})()

const COMPARE: Record<string, [string, string]> = {
  blocks: ['GPT-2’s MLP', 'anatomy/mlp'], route: ['GPT-2’s MLP', 'anatomy/mlp'], dispatch: ['GPT-2’s MLP', 'anatomy/mlp?phase=up'],
  params: ['GPT-2’s forward pass', 'anatomy?phase=stack'], balance: ['GPT-2’s training loss', 'training/loss'],
}

const CAPS: Record<string, [string, string]> = {
  blocks: ['Mixtral’s block is LLaMA’s (RMSNorm, RoPE, grouped-query attention) except for the MLP: each layer has 8 expert MLPs and a small router that sends each token to 2 of them. Click a label to jump to that part.', '8 experts per layer · top 2 · 32 layers'],
  route: ['The router is one small matrix: a token’s vector times W_g gives one score per expert. The two highest scores win, and a softmax over just those two sets how much each winner counts. Hover the cells.', 'g = softmax(top-2(x · W_g))'],
  dispatch: ['Each token runs through its two experts only, and the two outputs are added with the router’s weights. Different tokens pick different experts, so some experts get more of the batch than others, and some get none.', 'y = g_a · E_a(x) + g_b · E_b(x)'],
  params: ['All 8 experts must sit in memory: 46.7B parameters. But each token runs only 2 of them, 12.9B parameters, so it costs about as much compute as a 13B dense model.', '46.7B stored · 12.9B used per token'],
  balance: ['Left alone, a router keeps choosing the experts that are already good, and the others stop learning. MoE training adds a small loss that is lowest when both the tokens and the router’s probability are spread evenly over the experts.', 'L_aux = N · Σ f_i · P_i'],
}

export function mountMixtral(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: '',
      eyebrow: 'Architectures · Decoder-only',
      title: 'Mixtral',
      subtitle: 'Mixtral 8x7B · a mixture of experts in place of the MLP',
      specs: [
        { label: 'compared', value: 'Mixtral 8x7B', real: 'GPT-2 small', realLabel: 'vs' },
        { label: 'layers', value: '32', real: '12' },
        { label: 'd_model', value: '4,096', real: '768' },
        { label: 'MLP', value: '8 experts · top 2', real: '1 dense' },
        { label: 'd_ff', value: '14,336 each', real: '3,072' },
        { label: 'params', value: '46.7B · 12.9B active', real: '124M' },
        { label: 'context', value: '32,768', real: '1,024' },
      ],
    },
    size: [1040, 470],
    aria: 'Mixtral compared with GPT-2: the MLP of each block becomes eight expert MLPs and a router that sends each token to the two best-scoring experts, so the model stores 46.7 billion parameters but uses 12.9 billion per token.',
    phases: PHASES, learn: 'mixtral', tokens: TOKS, compare: COMPARE, caps: CAPS,
    still: ['dispatch', 9],
    hints: {
      blocks: 'Click a dark label on the drawing to jump to that part.',
      route: 'Hover a score or a gate to see how it is computed; click or tap to pin it.',
    },
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { pill, addNode, caption, title, arrow, lane, glass } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46
  const diamond = (x: number, y: number, r: number, a: number, on = false) => {
    ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath()
    ctx.fillStyle = rgba(on ? C.ink : C.bg, a * (on ? 0.9 : 1)); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.8 * a); ctx.lineWidth = 1.2; ctx.stroke()
  }

  /* ---------- scene 1: the blocks ---------- */
  function sceneBlocks(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const sx0 = pad + 130, sx1 = W - pad - 16, fx = (f: number) => lerp(sx0, sx1, f)
    const ys = [top + avail * 0.2, top + avail * 0.58], as = [eout(clamp(p / 0.3)), eout(clamp((p - 0.25) / 0.3))]
    const X = { pos: fx(0.04), n1: fx(0.17), at: fx(0.33), a1: fx(0.47), n2: fx(0.6), ml: fx(0.78), rt: fx(0.7), ex: fx(0.815), a2: fx(0.95) }
    const exX = (e: number) => X.ex + (e - 3.5) * 22
    ys.forEach((y, r) => {
      const a = as[r]
      if (a <= 0) return
      k.rowName(r ? 'Mixtral' : 'GPT-2', r ? '8x7B · 46.7B' : 'small · 124M', pad, y, a)
      for (let i = 0; i < N; i++) lane(sx0, lerp(sx0, sx1, a), y + (i - (N - 1) / 2) * 4, hue(i), a)
      if (!r) {
        rr(X.pos - 26, y - 15, 52, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.7 * a); ctx.lineWidth = 1; ctx.stroke()
        caption('+ W_P', X.pos, y + 4, a, C.ink)
        caption('learned positions', X.pos, y + 44, a)
      } else caption('no position vector', X.pos, y + 44, a)
      const labs = r ? ['RMSNorm', 'GQA + RoPE', 'RMSNorm'] : ['LayerNorm', 'MHA', 'LayerNorm']
      ;[X.n1, X.at, X.n2].forEach((x, j) => { glass(x, y - 26, y + 26, 0.2, { w: 8, d: 9 }, a); caption(labs[j], x, y + 44, a) })
      addNode(X.a1, y, a); addNode(X.a2, y, a)
      if (!r) { glass(X.ml, y - 26, y + 26, 0.2, { w: 8, d: 9 }, a); caption('MLP · GELU', X.ml, y + 44, a); return }
      // the MoE layer: a router, then 8 experts of which 2 run for this token
      const pick = R.top[0], ey = y + 62
      for (let e = 0; e < E; e++) {
        const on = pick.includes(e), ex = exX(e)
        if (on) {
          ctx.strokeStyle = rgba(hue(0), 0.9 * a); ctx.lineWidth = 1 + 3 * R.G[0][e]
          ctx.beginPath(); ctx.moveTo(X.rt, y + 11); ctx.bezierCurveTo(X.rt, y + 40, ex, y + 22, ex, ey - 17); ctx.stroke()
        }
        glass(ex, ey - 12, ey + 12, on ? 0.7 : 0.12, { w: 5, d: 5 }, a)
        ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(on ? C.ink : C.mute, a); ctx.fillText(eName(e), ex, ey + 22)
      }
      diamond(X.rt, y, 11, a)
      caption('8 expert MLPs, 2 run per token', X.ex, ey + 54, a, C.ink2)
    })
    const la = eout(clamp((p - 0.55) / 0.25)), [yA, yB] = ys
    if (la > 0) {
      ctx.setLineDash([2, 4]); ctx.strokeStyle = rgba(C.ink, 0.35 * la); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(X.ml, yA + 52); ctx.lineTo(X.ml, yB - 58); ctx.stroke()
      ctx.setLineDash([])
      caption('becomes', X.ml + 6, (yA + yB) / 2 + 8, la, C.ink2, 'left')
      pill('router', X.rt, yB - 44, 'route', la)
      pill('top-2', X.ex, yB - 44, 'dispatch', la)
      pill('load', X.a2 - 10, yB - 44, 'balance', la)
      pill('46.7B', pad + 30, yB + 48, 'params', la)
    }
    mk.formula = { segs: [['GPT-2  y = MLP(x)', C.ink2], ['     ·     ', C.mute], ['Mixtral  y = Σ over top-2 e:  g_e(x) · E_e(x)', C.ink]], note: 'Everything else in the block is LLaMA’s: RMSNorm, rotary positions, 32 query and 8 key/value heads. Each expert is a SwiGLU MLP of width 14,336. Click a label to open that part.' }
  }

  /* ---------- scene 2: the router, a small GEMM and a top-2 pick ---------- */
  function sceneRoute(p: number) {
    const fin = eout(clamp(p / 0.06)), c = 26
    const rX: Rect = { x: pad + 76, y: top + 164, c }, rZ: Rect = { x: rX.x + D * c + 34, y: rX.y, c }, rW: Rect = { x: rZ.x, y: rX.y - D * c - 30, c }
    const rG: Rect = { x: rZ.x + E * c + 58, y: rX.y, c }
    const g = gemm(clamp((p - 0.06) / 0.42), N, E, D, 'fast')
    TOKS.forEach((t, i) => drawChip(rX.x - 10 - chipW(t.text), rX.y + (i + 0.5) * c, t, fin, Math.min(20, c * 0.72)))
    mk.drawMat({ r: rX, vals: R.X, kind: 'row', alpha: fin, name: 'x', shape: '6 × 4', real: '6 × 4,096', labelW: D * c + 30 })
    mk.drawMat({ r: rW, vals: R.Wg, kind: 'w', alpha: fin, name: 'W_g', shape: '4 × 8', real: '4,096 × 8' })
    mk.drawMat({ r: rZ, vals: R.Z, kind: 'score', alpha: fin, name: 'scores', shape: '6 × 8', label: 'bottom', reveal: g.rev })
    const heads = (r: Rect, a: number) => {
      ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
      for (let e = 0; e < E; e++) ctx.fillText(eName(e), r.x + (e + 0.5) * c, r.y - 7)
    }
    heads(rZ, fin * clamp((p - 0.4) / 0.08))
    mk.hit('z', rZ, N, E)
    // the two best scores per token, then the softmax over just those two
    for (let i = 0; i < N; i++) {
      const ra = eout(clamp((p - 0.5 - i * 0.02) / 0.05))
      if (ra <= 0) continue
      ctx.strokeStyle = rgba(C.ink, 0.95 * ra); ctx.lineWidth = 2
      for (const e of R.top[i]) ctx.strokeRect(rZ.x + e * c + 1, rZ.y + i * c + 1, c - 2, c - 2)
    }
    const ga = fin * eout(clamp((p - 0.62) / 0.06))
    if (ga > 0) {
      arrow([[rZ.x + E * c + 12, rZ.y + (N * c) / 2], [rG.x - 12, rG.y + (N * c) / 2]], ga)
      caption('top 2,', (rZ.x + E * c + rG.x) / 2, rZ.y + (N * c) / 2 - 22, ga, C.ink2)
      caption('softmax', (rZ.x + E * c + rG.x) / 2, rZ.y + (N * c) / 2 - 8, ga, C.ink2)
      mk.drawMat({ r: rG, vals: R.G, kind: 'score', alpha: ga, name: 'g', shape: '6 × 8', label: 'bottom', noText: true,
        reveal: (i) => clamp((p - 0.64 - i * 0.025) / 0.04),
        paint: (x, y, cc, i, e, a) => { if (!R.G[i][e]) { mk.hatch(x, y, cc, a * 0.6); return 0 } return mk.paintAttn(x, y, cc, R.G[i][e], hue(i), a) } })
      heads(rG, ga)
      mk.hit('g', rG, N, E)
      // in words: which two experts, and how much each counts
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
      for (let i = 0; i < N; i++) {
        const ra = clamp((p - 0.66 - i * 0.025) / 0.04)
        if (ra <= 0) continue
        const [a, b] = R.top[i]
        ctx.fillStyle = rgba(hue(i), ga * ra); ctx.fillText(`${eName(a)} ${fmt(R.G[i][a])}  ${eName(b)} ${fmt(R.G[i][b])}`, rG.x + E * c + 18, rG.y + (i + 0.5) * c)
      }
    }
    const f = mk.focus
    const fs = mk.resolve({ z: { g, K: D } })
    if (fs && (!f || f.key === 'z')) mk.gemmOverlay({ A: rX, Av: R.X, B: rW, Bv: R.Wg, C: rZ, f: fs, names: ['score', 'x', 'W_g'], note: `How much the router likes ${eName(fs.j)} for ‘${TOKS[fs.i].text}’. Toy d_model 4; Mixtral’s router is 4,096 × 8, one column per expert.` })
    else if (f && f.key === 'g') {
      const [a, b] = R.top[f.i], v = R.G[f.i][f.j]
      ctx.strokeStyle = rgba(C.ink, 1); ctx.lineWidth = 2; ctx.strokeRect(rG.x + f.j * c, rG.y + f.i * c, c, c)
      mk.formula = v
        ? { segs: [[`g[${TOKS[f.i].text}, ${eName(f.j)}]`, C.ink], ['  =  ', C.mute], [`e^${fmtF(R.Z[f.i][f.j])} / (e^${fmtF(R.Z[f.i][a])} + e^${fmtF(R.Z[f.i][b])})`, C.ink2], ['  =  ', C.mute], [fmtF(v), C.ink]], note: `A softmax over the two winning scores only; the other six experts get exactly 0 and do not run for ‘${TOKS[f.i].text}’.` }
        : { segs: [[`g[${TOKS[f.i].text}, ${eName(f.j)}]`, C.ink], ['  =  ', C.mute], ['0', C.ink]], note: `${eName(f.j)} is not in the top 2 for ‘${TOKS[f.i].text}’, so it is skipped: no compute spent on it.` }
    } else mk.formula = { segs: [['scores = x · W_g', C.ink2], ['     ·     ', C.mute], ['g = softmax over the top 2 scores, 0 elsewhere', C.ink]], note: 'Each row is one token and each column one expert. Both ‘the’s pick E7 and E6: routing depends on the token’s vector, and repeated tokens often land on the same experts.' }
  }

  /* ---------- scene 3: dispatch to the experts and combine ---------- */
  function sceneDispatch(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.06)), xE = 470, xO = W - pad - 190, sq = 12
    const yt = (i: number) => top + 40 + i * ((avail - 64) / (N - 1)), ye = (e: number) => top + 34 + e * ((avail - 52) / (E - 1)), xs = xE + 50
    const win = (i: number) => clamp((p - 0.06 - i * 0.13) / 0.13) // token i's turn
    title('tokens', pad, top + 8, fin)
    title('experts · swiglu mlps', xE - 10, top + 8, fin)
    title('outputs', xO, top + 8, fin)
    const got = Array.from({ length: E }, () => [] as number[])
    for (let i = 0; i < N; i++) {
      const w = win(i), cur = w > 0 && w < 1
      drawChip(pad, yt(i), TOKS[i], fin, 20, cur)
      if (w <= 0) continue
      for (const e of R.top[i]) {
        const g = R.G[i][e], inA = clamp(w / 0.4), outA = clamp((w - 0.55) / 0.3), x0 = pad + chipW(TOKS[i].text) + 6
        ctx.strokeStyle = rgba(hue(i), (cur ? 0.9 : 0.35) * fin); ctx.lineWidth = 0.8 + 4 * g
        ctx.beginPath(); ctx.moveTo(x0, yt(i)); ctx.bezierCurveTo(lerp(x0, xE, 0.5), yt(i), lerp(x0, xE, 0.5), ye(e), lerp(x0, xE - 11, inA), lerp(yt(i), ye(e), inA)); ctx.stroke()
        if (inA >= 1) got[e].push(i)
        if (outA > 0) {
          const xo = xs + 3 * (sq + 3) + 6
          ctx.beginPath(); ctx.moveTo(xo, ye(e)); ctx.bezierCurveTo(lerp(xo, xO, 0.5), ye(e), lerp(xo, xO, 0.5), yt(i), lerp(xo, xO - 8, outA), lerp(ye(e), yt(i), outA)); ctx.stroke()
        }
      }
      const oa = clamp((win(i) - 0.8) / 0.2)
      if (oa > 0) {
        drawChip(xO, yt(i), TOKS[i], fin * oa, 20, cur)
        const [a, b] = R.top[i]
        ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, fin * oa)
        ctx.fillText(`${fmt(R.G[i][a])}·${eName(a)} + ${fmt(R.G[i][b])}·${eName(b)}`, xO + chipW(TOKS[i].text) + 10, yt(i))
      }
    }
    for (let e = 0; e < E; e++) {
      const y = ye(e), busy = got[e].length
      glass(xE, y - 15, y + 15, busy ? 0.55 : 0.1, { w: 8, d: 7 }, fin)
      mathName(`E_${e + 1}`, xE + 16, y + 5, fin * (busy ? 1 : 0.55), 17)
      got[e].forEach((i, n) => { rr(xs + n * (sq + 3), y - sq / 2, sq, sq, 2); ctx.fillStyle = rgba(hue(i), 0.85 * fin); ctx.fill() })
    }
    const sa = eout(clamp((p - 0.86) / 0.08))
    caption(`load ${R.load.join(' · ')}: ${eName(R.load.indexOf(0))} gets no token, ${eName(R.load.indexOf(Math.max(...R.load)))} gets ${Math.max(...R.load)}`, xE - 10, top + avail + 18, sa, C.ink2, 'left')
    const cur = Math.min(N - 1, Math.max(0, Math.floor((p - 0.06) / 0.13)))
    const [a, b] = R.top[cur]
    mk.formula = { segs: [['y = g_a · E_a(x) + g_b · E_b(x)', C.ink], ['     ·     ', C.mute], [`‘${TOKS[cur].text}’:  ${fmt(R.G[cur][a])} · ${eName(a)}(x) + ${fmt(R.G[cur][b])} · ${eName(b)}(x)`, hue(cur)]], note: 'Each expert only sees the tokens routed to it (the squares), as one small batch. The other six experts do no work for a token.' }
  }

  /* ---------- scene 4: stored vs used parameters ---------- */
  function sceneParams(p: number) {
    const { W } = stage
    const x0 = pad + 170, w = W - pad - x0 - 10, sc = w / PB.total, bh = 30
    const rows: [string, string, number][] = [
      ['stored', `${PB.total.toFixed(1)}B · ${Math.round(PB.total * 2)} GB in BF16`, top + 60],
      ['per token', `${PB.active.toFixed(1)}B · ≈ ${Math.round(PB.active * 2)} GFLOPs`, top + 170],
      ['GPT-2 small', '0.124B · ≈ 0.25 GFLOPs', top + 280],
    ]
    const grow = [eio(clamp((p - 0.04) / 0.3)), eio(clamp((p - 0.38) / 0.25)), eio(clamp((p - 0.68) / 0.15))]
    rows.forEach(([name, sub, y], r) => {
      const a = eout(clamp(grow[r] * 3))
      if (a <= 0) return
      ctx.font = serifAt(21); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(name, pad, y + bh / 2 + 2)
      caption(sub, pad, y + bh / 2 + 20, a, C.mute, 'left')
    })
    const seg = (x: number, y: number, len: number, label: string, fill: number, a: number, dashed = false) => {
      if (len <= 0.5 || a <= 0) return
      ctx.fillStyle = rgba(C.ink, fill * a); ctx.fillRect(x, y, len, bh)
      ctx.strokeStyle = rgba(C.ink, 0.45 * a); ctx.lineWidth = 1
      if (dashed) ctx.setLineDash([3, 3])
      ctx.strokeRect(x + 0.5, y + 0.5, len - 1, bh - 1); ctx.setLineDash([])
      if (label) { ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(fill > 0.5 ? C.bg : C.ink2, a); if (ctx.measureText(label).width < len - 6) ctx.fillText(label, x + len / 2, y + bh / 2 + 0.5) }
    }
    const shared = (PB.emb + PB.attn) * sc, ew = PB.expert * sc
    // stored: embeddings and attention, then all 8 experts
    let y = rows[0][2], L = w * grow[0]
    seg(x0, y, Math.min(L, shared), '', 0.6, 1)
    for (let e = 0; e < E; e++) seg(x0 + shared + e * ew, y, clamp(L - shared - e * ew, 0, ew), eName(e), 0.35, 1)
    if (grow[0] >= 1) { caption(`attention, embeddings: ${(PB.emb + PB.attn).toFixed(1)}B`, x0, y - 10, 1, C.ink2, 'left'); caption(`8 experts × 32 layers: ${(E * PB.expert).toFixed(1)}B`, x0 + shared + 4 * ew, y - 10, 1, C.ink2, 'center') }
    // used per token: the same shared part and any 2 experts per layer
    y = rows[1][2]; L = (shared + TOP * ew) * grow[1]
    if (grow[1] > 0) {
      seg(x0, y, Math.min(L, shared), '', 0.6, 1)
      for (let e = 0; e < E; e++) {
        if (e < TOP) seg(x0 + shared + e * ew, y, clamp(L - shared - e * ew, 0, ew), '2 of 8', 0.8, 1)
        else seg(x0 + shared + e * ew, y, ew, '', 0, 0.35 * grow[1], true)
      }
      if (grow[1] >= 1) caption('any 2 of the 8, chosen per token in every layer', x0 + shared, y + bh + 18, 1, C.mute, 'left')
    }
    y = rows[2][2]
    if (grow[2] > 0) {
      ctx.fillStyle = rgba(C.ink, 0.8); ctx.fillRect(x0, y, Math.max(2, 0.124 * sc) * grow[2], bh)
      caption('the width of one line at this scale', x0 + 12, y + bh / 2 + 4, grow[2], C.mute, 'left')
    }
    mk.formula = { segs: [['total = 1.6B + 8 × 5.64B = 46.7B', C.ink2], ['     ·     ', C.mute], ['per token = 1.6B + 2 × 5.64B = 12.9B', C.ink]], note: 'One expert is 3 × 4,096 × 14,336 = 176M weights per layer, 5.64B over 32 layers. Compute per token is about 2 FLOPs per weight used, but all 46.7B must be in GPU memory.' }
  }

  /* ---------- scene 5: load balancing ---------- */
  function sceneBalance(p: number) {
    const { H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.06)), x0 = pad + 40, bw = 22, gap = 16, yb = top + avail - 40, hMax = avail - 110, vMax = 0.3
    const Y = (v: number) => yb - (v / vMax) * hMax
    const ga = eio(clamp((p - 0.08) / 0.35))
    title('per expert, this batch', x0, top + 8, fin)
    ctx.strokeStyle = rgba(C.ink, 0.25 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 - 8, yb + 0.5); ctx.lineTo(x0 + E * (2 * bw + gap), yb + 0.5); ctx.stroke()
    for (let e = 0; e < E; e++) {
      const x = x0 + e * (2 * bw + gap)
      const hf = (R.f[e] / vMax) * hMax * ga, hp = (R.Pm[e] / vMax) * hMax * ga
      ctx.fillStyle = rgba(C.ink, 0.8 * fin); ctx.fillRect(x, yb - hf, bw - 2, hf)
      ctx.strokeStyle = rgba(C.ink2, 0.9 * fin); ctx.lineWidth = 1.2; ctx.strokeRect(x + bw + 0.5, yb - hp + 0.5, bw - 3, hp)
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, fin); ctx.fillText(eName(e), x + bw, yb + 6)
    }
    ctx.setLineDash([4, 4]); ctx.strokeStyle = rgba(C.ink2, 0.7 * fin * ga); ctx.beginPath(); ctx.moveTo(x0 - 8, Y(1 / E)); ctx.lineTo(x0 + E * (2 * bw + gap), Y(1 / E)); ctx.stroke(); ctx.setLineDash([])
    caption('even: 1/8', x0 + E * (2 * bw + gap) + 4, Y(1 / E) + 4, fin * ga, C.ink2, 'left')
    // key
    const kx = x0, ky = yb + 34
    ctx.fillStyle = rgba(C.ink, 0.8 * fin); ctx.fillRect(kx, ky - 8, 10, 10); caption('f: share of the 12 routing slots', kx + 16, ky + 1, fin, C.ink2, 'left')
    ctx.strokeStyle = rgba(C.ink2, 0.9 * fin); ctx.strokeRect(kx + 250.5, ky - 7.5, 9, 9); caption('P: mean router probability', kx + 266, ky + 1, fin, C.ink2, 'left')
    // the loss, computed for this batch
    const la = eout(clamp((p - 0.45) / 0.15)), rx = 690
    if (la > 0) {
      title('balancing loss', rx, top + 40, la)
      mathName('L_aux', rx, top + 80, la, 22)
      ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink2, la)
      fillRich('= 8 · Σ f_i · P_i', rx + 52, top + 78)
      ctx.fillStyle = rgba(C.ink, la); ctx.font = F.mono(13, 500)
      ctx.fillText(`= ${fmtF(R.aux)}  this batch`, rx + 52, top + 106)
      ctx.fillStyle = rgba(C.ink2, la); ctx.font = F.mono(12)
      ctx.fillText('= 1     when f and P are even', rx + 52, top + 132)
    }
    const na = eout(clamp((p - 0.65) / 0.15))
    if (na > 0) {
      caption('an expert that wins early gets more', rx, top + 190, na, C.mute, 'left')
      caption('tokens, trains more and wins more;', rx, top + 208, na, C.mute, 'left')
      caption('the loss pushes toward idle experts', rx, top + 226, na, C.mute, 'left')
      caption('Mixtral’s experts do not split by', rx, top + 268, na, C.ink2, 'left')
      caption('topic: routing follows syntax more', rx, top + 286, na, C.ink2, 'left')
      caption('than subject matter', rx, top + 304, na, C.ink2, 'left')
    }
    mk.formula = { segs: [['L_aux = N · Σ_i f_i · P_i', C.ink], ['     ·     ', C.mute], [`= 8 × ${fmtF(R.aux / E)} = ${fmtF(R.aux)}`, C.ink2]], note: 'f_i is the share of routing slots (6 tokens × 2) that went to expert i; P_i is its router probability averaged over the batch. Both even gives 1, the minimum. Added to the loss with a small weight, it spreads tokens over the experts.' }
  }

  return { blocks: sceneBlocks, route: sceneRoute, dispatch: sceneDispatch, params: sceneParams, balance: sceneBalance }
}
