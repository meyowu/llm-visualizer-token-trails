import { F, chipW, ctx, drawChip, mathName, rr } from '../../core/draw'
import { fmt, fmtF, gemm, type M, type Rect } from '../../core/matrix'
import { C, rgba } from '../../core/theme'
import { clamp, eout } from '../../core/util'
import { bp, type Kind } from '../../lib/backprop/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * Backpropagation: the loss's gradient flows from the logits back through every block to every weight, one layer at
 * a time by the chain rule. Every number is GPT-2 small's, computed offline in float64 (scripts/backprop-export.ts)
 * and checked against finite differences.
 */

const PHASES = [
  { id: 'chain', name: 'Blame flows backwards', short: 'Chain rule', dur: 9 },
  { id: 'linear', name: 'A linear layer: dW = Xᵀ · dY', short: 'dW', dur: 12 },
  { id: 'down', name: 'Passing it down: dX = dY · Wᵀ', short: 'dX', dur: 10 },
  { id: 'blame', name: 'The causal mask, backwards', short: 'Positions', dur: 9 },
  { id: 'depth', name: 'Through twelve blocks', short: 'Depth', dur: 10 },
  { id: 'check', name: 'Check it by nudging', short: 'Check', dur: 9 },
]

const T = bp.tokens, N = T.length - 1, NL = bp.layers.length
const S = 1e3 // dY and dW are drawn ×1000
const X: M = bp.linear.x, DY: M = bp.linear.dy.map((r) => r.map((v) => v * S)), DW: M = bp.linear.dW.map((r) => r.map((v) => v * S))
const XT: M = X[0].map((_, i) => X.map((r) => r[i]))
const nice = (s: string) => s.replace(/^Ġ/, ' ')
const KINDS: [Kind, string][] = [['attn.c_attn.weight', 'W_qkv'], ['attn.c_proj.weight', 'W_o'], ['mlp.c_fc.weight', 'W_up'], ['mlp.c_proj.weight', 'W_down']]
const last = bp.layers[NL - 1]

const COMPARE: Record<string, [string, string]> = {
  chain: ['where the gradient starts: p − y', 'training/loss?phase=grad'],
  linear: ['the same layer, forwards', 'anatomy/mlp?phase=down'],
  down: ['GELU, forwards', 'anatomy/mlp?phase=gelu'],
  blame: ['the causal mask, forwards', 'anatomy/attention?phase=mask'],
  depth: ['the residual stream', 'anatomy/layernorm?phase=stream'],
  check: ['the loss being nudged', 'training/loss?phase=loss'],
}

const CAPS: Record<string, [string, string]> = {
  chain: [`GPT-2 small reads “${bp.text}” and scores its ${N} next-token guesses: mean loss ${bp.loss}. Training needs, for each of its 124M weights, how the loss would change if that weight moved. Backpropagation gets all of them in one pass from the loss back to the input.`, `loss ${bp.loss} · 124M gradients in one backward pass`],
  linear: [`Take block 12’s last matrix, y = x · W. Its gradient is dW = Xᵀ · dY: each weight’s blame is its input times the gradient arriving at its output, summed over the ${N} positions. Hover a cell for its sum.`, `dW = Xᵀ · dY · summed over ${N} positions`],
  down: [`The same layer also passes blame to its input, dX = dY · Wᵀ, and so on down: through GELU (times its slope), the up-projection, LayerNorm, and attention. Every weight matrix gets its dW on the way. The residual add copies the gradient straight past each sub-layer.`, 'each layer: keep dW, pass dX down'],
  blame: [`Position i’s loss can only reach tokens at or before i, because attention never looked ahead. Each row is one position’s loss; each column is how strongly it reaches that token’s embedding.`, 'loss at i → embeddings at j ≤ i'],
  depth: [`The gradient reaches every block. Its size on the residual stream barely shrinks from block 12 to block 1, because each block adds to the stream and the gradient flows back through the adds unchanged. Only the embeddings see it grow, as LayerNorm divides their small vectors less.`, 'residual adds keep the gradient alive'],
  check: [`To check, nudge one weight by ±0.0001, run the model forwards twice, and see how the loss moves. The slope matches backpropagation to six digits. That takes two passes per weight; backpropagation gets all 124M at the cost of about two.`, '(L(w + ε) − L(w − ε)) / 2ε = ∂L/∂w'],
}

export function mountBackprop(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'Real gradients of GPT-2 small on the Training sentence, computed offline in float64 and checked against finite differences.',
      eyebrow: 'Training',
      title: 'Backprop',
      subtitle: 'the chain rule, from the loss back to every weight',
      specs: [
        { label: 'model', value: 'GPT-2 small', real: '124M gradients', realLabel: '' },
        { label: 'sentence', value: `${N} predictions`, real: `loss ${bp.loss}`, realLabel: '' },
        { label: 'drawn', value: 'dims 1–8', real: 'of 3,072 × 768', realLabel: '' },
      ],
    },
    size: [1040, 480],
    aria: 'Backpropagation through GPT-2 small: the gradient of the loss flows from the logits back through twelve blocks; for a linear layer, the weight gradient is the input transposed times the output gradient, and the input gradient is the output gradient times the weights transposed; checked against finite differences.',
    phases: PHASES, learn: 'backprop', tokens: words(T.slice(0, N).map(nice)), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    hints: { linear: 'Hover or tap a cell of dW for its sum over the positions.' },
    still: ['linear', 11],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption, arrow, glass } = k
  const pad = 36, top = 56
  const tok = (i: number) => ({ text: T[i], c: i })

  /* ---------- 1: the chain ---------- */
  function sceneChain(p: number) {
    const { W } = stage, y = top + 150, x0 = pad + 80, x1 = W - pad - 170, step = (x1 - x0) / (NL + 1)
    let tx = pad
    T.slice(0, N).forEach((_t, i) => { tx += drawChip(tx, top + 30, tok(i), 1, 22) + 5 })
    caption('the sentence, 5 predictions', pad, top + 60, 1, C.mute, 'left')
    const fw = eout(clamp(p / 0.3)), bw = eout(clamp((p - 0.35) / 0.45))
    for (let l = 0; l < NL; l++) glass(x0 + (l + 1) * step, y - 50, y + 50, bw > 0 ? 0.15 + 0.6 * clamp(bw * (NL + 1) - (NL - l)) : 0.15, { w: 7, d: 8 }, 1)
    caption('embed', x0, y + 74, 1, C.mute)
    caption('12 blocks', (x0 + x1) / 2, y + 74, 1, C.mute)
    // loss box
    rr(x1 + 30, y - 22, 120, 44, 8); ctx.fillStyle = rgba(C.ink, 0.06); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.6); ctx.lineWidth = 1; ctx.stroke()
    ctx.font = F.mono(12, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, fw); ctx.fillText(`loss ${bp.loss}`, x1 + 90, y + 1)
    arrow([[x0 - 10, y - 30], [x1 + 24, y - 30]], fw * 0.7)
    caption('forward: compute the loss', (x0 + x1) / 2, y - 64, fw, C.ink2)
    if (bw > 0) {
      const xe = x1 + 24 - bw * (x1 + 34 - x0)
      ctx.strokeStyle = rgba(C.ink, 0.9); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(x1 + 24, y + 30); ctx.lineTo(xe, y + 30); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(xe + 7, y + 25); ctx.lineTo(xe, y + 30); ctx.lineTo(xe + 7, y + 35); ctx.stroke()
      caption('backward: ∂L/∂(everything), last layer first', (x0 + x1) / 2, y + 104, bw, C.ink)
    }
    mk.formula = { segs: [['∂L/∂W', C.ink], ['  =  ', C.mute], ['∂L/∂y', C.ink2], [' · ', C.mute], ['∂y/∂W', C.ink2]], note: 'The chain rule: a weight affects the loss only through the layer output it feeds, so its gradient is the gradient at that output times how the output depends on it. Going backwards reuses each output gradient for everything below it.' }
  }

  /* ---------- 2: dW = Xᵀ · dY ---------- */
  const c = 24
  const lay = () => {
    const A: Rect = { x: (stage.W - (N + 8) * c - 40) / 2, y: top + 194, c }, B: Rect = { x: A.x + N * c + 40, y: A.y - N * c - 40, c }, Cr: Rect = { x: B.x, y: A.y, c }
    return { A, B, Cr }
  }
  function sceneLinear(p: number) {
    const { A, B, Cr } = lay(), fin = eout(clamp(p / 0.08)), g = gemm(clamp((p - 0.1) / 0.8), 8, 8, N, 'slow')
    mk.drawMat({ r: A, vals: XT, kind: 'w', alpha: fin, name: 'Xᵀ', shape: `8 × ${N}`, real: `3,072 × ${N}`, label: 'bottom' })
    mk.drawMat({ r: B, vals: DY, kind: 'row', alpha: fin, name: 'dY ×10³', shape: `${N} × 8`, real: `${N} × 768` })
    mk.drawMat({ r: Cr, vals: DW, kind: 'w', alpha: fin, name: 'dW ×10³', shape: '8 × 8', real: '3,072 × 768', label: 'bottom', reveal: g.rev })
    // the positions label the shared axis
    for (let n = 0; n < N; n++) {
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.tok[n % 7], fin)
      ctx.fillText(nice(T[n]).trim(), A.x + (n + 0.5) * c, A.y - 8)
      drawChip(B.x - 14 - chipW(T[n]), B.y + (n + 0.5) * c, tok(n), fin, 20)
    }
    mk.hit('dw', Cr, 8, 8)
    const fs = mk.resolve({ dw: { g, K: N } })
    if (fs) mk.gemmOverlay({ A, Av: XT, B, Bv: DY, C: Cr, f: fs, names: ['dW', 'Xᵀ', 'dY'], note: `Weight (${fs.i + 1}, ${fs.j + 1}) of block 12’s down-projection: its input at dimension ${fs.i + 1} times the gradient at output ${fs.j + 1}, summed over the ${N} positions. A batch adds more positions to the same sum.` })
    else mk.formula = { segs: [['dW', C.ink], [' = ', C.mute], ['Xᵀ · dY', C.ink2]], note: 'X is what the layer read going forwards (kept from the forward pass); dY is the gradient that arrived from above.' }
  }

  /* ---------- 3: passing it down ---------- */
  function sceneDown(p: number) {
    const { W } = stage, y0 = top + 70, bw = 150, gap = (W - 2 * pad - 5 * bw) / 4
    const steps: [string, string, number | null][] = [
      ['W_down', 'dW, then dX = dY · Wᵀ', last['mlp.c_proj.weight']],
      ['× GELU′(h)', 'GELU’s slope', null],
      ['W_up', 'dW, then dX', last['mlp.c_fc.weight']],
      ['LayerNorm', 'dγ, dβ, then dX', last['ln_2.weight']],
      ['+ residual', 'gradient copied past', null],
    ]
    title('block 12’s MLP, backwards', pad, y0 - 30, 1)
    steps.forEach(([a, b, n], i) => {
      const on = eout(clamp((p - 0.05 - i * 0.12) / 0.1)), x = pad + i * (bw + gap)
      rr(x, y0, bw, 62, 8); ctx.fillStyle = rgba(C.ink, 0.04 + 0.06 * on); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.25 + 0.6 * on); ctx.lineWidth = 1; ctx.stroke()
      const nw = mathName(a, -1e4, 0, 0, 17)
      mathName(a, x + (bw - nw) / 2, y0 + 26, 0.5 + 0.5 * on, 17)
      ctx.font = F.small; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, 1); ctx.fillText(b, x + bw / 2, y0 + 46)
      if (n !== null) { ctx.font = F.mono(11); ctx.fillStyle = rgba(C.ink2, on); ctx.fillText(`‖dW‖ = ${fmtF(n)}`, x + bw / 2, y0 + 84) }
      if (i < steps.length - 1) arrow([[x + bw + 4, y0 + 31], [x + bw + gap - 4, y0 + 31]], on)
    })
    // the whole block
    const ya = y0 + 150, sa = eout(clamp((p - 0.7) / 0.12))
    title('then attention: every weight of block 12 gets its gradient', pad, ya - 14, sa)
    KINDS.forEach(([kd, name], i) => {
      const x = pad + i * 230, v = last[kd], mx = Math.max(...KINDS.map(([kk]) => last[kk]))
      mathName(name, x, ya + 18, sa, 17)
      ctx.fillStyle = rgba(C.ink, 0.07 * sa); ctx.fillRect(x, ya + 28, 180, 8)
      ctx.fillStyle = rgba(C.ink, 0.7 * sa); ctx.fillRect(x, ya + 28, (v / mx) * 180, 8)
      ctx.font = F.small; ctx.fillStyle = rgba(C.mute, sa); ctx.fillText(`‖dW‖ ${fmtF(v)} · ${(bp.params[kd] / 1e6).toFixed(2)}M weights`, x, ya + 52)
    })
    mk.formula = { segs: [['dX', C.ink], [' = ', C.mute], ['dY · Wᵀ', C.ink2], ['     ', C.mute], ['GELU: ', C.mute], ['dh = dg ⊙ GELU′(h)', C.ink2]], note: 'Each backward step is about as costly as its forward step, and needs what the forward pass saw (X, h), which is why training keeps activations in memory.' }
  }

  /* ---------- 4: blame by position ---------- */
  function sceneBlame(p: number) {
    const cs = 44, x0 = pad + 170, y0 = top + 80, mx = Math.max(...bp.blame.flat())
    title('loss at position i (rows) → embedding at j (columns)', x0, y0 - 44, 1)
    for (let j = 0; j < N; j++) {
      ctx.font = F.mono(11); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.tok[j % 7], 1)
      ctx.fillText(nice(T[j]).trim(), x0 + (j + 0.5) * cs, y0 - 8)
    }
    for (let i = 0; i < N; i++) {
      const a = eout(clamp((p - 0.05 - i * 0.1) / 0.12))
      drawChip(x0 - 74 - chipW(T[i]), y0 + (i + 0.5) * cs, tok(i), 1, 22)
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, 1)
      ctx.fillText(`→ ${nice(T[i + 1]).trim()}`, x0 - 66, y0 + (i + 0.5) * cs)
      for (let j = 0; j < N; j++) {
        const x = x0 + j * cs, y = y0 + i * cs
        if (j > i) { mk.hatch(x + 1, y + 1, cs - 2, 0.7); continue }
        const v = bp.blame[i][j] / mx, s = 5 + (cs - 22) * Math.sqrt(v) * a
        ctx.fillStyle = rgba(C.tok[j % 7], 0.85 * a); ctx.fillRect(x + (cs - s) / 2, y + (cs - 14 - s) / 2 + 1, s, s)
        ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(fmt(bp.blame[i][j]), x + cs / 2, y + cs - 3)
      }
    }
    caption('hatched: masked, no gradient', x0 + N * cs + 20, y0 + 20, eout(clamp((p - 0.6) / 0.1)), C.mute, 'left')
    mk.formula = { segs: [['‖∂L_i / ∂x_j‖', C.ink], ['  =  0  for  j > i', C.ink2]], note: `Row “${nice(T[1]).trim()} → ${nice(T[2]).trim()}”: its loss reaches both earlier tokens. Every position is trained at once, from the same pass.` }
  }

  /* ---------- 5: through the depth ---------- */
  function sceneDepth(p: number) {
    const { W, H } = stage, x0 = pad + 70, x1 = W - pad - 40, yB = H - 90, hMax = 200, bw = (x1 - x0) / NL
    const mx = Math.max(...bp.layers.flatMap((l) => KINDS.map(([kd]) => l[kd])))
    title('‖dW‖ per weight matrix, block by block', x0, top + 24, 1)
    KINDS.forEach(([, name], j) => {
      const kx = x0 + 330 + j * 110
      ctx.fillStyle = rgba(C.ink, [0.3, 0.5, 0.7, 0.9][j]); ctx.fillRect(kx, top + 14, 14, 10)
      mathName(name, kx + 20, top + 25, 1, 15, C.ink2)
    })
    bp.layers.forEach((l, i) => {
      const a = eout(clamp((p - 0.05 - (NL - 1 - i) * 0.035) / 0.1))
      KINDS.forEach(([kd], j) => {
        const h = (l[kd] / mx) * hMax * a, x = x0 + i * bw + 8 + j * ((bw - 16) / 4)
        ctx.fillStyle = rgba(C.ink, [0.3, 0.5, 0.7, 0.9][j] * a); ctx.fillRect(x, yB - h, (bw - 16) / 4 - 2, h)
      })
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, 1); ctx.fillText(String(i + 1), x0 + (i + 0.5) * bw, yB + 6)
    })
    caption('block', x0 - 10, yB + 18, 1, C.mute, 'right')
    // the residual stream's gradient, as a line on its own scale
    const sa = eout(clamp((p - 0.5) / 0.15)), sm = Math.max(...bp.stream.slice(1))
    if (sa > 0) {
      ctx.strokeStyle = rgba(C.tok[0], sa); ctx.lineWidth = 2; ctx.beginPath()
      bp.stream.slice(1).forEach((v, i) => { const x = x0 + (i + 0.5) * bw, y = yB - hMax - 30 + (1 - v / sm) * 60; if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y) })
      ctx.stroke()
      caption(`‖∂L/∂x‖ on the stream after each block: ${fmtF(bp.stream[1])} … ${fmtF(bp.stream[NL])}`, x0, yB - hMax - 48, sa, C.ink2, 'left')
    }
    mk.formula = { segs: [['x_{l+1} = x_l + f(x_l)', C.ink2], ['  ⇒  ', C.mute], ['∂L/∂x_l = ∂L/∂x_{l+1} · (1 + ∂f/∂x_l)', C.ink]], note: `The 1 is the residual add. At the embeddings the gradient is ${fmtF(bp.stream[0])}, larger again: GPT-2’s embedding vectors are short, so block 1’s LayerNorm scales their gradient up.` }
  }

  /* ---------- 6: finite differences ---------- */
  function sceneCheck(p: number) {
    const x0 = pad + 20, y0 = top + 60
    title('three weights · backprop vs nudging', x0, y0 - 24, 1)
    const label = (n: string) => n.replace(/^h\.(\d+)\./, (_, l) => `block ${Number(l) + 1} `).replace('.weight', '').replace('mlp.c_proj', 'W_down').replace('attn.c_attn', 'W_qkv').replace('wpe', 'W_P')
    bp.checks.forEach((c2, i) => {
      const a = eout(clamp((p - 0.05 - i * 0.2) / 0.15)), y = y0 + i * 96
      if (a <= 0) return
      ctx.font = F.mono(12, 600); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a)
      ctx.fillText(`${label(c2.name)} [${c2.row}, ${c2.col}]`, x0, y)
      ctx.font = F.mono(12); ctx.fillStyle = rgba(C.ink2, a)
      const sup = (d: string) => [...d].map((ch) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(ch)]).join('')
      const num = (v: number) => { const [m, e] = Math.abs(v).toExponential(5).split('e'); return `${v < 0 ? '−' : ' '}${m} × 10${e.startsWith('-') ? '⁻' : ''}${sup(e.replace(/[-+]/, ''))}` }
      ctx.fillText('backprop'.padEnd(24) + num(c2.backprop), x0 + 20, y + 26)
      ctx.fillText('(L(w+ε) − L(w−ε)) / 2ε'.padEnd(24) + num(c2.finite), x0 + 20, y + 48)
    })
    mk.formula = { segs: [['ε = 10⁻⁴', C.ink2], ['   ·   ', C.mute], ['2 forward passes per weight', C.ink2], ['   vs   ', C.mute], ['1 backward pass for all 124M', C.ink]], note: 'Numerical checks like this are how backpropagation code is tested. Frameworks (PyTorch’s autograd) derive the backward pass from the forward code automatically.' }
  }

  return { chain: sceneChain, linear: sceneLinear, down: sceneDown, blame: sceneBlame, depth: sceneDepth, check: sceneCheck }
}
