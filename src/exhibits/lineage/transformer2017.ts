import { F, chipW, drawChip, fillRich, mathRun, rr, serifAt, type TokLike } from '../../core/draw'
import { fmt, fmtF, gemm, type Rect } from '../../core/matrix'
import { C, blend, mixc, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, gauss, lerp, rng } from '../../core/util'
import { streamNorms, wpeSlice } from '../../lib/gpt2/data'
import type { Nav } from '../registry'
import { mountExhibit, words, type Env } from '../kit'

/*
 * The original Transformer (Vaswani et al. 2017) as a diff against GPT-2. Two stacks instead of
 * one: an encoder reads the source sentence and a decoder writes the translation, reading the
 * encoder's output through cross-attention. LayerNorm comes after each residual add (post-LN), and
 * positions are fixed sinusoids. The attention numbers are toy vectors on an EN → DE example; the
 * GPT-2 numbers (stream lengths, W_P) are real.
 */

const PHASES = [
  { id: 'rnn', name: 'Before: one step at a time', short: 'RNN → attention', dur: 9 },
  { id: 'blocks', name: 'GPT-2 vs the 2017 Transformer', short: 'Blocks', dur: 8 },
  { id: 'translate', name: 'Translating a sentence', short: 'Translate', dur: 12 },
  { id: 'masks', name: 'Three kinds of attention', short: 'Masks', dur: 7 },
  { id: 'cross', name: 'Cross-attention', short: 'Cross', dur: 12 },
  { id: 'postln', name: 'Post-LN vs pre-LN', short: 'Post-LN', dur: 9 },
  { id: 'pos', name: 'Sinusoidal positions', short: 'Sinusoids', dur: 9 },
]

const SRC = words(['I', 'have', 'seen', 'the', 'cat', '.'])
const TIN = words(['<s>', 'Ich', 'habe', 'die', 'Katze', 'gesehen', '.'])
const TOUT = words(['Ich', 'habe', 'die', 'Katze', 'gesehen', '.', '</s>'])
const NS = SRC.length, NT = TIN.length, DK = 4
/** The source word each target position translates next (German puts the participle last). */
const ALIGN = [0, 1, 3, 4, 2, 5, 5]
/** Seed for readable toy cross-attention: every row peaks on its ALIGN word, with some spread. */
const XSEED = 3175
const D_MODEL = 512, BASE = 10000

const dot = (a: number[], b: number[]) => a.reduce((s, v, k) => s + v * b[k], 0)
const softmax = (row: number[]) => {
  const m = Math.max(...row.filter(isFinite)), e = row.map((v) => (isFinite(v) ? Math.exp(v - m) : 0)), z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
}

/** Toy attention: keys say what each source word offers, queries what each target position looks for. */
const TOY = (() => {
  const r = rng(XSEED)
  const unit = () => { const v = Array.from({ length: DK }, () => gauss(r)), n = Math.hypot(...v); return v.map((x) => x / n) }
  const U = SRC.map(unit)
  const K = U.map((u) => u.map((x) => x * 2.5 + 0.2 * gauss(r)))
  const Q = ALIGN.map((i, j) => U[i].map((x) => x * (j === NT - 1 ? 1.7 : 2.5) + 0.3 * gauss(r)))
  const S = Q.map((q) => K.map((k) => dot(q, k)))
  const A = S.map((row) => softmax(row.map((v) => v / Math.sqrt(DK))))
  // self-attention in the encoder (no mask) and the decoder (causal), and the encoder output
  const r2 = rng(71), vecs = (n: number) => Array.from({ length: n }, () => Array.from({ length: DK }, () => gauss(r2) * 1.3))
  const qe = vecs(NS), ke = vecs(NS), qd = vecs(NT), kd = vecs(NT)
  const Aenc = qe.map((q) => softmax(ke.map((k) => dot(q, k) / 2)))
  const Adec = qd.map((q, i) => softmax(kd.map((k, j) => (j <= i ? dot(q, k) / 2 : -Infinity))))
  const mem = vecs(NS)
  return { K, KT: K[0].map((_, k) => K.map((row) => row[k])), Q, S, A, Aenc, Adec, mem }
})()

/** Column j of the sinusoid map: every 8th sin/cos pair of the 512 dimensions. */
const peDim = (j: number) => 16 * (j >> 1) + (j & 1)
const pe = (pos: number, dim: number) => {
  const w = Math.pow(BASE, -(dim - (dim & 1)) / D_MODEL)
  return dim & 1 ? Math.cos(pos * w) : Math.sin(pos * w)
}
const MAP = 64

const COMPARE: Record<string, [string, string]> = {
  rnn: ['GPT-2’s attention', 'anatomy/attention'], blocks: ['GPT-2’s block', 'anatomy/layernorm?phase=stream'],
  translate: ['GPT-2’s sampling', 'anatomy/unembed?phase=sample'], masks: ['GPT-2’s causal mask', 'anatomy/attention?phase=mask'],
  cross: ['GPT-2’s scores', 'anatomy/attention?phase=scores'], postln: ['GPT-2’s pre-LN', 'anatomy/layernorm?phase=stream'],
  pos: ['GPT-2’s positions', 'anatomy/embedding?phase=pos'],
}

const CAPS: Record<string, [string, string]> = {
  rnn: ['Before 2017, translation models were recurrent neural networks (RNNs): they read a sentence one token at a time, and each step needs the result of the one before, so a GPU cannot run the steps side by side. Self-attention links every pair of positions in one matrix product, so a whole sentence is processed at once.', 'RNN: n steps · attention: 1'],
  blocks: ['GPT-2 is one stack. The 2017 Transformer has two: an encoder that reads the source sentence and a decoder that writes the translation, reading the encoder’s output through cross-attention. It also normalises after each residual add, uses fixed sinusoids for positions and ReLU in the MLP. Click a label to jump to that change.', '6 encoder + 6 decoder layers · d_model 512'],
  translate: ['The encoder reads the English sentence once. The decoder then writes German one token at a time, like GPT-2, and each pass reads the encoder’s output. In training, the correct translation is fed in (teacher forcing) and the causal mask hides the future, so all positions run in one pass.', 'encoder once · decoder once per token'],
  masks: ['There are three attentions, and they all compute softmax(Q·Kᵀ / √d_k) · V. The encoder sees the whole source, the decoder sees only earlier target tokens, and cross-attention lets each target token see the whole source. GPT-2 has only the middle kind.', 'source × source · target × target · target × source'],
  cross: ['In cross-attention the queries come from the decoder and the keys and values from the encoder output, so the score matrix is target × source and is not square. Each row looks for the source word it needs next: ‘gesehen’ goes back to ‘seen’, although German moves it to the end. Hover the cells.', 'Q 7 × d_k · Kᵀ d_k × 6 → 7 × 6'],
  postln: ['The 2017 model normalises after each residual add (post-LN), so LayerNorm sits on the stream’s main path. Deep post-LN models train poorly without a long learning-rate warmup (4,000 steps in the paper). GPT-2 normalises the copy each sub-layer reads (pre-LN) instead: the main path only adds, the stream grows, and one final LayerNorm, ln_f, tidies it up.', 'LN(x + f(x)) → x + f(LN(x))'],
  pos: ['Attention ignores order, so both models add a position vector to each token. The 2017 model computes it from sine and cosine waves: no parameters, and a vector for any position. GPT-2 learns a table of 1,024 rows instead. Each sin/cos pair turns like a clock hand, so a shift by k positions is a fixed rotation, the idea RoPE later moved into attention.', 'PE(pos, 2i) = sin(pos / 10000^(2i/d_model))'],
}

export function mountTransformer2017(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: '',
      eyebrow: 'Lineage · Origin',
      title: 'Transformer (2017)',
      subtitle: 'Vaswani et al., “Attention Is All You Need” · what GPT-2 changed',
      specs: [
        { label: 'compared', value: 'base 2017', real: 'GPT-2 small', realLabel: 'vs' },
        { label: 'layers', value: '6 enc + 6 dec', real: '12' },
        { label: 'd_model', value: '512', real: '768' },
        { label: 'heads', value: '8', real: '12' },
        { label: 'd_ff', value: '2,048', real: '3,072' },
        { label: 'vocab', value: '~37,000 shared', real: '50,257' },
        { label: 'params', value: '65M', real: '124M' },
      ],
    },
    size: [1040, 480],
    aria: 'The original 2017 Transformer compared with GPT-2: an encoder reads the source sentence and a decoder writes the translation token by token through cross-attention; LayerNorm comes after each residual add; positions are fixed sine and cosine waves.',
    phases: PHASES, learn: 'transformer2017', tokens: SRC, compare: COMPARE, caps: CAPS,
    still: ['cross', 11],
    hints: {
      blocks: 'Click a dark label on the drawing to jump to that part.',
      masks: 'Hover a cell to read its weight; click or tap to pin it.',
      cross: 'Hover a cell to see how it is computed; click or tap to pin it.',
      pos: 'Hover a cell of either map to read it; click or tap to pin it.',
    },
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { pill, addNode, caption, title, arrow, lane, glass } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46
  /** What a target row reads from the source: the source hues blended by its cross-attention weights. */
  const readCol = (j: number) => blend(SRC.map((_, i) => hue(i)), TOY.A[j])
  const memCol = (i: number) => blend(SRC.map((_, k) => hue(k)), TOY.Aenc[i].map((w, k) => 0.5 * w + (k === i ? 0.5 : 0)))
  const rowName = (name: string, sub: string, y: number, a: number) => k.rowName(name, sub, pad, y, a)
  const chipAt = (t: TokLike, cx: number, y: number, a: number, h = 18, hl = false) => drawChip(cx - chipW(t.text) / 2, y, t, a, h, hl)

  /* ---------- scene 1: RNN, one step at a time, vs attention, all at once ---------- */
  function sceneRnn(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.06)), bs = 24, x0 = pad + 170, gx = 78, xs = SRC.map((_, i) => x0 + i * gx)
    const yR = top + avail * 0.2, yA = top + avail * 0.6, cyR = yR + 50, cyA = yA + 78
    const T = (k: number) => 0.1 + k * 0.09 // RNN step k is done
    rowName('RNN', 'before 2017', yR + 8, fin)
    rowName('Attention', '2017', yA + 8, fin)
    const done = SRC.filter((_, k) => p >= T(k)).length
    // RNN: h[t] = f(h[t−1], x[t]); a step cannot start before the previous one is done
    SRC.forEach((t, k) => {
      const x = xs[k], on = p >= T(k), run = clamp((p - (T(k) - 0.09)) / 0.09)
      chipAt(t, x, cyR, fin)
      arrow([[x, cyR - 11], [x, yR + bs / 2 + 3]], fin * (run > 0 ? 1 : 0.4))
      if (k > 0) {
        arrow([[xs[k - 1] + bs / 2 + 2, yR], [x - bs / 2 - 3, yR]], fin * (run > 0 ? 1 : 0.4))
        if (run > 0 && run < 1) { ctx.fillStyle = rgba(C.ink, fin); ctx.beginPath(); ctx.arc(lerp(xs[k - 1] + bs / 2 + 2, x - bs / 2 - 3, run), yR, 3, 0, 7); ctx.fill() }
      }
      const col = blend(SRC.slice(0, k + 1).map((_, i) => hue(i)), SRC.slice(0, k + 1).map((_, i) => Math.pow(0.55, k - i) * (1 - 0.55) / (1 - Math.pow(0.55, k + 1))))
      rr(x - bs / 2, yR - bs / 2, bs, bs, 4)
      ctx.fillStyle = rgba(on ? col : C.ink, (on ? 0.8 : 0.04) * fin); ctx.fill()
      ctx.strokeStyle = rgba(C.ink, (on ? 0.55 : 0.25) * fin); ctx.lineWidth = 1; ctx.stroke()
      mathRun([['h', false], [String(k + 1), true]], x - 6, yR - bs / 2 - 7, fin * (on ? 1 : 0.5), 16)
    })
    const cx = xs[NS - 1] + 44
    ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, fin)
    ctx.fillText(`step ${Math.max(1, done)} / ${NS}`, cx, yR + 4)
    caption('one after another', cx, yR + 22, fin, C.mute, 'left')
    // attention: every output reads every input directly, all in one matrix product
    const la = eout(clamp((p - 0.1) / 0.08))
    SRC.forEach((t, k) => {
      chipAt(t, xs[k], cyA, fin)
      for (let j = 0; j < NS; j++) {
        ctx.strokeStyle = rgba(hue(k), 0.35 * la * fin); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(xs[k], cyA - 11); ctx.lineTo(xs[j], yA + bs / 2 + 3); ctx.stroke()
      }
    })
    SRC.forEach((_, j) => {
      rr(xs[j] - bs / 2, yA - bs / 2, bs, bs, 4)
      ctx.fillStyle = rgba(la > 0.5 ? memCol(j) : C.ink, (la > 0.5 ? 0.8 : 0.04) * fin); ctx.fill()
      ctx.strokeStyle = rgba(C.ink, 0.5 * fin); ctx.lineWidth = 1; ctx.stroke()
    })
    ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink, fin * la)
    ctx.fillText('1 step', cx, yA + 4)
    caption('all positions at once', cx, yA + 22, fin * la, C.mute, 'left')
    // the longest route between two tokens: through every hidden state, or one direct link
    const pa = eout(clamp((p - 0.64) / 0.1))
    if (pa > 0) {
      ctx.strokeStyle = rgba(C.ink, 0.95 * pa); ctx.lineWidth = 2.2
      ctx.beginPath(); ctx.moveTo(xs[0], cyR - 11); ctx.lineTo(xs[0], yR); ctx.lineTo(xs[4], yR); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(xs[0], cyA - 11); ctx.lineTo(xs[4], yA + bs / 2 + 3); ctx.stroke()
      caption('‘I’ reaches ‘cat’ after 4 steps', xs[2], yR - 38, pa, C.ink2)
      caption('‘I’ reaches ‘cat’ in one step', xs[2], yA - 30, pa, C.ink2)
    }
    // Vaswani et al., Table 1
    const ta = eout(clamp((p - 0.74) / 0.12))
    if (ta > 0) {
      const tx = W - pad - 214, ty = top + avail * 0.36, cols = [tx + 100, tx + 156, tx + 200]
      title('Vaswani et al. · table 1', tx, ty - 30, ta)
      ;['cost', 'steps', 'path'].forEach((h, k) => title(h, cols[k], ty, ta, 'center'))
      const rows: [string, string[]][] = [['attention', ['O(n²·d)', 'O(1)', 'O(1)']], ['RNN', ['O(n·d²)', 'O(n)', 'O(n)']]]
      rows.forEach(([name, vals], r) => {
        const y = ty + 28 + r * 26
        ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(r ? C.ink2 : C.ink, ta); ctx.fillText(name, tx, y)
        vals.forEach((v, k) => { ctx.textAlign = 'center'; ctx.fillText(v, cols[k], y) })
      })
      ctx.strokeStyle = rgba(C.ink, 0.15 * ta); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(tx, ty + 10); ctx.lineTo(W - pad, ty + 10); ctx.stroke()
      caption('n tokens of d numbers each', tx, ty + 84, ta, C.mute, 'left')
    }
    mk.formula = { segs: [['RNN  h[t] = f(h[t−1], x[t])', C.ink2], ['     ·     ', C.mute], ['attention  softmax(Q·Kᵀ / √d_k) · V, every row at once', C.ink]], note: 'Table 1 of the paper: work per layer, sequential steps, and the longest path a signal travels between two positions. Attention costs more per layer once n exceeds d, but it has no chain of steps to wait for.' }
  }

  /* ---------- scene 2: one stack vs encoder + decoder ---------- */
  type El = [kind: 'pos' | 'ln' | 'attn' | 'cross' | 'ffn' | 'add', f: number, label: string]
  const ROWS: [string, string, El[], number][] = [
    ['GPT-2', 'decoder-only · × 12', [['pos', 0.03, '+ W_P'], ['ln', 0.12, 'LN'], ['attn', 0.21, 'self · causal'], ['add', 0.3, ''], ['ln', 0.66, 'LN'], ['ffn', 0.76, 'MLP · GELU'], ['add', 0.87, '']], 5],
    ['Encoder', '2017 · × 6', [['pos', 0.03, '+ PE'], ['attn', 0.21, 'self · all'], ['add', 0.3, ''], ['ln', 0.37, 'LN'], ['ffn', 0.76, 'FFN · ReLU'], ['add', 0.87, ''], ['ln', 0.94, 'LN']], NS],
    ['Decoder', '2017 · × 6', [['pos', 0.03, '+ PE'], ['attn', 0.21, 'self · causal'], ['add', 0.3, ''], ['ln', 0.37, 'LN'], ['cross', 0.47, 'cross'], ['add', 0.56, ''], ['ln', 0.63, 'LN'], ['ffn', 0.76, 'FFN · ReLU'], ['add', 0.87, ''], ['ln', 0.94, 'LN']], NT],
  ]
  function sceneBlocks(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const sx0 = pad + 130, sx1 = W - pad - 16, fx = (f: number) => lerp(sx0, sx1, f)
    const ys = [top + avail * 0.14, top + avail * 0.52, top + avail * 0.87]
    const as = [eout(clamp(p / 0.25)), eout(clamp((p - 0.2) / 0.25)), eout(clamp((p - 0.3) / 0.25))]
    ROWS.forEach(([name, sub, els, lanes], r) => {
      const y = ys[r], a = as[r]
      if (a <= 0) return
      rowName(name, sub, y, a)
      for (let i = 0; i < lanes; i++) lane(sx0, lerp(sx0, sx1, a), y + (i - (lanes - 1) / 2) * 4, hue(i), a)
      for (const [kind, f, lab] of els) {
        const x = fx(f)
        if (kind === 'add') { addNode(x, y, a); continue }
        if (kind === 'pos') {
          rr(x - 24, y - 15, 48, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.7 * a); ctx.lineWidth = 1; ctx.stroke()
          ctx.font = F.small; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a); fillRich(lab, x, y + 0.5)
          caption(r ? 'fixed' : 'learned', x, y + 44, a, r ? C.ink2 : C.mute)
          continue
        }
        const changed = r > 0 && (kind === 'ln' || kind === 'cross')
        if (kind === 'ln') glass(x, y - 20, y + 20, changed ? 0.55 : 0.2, { w: 5, d: 6 }, a)
        else glass(x, y - 26, y + 26, changed ? 0.55 : 0.2, { w: 8, d: 9 }, a)
        caption(lab, x, y + 44, a, changed ? C.ink2 : C.mute)
      }
    })
    // the encoder's output feeds every decoder layer's cross-attention
    const ea = eout(clamp((p - 0.45) / 0.2)), yE = ys[1], yD = ys[2], xc = fx(0.47)
    if (ea > 0) {
      const yM = yE + 62
      arrow([[sx1, yE], [sx1 + 8, yE], [sx1 + 8, yM], [xc, yM], [xc, yD - 34]], ea)
      caption('encoder output → keys, values', (xc + sx1) / 2 + 40, yM + 15, ea, C.ink2)
      title('transformer 2017 · base', pad, yE - 70, ea)
      ctx.setLineDash([2, 4]); ctx.strokeStyle = rgba(C.ink, 0.2 * ea); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(pad, yE - 62); ctx.lineTo(W - pad, yE - 62); ctx.stroke(); ctx.setLineDash([])
    }
    // what changed: pills that open each step
    const la = eout(clamp((p - 0.6) / 0.2))
    if (la > 0) {
      pill('sinusoids', fx(0.03), yE - 47, 'pos', la)
      pill('no mask', fx(0.21), yE - 47, 'masks', la)
      pill('post-LN', fx(0.37), yE - 47, 'postln', la)
      pill('cross-attention', xc + 70, yD - 47, 'cross', la)
    }
    mk.formula = { segs: [['GPT-2  h′ = h + f(LN(h))', C.ink2], ['     ·     ', C.mute], ['2017  h′ = LN(h + f(h))', C.ink]], note: 'f is a sub-layer: attention or the MLP. The encoder repeats self-attention and FFN 6 times; the decoder adds cross-attention between them. Click a label to open that change.' }
  }

  /* ---------- scene 3: translating, encoder once, decoder once per token ---------- */
  function sceneTranslate(p: number) {
    const { H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.06)), ls = 18
    const yE0 = top + 40, yD0 = top + avail * 0.5 + 6, xl = pad + 76
    const encX = [216, 232, 248, 264, 280, 296], mx = 336, decX = [474, 490, 506, 522, 538, 554], xo = 640
    const ly = (y0: number, i: number) => y0 + i * ls
    // encoder: runs once over the whole source
    const ea = eout(clamp((p - 0.03) / 0.1))
    title('encoder · runs once', pad, top + 10, fin)
    SRC.forEach((t, i) => {
      drawChip(pad, ly(yE0, i), t, fin, 15)
      lane(xl, lerp(xl, encX[0], ea), ly(yE0, i), hue(i), fin)
      lane(encX[5] + 6, lerp(encX[5] + 6, mx - 4, clamp(ea * 2 - 1)), ly(yE0, i), memCol(i), fin)
    })
    encX.forEach((x) => glass(x, yE0 - 16, ly(yE0, NS - 1) + 16, 0.15 + 0.35 * ea, { w: 4, d: 6 }, fin))
    caption('× 6', (encX[0] + encX[5]) / 2, yE0 - 26, fin)
    const mR: Rect = { x: mx, y: yE0 - ls / 2, c: ls }
    mk.drawMat({ r: mR, vals: TOY.mem, kind: 'row', rowCols: SRC.map((_, i) => memCol(i)), alpha: fin * clamp(ea * 2 - 1), name: 'memory', shape: '6 × 4', real: '6 × 512', noText: true, labelW: 200 })
    // decoder: one pass per new token, reading the memory through cross-attention
    const sp = clamp((p - 0.14) / 0.83) * NT, s = Math.min(NT - 1, Math.floor(sp)), f = sp >= NT ? 1 : sp - s
    title('decoder · one pass per token', pad, yD0 - 30, fin)
    decX.forEach((x) => glass(x, yD0 - 16, ly(yD0, NT - 1) + 16, 0.25, { w: 4, d: 6 }, fin))
    caption('× 6', (decX[0] + decX[5]) / 2, yD0 - 26, fin)
    for (let j = 0; j <= s; j++) {
      const cur = j === s, y = ly(yD0, j), a = fin * (cur ? 1 : 0.55)
      const grow = cur ? clamp(f / 0.3) : 1
      drawChip(pad, y, TIN[j], fin, 15, cur)
      lane(xl, lerp(xl, decX[0], grow), y, hue(j), a)
      if (grow >= 1) lane(decX[5] + 6, lerp(decX[5] + 6, xo - 8, cur ? clamp((f - 0.3) / 0.2) : 1), y, mixc(hue(j), readCol(j), 0.45), a)
      const oa = cur ? clamp((f - 0.5) / 0.1) : 1
      if (oa > 0) drawChip(xo, y, TOUT[j], fin * oa * (cur ? 1 : 0.55), 15, cur)
    }
    // the current position's cross-attention into the memory rows
    const xa = fin * clamp((f - 0.08) / 0.2) * (sp >= NT ? 0 : 1)
    if (xa > 0) {
      for (let i = 0; i < NS; i++) {
        const w = TOY.A[s][i], y0 = ly(yE0, i), y1 = ly(yD0, s)
        ctx.strokeStyle = rgba(hue(i), (0.08 + 0.9 * w) * xa); ctx.lineWidth = 0.6 + 4 * w
        ctx.beginPath(); ctx.moveTo(mx + DK * ls + 4, y0); ctx.bezierCurveTo(mx + DK * ls + 60, y0, decX[2], y1 - 60, (decX[2] + decX[3]) / 2, y1); ctx.stroke()
      }
    }
    // the new token is appended to the input for the next pass
    const ra = fin * (sp >= NT || s >= NT - 1 ? 0 : clamp((f - 0.62) / 0.2))
    if (ra > 0) {
      const yb = ly(yD0, NT - 1) + 22, xm = xo + chipW(TOUT[s].text) / 2
      arrow([[xm, ly(yD0, s) + 9], [xm, yb], [pad + 16, yb], [pad + 16, ly(yD0, s + 1) + 9]], ra, true)
      caption('appended, then the next pass', (pad + xm) / 2, yb + 15, ra, C.mute)
    }
    // the sentences, and how many passes each stack ran
    const rx = 740
    title('source', rx, top + 10, fin)
    ctx.font = serifAt(19); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, fin)
    ctx.fillText('I have seen the cat.', rx, top + 38)
    title('translation', rx, top + 76, fin)
    let x = rx
    TOUT.forEach((t, j) => {
      if (j > s || (j === s && f < 0.5 && sp < NT) || t.text === '</s>') return
      ctx.font = serifAt(19); ctx.fillStyle = rgba(j === s && sp < NT ? hue(j) : C.ink, fin)
      const txt = (t.text === '.' || j === 0 ? '' : ' ') + t.text
      ctx.fillText(txt, x, top + 104); x += ctx.measureText(txt).width
    })
    ctx.font = F.mono(12); ctx.fillStyle = rgba(C.ink2, fin)
    ctx.fillText(`encoder passes  1`, rx, top + 150)
    ctx.fillText(`decoder passes  ${Math.min(NT, s + 1)} / ${NT}`, rx, top + 172)
    caption('training runs all 7 decoder', rx, top + 206, fin, C.mute, 'left')
    caption('positions in one pass', rx, top + 222, fin, C.mute, 'left')
    mk.formula = { segs: [['memory = Encoder(source)', C.ink2], ['     ·     ', C.mute], ['next token = Decoder(target so far, memory)', C.ink]], note: `The memory is computed once and read by every decoder pass. The lines show where position ${s} (‘${TIN[s].text}’) looks in the source while it writes ‘${TOUT[s].text}’. Greedy decoding shown; the paper used beam search with 4 beams.` }
  }

  /* ---------- scene 4: the three attentions ---------- */
  function sceneMasks(p: number) {
    const { W } = stage
    const c = 28, y0 = top + 110, specs: { key: string; rows: TokLike[]; cols: TokLike[]; vals: number[][]; masked: boolean; name: string; sub: string; t0: number }[] = [
      { key: 'enc', rows: SRC, cols: SRC, vals: TOY.Aenc, masked: false, name: 'encoder self-attention', sub: 'source × source · no mask', t0: 0 },
      { key: 'dec', rows: TIN, cols: TIN, vals: TOY.Adec, masked: true, name: 'decoder self-attention', sub: 'target × target · causal', t0: 0.22 },
      { key: 'x', rows: TIN, cols: SRC, vals: TOY.A, masked: false, name: 'cross-attention', sub: 'target × source · no mask', t0: 0.44 },
    ]
    ctx.font = F.mono(10.5)
    const labW = (ts: TokLike[]) => Math.max(...ts.map((t) => ctx.measureText(t.text).width)) + 10
    const widths = specs.map((m) => labW(m.rows) + m.cols.length * c)
    const gap = (W - 2 * pad - widths.reduce((a, b) => a + b, 0)) / 2
    let x = pad
    const f = mk.focus
    specs.forEach((m, k) => {
      const a = eout(clamp((p - m.t0) / 0.18)), lw = labW(m.rows), mx = x + lw
      x += widths[k] + gap
      if (a <= 0) return
      title(m.name, mx, y0 - 78, a)
      caption(m.sub, mx, y0 - 62, a, C.ink2, 'left')
      const r: Rect = { x: mx, y: y0, c }
      mk.drawMat({ r, vals: m.vals, kind: 'attn', alpha: a, name: '', shape: '', label: 'none', noText: true, paint: (px, py, cc, i, j, aa) => {
        if (m.masked && j > i) { mk.hatch(px, py, cc, aa); return 0 }
        return mk.paintAttn(px, py, cc, m.vals[i][j], hue(j), aa)
      } })
      mk.hit(m.key, r, m.rows.length, m.cols.length)
      ctx.font = F.mono(10.5); ctx.textBaseline = 'middle'
      m.rows.forEach((t, i) => { ctx.textAlign = 'right'; ctx.fillStyle = rgba(hue(i), a * (f && f.key === m.key && f.i !== i ? 0.45 : 1)); ctx.fillText(t.text, mx - 6, y0 + (i + 0.5) * c) })
      m.cols.forEach((t, j) => {
        ctx.save(); ctx.translate(mx + (j + 0.5) * c, y0 - 6); ctx.rotate(-Math.PI / 4)
        ctx.textAlign = 'left'; ctx.fillStyle = rgba(hue(j), a * (f && f.key === m.key && f.j !== j ? 0.45 : 1)); ctx.fillText(t.text, 0, 0); ctx.restore()
      })
      if (f && f.key === m.key) { ctx.strokeStyle = rgba(C.ink, a); ctx.lineWidth = 2; ctx.strokeRect(mx + f.j * c, y0 + f.i * c, c, c) }
      if (k === 1) caption('GPT-2 has only this kind', mx + (m.cols.length * c) / 2, y0 + m.rows.length * c + 30, a, C.ink2)
    })
    const h = f && specs.find((m) => m.key === f.key)
    if (h && f) {
      const blocked = h.masked && f.j > f.i, v = h.vals[f.i][f.j]
      mk.formula = blocked
        ? { segs: [[`${h.name}  A[${f.i},${f.j}]`, C.ink], ['  =  ', C.mute], ['0  (masked)', C.ink]], note: `‘${h.rows[f.i].text}’ may not look at the later ‘${h.cols[f.j].text}’: its score is set to −∞ before the softmax.` }
        : { segs: [[`${h.name}  A[${f.i},${f.j}]`, C.ink], ['  =  ', C.mute], [fmt(v), C.ink]], note: `‘${h.rows[f.i].text}’ puts ${Math.round(v * 100)}% of its attention on ‘${h.cols[f.j].text}’. Toy weights.` }
    } else mk.formula = { segs: [['all three:  softmax(Q·Kᵀ / √d_k) · V', C.ink]], note: 'The same computation three times. What differs is where Q, K and V come from and which cells are masked. Toy weights; the colour of a cell is the hue of the token being read.' }
  }

  /* ---------- scene 5: cross-attention, Q from the decoder, K and V from the encoder ---------- */
  function sceneCross(p: number) {
    const { W } = stage
    const fin = eout(clamp(p / 0.06)), c = 28
    const rQ: Rect = { x: pad + 84, y: top + 176, c }, rS: Rect = { x: rQ.x + DK * c + 34, y: rQ.y, c }, rK: Rect = { x: rS.x, y: rQ.y - DK * c - 26, c }
    const g = gemm(clamp((p - 0.08) / 0.37), NT, NS, DK, 'fast'), sm = eio(clamp((p - 0.48) / 0.1))
    TIN.forEach((t, i) => drawChip(rQ.x - 10 - chipW(t.text), rQ.y + (i + 0.5) * c, t, fin, Math.min(20, c * 0.72)))
    mk.drawMat({ r: rQ, vals: TOY.Q, kind: 'row', rowCols: TIN.map((_, i) => hue(i)), alpha: fin, name: 'Q', shape: '7 × 4', real: '7 × 64', labelW: DK * c + 30 })
    mk.drawMat({ r: rK, vals: TOY.KT, kind: 'col', alpha: fin, name: 'Kᵀ', shape: '4 × 6', real: '64 × 6', colToks: true })
    caption('from the decoder', rQ.x + (DK * c) / 2, rQ.y + NT * c + 22, fin, C.ink2)
    caption('from the encoder', rK.x - 12, rK.y + (DK * c) / 2 - 4, fin, C.ink2, 'right')
    caption('output (memory)', rK.x - 12, rK.y + (DK * c) / 2 + 12, fin, C.ink2, 'right')
    if (sm < 1) mk.drawMat({ r: rS, vals: TOY.S, kind: 'score', alpha: fin * (1 - sm), name: 'S', shape: '7 × 6', label: 'bottom', reveal: g.rev })
    if (sm > 0) mk.drawMat({ r: rS, vals: TOY.A, kind: 'attn', alpha: fin * sm, name: 'A = softmax(S / √d_k)', shape: '7 × 6', label: 'bottom', labelW: 400 })
    const key = sm < 0.5 ? 'S' : 'A'
    mk.hit(key, rS, NT, NS)
    // right: the same weights as links from each target position to the source words
    const bx0 = 520, bx1 = W - pad - 20, sy = top + 64, ty = top + 300
    const bxS = (i: number) => lerp(bx0, bx1, (i + 0.5) / NS), bxT = (j: number) => lerp(bx0, bx1, (j + 0.5) / NT)
    const ba = fin * eout(clamp((p - 0.5) / 0.08))
    title('source (encoder)', bx0, sy - 26, ba)
    title('decoder position → word it writes', bx0, ty + 60, ba)
    const f = mk.focus && mk.focus.key === 'A' ? mk.focus : null
    for (let j = 0; j < NT; j++) {
      const la = eout(clamp((p - 0.6 - j * 0.045) / 0.06))
      if (la <= 0) continue
      for (let i = 0; i < NS; i++) {
        const w = TOY.A[j][i], focusOn = f ? f.i === j : true
        ctx.strokeStyle = rgba(hue(i), (0.05 + 0.9 * w) * la * ba * (focusOn ? 1 : 0.12)); ctx.lineWidth = 0.6 + 5 * w
        ctx.beginPath(); ctx.moveTo(bxT(j), ty - 11); ctx.bezierCurveTo(bxT(j), ty - 90, bxS(i), sy + 90, bxS(i), sy + 11); ctx.stroke()
      }
      rr(bxT(j) - 14, ty + 30, 28, 8, 3); ctx.fillStyle = rgba(readCol(j), 0.85 * la * ba); ctx.fill()
    }
    SRC.forEach((t, i) => chipAt(t, bxS(i), sy, ba, 20, !!f && TOY.A[f.i][i] > 0.3))
    TIN.forEach((t, j) => {
      chipAt(t, bxT(j), ty, ba, 20, !!f && f.i === j)
      caption(TOUT[j].text, bxT(j), ty + 24, ba, C.mute)
    })
    // formula: the GEMM cell being computed, or the weight under the pointer
    const fs = mk.resolve({ S: { g, K: DK } })
    if (fs && key === 'S') mk.gemmOverlay({ A: rQ, Av: TOY.Q, B: rK, Bv: TOY.KT, C: rS, f: fs, names: ['S', 'Q', 'Kᵀ'], note: `Row ${fs.i} is the query of ‘${TIN[fs.i].text}’, column ${fs.j} the key of ‘${SRC[fs.j].text}’. Toy numbers, set up so each query looks for the source word to translate next; a trained model learns this.` })
    else if (f) {
      const v = TOY.A[f.i][f.j]
      ctx.strokeStyle = rgba(C.ink, 1); ctx.lineWidth = 2; ctx.strokeRect(rS.x + f.j * c, rS.y + f.i * c, c, c)
      mk.formula = { segs: [[`A[${f.i},${f.j}]`, C.ink], ['  =  ', C.mute], [`softmax(S[${f.i},:] / 2)[${f.j}]`, C.mute], ['  =  ', C.mute], [fmtF(v), C.ink]], note: `The query at ‘${TIN[f.i].text}’, about to write ‘${TOUT[f.i].text}’, puts ${Math.round(v * 100)}% of its attention on ‘${SRC[f.j].text}’.` }
    } else mk.formula = { segs: [['A = softmax(Q_dec · K_encᵀ / √d_k)', C.ink], ['     ·     ', C.mute], ['out = A · V_enc', C.ink2]], note: 'No mask: every target position may read the whole source. ‘gesehen’ comes last in German, yet its row goes back to ‘seen’. The bar under each target token is the blend of source hues it reads.' }
  }

  /* ---------- scene 6: where LayerNorm sits ---------- */
  const NORMS = streamNorms()
  function scenePostLn(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.06)), x0 = pad + 8, x1 = 500, bx = 150
    const run = (p * 2.2) % 1 // a pulse along the main path
    const schem = (y: number, name: string, post: boolean) => {
      title(name, pad, y - 74, fin)
      for (let i = 0; i < 5; i++) lane(x0, x1, y + (i - 2) * 4, hue(i), fin)
      const fx = post ? 250 : 290, ax = post ? 340 : 380, by = y - 50
      ctx.strokeStyle = rgba(C.ink, 0.5 * fin); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(bx, y - 10); ctx.lineTo(bx, by); ctx.lineTo(ax, by); ctx.stroke()
      arrow([[ax, by], [ax, y - 11]], fin)
      if (!post) { glass(200, by - 20, by + 20, 0.55, { w: 5, d: 6 }, fin); caption('LN', 200, by - 34, fin, C.ink2) }
      glass(fx, by - 24, by + 24, 0.2, { w: 8, d: 9 }, fin); caption('attn or MLP', fx, by - 42, fin)
      addNode(ax, y, fin)
      if (post) { glass(420, y - 20, y + 20, 0.55, { w: 5, d: 6 }, fin); caption('LN', 420, y + 36, fin, C.ink2) }
      caption(post ? 'LayerNorm on the main path' : 'main path: only adds', (x0 + x1) / 2, y + 36 + (post ? 18 : 0), fin, post ? C.ink2 : C.mute)
      const px = lerp(x0, x1, run)
      ctx.fillStyle = rgba(C.ink, fin * (1 - Math.abs(run - 0.5) * 1.6)); ctx.beginPath(); ctx.arc(px, y, 3.2, 0, 7); ctx.fill()
    }
    schem(top + avail * 0.3, 'pre-LN · GPT-2', false)
    schem(top + avail * 0.78, 'post-LN · 2017', true)
    // the length of the residual stream, layer by layer
    const cx0 = 600, cx1 = W - pad - 110, cy0 = top + 36, cy1 = top + avail - 34
    const lo = Math.log(3), hi = Math.log(600), Y = (v: number) => lerp(cy1, cy0, (Math.log(v) - lo) / (hi - lo)), X = (l: number) => lerp(cx0, cx1, l / 12)
    const ca = eout(clamp((p - 0.25) / 0.1))
    title('length of the stream, ‖x‖', cx0, cy0 - 18, ca)
    ctx.strokeStyle = rgba(C.ink, 0.25 * ca); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy1); ctx.lineTo(cx1, cy1); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, ca)
    for (const v of [10, 30, 100, 300]) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(v), cx0 - 6, Y(v)); ctx.strokeStyle = rgba(C.ink, 0.07 * ca); ctx.beginPath(); ctx.moveTo(cx0, Y(v)); ctx.lineTo(cx1, Y(v)); ctx.stroke() }
    for (const l of [0, 3, 6, 9, 12]) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(String(l), X(l), cy1 + 6) }
    caption('layer (0 = embedding)', (cx0 + cx1) / 2, cy1 + 32, ca)
    const ga = clamp((p - 0.35) / 0.4) * 12
    ctx.strokeStyle = rgba(C.ink, 0.95 * ca); ctx.lineWidth = 1.8; ctx.beginPath()
    for (let l = 0; l <= Math.floor(ga); l++) (l ? ctx.lineTo(X(l), Y(NORMS[l])) : ctx.moveTo(X(l), Y(NORMS[l])))
    if (ga < 12) { const l = Math.floor(ga), t = ga - l; ctx.lineTo(X(l + t), Y(Math.exp(lerp(Math.log(NORMS[l]), Math.log(NORMS[l + 1]), t)))) }
    ctx.stroke()
    for (let l = 0; l <= Math.floor(ga); l++) { ctx.fillStyle = rgba(C.ink, ca); ctx.beginPath(); ctx.arc(X(l), Y(NORMS[l]), 2.4, 0, 7); ctx.fill() }
    const ea = clamp((ga - 11.5) * 2)
    caption(`GPT-2 ${fmt(NORMS[12])}`, X(12) + 10, Y(NORMS[12]) + 4, ca * ea, C.ink, 'left')
    caption('pre-LN, real', X(12) + 10, Y(NORMS[12]) + 20, ca * ea, C.mute, 'left')
    const pa = eout(clamp((p - 0.72) / 0.12)), flat = Math.sqrt(768)
    if (pa > 0) {
      ctx.setLineDash([5, 4]); ctx.strokeStyle = rgba(C.ink2, pa); ctx.lineWidth = 1.6
      ctx.beginPath(); ctx.moveTo(X(0), Y(flat)); ctx.lineTo(lerp(X(0), X(12), pa), Y(flat)); ctx.stroke(); ctx.setLineDash([])
      caption(`post-LN ${fmt(flat)}`, X(12) + 10, Y(flat) + 4, pa, C.ink2, 'left')
      caption('= √768, γ = 1', X(12) + 10, Y(flat) + 20, pa, C.mute, 'left')
    }
    mk.formula = { segs: [['pre-LN  x + f(LN(x))', C.ink2], ['     ·     ', C.mute], ['post-LN  LN(x + f(x))', C.ink]], note: 'The GPT-2 line is real: the mean length of its residual stream on “The cat sat on the floor”, after the embedding and after each block (the first position left out). Post-LN resets the length after every sub-layer; the dashed line assumes γ = 1 at GPT-2’s width.' }
  }

  /* ---------- scene 7: sinusoids vs learned positions ---------- */
  const WP = wpeSlice()
  const wpMax = (() => { const v = WP.slice(1).flat().map(Math.abs).sort((a, b) => a - b); return v[Math.floor(v.length * 0.98)] })()
  const PEV = Array.from({ length: MAP }, (_, pos) => Array.from({ length: MAP }, (_, j) => pe(pos, peDim(j))))
  function heat(r: Rect, vals: number[][], vmax: number, a: number, reveal: (j: number) => number, row: number) {
    const c = r.c
    mk.slab(r.x, r.y, MAP * c, MAP * c, a)
    for (let i = 0; i < MAP; i++) for (let j = 0; j < MAP; j++) {
      const rv = reveal(j)
      if (rv <= 0) continue
      const v = vals[i][j], m = Math.min(1, Math.abs(v) / vmax), x = r.x + j * c, y = r.y + i * c, aa = a * rv * (row < 0 || row === i ? 1 : 0.7)
      if (v >= 0) { ctx.fillStyle = rgba(C.ink, (0.05 + 0.8 * m) * aa); ctx.fillRect(x, y, c, c) }
      else { ctx.strokeStyle = rgba(C.neg, (0.1 + 0.75 * m) * aa); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, c - 1, c - 1) }
    }
    ctx.strokeStyle = rgba(C.ink, 0.22 * a); ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, MAP * c - 1, MAP * c - 1)
    if (row >= 0) { ctx.strokeStyle = rgba(C.ink, 0.9 * a); ctx.lineWidth = 1.2; ctx.strokeRect(r.x - 1, r.y + row * c - 1, MAP * c + 2, c + 2) }
  }
  function scenePos(p: number) {
    const fin = eout(clamp(p / 0.06)), c = 4, y0 = top + 60
    const rP: Rect = { x: pad + 40, y: y0, c }, rW: Rect = { x: rP.x + MAP * c + 64, y: y0, c }
    const pw = clamp((p - 0.04) / 0.3), wa = fin * eout(clamp((p - 0.3) / 0.12))
    const sweep = p > 0.5 ? Math.min(MAP - 1, Math.floor(eio(clamp((p - 0.5) / 0.45)) * MAP)) : -1
    const f = mk.focus && (mk.focus.key === 'pe' || mk.focus.key === 'wp') ? mk.focus : null
    const row = f ? f.i : sweep
    mathRun([['PE', false]], rP.x, y0 - 30, fin, 19)
    caption('sinusoids · 2017', rP.x + 30, y0 - 30, fin, C.ink2, 'left')
    caption('64 of 512 dims: every 8th sin/cos pair', rP.x, y0 + MAP * c + 20, fin, C.mute, 'left')
    heat(rP, PEV, 1, fin, (j) => clamp(pw * MAP * 1.2 - j), row)
    if (wa > 0) {
      mathRun([['W', false], ['P', true]], rW.x, y0 - 30, wa, 19)
      caption('learned · GPT-2, real', rW.x + 34, y0 - 30, wa, C.ink2, 'left')
      caption('dims 0–63 of 768', rW.x, y0 + MAP * c + 20, wa, C.mute, 'left')
      heat(rW, WP, wpMax, wa, () => 1, row)
    }
    ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, fin)
    for (const v of [0, 16, 32, 48, 63]) ctx.fillText(String(v), rP.x - 6, y0 + (v + 0.5) * c)
    ctx.save(); ctx.translate(rP.x - 30, y0 + (MAP * c) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('position', 0, 0); ctx.restore()
    mk.hit('pe', rP, MAP, MAP)
    if (wa > 0.3) mk.hit('wp', rW, MAP, MAP)
    // each sin/cos pair is a point turning around a circle as the position grows
    const ra = fin * eout(clamp((p - 0.45) / 0.1)), pos = row < 0 ? 0 : row
    const px0 = rW.x + MAP * c + 70, R = 30, pairs = [0, 32, 96]
    if (ra > 0) {
      title('one sin/cos pair per circle', px0, y0 - 26, ra)
      pairs.forEach((i, k) => {
        const cx = px0 + R + k * (2 * R + 30), cy = y0 + 40, w = Math.pow(BASE, (-2 * i) / D_MODEL), ang = pos * w
        ctx.strokeStyle = rgba(C.ink, 0.2 * ra); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke()
        ctx.strokeStyle = rgba(C.ink, 0.9 * ra); ctx.lineWidth = 1.8
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.sin(ang) * R, cy - Math.cos(ang) * R); ctx.stroke()
        ctx.fillStyle = rgba(C.ink, ra); ctx.beginPath(); ctx.arc(cx + Math.sin(ang) * R, cy - Math.cos(ang) * R, 2.6, 0, 7); ctx.fill()
        caption(`pair ${i}`, cx, cy + R + 20, ra, C.ink2)
        caption(`ω ${w >= 1 ? '1' : w.toFixed(w >= 0.1 ? 2 : 3)}`, cx, cy + R + 36, ra)
      })
      ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, ra)
      ctx.fillText(`pos ${pos}`, px0, y0 + 150)
      caption('angle = pos × ω: a shift by k', px0, y0 + 176, ra, C.mute, 'left')
      caption('turns each pair by k × ω', px0, y0 + 192, ra, C.mute, 'left')
      caption('GPT-2 has no row past 1,023;', px0, y0 + 226, ra, C.ink2, 'left')
      caption('the formula works at any position', px0, y0 + 242, ra, C.ink2, 'left')
    }
    if (f && f.key === 'pe') {
      const dim = peDim(f.j), i2 = dim - (dim & 1)
      mk.formula = { segs: [[`PE[${f.i}, ${dim}]`, C.ink], ['  =  ', C.mute], [`${dim & 1 ? 'cos' : 'sin'}(${f.i} / 10000^(${i2}/512))`, C.ink2], ['  =  ', C.mute], [fmtF(PEV[f.i][f.j]), C.ink]], note: 'Computed, not learned: the same formula gives a vector for any position, with no parameters.' }
    } else if (f) mk.formula = { segs: [[`W_P[${f.i}, ${f.j}]`, C.ink], ['  =  ', C.mute], [fmtF(WP[f.i][f.j]), C.ink]], note: 'Learned during training, one row per position up to 1,023. Row 0 has a few very large values (drawn at full strength).' }
    else mk.formula = { segs: [['PE(pos, 2i) = sin(pos / 10000^(2i / d_model))', C.ink], ['     ·     ', C.mute], ['PE(pos, 2i+1) = cos(…)', C.ink]], note: 'Wavelengths grow from 2π to 10,000 · 2π across the dimensions. Vaswani et al. also tried learned positions and got nearly the same results; they kept sinusoids in the hope that they extend to sentences longer than any seen in training.' }
  }

  return { rnn: sceneRnn, blocks: sceneBlocks, translate: sceneTranslate, masks: sceneMasks, cross: sceneCross, postln: scenePostLn, pos: scenePos }
}
