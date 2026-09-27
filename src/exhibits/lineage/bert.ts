import { F, chipW, drawChip, rr, tokLabel, type TokLike } from '../../core/draw'
import { stepper } from '../../core/frame'
import { fmt, fmtF, gemm, type Rect } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, gauss, lerp, rng } from '../../core/util'
import { bert } from '../../lib/bert/data'
import { leftOnly, presets } from '../../lib/gpt2/data'
import type { Nav } from '../registry'
import { mountLineage, words, type Env } from './kit'

/*
 * BERT-base as a diff against GPT-2. Same shape (12 layers, 768, 12 heads, 3,072); what changes is the
 * direction: no causal mask, a masked-word objective, post-LN, segment embeddings, and a [CLS] vector
 * for classification. The attention heads and masked-word predictions are real (bert-base-uncased and
 * GPT-2 small, exported offline); the embedding sums and the classifier use toy numbers.
 */

const PHASES = [
  { id: 'blocks', name: 'Same shape, both directions', short: 'Blocks', dur: 7 },
  { id: 'masks', name: 'Looking both ways', short: 'Masks', dur: 9 },
  { id: 'inputs', name: 'What goes in', short: 'Inputs', dur: 8 },
  { id: 'mlm', name: 'Fill in the blank', short: 'MLM', dur: 12 },
  { id: 'use', name: 'Reading, not writing', short: '[CLS]', dur: 9 },
]

const SEQ = words(bert.examples[0].toks)
const tok = (text: string, c: number): TokLike => ({ text, c })

/** GPT-2's clearest previous-token head, found on its longest run of the cat prompt. */
const GPT = (() => {
  const pass = presets()[0].passes.at(-1)!, N = pass.texts.length
  let best = { l: 0, h: 0, s: -1 }
  pass.att.forEach((heads, l) => heads.forEach((A, h) => {
    let s = 0
    for (let i = 1; i < N; i++) s += A[i][i - 1]
    s /= N - 1
    if (s > best.s) best = { l, h, s }
  }))
  // rows are stored up to the diagonal only; pad the masked part with zeros
  const A = pass.att[best.l][best.h].map((row) => [...row, ...Array(N - row.length).fill(0)])
  return { texts: pass.texts, A, layer: best.l, head: best.h, score: best.s }
})()

/** Toy classifier on the [CLS] vector: 8 numbers → 2 labels. */
const CLS = (() => {
  const r = rng(77), h = [Array.from({ length: 8 }, () => gauss(r))]
  const Wc = Array.from({ length: 8 }, (_, k) => [h[0][k] * 0.18 + gauss(r) * 0.15, -h[0][k] * 0.1 + gauss(r) * 0.15])
  const z = [[0, 1].map((j) => h[0].reduce((s, v, k) => s + v * Wc[k][j], 0))]
  const m = Math.max(...z[0]), e = z[0].map((v) => Math.exp(v - m)), s = e[0] + e[1]
  return { h, Wc, z, p: e.map((v) => v / s) }
})()

const COMPARE: Record<string, [string, string]> = {
  blocks: ['GPT-2’s block', 'anatomy/layernorm?phase=stream'], masks: ['GPT-2’s causal mask', 'anatomy/attention?phase=mask'],
  inputs: ['GPT-2’s embedding', 'anatomy/embedding'], mlm: ['GPT-2’s next-token loss', 'training/loss'], use: ['GPT-2’s output', 'anatomy/unembed'],
}

const CAPS: Record<string, [string, string]> = {
  blocks: ['BERT-base has exactly GPT-2 small’s shape: 12 layers, 768 wide, 12 heads. What differs is the direction: there is no causal mask, so every token sees the whole sentence, and it is trained to fill in hidden words instead of predicting the next one. Like the 2017 Transformer, it normalises after each residual add.', '12 × 768 · 110M · encoder-only'],
  masks: ['Two real heads. GPT-2’s clearest previous-token head can only look back. BERT’s layer 3 head 1 is its mirror image: nearly all of its attention goes to the next token, which a causal mask would forbid. Hover the cells.', 'causal triangle · full square'],
  inputs: ['BERT reads one or two sentences at a time: [CLS] first and [SEP] after each sentence. Each input vector is the sum of a WordPiece token embedding, a segment embedding (sentence A or B) and a learned position embedding. WordPiece splits rare words: mailman → mail ##man.', 'token + segment + position'],
  mlm: ['In training, 15% of the tokens are chosen and hidden (mostly as [MASK]), and BERT predicts them from both sides. These are real predictions from BERT-base, next to GPT-2’s guesses from only the words before the blank. Use the arrows below to switch sentences.', 'predict [MASK] from both sides'],
  use: ['BERT is built to read, not to write: with no left-to-right order it has no natural way to generate text. Instead a small classifier is trained on top, on the final [CLS] vector for a label per sentence (such as sentiment) or on each token’s vector for tagging.', 'label = softmax(h_CLS · W)'],
}

export function mountBert(root: HTMLElement, nav: Nav): () => void {
  return mountLineage(root, nav, {
    frame: {
      formulaHint: 'Hover a cell of an attention map or a bar to read it; click or tap to pin it.',
      eyebrow: 'Lineage · Encoder',
      title: 'BERT',
      subtitle: 'BERT-base · the same shape as GPT-2, reading both ways',
      specs: [
        { label: 'compared', value: 'BERT-base', real: 'GPT-2 small', realLabel: 'vs' },
        { label: 'layers', value: '12', real: '12' },
        { label: 'd_model', value: '768', real: '768' },
        { label: 'heads', value: '12', real: '12' },
        { label: 'd_ff', value: '3,072', real: '3,072' },
        { label: 'vocab', value: '30,522 WordPiece', real: '50,257' },
        { label: 'params', value: '110M', real: '124M' },
      ],
    },
    size: [1040, 480],
    aria: 'BERT compared with GPT-2: the same size, but no causal mask, so every token sees the whole sentence; it is trained to fill in masked words from both sides and used to classify text through its [CLS] vector.',
    phases: PHASES, learn: 'bert', tokens: SEQ, compare: COMPARE, caps: CAPS,
    still: ['mlm', 10],
    scenes,
  })
}

function scenes({ stage, ctx, mk, k, player, onFrame }: Env) {
  const { pill, addNode, caption, title, lane, glass, arrow } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46
  const chipAt = (t: TokLike, cx: number, y: number, a: number, h = 18, hl = false) => drawChip(cx - chipW(t.text) / 2, y, t, a, h, hl)

  // choose the masked sentence; it also advances by itself during the phase until you pick one
  const EX = bert.examples, LEFT = leftOnly()
  const sel = { i: 0, manual: false }
  const exStep = stepper(player.meta, 'sentence', EX.length, 0, (i) => { sel.i = i; sel.manual = true; if (player.cur().id === 'mlm') player.setPlaying(false) })
  const exEl = player.meta.querySelector('.stepper') as HTMLElement
  exEl.hidden = true
  onFrame(() => { const on = player.cur().id === 'mlm'; if (exEl.hidden === on) exEl.hidden = !on })

  /** A token-by-token attention map with row and rotated column labels; masked cells hatched. */
  function attnMap(key: string, x: number, y: number, c: number, labels: string[], A: number[][], causal: boolean, a: number) {
    const n = labels.length, r: Rect = { x, y, c }
    mk.drawMat({ r, vals: A, kind: 'attn', alpha: a, name: '', shape: '', label: 'none', noText: true, paint: (px, py, cc, i, j, aa) => {
      if (causal && j > i) { mk.hatch(px, py, cc, aa); return 0 }
      return mk.paintAttn(px, py, cc, A[i][j], hue(j), aa)
    } })
    mk.hit(key, r, n, n)
    const f = mk.focus?.key === key ? mk.focus : null
    ctx.font = F.mono(10.5); ctx.textBaseline = 'middle'
    labels.forEach((t, i) => { ctx.textAlign = 'right'; ctx.fillStyle = rgba(hue(i), a * (f && f.i !== i ? 0.45 : 1)); ctx.fillText(t, x - 6, y + (i + 0.5) * c) })
    labels.forEach((t, j) => {
      ctx.save(); ctx.translate(x + (j + 0.5) * c, y - 6); ctx.rotate(-Math.PI / 4)
      ctx.textAlign = 'left'; ctx.fillStyle = rgba(hue(j), a * (f && f.j !== j ? 0.45 : 1)); ctx.fillText(t, 0, 0); ctx.restore()
    })
    if (f) { ctx.strokeStyle = rgba(C.ink, a); ctx.lineWidth = 2; ctx.strokeRect(x + f.j * c, y + f.i * c, c, c) }
    return f
  }

  /* ---------- scene 1: the blocks ---------- */
  function sceneBlocks(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const sx0 = pad + 130, sx1 = W - pad - 90, fx = (f: number) => lerp(sx0, sx1, f)
    const ys = [top + avail * 0.2, top + avail * 0.6], as = [eout(clamp(p / 0.3)), eout(clamp((p - 0.25) / 0.3))]
    ys.forEach((y, r) => {
      const a = as[r]
      if (a <= 0) return
      k.rowName(r ? 'BERT' : 'GPT-2', r ? 'base · 110M' : 'small · 124M', pad, y, a)
      for (let i = 0; i < 5; i++) lane(sx0, lerp(sx0, sx1, a), y + (i - 2) * 4, hue(i), a)
      const box = (x: number, w: number, t: string, sub: string, hl: boolean) => {
        rr(x - w / 2, y - 15, w, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, (hl ? 0.9 : 0.6) * a); ctx.lineWidth = 1; ctx.stroke()
        caption(t, x, y + 4, a, C.ink); caption(sub, x, y + 44, a, hl ? C.ink2 : C.mute)
      }
      box(fx(0.03), r ? 72 : 52, r ? '+ P + seg' : '+ W_P', r ? 'positions, segments' : 'positions', r === 1)
      const plates: [number, string, boolean][] = r
        ? [[0.24, 'self · all', true], [0.43, 'LN', true], [0.62, 'MLP', false], [0.81, 'LN', true]]
        : [[0.14, 'LN', false], [0.24, 'self · causal', false], [0.52, 'LN', false], [0.62, 'MLP', false]]
      for (const [f, t, hl] of plates) {
        const ln = t === 'LN'
        glass(fx(f), y - (ln ? 20 : 26), y + (ln ? 20 : 26), hl ? 0.55 : 0.2, ln ? { w: 5, d: 6 } : { w: 8, d: 9 }, a)
        caption(t, fx(f), y + 44, a, hl ? C.ink2 : C.mute)
      }
      addNode(fx(0.35), y, a); addNode(fx(0.73), y, a)
      // output head: the next token, or the blanks and [CLS]
      const hx = sx1 + 44
      arrow([[sx1, y], [hx - 30, y]], a)
      rr(hx - 36, y - 15, 92, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.7 * a); ctx.stroke()
      caption(r ? 'fill blanks' : 'next token', hx + 10, y + 4, a, C.ink)
      caption(r ? 'or [CLS] → label' : 'W_Eᵀ, softmax', hx + 10, y + 44, a, r ? C.ink2 : C.mute)
    })
    const la = eout(clamp((p - 0.55) / 0.25)), yB = ys[1]
    if (la > 0) {
      pill('no mask', fx(0.24), yB - 46, 'masks', la)
      pill('inputs', fx(0.03), yB - 46, 'inputs', la)
      pill('[MASK]', sx1 + 30, yB - 46, 'mlm', la)
      pill('[CLS]', sx1 + 30, yB + 70, 'use', la)
    }
    mk.formula = { segs: [['GPT-2  x + f(LN(x)), causal', C.ink2], ['     ·     ', C.mute], ['BERT  LN(x + f(x)), every token sees every token', C.ink]], note: 'Same widths, same number of layers and heads, GELU in both. BERT has 30,522 WordPiece tokens, 512 positions and two segment embeddings; it has no causal mask and no next-token head.' }
  }

  /* ---------- scene 2: two real heads ---------- */
  function sceneMasks(p: number) {
    const c = 28, y0 = top + 120
    const ga = eout(clamp(p / 0.2)), ba = eout(clamp((p - 0.3) / 0.2))
    const gl = GPT.texts.map((t) => tokLabel(t)), bl = bert.ahead.toks
    const gx = pad + 80, bx = 590
    title(`GPT-2 · layer ${GPT.layer + 1}, head ${GPT.head + 1}`, gx, y0 - 78, ga)
    caption(`looks back one token (${Math.round(GPT.score * 100)}% on average)`, gx, y0 - 60, ga, C.ink2, 'left')
    const f1 = attnMap('g', gx, y0, c, gl, GPT.A, true, ga)
    let f2 = null
    if (ba > 0) {
      title(`BERT · layer ${bert.ahead.layer + 1}, head ${bert.ahead.head + 1}`, bx, y0 - 78, ba)
      caption(`looks ahead one token (${Math.round(bert.ahead.score * 100)}% on average)`, bx, y0 - 60, ba, C.ink2, 'left')
      f2 = attnMap('b', bx, y0, c, bl, bert.ahead.A, false, ba)
    }
    // the off-diagonal each head follows
    const da = eout(clamp((p - 0.62) / 0.12))
    if (da > 0) {
      ctx.strokeStyle = rgba(C.ink, 0.7 * da); ctx.lineWidth = 1.2; ctx.setLineDash([3, 3])
      ctx.beginPath(); ctx.moveTo(gx + 0.5 * c, y0 + 1.5 * c); ctx.lineTo(gx + (gl.length - 1.5) * c, y0 + (gl.length - 0.5) * c); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(bx + 1.5 * c, y0 + 0.5 * c); ctx.lineTo(bx + (bl.length - 0.5) * c, y0 + (bl.length - 1.5) * c); ctx.stroke()
      ctx.setLineDash([])
      caption('below the diagonal: earlier tokens', gx, y0 + gl.length * c + 26, da, C.mute, 'left')
      caption('above the diagonal: later tokens, masked in GPT-2', bx, y0 + bl.length * c + 26, da, C.mute, 'left')
    }
    const f = f1 ?? f2
    if (f) {
      const g = f.key === 'g', A = g ? GPT.A : bert.ahead.A, L = g ? gl : bl, v = A[f.i][f.j]
      mk.formula = g && f.j > f.i
        ? { segs: [[`GPT-2  A[${f.i},${f.j}]`, C.ink], ['  =  ', C.mute], ['0  (masked)', C.ink]], note: `‘${L[f.i]}’ may not look at the later ‘${L[f.j]}’.` }
        : { segs: [[`${g ? 'GPT-2' : 'BERT'}  A[${f.i},${f.j}]`, C.ink], ['  =  ', C.mute], [fmtF(v), C.ink]], note: `‘${L[f.i]}’ puts ${Math.round(v * 100)}% of this head’s attention on ‘${L[f.j]}’. Real weights.` }
    } else mk.formula = { segs: [['GPT-2  A[i, j] = 0 for j > i', C.ink2], ['     ·     ', C.mute], ['BERT  no mask', C.ink]], note: `Real attention: GPT-2 small on its own continuation of “The cat sat on the”, BERT-base on “The cat sat on the floor.” Both heads found by the export as the strongest of their kind.` }
  }

  /* ---------- scene 3: what goes in ---------- */
  const PAIR = bert.pair
  const EMB = (() => {
    const r = rng(41), v = (n: number) => Array.from({ length: n }, () => gauss(r))
    const seg = [v(4), v(4)]
    return { tok: PAIR.toks.map(() => v(4)), seg, pos: PAIR.toks.map((_, i) => [0, 1, 2, 3].map((k2) => Math.sin(i / (1 + 2 * k2) + k2))) }
  })()
  function sceneInputs(p: number) {
    const { W } = stage
    const n = PAIR.toks.length, x0 = pad + 110, span = W - pad - 20 - x0, dx = span / n, cs = Math.min(12, (dx - 8) / 4)
    const y0 = top + 50, rows = [y0 + 58, y0 + 100, y0 + 142, y0 + 206]
    const names = ['token', 'segment', 'position', 'input']
    const ra = (r: number) => eout(clamp((p - 0.06 - r * 0.16) / 0.12))
    PAIR.toks.forEach((t, i) => chipAt(tok(t, i), x0 + (i + 0.5) * dx, y0, eout(clamp(p / 0.08)), 20, t === '##man' || t === 'mail'))
    names.forEach((nm, r) => {
      const a = ra(r)
      if (a <= 0) return
      title(nm, pad, rows[r] + cs / 2 + 4, a)
      if (r === 3) { ctx.strokeStyle = rgba(C.ink, 0.25 * a); ctx.beginPath(); ctx.moveTo(x0, rows[3] - 22); ctx.lineTo(x0 + span, rows[3] - 22); ctx.stroke() }
      PAIR.toks.forEach((_, i) => {
        const vals = r === 0 ? EMB.tok[i] : r === 1 ? EMB.seg[PAIR.seg[i]] : r === 2 ? EMB.pos[i] : EMB.tok[i].map((v, k2) => v + EMB.seg[PAIR.seg[i]][k2] + EMB.pos[i][k2])
        const col = r === 0 || r === 3 ? hue(i) : null, vmax = r === 3 ? 3 : 2
        const bx = x0 + (i + 0.5) * dx - (4 * cs) / 2
        vals.forEach((v, k2) => mk.paintCell(bx + k2 * cs, rows[r], cs, v, vmax, col, 0.6, a))
        if (r === 1 && (i === 0 || PAIR.seg[i] !== PAIR.seg[i - 1])) caption(PAIR.seg[i] ? 'B' : 'A', bx - 8, rows[r] + cs - 1, a, C.ink2, 'right')
        if (r === 2 && i % 3 === 0) caption(String(i), x0 + (i + 0.5) * dx, rows[r] + cs + 16, a, C.mute)
      })
      if (r > 0 && r < 3) caption('+', x0 - 30, rows[r] + cs, a, C.ink2)
    })
    const ba = eout(clamp((p - 0.7) / 0.1))
    if (ba > 0) {
      const iSep = PAIR.toks.indexOf('[SEP]'), iMail = PAIR.toks.indexOf('mail')
      caption('sentence A', x0 + (iSep / 2 + 0.5) * dx, y0 - 26, ba, C.ink2)
      caption('sentence B', x0 + ((iSep + n) / 2 + 0.5) * dx, y0 - 26, ba, C.ink2)
      caption('mailman → mail ##man', x0 + (iMail + 1) * dx, rows[3] + cs + 40, ba, C.ink2)
    }
    mk.formula = { segs: [['input = E_token[id] + E_segment[A or B] + E_position[i]', C.ink], ['     ·     ', C.mute], ['then LayerNorm', C.ink2]], note: 'The tokens are BERT’s real WordPiece split (lower-cased, ## marks a piece inside a word). The small vectors are toy: 4 numbers where BERT has 768. GPT-2 adds only the token and position rows.' }
  }

  /* ---------- scene 4: fill in the blank, real predictions ---------- */
  function sceneMlm(p: number) {
    const { W } = stage
    const seg = p * EX.length
    if (!sel.manual) { const i = Math.min(EX.length - 1, Math.floor(seg)); if (i !== sel.i) { sel.i = i; exStep.set(i) } }
    const ex = EX[sel.i], lo = LEFT[sel.i], toks = ex.toks.slice(1, -1), mi = ex.mask - 1
    const local = sel.manual ? 1 : clamp(seg - sel.i)
    const fin = eout(clamp(local / 0.08)), n = toks.length
    const cw = toks.map((t) => chipW(t) + 10), total = cw.reduce((a, b) => a + b, 0), sx = (W - total) / 2, ys = top + 110
    const cx = (i: number) => sx + cw.slice(0, i).reduce((a, b) => a + b, 0) + cw[i] / 2
    toks.forEach((t, i) => chipAt(tok(t, i), cx(i), ys, fin, 22, i === mi))
    // GPT-2 above: only the words before the blank; BERT below: both sides
    const ga = eout(clamp((local - 0.08) / 0.12)), ba = eout(clamp((local - 0.2) / 0.12))
    for (let i = 0; i < n; i++) {
      if (i === mi) continue
      if (i < mi) { ctx.strokeStyle = rgba(hue(i), 0.7 * ga); ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(cx(i), ys - 12); ctx.quadraticCurveTo((cx(i) + cx(mi)) / 2, ys - 46 - 6 * (mi - i), cx(mi), ys - 12); ctx.stroke() }
      else { ctx.fillStyle = rgba(C.bg, 0.55 * ga); ctx.fillRect(cx(i) - cw[i] / 2 + 3, ys - 12, cw[i] - 6, 24) }
      ctx.strokeStyle = rgba(hue(i), 0.7 * ba); ctx.lineWidth = 1.4
      ctx.beginPath(); ctx.moveTo(cx(i), ys + 12); ctx.quadraticCurveTo((cx(i) + cx(mi)) / 2, ys + 46 + 6 * Math.abs(mi - i), cx(mi), ys + 12); ctx.stroke()
    }
    caption('GPT-2 sees only this side', (sx + cx(mi)) / 2, ys - 62, ga, C.ink2)
    caption('BERT sees both sides', W / 2, ys + 86, ba, C.ink2)
    // the two top-6 lists
    const bars = (x: number, y: number, head: string, rows: { t: string; p: number }[], a: number, strong: boolean) => {
      title(head, x, y, a)
      rows.slice(0, 6).forEach((r, j) => {
        const yy = y + 22 + j * 24, g = eio(clamp((local - 0.35 - j * 0.03) / 0.15))
        ctx.font = F.mono(12, j === 0 ? 500 : 400); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(j === 0 ? C.ink : C.ink2, a)
        ctx.fillText(r.t, x, yy)
        ctx.fillStyle = rgba(C.ink, (strong ? 0.75 : 0.45) * a); ctx.fillRect(x + 110, yy - 7, Math.max(1.5, r.p * 260 * g), 14)
        ctx.fillStyle = rgba(C.ink2, a * g); ctx.fillText(`${(r.p * 100).toFixed(1)}%`, x + 118 + r.p * 260 * g, yy)
      })
    }
    const la = eout(clamp((local - 0.3) / 0.1)), by = ys + 118
    bars(W / 2 - 440, by, 'bert · both sides', ex.top, la, true)
    bars(W / 2 + 60, by, `gpt-2 · after “${lo.left}”`, lo.top.map((t) => ({ t: t.s, p: t.p })), la, false)
    mk.formula = { segs: [[`BERT  p(“${ex.top[0].t}” | both sides) = ${fmt(ex.top[0].p * 100)}%`, C.ink], ['     ·     ', C.mute], [`GPT-2  top guess after “${lo.left}”: ${lo.top[0].s} ${fmt(lo.top[0].p * 100)}%`, C.ink2]], note: 'Real outputs of bert-base-uncased and GPT-2 small. In pretraining, of the 15% of tokens chosen, 80% become [MASK], 10% a random token and 10% stay as they are, so the model cannot rely on seeing [MASK].' }
  }

  /* ---------- scene 5: a classifier on [CLS] ---------- */
  function sceneUse(p: number) {
    const { H } = stage
    const fin = eout(clamp(p / 0.08)), toks = bert.cls, n = toks.length
    const x0 = pad + 20, dy = 32, y0 = top + 40
    toks.forEach((t, i) => drawChip(x0, y0 + i * dy, tok(t, i), fin, 20, i === 0))
    const sx = x0 + 90, ex = sx + 200
    for (let i = 0; i < n; i++) lane(sx, ex, y0 + i * dy, hue(i), fin * (i ? 0.5 : 1))
    for (let l = 0; l < 6; l++) glass(sx + 30 + l * 26, y0 - 18, y0 + (n - 1) * dy + 18, 0.2, { w: 5, d: 6 }, fin)
    caption('BERT, 12 layers', sx + 95, y0 + (n - 1) * dy + 46, fin)
    // [CLS]'s final vector × W → one score per label (A left, B above, C where they meet)
    const c = 26, rZ: Rect = { x: ex + 330, y: top + 8 * c + 70, c }, rH: Rect = { x: rZ.x - 8 * c - 40, y: rZ.y, c }, rW: Rect = { x: rZ.x, y: rZ.y - 8 * c - 26, c }
    const ha = fin * eout(clamp((p - 0.12) / 0.1)), g = gemm(clamp((p - 0.25) / 0.35), 1, 2, 8, 'slow')
    ctx.strokeStyle = rgba(hue(0), 0.8 * ha); ctx.lineWidth = 1.3
    ctx.beginPath(); ctx.moveTo(ex, y0); ctx.bezierCurveTo(ex + 80, y0, rH.x - 80, rH.y + c / 2, rH.x - 8, rH.y + c / 2); ctx.stroke()
    mk.drawMat({ r: rH, vals: CLS.h, kind: 'row', rowCols: [hue(0)], alpha: ha, name: 'h_CLS', shape: '1 × 8', real: '1 × 768', label: 'bottom', labelW: 8 * c + 30 })
    mk.drawMat({ r: rW, vals: CLS.Wc, kind: 'w', alpha: ha, name: 'W', shape: '8 × 2', real: '768 × 2', labelW: 200 })
    mk.drawMat({ r: rZ, vals: CLS.z, kind: 'score', alpha: ha, name: 'z', shape: '1 × 2', label: 'bottom', reveal: g.rev, labelW: 80 })
    mk.hit('z', rZ, 1, 2)
    const fs = mk.resolve({ z: { g, K: 8 } })
    if (fs) mk.gemmOverlay({ A: rH, Av: CLS.h, B: rW, Bv: CLS.Wc, C: rZ, f: fs, names: ['z', 'h_CLS', 'W'], note: `The score for “${fs.j ? 'negative' : 'positive'}”. A trained classifier head reads the [CLS] vector; toy numbers, and in practice BERT itself is fine-tuned together with this small matrix.` })
    else mk.formula = { segs: [['p(label) = softmax(h_CLS · W)', C.ink]], note: 'The [CLS] position has no word of its own; during fine-tuning its final vector learns to summarise the sentence for the task. Toy numbers.' }
    const pa = eout(clamp((p - 0.65) / 0.1))
    if (pa > 0) {
      const px = rZ.x + 2 * c + 44
      arrow([[rZ.x + 2 * c + 8, rZ.y + c / 2], [px - 8, rZ.y + c / 2]], pa)
      ;['positive', 'negative'].forEach((lab, j) => {
        const y = rZ.y + c / 2 + (j - 0.5) * 26
        ctx.font = F.mono(12, j === 0 ? 500 : 400); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(j === 0 ? C.ink : C.ink2, pa)
        ctx.fillText(lab, px, y)
        ctx.fillStyle = rgba(C.ink, 0.7 * pa); ctx.fillRect(px + 76, y - 6, Math.max(1.5, CLS.p[j] * 90), 12)
        ctx.fillStyle = rgba(C.ink2, pa); ctx.fillText(fmt(CLS.p[j]), px + 82 + CLS.p[j] * 90, y)
      })
    }
    const na = eout(clamp((p - 0.8) / 0.1))
    caption('each token’s own vector can be classified too: names, places, parts of speech', x0, H - bot - 10, na, C.mute, 'left')
  }

  return { blocks: sceneBlocks, masks: sceneMasks, inputs: sceneInputs, mlm: sceneMlm, use: sceneUse }
}
