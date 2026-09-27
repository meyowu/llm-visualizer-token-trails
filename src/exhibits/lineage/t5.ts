import { F, chipW, drawChip, fillRich, rr } from '../../core/draw'
import { fmtF } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eout, lerp } from '../../core/util'
import { bucket, t5 } from '../../lib/t5/data'
import type { Nav } from '../registry'
import { mountExhibit, words, type Env } from '../kit'

/*
 * T5 as a diff against GPT-2 and the 2017 Transformer. An encoder–decoder where every task is text in,
 * text out; pretrained by span corruption; pre-LN with a scale-only norm, no biases, and positions as a
 * learned bias per relative-distance bucket. The task outputs, the span fill and the biases are real
 * (T5-small, exported offline); the paper's span-corruption sentence is its Figure 2 example.
 */

const PHASES = [
  { id: 'tasks', name: 'Text in, text out', short: 'Text to text', dur: 12 },
  { id: 'spans', name: 'Span corruption', short: 'Spans', dur: 10 },
  { id: 'blocks', name: 'GPT-2 vs T5', short: 'Blocks', dur: 8 },
  { id: 'buckets', name: 'Relative position buckets', short: 'Buckets', dur: 9 },
  { id: 'bias', name: 'Learned position biases', short: 'Bias', dur: 9 },
]

const PAPER = ['Thank', 'you', 'for', 'inviting', 'me', 'to', 'your', 'party', 'last', 'week.']
/** The paper's Figure 2: spans dropped from the sentence, with the sentinel that replaces each. */
const DROPS: [number[], string][] = [[[2, 3], '<X>'], [[8], '<Y>']]
const SEQ = words(PAPER)
const H = 8

const COMPARE: Record<string, [string, string]> = {
  tasks: ['GPT-2’s output', 'anatomy/unembed'], spans: ['BERT’s masked words', 'lineage/bert?phase=mlm'],
  blocks: ['the 2017 blocks', 'lineage/transformer-2017?phase=blocks'], buckets: ['2017 sinusoids', 'lineage/transformer-2017?phase=pos'],
  bias: ['GPT-2’s scores', 'anatomy/attention?phase=scores'],
}

const CAPS: Record<string, [string, string]> = {
  tasks: ['T5 treats every task the same way: text in, text out. A short prefix names the task, and the answer is always generated as text, even a label (“negative”) or a score (“3.6”). These are real outputs of T5-small.', 'prefix: input → output text'],
  spans: ['T5 is pretrained by span corruption: about 15% of the tokens, in spans of 3 on average, are dropped and replaced by sentinel tokens, and the decoder writes out only what was dropped, each span after its sentinel. The last lines are T5-small’s real output.', 'input with sentinels → the missing spans'],
  blocks: ['T5 is an encoder–decoder like the 2017 Transformer, with changes that became standard: LayerNorm before each sub-layer and scale-only (no mean, no β: an RMSNorm), no bias terms anywhere, and position as a learned bias on the attention scores instead of added sinusoids. Click a label to jump.', '6 + 6 layers · d_model 512 · 60M'],
  buckets: ['Instead of a vector per position, T5 sorts each query–key pair by distance into one of 32 buckets: one per distance up to 7, then wider and wider, and everything from 91 on shares the last. The encoder keeps separate buckets for keys before and after the query; the decoder only looks back.', '32 buckets · exact below 8 · log-spaced to 128'],
  bias: ['Each bucket has one learned number per head, added to the attention score of every pair in that bucket, and shared by all layers. These are T5-small’s real biases: some encoder heads lean toward earlier tokens, some toward later ones, and one strongly avoids its own position. Hover the cells.', 'score = q · k + b[bucket(j − i), head]'],
}

export function mountT5(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: '',
      eyebrow: 'Lineage · Encoder–decoder',
      title: 'T5',
      subtitle: 'T5-small · every task as text in, text out',
      specs: [
        { label: 'compared', value: 'T5-small', real: 'GPT-2 small', realLabel: 'vs' },
        { label: 'layers', value: '6 enc + 6 dec', real: '12' },
        { label: 'd_model', value: '512', real: '768' },
        { label: 'heads', value: '8', real: '12' },
        { label: 'd_ff', value: '2,048 · ReLU', real: '3,072 · GELU' },
        { label: 'vocab', value: '32,128 SentencePiece', real: '50,257' },
        { label: 'params', value: '60M', real: '124M' },
      ],
    },
    size: [1040, 480],
    aria: 'T5 compared with GPT-2: an encoder–decoder that treats every task as text in and text out, pretrained by filling in dropped spans, with position given as a learned bias per bucket of relative distance.',
    phases: PHASES, learn: 't5', tokens: SEQ, compare: COMPARE, caps: CAPS,
    still: ['tasks', 11.5],
    hints: {
      blocks: 'Click a dark label on the drawing to jump to that part.',
      buckets: 'Hover a distance to see its bucket; click or tap to pin it.',
      bias: 'Hover a cell to read its bias; click or tap to pin it.',
    },
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { pill, addNode, caption, title, lane, glass, arrow } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46
  const strip = (s: string) => s.replace(/▁/g, ' ').replace('</s>', '')
  /** A sentinel: a dark pill, so it reads as a placeholder rather than a word. */
  function sentinel(x: number, y: number, label: string, a: number, h = 22) {
    ctx.font = F.mono(10.5, 500)
    const w = ctx.measureText(label).width + 14
    rr(x, y - h / 2, w, h, 5); ctx.fillStyle = rgba(C.ink, 0.85 * a); ctx.fill()
    ctx.fillStyle = rgba(C.bg, a); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(label, x + 7, y + 0.5)
    return w
  }
  /** Lay out a row of words and sentinels from x; returns the x of each item. */
  function row(items: (string | { s: string })[], x: number, y: number, a: number, hues: (number | null)[]) {
    const xs: number[] = []
    items.forEach((it, i) => {
      xs.push(x)
      if (typeof it === 'string') { const h = hues[i]; x += drawChip(x, y, { text: it, c: h ?? 0 }, a, 22) + 6 }
      else x += sentinel(x, y, it.s, a) + 6
    })
    return xs
  }

  /* ---------- scene 1: tasks as text ---------- */
  function sceneTasks(p: number) {
    const { W } = stage
    const xo = 640, rowH = 60, y0 = top + 46
    title('input · the prefix names the task', pad, top + 12, eout(clamp(p / 0.05)))
    title('output · t5-small, real', xo, top + 12, eout(clamp(p / 0.05)))
    t5.tasks.forEach((t, r) => {
      const a = eout(clamp((p - 0.03 - r * 0.14) / 0.06)), y = y0 + r * rowH
      if (a <= 0) return
      // input: the prefix, then the rest, cut to fit
      const cut = t.input.indexOf(':') + 1, pre = t.input.slice(0, cut), rest = t.input.slice(cut)
      ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a)
      ctx.fillText(pre, pad, y)
      const pw = ctx.measureText(pre).width
      ctx.font = F.mono(12); ctx.fillStyle = rgba(C.ink2, a)
      const room = Math.floor((xo - 60 - pad - pw) / ctx.measureText('m').width)
      ctx.fillText(rest.length > room ? rest.slice(0, room - 1) + '…' : rest, pad + pw, y)
      arrow([[xo - 46, y - 4], [xo - 14, y - 4]], a)
      // output: the pieces as the decoder writes them
      const n = t.out.length, shown = Math.round(clamp((p - 0.05 - r * 0.14) / 0.12) * n)
      const text = t.out.slice(0, shown).map(strip).join('').trim()
      const maxC = Math.floor((W - pad - xo) / 7.2), lines: string[] = []
      let cur = ''
      for (const w of text.split(' ')) { if ((cur + ' ' + w).trim().length > maxC) { lines.push(cur); cur = w } else cur = (cur + ' ' + w).trim() }
      if (cur) lines.push(cur)
      ctx.font = F.mono(12, 500); ctx.fillStyle = rgba(C.ink, a)
      lines.slice(0, 3).forEach((l, i) => ctx.fillText(l, xo, y + i * 16))
    })
    mk.formula = { segs: [['“translate English to German: I have seen the cat.”', C.ink2], ['  →  ', C.mute], [`“${t5.tasks[0].text}”`, C.ink]], note: 'The same weights and the same loss for every task; only the prefix changes. Compare GPT-2, which can only continue text, and BERT, which needs a new output layer per task. Greedy decoding.' }
  }

  /* ---------- scene 2: span corruption ---------- */
  function sceneSpans(p: number) {
    const x0 = pad + 110, ys = [top + 44, top + 116, top + 188], yr = [top + 290, top + 352]
    const dropped = new Set(DROPS.flatMap(([ix]) => ix))
    const ra = (t0: number) => eout(clamp((p - t0) / 0.08))
    ;['original', 'input', 'target'].forEach((nm, r) => title(nm, pad, ys[r] + 4, ra([0, 0.2, 0.36][r])))
    // original sentence, dropped spans outlined
    const xsO = row(PAPER, x0, ys[0], ra(0), PAPER.map((_, i) => i))
    const oa = ra(0.1)
    if (oa > 0) for (const [ix] of DROPS) {
      const a0 = xsO[ix[0]], a1 = xsO[ix[ix.length - 1]] + chipW(PAPER[ix[ix.length - 1]])
      ctx.setLineDash([3, 3]); ctx.strokeStyle = rgba(C.ink, 0.8 * oa); ctx.lineWidth = 1.2; rr(a0 - 4, ys[0] - 16, a1 - a0 + 8, 32, 7); ctx.stroke(); ctx.setLineDash([])
    }
    // input: sentinels where the spans were
    if (ra(0.2) > 0) {
      const items: (string | { s: string })[] = [], hues: (number | null)[] = []
      PAPER.forEach((w, i) => {
        const d = DROPS.find(([ix]) => ix[0] === i)
        if (d) { items.push({ s: d[1] }); hues.push(null) } else if (!dropped.has(i)) { items.push(w); hues.push(i) }
      })
      row(items, x0, ys[1], ra(0.2), hues)
    }
    // target: each sentinel, then the words it hides, and a final sentinel
    if (ra(0.36) > 0) {
      const items: (string | { s: string })[] = [], hues: (number | null)[] = []
      for (const [ix, s] of DROPS) { items.push({ s }); hues.push(null); for (const i of ix) { items.push(PAPER[i]); hues.push(i) } }
      items.push({ s: '<Z>' }); hues.push(null)
      row(items, x0, ys[2], ra(0.36), hues)
      caption('Raffel et al., figure 2', x0, ys[2] + 34, ra(0.36), C.mute, 'left')
    }
    // the real model on the docs example
    const qa = ra(0.58)
    if (qa > 0) {
      title('t5-small, real', pad, yr[0] - 34, qa)
      const piece = (s: string) => (/^<extra_id_\d+>$/.test(s) ? { s } : strip(s).trim())
      const srcItems = t5.spans.src.filter((s) => s !== '</s>').map(piece), outItems = t5.spans.out.filter((s) => s !== '</s>').map(piece)
      caption('in', pad, yr[0] + 4, qa, C.mute, 'left'); caption('out', pad, yr[1] + 4, ra(0.68), C.mute, 'left')
      row(srcItems, x0 - 60, yr[0], qa, srcItems.map((_, i) => i))
      if (ra(0.68) > 0) row(outItems, x0 - 60, yr[1], ra(0.68), outItems.map((_, i) => i))
    }
    mk.formula = { segs: [['input  Thank you <X> me to your party <Y> week.', C.ink2], ['     →     ', C.mute], ['target  <X> for inviting <Y> last <Z>', C.ink]], note: 'The target is short: only the dropped spans, not the whole sentence, which makes pretraining cheaper. In the vocabulary the sentinels are <extra_id_0>, <extra_id_1>, … (100 of them). Compare BERT, which predicts one token per [MASK].' }
  }

  /* ---------- scene 3: the blocks ---------- */
  type El = [kind: 'pos' | 'ln' | 'attn' | 'cross' | 'ffn' | 'add', f: number, label: string]
  const ROWS: [string, string, El[], number][] = [
    ['GPT-2', 'decoder-only · × 12', [['pos', 0.03, '+ W_P'], ['ln', 0.12, 'LN'], ['attn', 0.21, 'self · causal'], ['add', 0.3, ''], ['ln', 0.66, 'LN'], ['ffn', 0.76, 'MLP · GELU'], ['add', 0.87, '']], 5],
    ['Encoder', 'T5 · × 6', [['ln', 0.12, 'RMS'], ['attn', 0.21, 'self + bias'], ['add', 0.3, ''], ['ln', 0.66, 'RMS'], ['ffn', 0.76, 'FFN · ReLU'], ['add', 0.87, ''], ['ln', 0.95, 'RMS']], 6],
    ['Decoder', 'T5 · × 6', [['ln', 0.12, 'RMS'], ['attn', 0.21, 'causal + bias'], ['add', 0.3, ''], ['ln', 0.38, 'RMS'], ['cross', 0.47, 'cross'], ['add', 0.56, ''], ['ln', 0.66, 'RMS'], ['ffn', 0.76, 'FFN · ReLU'], ['add', 0.87, ''], ['ln', 0.95, 'RMS']], 7],
  ]
  function sceneBlocks(p: number) {
    const { W, H: Hh } = stage, avail = Hh - top - bot
    const sx0 = pad + 130, sx1 = W - pad - 16, fx = (f: number) => lerp(sx0, sx1, f)
    const ys = [top + avail * 0.14, top + avail * 0.52, top + avail * 0.87]
    const as = [eout(clamp(p / 0.25)), eout(clamp((p - 0.2) / 0.25)), eout(clamp((p - 0.3) / 0.25))]
    ROWS.forEach(([name, sub, els, lanes], r) => {
      const y = ys[r], a = as[r]
      if (a <= 0) return
      k.rowName(name, sub, pad, y, a)
      for (let i = 0; i < lanes; i++) lane(sx0, lerp(sx0, sx1, a), y + (i - (lanes - 1) / 2) * 4, hue(i), a)
      if (r) caption('no W_P', fx(0.03), y + 44, a)
      for (const [kind, f, lab] of els) {
        const x = fx(f)
        if (kind === 'add') { addNode(x, y, a); continue }
        if (kind === 'pos') {
          rr(x - 24, y - 15, 48, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.7 * a); ctx.lineWidth = 1; ctx.stroke()
          ctx.font = F.small; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a); fillRich(lab, x, y + 0.5)
          caption('learned', x, y + 44, a)
          continue
        }
        const changed = r > 0 && (kind === 'ln' || kind === 'attn')
        if (kind === 'ln') glass(x, y - 20, y + 20, changed ? 0.55 : 0.2, { w: 5, d: 6 }, a)
        else glass(x, y - 26, y + 26, changed ? 0.55 : 0.2, { w: 8, d: 9 }, a)
        caption(lab, x, y + 44, a, changed ? C.ink2 : C.mute)
      }
    })
    const ea = eout(clamp((p - 0.45) / 0.2)), yE = ys[1], yD = ys[2], xc = fx(0.47)
    if (ea > 0) {
      const yM = yE + 62
      arrow([[sx1, yE], [sx1 + 8, yE], [sx1 + 8, yM], [xc, yM], [xc, yD - 34]], ea)
      caption('encoder output → keys, values', (xc + sx1) / 2 + 40, yM + 15, ea, C.ink2)
      title('t5 · small', pad, yE - 70, ea)
      ctx.setLineDash([2, 4]); ctx.strokeStyle = rgba(C.ink, 0.2 * ea); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(pad, yE - 62); ctx.lineTo(W - pad, yE - 62); ctx.stroke(); ctx.setLineDash([])
    }
    const la = eout(clamp((p - 0.6) / 0.2))
    if (la > 0) {
      pill('buckets', fx(0.21) - 34, yE - 47, 'buckets', la)
      pill('bias', fx(0.21) + 34, yE - 47, 'bias', la)
      pill('text to text', fx(0.03) + 30, yE + 70, 'tasks', la)
    }
    mk.formula = { segs: [['GPT-2  x + f(LN(x))', C.ink2], ['     ·     ', C.mute], ['T5  x + f(RMS(x))', C.ink], ['     ·     ', C.mute], ['2017  LN(x + f(x))', C.ink2]], note: 'RMS(x) = γ ⊙ x / √(mean(x²) + ε): T5 calls it LayerNorm, but it has no mean and no β. Position enters only through the bias on the scores, so the stream carries no position vector at all.' }
  }

  /* ---------- scene 4: relative position buckets ---------- */
  const R = 64
  function sceneBuckets(p: number) {
    const { W } = stage
    const cw = 7, x0 = Math.round((W - (2 * R + 1) * cw) / 2), rows = [{ y: top + 86, bi: true }, { y: top + 206, bi: false }], ch = 24
    const reveal = eout(clamp((p - 0.04) / 0.5)) * R
    const f = mk.focus && (mk.focus.key === 'enc' || mk.focus.key === 'dec') ? mk.focus : null
    title('encoder · keys before and after the query', x0, rows[0].y - 44, eout(clamp(p / 0.06)))
    title('decoder · keys before the query only', x0, rows[1].y - 44, eout(clamp((p - 0.1) / 0.06)))
    rows.forEach(({ y, bi }, r) => {
      const a = eout(clamp((p - r * 0.1) / 0.06))
      if (a <= 0) return
      let runStart = -R
      for (let d = -R; d <= R; d++) {
        const x = x0 + (d + R) * cw
        if (Math.abs(d) > reveal) continue
        if (!bi && d > 0) { mk.hatch(x, y, cw, a); continue }
        const b = bucket(d, bi)
        ctx.fillStyle = rgba(C.ink, (b % 2 ? 0.42 : 0.16) * a); ctx.fillRect(x, y, cw - (d < R && bucket(d + 1, bi) !== b ? 1 : 0), ch)
        // label each bucket run that is wide enough
        const next = d < R ? bucket(d + 1, bi) : -1
        if (next !== b || d === R || (!bi && d === 0)) {
          const wRun = (d - runStart + 1) * cw
          if (wRun >= 14) { ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(String(b), x0 + ((runStart + d + 1) / 2 + R) * cw, y - 7) }
          runStart = d + 1
        }
      }
      // the query itself
      ctx.strokeStyle = rgba(C.ink, a); ctx.lineWidth = 1.5; ctx.strokeRect(x0 + R * cw - 0.5, y - 1.5, cw + 1, ch + 3)
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, a)
      for (const d of [-64, -32, -8, 0, 8, 32, 64]) if (bi || d <= 0) ctx.fillText(String(d).replace('-', '−'), x0 + (d + R + 0.5) * cw, y + ch + 6)
      mk.hit(bi ? 'enc' : 'dec', { x: x0, y, c: cw }, 1, 2 * R + 1)
    })
    caption('key position − query position', W / 2, rows[1].y + ch + 40, eout(clamp((p - 0.2) / 0.1)), C.mute)
    // the ranges, one direction
    const la = eout(clamp((p - 0.6) / 0.12))
    if (la > 0) {
      const ranges: string[] = []
      let s = 0
      for (let d = 1; d <= 130; d++) if (bucket(d, true) !== bucket(d - 1, true) || d === 130) { const e = d - 1; ranges.push(s === e ? `${s}` : e >= 129 ? `${s}+` : `${s}–${e}`); s = d }
      caption(`distances per bucket (each way): ${ranges.join(' · ')}`, W / 2, rows[1].y + ch + 70, la, C.ink2)
    }
    if (f) {
      const d = f.j - R, bi = f.key === 'enc'
      ctx.strokeStyle = rgba(C.ink, 1); ctx.lineWidth = 2; ctx.strokeRect(x0 + f.j * cw - 1, rows[bi ? 0 : 1].y - 1, cw + 2, ch + 2)
      mk.formula = !bi && d > 0
        ? { segs: [[`decoder  j − i = ${d}`, C.ink], ['  →  masked', C.ink2]], note: 'A later token: the decoder may not look at it.' }
        : { segs: [[`${bi ? 'encoder' : 'decoder'}  j − i = ${d}`, C.ink], ['  →  bucket ', C.mute], [String(bucket(d, bi)), C.ink]], note: `Every pair of tokens this far apart ${d > 0 ? '(key after the query) ' : d < 0 ? '(key before the query) ' : ''}shares this bucket, and so the same learned bias.` }
    } else mk.formula = { segs: [['bucket(d) = |d| if |d| < 8, else 8 + ⌊log(|d| / 8) / log(128 / 8) × 8⌋, at most 15', C.ink], ['     ·     ', C.mute], ['+ 16 if the key is after the query', C.ink2]], note: 'For the encoder. The decoder uses all 32 buckets for keys before the query: exact up to 15, log-spaced up to 128. Beyond 128 everything shares the last bucket, so T5 can run on inputs longer than it was trained on.' }
  }

  /* ---------- scene 5: the learned biases ---------- */
  const SPAN = 24
  const leanOf = (B: number[][], bi: boolean, h: number) => {
    const at = (d: number) => B[bucket(d, bi)][h]
    const left = [1, 2, 3, 4].reduce((s, d) => s + at(-d), 0) / 4, right = bi ? [1, 2, 3, 4].reduce((s, d) => s + at(d), 0) / 4 : left
    if (at(0) < -8) return 'avoids itself'
    if (bi && left - right > 1.5) return 'leans earlier'
    if (bi && right - left > 1.5) return 'leans later'
    if (!bi && at(-1) > at(-8) + 2) return 'favours recent'
    return 'broad'
  }
  function sceneBias(p: number) {
    const { W } = stage
    const c = 15, n = 2 * SPAN + 1, x0 = Math.round((W - n * c) / 2) - 30
    const blocks = [{ key: 'be', B: t5.bias.encoder, bi: true, y: top + 52, name: 'encoder, 8 heads' }, { key: 'bd', B: t5.bias.decoder, bi: false, y: top + 52 + H * c + 64, name: 'decoder, 8 heads' }]
    const f = mk.focus && (mk.focus.key === 'be' || mk.focus.key === 'bd') ? mk.focus : null
    blocks.forEach(({ key, B, bi, y, name }, r) => {
      const a = eout(clamp((p - r * 0.25) / 0.12))
      if (a <= 0) return
      title(name, x0, y - 30, a)
      const vals = Array.from({ length: H }, (_, h) => Array.from({ length: n }, (_, j) => B[bucket(j - SPAN, bi)][h]))
      for (let h = 0; h < H; h++) {
        for (let j = 0; j < n; j++) {
          const x = x0 + j * c, yy = y + h * c
          if (!bi && j > SPAN) { mk.hatch(x, yy, c, a); continue }
          mk.paintCell(x, yy, c, vals[h][j], 6, null, 0.85, a)
        }
        ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(f?.key === key && f.i === h ? C.ink : C.mute, a); ctx.fillText(`h${h + 1}`, x0 - 8, y + (h + 0.5) * c)
        const la = eout(clamp((p - 0.55 - r * 0.1) / 0.1))
        if (la > 0) { ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink2, la * a); ctx.fillText(leanOf(B, bi, h), x0 + n * c + 12, y + (h + 0.5) * c) }
      }
      ctx.strokeStyle = rgba(C.ink, 0.22 * a); ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y + 0.5, n * c - 1, H * c - 1)
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
      for (const d of [-24, -12, -1, 0, 1, 12, 24]) if (bi || d <= 0) ctx.fillText(String(d).replace('-', '−'), x0 + (d + SPAN + 0.5) * c, y - 6)
      mk.hit(key, { x: x0, y, c }, H, n)
      if (f?.key === key) { ctx.strokeStyle = rgba(C.ink, 1); ctx.lineWidth = 2; ctx.strokeRect(x0 + f.j * c, y + f.i * c, c, c) }
    })
    if (f) {
      const blk = blocks.find((b) => b.key === f.key)!, d = f.j - SPAN
      mk.formula = !blk.bi && d > 0
        ? { segs: [[`decoder head ${f.i + 1}, j − i = ${d}`, C.ink], ['  →  masked', C.ink2]], note: 'Later tokens are hidden from the decoder.' }
        : { segs: [[`${blk.bi ? 'encoder' : 'decoder'} head ${f.i + 1}, j − i = ${d}`, C.ink], ['  →  bucket ', C.mute], [String(bucket(d, blk.bi)), C.ink2], ['  →  b = ', C.mute], [fmtF(blk.B[bucket(d, blk.bi)][f.i]), C.ink]], note: `Added to this head’s score for every pair at this distance${Math.abs(d) >= 8 ? ' (and to all distances in the same bucket)' : ''}, in all 6 layers. Real T5-small weights.` }
    } else mk.formula = { segs: [['score(i, j) = q_i · k_j + b[bucket(j − i), head]', C.ink]], note: 'Filled cells are positive (the head is drawn toward that distance), outlined cells negative. Real T5-small weights, from the first layer, where T5 computes the bias once and reuses it in every layer.' }
  }

  return { tasks: sceneTasks, spans: sceneSpans, blocks: sceneBlocks, buckets: sceneBuckets, bias: sceneBias }
}
