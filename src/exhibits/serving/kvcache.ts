import { F, drawChip, rr, serifAt, type TokLike } from '../../core/draw'
import { fmt, fmtF, gemm, type Rect } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp } from '../../core/util'
import { kvSlice } from '../../lib/gpt2/data'
import { mountExhibit, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * The KV cache: why generation keeps every earlier token's keys and values, how prefill fills the cache
 * and a decode step reads it, how large it grows, and why decoding is memory-bound. The q, k, v rows
 * are real GPT-2 small numbers for one head (exported offline); the rest is arithmetic on real sizes.
 */

const PHASES = [
  { id: 'loop', name: 'One token at a time', short: 'Why cache', dur: 9 },
  { id: 'prefill', name: 'Prefill', dur: 8 },
  { id: 'step', name: 'A decode step', short: 'Decode', dur: 13 },
  { id: 'size', name: 'How big the cache gets', short: 'Size', dur: 8 },
  { id: 'bound', name: 'Compute-bound and memory-bound', short: 'Roofline', dur: 10 },
]

const KV = kvSlice(), N = KV.ids.length, P = N - 1, SHOW = 8, DH = 64
const TOKS: TokLike[] = KV.syms.map((s, i) => ({ text: s.replace(/^Ġ/, ' '), c: i }))
const cut = (m: number[][]) => m.map((r) => r.slice(0, SHOW))
const Kt = Array.from({ length: SHOW }, (_, d) => KV.k.map((r) => r[d])) // Kᵀ, 8 × N
const q = KV.q[N - 1]
const S = KV.k.map((k) => k.reduce((s, v, d) => s + v * q[d], 0))
const W = (() => { const z = S.map((v) => v / Math.sqrt(DH)), m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), t = e.reduce((a, b) => a + b, 0); return e.map((v) => v / t) })()
const OUT = Array.from({ length: SHOW }, (_, d) => W.reduce((s, w, j) => s + w * KV.v[j][d], 0))

/** KV cache per token, in KiB, 16-bit numbers. */
const kib = (layers: number, kvHeads: number, dHead: number) => (2 * layers * kvHeads * dHead * 2) / 1024
const MODELS: [string, number, string][] = [
  ['GPT-2 small', kib(12, 12, 64), '12 layers · 12 heads × 64'],
  ['LLaMA 3 8B', kib(32, 8, 128), '32 layers · 8 kv heads × 128'],
  ['LLaMA 3 70B', kib(80, 8, 128), '80 layers · 8 kv heads × 128'],
  ['DeepSeek-V3', ((512 + 64) * 61 * 2) / 1024, '61 layers · MLA latent 576'],
]

const COMPARE: Record<string, [string, string]> = {
  loop: ['GPT-2’s sampling loop', 'anatomy/unembed?phase=sample'], prefill: ['GPT-2’s q, k, v', 'anatomy/attention?phase=qkv'], step: ['GPT-2’s scores', 'anatomy/attention?phase=scores'],
  size: ['LLaMA’s GQA', 'lineage/llama?phase=gqa'], bound: ['Mixtral’s active parameters', 'lineage/mixtral?phase=params'],
}

const CAPS: Record<string, [string, string]> = {
  loop: ['GPT-2 writes one token at a time, and each new token attends to every earlier one. Those tokens’ keys and values never change once computed, so a serving system keeps them in a KV cache instead of recomputing the whole prefix at every step.', 'K, V rows computed: n(n + 1) / 2 → n'],
  prefill: ['The prompt is processed in one pass: all its tokens go through each layer together, as matrix products, and every layer writes their keys and values to the cache. These are real GPT-2 values for one head (the first 8 of its 64 numbers).', 'K, V = X · W_K, X · W_V for the whole prompt'],
  step: [`Each decode step runs one token through the model. It computes its own q, k and v, appends k and v to the cache, and attends over all the cached keys: one row of scores instead of a matrix. Real GPT-2 numbers from layer ${KV.layer + 1}, head ${KV.head + 1}; hover the cells.`, 'q · Kᵀ → softmax → · V'],
  size: ['The cache holds two vectors (K and V) per token, per layer, per key/value head. It grows with every token of every sequence being served, and at long contexts it outgrows the model’s weights; grouped-query attention, latent attention and paging all attack it.', '2 × layers × kv heads × d_head × 2 bytes'],
  bound: ['Prefill does many operations for each byte it reads, so it is limited by compute. A decode step reads every weight, and the cache, to produce a single token, so it is limited by memory bandwidth; serving many sequences in one batch is how a GPU’s compute gets used.', 'FLOPs per byte ≈ tokens per weight read'],
}

export function mountKvCache(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: '',
      eyebrow: 'Serving · Memory',
      title: 'KV Cache',
      subtitle: 'prefill once, then one token per step',
      specs: [
        { label: 'model', value: 'GPT-2 small' },
        { label: 'per token', value: '36 KiB', real: '2 × 12 × 768 × 2 B', realLabel: '' },
        { label: 'at 1,024 tokens', value: '36 MiB' },
        { label: 'head shown', value: `layer ${KV.layer + 1} · head ${KV.head + 1}`, real: '64 dims, 8 drawn', realLabel: '' },
      ],
    },
    size: [1040, 480],
    aria: 'The KV cache: generation keeps the keys and values of all earlier tokens so each new token computes only its own; prefill fills the cache in one pass and each decode step attends over it. Real GPT-2 numbers for one attention head.',
    phases: PHASES, learn: 'kvcache', tokens: TOKS, compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['step', 12],
    hints: {
      prefill: 'Hover a cell of K or V to read it; click or tap to pin it.',
      step: 'Hover a score or an output cell to see how it is computed; click or tap to pin it.',
    },
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { caption, title, arrow } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46

  /* ---------- scene 1: recomputing vs caching ---------- */
  /** A generic run: a 3-token prompt, then 7 generated tokens. */
  const GP = 3, GN = 10
  function sceneLoop(p: number) {
    const { W: Wd } = stage
    const c = 30, rowsN = GN - GP + 1, gy = top + 70, gx = [pad + 130, Wd / 2 + 110]
    const shown = Math.min(rowsN, Math.floor(clamp((p - 0.05) / 0.6) * rowsN) + 1)
    ;['without a cache', 'with a kv cache'].forEach((name, g) => {
      const x0 = gx[g]
      title(name, x0, gy - 34, 1)
      for (let j = 0; j < GN; j++) { ctx.fillStyle = rgba(hue(j), 0.9); ctx.beginPath(); ctx.arc(x0 + (j + 0.5) * c, gy - 12, 5, 0, 7); ctx.fill() }
      let count = 0
      for (let r = 0; r < rowsN; r++) {
        const y = gy + r * c, a = r < shown ? 1 : 0.12, last = r === 0 ? GP - 1 : GP - 1 + r
        if (g === 0) { ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(r === 0 ? 'prefill' : `step ${r}`, x0 - 10, y + c / 2) }
        for (let j = 0; j <= last; j++) {
          // prefill computes the prompt; each step adds one token, and without a cache redoes all before it
          const computed = r === 0 || g === 0 || j === last, x = x0 + j * c
          if (computed) { ctx.fillStyle = rgba(hue(j), 0.75 * a); ctx.fillRect(x + 2, y + 2, c - 4, c - 4); if (r < shown) count++ }
          else { ctx.strokeStyle = rgba(hue(j), 0.6 * a); ctx.lineWidth = 1; ctx.strokeRect(x + 2.5, y + 2.5, c - 5, c - 5) }
        }
      }
      ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, 1)
      ctx.fillText(`${count} K, V rows computed`, x0, gy + rowsN * c + 30)
      caption(g === 0 ? 'every step redoes the whole prefix' : 'filled: computed · outlined: read from the cache', x0, gy + rowsN * c + 50, 1, C.mute, 'left')
    })
    const la = eout(clamp((p - 0.7) / 0.12))
    caption('for 1,024 tokens: 524,800 rows without a cache, 1,024 with one, in every layer and head', Wd / 2, top + 10, la, C.ink2)
    mk.formula = { segs: [['without  Σ n = n(n + 1) / 2', C.ink2], ['     ·     ', C.mute], ['with  n', C.ink]], note: 'Counting the key and value rows each layer must compute, for a 3-token prompt and 7 new tokens. The prompt is one pass either way; the difference is in the generated tokens.' }
  }

  /* ---------- scene 2: prefill ---------- */
  function scenePrefill(p: number) {
    const c = 24, x0 = pad + 110, y0 = top + 90, fin = eout(clamp(p / 0.08))
    const rows = eio(clamp((p - 0.1) / 0.4)) * P
    title('prompt', pad, y0 - 22, fin)
    TOKS.slice(0, P).forEach((t, i) => drawChip(pad, y0 + (i + 0.5) * c, t, fin, 18))
    const rK: Rect = { x: x0 + 90, y: y0, c }, rV: Rect = { x: rK.x + SHOW * c + 60, y: y0, c }
    arrow([[x0 - 6, y0 + (P * c) / 2], [rK.x - 14, y0 + (P * c) / 2]], fin)
    caption('· W_K, · W_V', (x0 + rK.x) / 2 - 8, y0 + (P * c) / 2 - 12, fin, C.ink2)
    const rev = (i: number) => clamp(rows - i)
    mk.drawMat({ r: rK, vals: cut(KV.k.slice(0, P)), kind: 'row', alpha: fin, name: 'K', shape: '5 × 8', real: '5 × 64', reveal: (i) => rev(i), noText: true })
    mk.drawMat({ r: rV, vals: cut(KV.v.slice(0, P)), kind: 'row', alpha: fin, name: 'V', shape: '5 × 8', real: '5 × 64', reveal: (i) => rev(i), noText: true })
    mk.hit('k', rK, P, SHOW); mk.hit('v', rV, P, SHOW)
    // the cache, drawn as one slab per layer
    const ca = eout(clamp((p - 0.55) / 0.12)), cx = rV.x + SHOW * c + 80, cy = y0 - 10
    if (ca > 0) {
      title('the cache', cx, cy - 12, ca)
      for (let l = 11; l >= 0; l--) {
        const x = cx + l * 5, y = cy + l * 5, on = l === KV.layer
        rr(x, y, 110, 120, 4); ctx.fillStyle = rgba(C.bg, ca); ctx.fill(); ctx.strokeStyle = rgba(C.ink, (on ? 0.9 : 0.25) * ca); ctx.lineWidth = 1; ctx.stroke()
      }
      caption('12 layers × 12 heads', cx, cy + 12 * 5 + 150, ca, C.ink2, 'left')
      caption('K and V for each prompt token', cx, cy + 12 * 5 + 166, ca, C.mute, 'left')
      arrow([[rV.x + SHOW * c + 12, y0 + (P * c) / 2], [cx - 8, y0 + (P * c) / 2]], ca)
    }
    const f = mk.focus && (mk.focus.key === 'k' || mk.focus.key === 'v') ? mk.focus : null
    if (f) {
      const m = f.key === 'k' ? KV.k : KV.v
      mk.formula = { segs: [[`${f.key.toUpperCase()}[${TOKS[f.i].text.trim()}, ${f.j}]`, C.ink], ['  =  ', C.mute], [fmtF(m[f.i][f.j]), C.ink]], note: `Real GPT-2: layer ${KV.layer + 1}, head ${KV.head + 1}, number ${f.j + 1} of 64.` }
    } else mk.formula = { segs: [['K = LN(X) · W_K + b', C.ink2], ['     ·     ', C.mute], ['V = LN(X) · W_V + b', C.ink2], ['     ·     ', C.mute], ['for all 5 prompt tokens at once', C.ink]], note: 'Prefill is one ordinary forward pass over the prompt; the only extra work is keeping K and V. It also produces the first new token.' }
  }

  /* ---------- scene 3: a decode step, with real numbers ---------- */
  function sceneStep(p: number) {
    const c = 22, ry = top + SHOW * c + 60
    const rQ: Rect = { x: pad + 110, y: ry, c }, rS: Rect = { x: rQ.x + SHOW * c + 40, y: ry, c }, rKt: Rect = { x: rS.x, y: ry - SHOW * c - 26, c }
    const rW: Rect = { x: rS.x + N * c + 60, y: ry, c }, rO: Rect = { x: rW.x + N * c + 40, y: ry, c }, rV: Rect = { x: rO.x, y: ry - N * c - 26, c }
    const fin = eout(clamp(p / 0.06)), app = eio(clamp((p - 0.08) / 0.14))
    const g1 = gemm(clamp((p - 0.24) / 0.24), 1, N, SHOW, 'fast'), sm = eio(clamp((p - 0.5) / 0.08)), g2 = gemm(clamp((p - 0.6) / 0.28), 1, SHOW, N, 'fast')
    drawChip(pad, ry + c / 2, TOKS[P], fin, 20, true)
    caption('new token', pad, ry + c + 20, fin, C.mute, 'left')
    mk.drawMat({ r: rQ, vals: [q.slice(0, SHOW)], kind: 'row', rowCols: [hue(P)], alpha: fin, name: 'q', shape: '1 × 8', real: '1 × 64', label: 'bottom', noText: true, labelW: SHOW * c + 30 })
    // Kᵀ: the cached columns, then the new column sliding in
    mk.drawMat({ r: rKt, vals: Kt.map((r) => r.slice(0, SHOW)), kind: 'col', alpha: fin, name: '', shape: '', label: 'none', noText: true, reveal: (_, j) => (j < P ? 1 : app) })
    mk.label('Kᵀ', '8 × 6', '64 × 6', rKt.x - 176, rKt.y + 18, fin)
    caption('the cached keys', rKt.x - 176, rKt.y + 38, fin, C.mute, 'left')
    TOKS.forEach((t, j) => {
      ctx.save(); ctx.translate(rKt.x + (j + 0.5) * c, rKt.y - 6); ctx.rotate(-Math.PI / 3)
      ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(hue(j), fin * (j < P ? 1 : app)); ctx.fillText(t.text.trim(), 0, 0); ctx.restore()
    })
    if (app > 0 && app < 1) { ctx.strokeStyle = rgba(hue(P), 1); ctx.lineWidth = 2; ctx.strokeRect(rKt.x + P * c, rKt.y - 30 * (1 - app), c, SHOW * c) }
    mk.drawMat({ r: rS, vals: [S], kind: 'score', alpha: fin, name: 's', shape: '1 × 6', label: 'bottom', reveal: g1.rev, noText: true })
    mk.hit('s', rS, 1, N)
    if (sm > 0) {
      arrow([[rS.x + N * c + 8, ry + c / 2], [rW.x - 8, ry + c / 2]], sm)
      caption('÷ 8, softmax', (rS.x + N * c + rW.x) / 2, ry - 8, sm, C.ink2)
      mk.drawMat({ r: rW, vals: [W], kind: 'attn', alpha: fin * sm, name: 'w', shape: '1 × 6', label: 'bottom', noText: true })
      mk.hit('w', rW, 1, N)
    }
    const va = fin * eout(clamp((p - 0.56) / 0.06))
    mk.drawMat({ r: rV, vals: cut(KV.v), kind: 'row', alpha: va, name: 'V', shape: '6 × 8', real: '6 × 64', noText: true, reveal: (i) => (i < P ? 1 : app) })
    TOKS.forEach((t, i) => { ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(hue(i), va); ctx.fillText(t.text.trim(), rV.x - 8, rV.y + (i + 0.5) * c) })
    mk.drawMat({ r: rO, vals: [OUT], kind: 'row', rowCols: [hue(P)], alpha: va, name: 'out', shape: '1 × 8', real: '1 × 64', label: 'bottom', reveal: g2.rev, noText: true, labelW: 200 })
    mk.hit('o', rO, 1, SHOW)
    caption('← appended', rKt.x + N * c + 10, rKt.y + (SHOW * c) / 2, fin * app, C.ink2, 'left')
    const fs = mk.resolve({ s: { g: g1, K: SHOW }, o: { g: g2, K: N } })
    const f = mk.focus
    if (fs?.key === 's') mk.gemmOverlay({ A: rQ, Av: [q.slice(0, SHOW)], B: rKt, Bv: Kt.map((r) => r.slice(0, SHOW)), C: rS, f: fs, names: ['s', 'q', 'Kᵀ'], note: `The new token’s query against the key of ‘${TOKS[fs.j].text.trim()}’, read from the cache. Real GPT-2, all 64 numbers counted.`, more: { n: DH - SHOW, sum: (_, j) => KV.k[j].slice(SHOW).reduce((s, v, d) => s + v * q[SHOW + d], 0) }, fmtC: fmtF })
    else if (fs?.key === 'o') mk.gemmOverlay({ A: rW, Av: [W], B: rV, Bv: cut(KV.v), C: rO, f: fs, names: ['out', 'w', 'V'], note: 'Each cached value row, weighted by how much the new token attends to it. Real GPT-2.', fmtC: fmtF })
    else if (f?.key === 'w') {
      ctx.strokeStyle = rgba(C.ink, 1); ctx.lineWidth = 2; ctx.strokeRect(rW.x + f.j * c, rW.y, c, c)
      mk.formula = { segs: [[`w[${TOKS[f.j].text.trim()}]`, C.ink], ['  =  ', C.mute], [`softmax(s / 8)`, C.mute], ['  =  ', C.mute], [fmtF(W[f.j]), C.ink]], note: `GPT-2’s own attention weight here is ${fmtF(KV.attn[f.j])}. This head spreads its attention over the whole sentence.` }
    } else mk.formula = { segs: [['s = q · Kᵀ', C.ink2], ['   ·   ', C.mute], ['w = softmax(s / √64)', C.ink2], ['   ·   ', C.mute], ['out = w · V', C.ink]], note: 'The new token computes one q, k and v per head; everything else is read from the cache. The work grows with the length of the cache, not with its square.' }
  }

  /* ---------- scene 4: size ---------- */
  function sceneSize(p: number) {
    const { W: Wd } = stage
    const x0 = pad + 200, w = Wd - pad - x0 - 270, max = 320, rowH = 62
    title('kv cache per token, 16-bit', x0, top + 18, 1)
    title('one 128K-token sequence', Wd - pad - 150, top + 18, 1)
    MODELS.forEach(([name, v, sub], r) => {
      const a = eout(clamp((p - 0.05 - r * 0.12) / 0.12)), g = eio(clamp((p - 0.08 - r * 0.12) / 0.2)), y = top + 44 + r * rowH
      if (a <= 0) return
      ctx.font = serifAt(20); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(name, pad, y + 18)
      caption(sub, pad, y + 36, a, C.mute, 'left')
      const bw = Math.max(2, (v / max) * w * g)
      ctx.fillStyle = rgba(C.ink, 0.7 * a); ctx.fillRect(x0, y + 4, bw, 20)
      ctx.font = F.mono(12); ctx.fillStyle = rgba(C.ink2, a * g); ctx.fillText(`${fmt(v)} KiB`, x0 + bw + 8, y + 19)
      const gib = (v * 131072) / 2 ** 20
      ctx.fillText(name === 'GPT-2 small' ? 'max 1,024 tokens' : `${gib >= 10 ? Math.round(gib) : gib.toFixed(1)} GiB`, Wd - pad - 150, y + 19)
    })
    const la = eout(clamp((p - 0.62) / 0.1))
    caption('LLaMA 3 70B’s weights take 140 GB in 16-bit; its cache for one 128K-token sequence takes 40 GiB,', x0, top + 44 + MODELS.length * rowH + 20, la, C.ink2, 'left')
    caption('and a server holds many sequences at once', x0, top + 44 + MODELS.length * rowH + 38, la, C.ink2, 'left')
    mk.formula = { segs: [['bytes per token = 2 × layers × kv heads × d_head × 2', C.ink], ['     ·     ', C.mute], ['GPT-2 small: 2 × 12 × 12 × 64 × 2 = 36 KiB', C.ink2]], note: 'The 2 in front is for K and V; the last 2 is bytes per 16-bit number. Grouped-query attention cuts the kv heads, latent attention replaces them with one small latent, quantization cuts the bytes.' }
  }

  /* ---------- scene 5: the roofline ---------- */
  function sceneBound(p: number) {
    const { W: Wd, H } = stage, avail = H - top - bot
    const cx0 = pad + 70, cx1 = Wd - pad - 250, cy0 = top + 30, cy1 = top + avail - 40
    const lx0 = Math.log(0.5), lx1 = Math.log(4096), ly0 = Math.log(0.5), ly1 = Math.log(600)
    const X = (v: number) => lerp(cx0, cx1, (Math.log(v) - lx0) / (lx1 - lx0)), Y = (v: number) => lerp(cy1, cy0, (Math.log(v) - ly0) / (ly1 - ly0))
    const PEAK = 312, BW = 2.0, fin = eout(clamp(p / 0.08))
    title('attainable tflop/s · a100 80 gb', cx0, cy0 - 12, fin)
    ctx.strokeStyle = rgba(C.ink, 0.25 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy1); ctx.lineTo(cx1, cy1); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, fin)
    for (const v of [1, 10, 100]) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(v), cx0 - 6, Y(v)) }
    for (const v of [1, 4, 16, 64, 256, 1024, 4096]) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(v.toLocaleString('en-US'), X(v), cy1 + 6) }
    caption('FLOPs per byte read', (cx0 + cx1) / 2, cy1 + 32, fin)
    // the roof: bandwidth-limited slope, then the compute ceiling
    const ra = eio(clamp((p - 0.06) / 0.25)), ridge = PEAK / BW
    ctx.strokeStyle = rgba(C.ink, 0.9 * fin); ctx.lineWidth = 2; ctx.beginPath()
    for (let s = 0; s <= 60 * ra; s++) { const v = Math.exp(lerp(lx0, lx1, s / 60)), y = Math.min(PEAK, BW * v); if (s) ctx.lineTo(X(v), Y(y)); else ctx.moveTo(X(v), Y(y)) }
    ctx.stroke()
    if (ra >= 1) { caption('2.0 TB/s × FLOPs per byte', X(6), Y(BW * 6) + 30, fin, C.ink2, 'left'); caption(`312 TFLOP/s (from ${Math.round(ridge)} FLOPs per byte)`, X(ridge) + 6, Y(PEAK) - 10, fin, C.ink2, 'left') }
    const pts: [string, number, number][] = [['decode, batch 1', 1, 0.35], ['decode, batch 32', 32, 0.5], ['prefill, 512 tokens', 512, 0.65]]
    pts.forEach(([name, v, t0]) => {
      const a = eout(clamp((p - t0) / 0.1))
      if (a <= 0) return
      const y = Math.min(PEAK, BW * v)
      ctx.fillStyle = rgba(C.tok[4], a); ctx.beginPath(); ctx.arc(X(v), Y(y), 5, 0, 7); ctx.fill()
      caption(`${name}: ${y < 10 ? y.toFixed(0) : Math.round(y)} TFLOP/s`, X(v) + 10, Y(y) + 22, a, C.ink, 'left')
    })
    const na = eout(clamp((p - 0.8) / 0.1)), nx = cx1 + 30, ny = cy0 + 150
    caption('one decode step at batch 1 uses', nx, ny, na, C.ink2, 'left')
    caption('under 1% of the GPU’s arithmetic:', nx, ny + 16, na, C.ink2, 'left')
    caption('it waits on memory, reading all', nx, ny + 32, na, C.ink2, 'left')
    caption('the weights for one token', nx, ny + 48, na, C.ink2, 'left')
    mk.formula = { segs: [['FLOPs per byte ≈ (2 FLOPs × tokens) / (2 bytes) per weight = tokens per weight read', C.ink]], note: 'A simple model of the weight matrix products in 16-bit, ignoring the cache reads (which make decode even more memory-bound) and attention’s own FLOPs. The ridge where the two limits meet is at 312 / 2.0 = 156 FLOPs per byte.' }
  }

  return { loop: sceneLoop, prefill: scenePrefill, step: sceneStep, size: sceneSize, bound: sceneBound }
}
