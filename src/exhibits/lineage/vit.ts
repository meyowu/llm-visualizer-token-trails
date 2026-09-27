import { F, fillRich, mathRun, rr, type TokLike } from '../../core/draw'
import { fmtF, gemm, type Rect } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, gauss, lerp, rng } from '../../core/util'
import { GRID, explained, filters, posSim } from '../../lib/vit/data'
import type { Nav } from '../registry'
import { mountExhibit, type Env } from '../kit'

/*
 * ViT-B/16 as a diff against GPT-2: the same width and depth, no mask, and tokens that are image
 * patches. The image and the patch-embedding product are toy; the position-embedding similarities and
 * the filter components are real (google/vit-base-patch16-224, exported offline).
 */

const PHASES = [
  { id: 'blocks', name: 'GPT-2 vs ViT', short: 'Blocks', dur: 7 },
  { id: 'patches', name: 'An image as a sequence', short: 'Patches', dur: 9 },
  { id: 'embed', name: 'Patch embedding', short: 'x · W', dur: 10 },
  { id: 'positions', name: '[CLS] and positions', short: 'Positions', dur: 11 },
  { id: 'head', name: 'One label for the image', short: 'Head', dur: 8 },
]

/** The toy image: 56 × 56 grey levels, 4 per patch side, so the 14 × 14 patch grid matches ViT's. */
const PX = 56, PP = PX / GRID
const IMG = (() => {
  const v = new Float32Array(PX * PX)
  for (let y = 0; y < PX; y++) for (let x = 0; x < PX; x++) {
    const u = (x + 0.5) / PX, w = (y + 0.5) / PX
    let g = 0.1 + 0.12 * (1 - w)
    if (Math.hypot(u - 0.74, w - 0.24) < 0.1) g = 0.95
    if (w > 0.64 + 0.07 * Math.sin(u * 6)) g = 0.42
    if (w > 0.78 + 0.05 * Math.cos(u * 9 + 1)) g = 0.28
    if (u > 0.2 && u < 0.44 && w > 0.5 && w < 0.8) g = 0.72
    if (w > 0.36 && w <= 0.5 && Math.abs(u - 0.32) < (w - 0.36) * 1.0) g = 0.58
    if (u > 0.29 && u < 0.35 && w > 0.66 && w < 0.8) g = 0.18
    v[y * PX + x] = g
  }
  return v
})()
/** Patch p's grey levels, row by row (PP × PP). */
const patchVals = (p: number) => { const r = Math.floor(p / GRID), c = p % GRID, out: number[] = []; for (let y = 0; y < PP; y++) for (let x = 0; x < PP; x++) out.push(IMG[(r * PP + y) * PX + c * PP + x]); return out }

/** Toy patch embedding: 6 patches × 16 values · W (16 × 8). */
const PICK = (() => {
  // the six patches with the most contrast (edges of the sun, roof, door and hills), in reading order
  const variance = (q: number) => { const v = patchVals(q), m = v.reduce((a, b) => a + b, 0) / v.length; return v.reduce((a, b) => a + (b - m) ** 2, 0) }
  const order = Array.from({ length: GRID * GRID }, (_, q) => q).sort((a, b) => variance(b) - variance(a)), out: number[] = []
  for (const q of order) if (out.length < 6 && out.every((o) => Math.abs(o - q) > 2)) out.push(q)
  return out.sort((a, b) => a - b)
})()
const EMB = (() => {
  const r = rng(16), X = PICK.map((p) => patchVals(p).map((v) => v - 0.4))
  const Wp = Array.from({ length: PP * PP }, () => Array.from({ length: 8 }, () => gauss(r) * 0.5))
  return { X, Wp, E: X.map((x) => Wp[0].map((_, j) => x.reduce((s, v, k) => s + v * Wp[k][j], 0))) }
})()
const SEQ: TokLike[] = PICK.map((p, i) => ({ text: `p${p + 1}`, c: i }))

const COMPARE: Record<string, [string, string]> = {
  blocks: ['GPT-2’s block', 'anatomy/layernorm?phase=stream'], patches: ['GPT-2’s tokenizer', 'anatomy/tokenizer'], embed: ['GPT-2’s embedding', 'anatomy/embedding'],
  positions: ['GPT-2’s positions', 'anatomy/embedding?phase=pos'], head: ['GPT-2’s output', 'anatomy/unembed'],
}

const CAPS: Record<string, [string, string]> = {
  blocks: ['ViT-B/16 has GPT-2 small’s width and depth, arranged like BERT (no mask), but its tokens are image patches: the vocabulary lookup becomes one matrix product on raw pixels, and the output is one class for the whole image, read from a [CLS] token. Click a label to jump.', '12 × 768 · 86M · 197 tokens'],
  patches: ['A 224 × 224 image is cut into a 14 × 14 grid of 16 × 16 patches. Read row by row, the 196 patches become the token sequence; each one is 16 × 16 × 3 = 768 numbers of red, green and blue. (A toy grey image here.)', '224² → 14 × 14 patches → 196 tokens'],
  embed: ['Each flattened patch is multiplied by one learned matrix, the same for every patch. This replaces GPT-2’s vocabulary lookup and equals a 16 × 16 convolution with stride 16. On the right, the top principal components of ViT-B/16’s real 768 filters.', 'x = patches · W_patch (768 × 768)'],
  positions: ['A learned [CLS] token goes in front, and each of the 197 slots gets a learned position vector, a plain 1D list. Yet in the real model each patch position’s vector is most like its neighbours’ and its own row’s and column’s: training rediscovered the 2D grid. Hover a small map.', 'x = [CLS; patches] + P (197 × 768)'],
  head: ['All 197 tokens pass through 12 encoder layers, every patch attending to every patch. For classification only the [CLS] output is used, mapped to 1,000 ImageNet classes. ViT needed very large pretraining sets (up to 300M images) to beat convolutional networks.', 'class = head(LN(x_CLS))'],
}

export function mountVit(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: '',
      eyebrow: 'Lineage · Vision & diffusion',
      title: 'Vision Transformer',
      subtitle: 'ViT-B/16 · image patches as tokens',
      specs: [
        { label: 'compared', value: 'ViT-B/16', real: 'GPT-2 small', realLabel: 'vs' },
        { label: 'layers', value: '12', real: '12' },
        { label: 'd_model', value: '768', real: '768' },
        { label: 'heads', value: '12', real: '12' },
        { label: 'tokens', value: '1 + 196 patches', real: '≤ 1,024 text' },
        { label: 'output', value: '1,000 classes', real: '50,257 tokens' },
        { label: 'params', value: '86M', real: '124M' },
      ],
    },
    size: [1040, 500],
    aria: 'The Vision Transformer compared with GPT-2: an image is cut into 16 by 16 patches that become the tokens, each patch is embedded by one matrix product, learned 1D positions end up encoding the 2D grid, and a class token is classified.',
    phases: PHASES, learn: 'vit', tokens: SEQ, compare: COMPARE, caps: CAPS,
    still: ['positions', 9],
    hints: {
      blocks: 'Click a dark label on the drawing to jump to that part.',
      patches: 'Hover a patch to see its place in the sequence; click or tap to pin it.',
      embed: 'Hover a cell of x to see how it is computed; click or tap to pin it.',
      positions: 'Hover a small map or a cell of the large one; click or tap to pin it.',
    },
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { pill, addNode, caption, title, lane, glass, arrow } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46
  /** The toy image, drawn at scale s per grey level; `dim` fades everything except patch `hl`. */
  function image(x: number, y: number, s: number, a: number, hl = -1, dim = 1) {
    for (let r = 0; r < PX; r++) for (let c = 0; c < PX; c++) {
      const p = Math.floor(r / PP) * GRID + Math.floor(c / PP)
      ctx.fillStyle = rgba(C.ink, IMG[r * PX + c] * a * (hl < 0 || p === hl ? 1 : dim))
      ctx.fillRect(x + c * s, y + r * s, s + 0.3, s + 0.3)
    }
  }
  function patchTile(p: number, x: number, y: number, s: number, a: number, col: RGB | null) {
    const v = patchVals(p)
    v.forEach((g, i) => { ctx.fillStyle = rgba(C.ink, g * a); ctx.fillRect(x + (i % PP) * s, y + Math.floor(i / PP) * s, s + 0.3, s + 0.3) })
    if (col) { ctx.strokeStyle = rgba(col, a); ctx.lineWidth = 1.5; ctx.strokeRect(x - 1, y - 1, PP * s + 2, PP * s + 2) }
  }
  // real filter components as tiny RGB images, made once (their colours are data, not theme)
  const tiles = filters.map((f) => {
    const cv = document.createElement('canvas'); cv.width = 16; cv.height = 16
    const c2 = cv.getContext('2d')!, img = c2.createImageData(16, 16)
    for (let i = 0; i < 256; i++) for (let ch = 0; ch < 3; ch++) img.data[i * 4 + ch] = Math.round(127.5 + 127.5 * f[ch * 256 + i])
    for (let i = 0; i < 256; i++) img.data[i * 4 + 3] = 255
    c2.putImageData(img, 0, 0)
    return cv
  })

  /* ---------- scene 1: the blocks ---------- */
  function sceneBlocks(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const sx0 = pad + 188, sx1 = W - pad - 110, fx = (f: number) => lerp(sx0, sx1, f)
    const ys = [top + avail * 0.22, top + avail * 0.64], as = [eout(clamp(p / 0.3)), eout(clamp((p - 0.25) / 0.3))]
    ys.forEach((y, r) => {
      const a = as[r]
      if (a <= 0) return
      k.rowName(r ? 'ViT' : 'GPT-2', r ? 'B/16 · 86M' : 'small · 124M', pad, y, a)
      // input: a lookup of text tokens, or a matrix product on patches
      if (r) { for (let i = 0; i < 2; i++) patchTile(PICK[i + 2], pad + 104 + i * 20, y - 8, 4, a, hue(i)) }
      else { ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText('text', pad + 106, y) }
      for (let i = 0; i < 5; i++) lane(sx0, lerp(sx0, sx1, a), y + (i - 2) * 4, hue(i), a)
      const box = (x: number, w: number, t: string, sub: string, hl: boolean) => {
        rr(x - w / 2, y - 15, w, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, (hl ? 0.9 : 0.6) * a); ctx.lineWidth = 1; ctx.stroke()
        caption(t, x, y + 4, a, C.ink); caption(sub, x, y + 44, a, hl ? C.ink2 : C.mute)
      }
      box(fx(0.03), 74, r ? '· W_patch' : 'W_E[id]', r ? 'per patch' : 'lookup', r === 1)
      box(fx(0.18), 50, '+ P', r ? '197 learned' : '1,024 learned', false)
      const plates: [number, string, boolean][] = [[0.3, 'LN', false], [0.4, r ? 'self · all' : 'self · causal', r === 1], [0.6, 'LN', false], [0.7, 'MLP · GELU', false]]
      for (const [f, t, hl] of plates) {
        const ln = t === 'LN'
        glass(fx(f), y - (ln ? 20 : 26), y + (ln ? 20 : 26), hl ? 0.55 : 0.2, ln ? { w: 5, d: 6 } : { w: 8, d: 9 }, a)
        caption(t, fx(f), y + 44, a, hl ? C.ink2 : C.mute)
      }
      addNode(fx(0.5), y, a); addNode(fx(0.82), y, a)
      const hx = sx1 + 50
      arrow([[sx1, y], [hx - 40, y]], a)
      rr(hx - 38, y - 15, 104, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, (r ? 0.9 : 0.6) * a); ctx.stroke()
      caption(r ? '[CLS] → class' : 'every → token', hx + 14, y + 4, a, C.ink)
      caption(r ? '1,000 labels' : '50,257 tokens', hx + 14, y + 44, a, r ? C.ink2 : C.mute)
    })
    const la = eout(clamp((p - 0.55) / 0.25)), yB = ys[1]
    if (la > 0) {
      pill('patches', pad + 124, yB + 70, 'patches', la)
      pill('· W_patch', fx(0.03), yB - 46, 'embed', la)
      pill('positions', fx(0.18) + 20, yB + 70, 'positions', la)
      pill('[CLS]', sx1 + 64, yB - 46, 'head', la)
    }
    mk.formula = { segs: [['GPT-2  x = W_E[ids] + W_P', C.ink2], ['     ·     ', C.mute], ['ViT  x = [CLS; patches · W_patch] + P', C.ink]], note: 'The encoder blocks are the same pre-LN blocks as GPT-2’s, with GELU, 12 heads and width 768; only the mask is gone. An MLP head on the final [CLS] vector gives the class.' }
  }

  /* ---------- scene 2: patches ---------- */
  function scenePatches(p: number) {
    const { W } = stage
    const s = 4, ix = pad + 20, iy = top + 36, side = PX * s
    const fin = eout(clamp(p / 0.08)), ga = eout(clamp((p - 0.12) / 0.12))
    const cur = p > 0.3 ? Math.min(GRID * GRID - 1, Math.floor(eio(clamp((p - 0.3) / 0.6)) * 28)) : -1
    const f = mk.focus?.key === 'img' ? mk.focus : null, hl = f ? f.i * GRID + f.j : cur
    image(ix, iy, s, fin, hl, hl >= 0 ? 0.45 : 1)
    if (ga > 0) {
      ctx.strokeStyle = rgba(C.bg, 0.8 * ga); ctx.lineWidth = 1
      for (let g = 1; g < GRID; g++) { ctx.beginPath(); ctx.moveTo(ix + g * PP * s, iy); ctx.lineTo(ix + g * PP * s, iy + side); ctx.moveTo(ix, iy + g * PP * s); ctx.lineTo(ix + side, iy + g * PP * s); ctx.stroke() }
      ctx.strokeStyle = rgba(C.ink, 0.3 * ga); ctx.strokeRect(ix - 0.5, iy - 0.5, side + 1, side + 1)
    }
    mk.hit('img', { x: ix, y: iy, c: PP * s }, GRID, GRID)
    if (hl >= 0) { ctx.strokeStyle = rgba(hue(hl), 1); ctx.lineWidth = 2; ctx.strokeRect(ix + (hl % GRID) * PP * s - 1, iy + Math.floor(hl / GRID) * PP * s - 1, PP * s + 2, PP * s + 2) }
    caption('224 × 224 pixels · 14 × 14 patches of 16 × 16', ix, iy + side + 22, fin, C.mute, 'left')
    // the patches, read row by row, as a token sequence
    const sa = eout(clamp((p - 0.25) / 0.1)), tx = ix + side + 70, ts = 5, tw = PP * ts + 8, perRow = Math.floor((W - pad - tx) / tw)
    if (sa > 0) {
      title('the sequence: 196 patch tokens', tx, iy + 8, sa)
      const n = Math.min(GRID * GRID, perRow * 5), shown = Math.max(0, Math.min(n, cur + 1))
      for (let q = 0; q < shown; q++) {
        const x = tx + (q % perRow) * tw, y = iy + 28 + Math.floor(q / perRow) * (tw + 18)
        patchTile(q, x, y, ts, sa, hue(q))
        if (q % perRow === 0) { ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, sa); ctx.fillText(String(q + 1), x - 8, y + (PP * ts) / 2) }
      }
      if (cur >= 27) caption(`… ${GRID * GRID - 28} more, 196 in all`, tx, iy + 28 + 2 * (tw + 18) + 10, sa, C.ink2, 'left')
    }
    const fp = f ? f.i * GRID + f.j : null
    mk.formula = fp !== null
      ? { segs: [[`patch ${fp + 1}`, hue(fp)], ['  =  row ', C.mute], [String(f!.i + 1), C.ink], [', column ', C.mute], [String(f!.j + 1), C.ink], ['  →  token ', C.mute], [String(fp + 1), C.ink]], note: 'Its 16 × 16 × 3 = 768 pixel values become one token. The order is row by row, left to right, like reading text.' }
      : { segs: [['N = (224 / 16)² = 196 tokens', C.ink], ['     ·     ', C.mute], ['each 16 · 16 · 3 = 768 values', C.ink2]], note: 'The toy image has 4 × 4 grey levels per patch where the real one has 16 × 16 × 3 colour values. Colour outlines follow the token hues, repeating every 7 tokens as on every page.' }
  }

  /* ---------- scene 3: patch embedding ---------- */
  function sceneEmbed(p: number) {
    const { W } = stage
    const c = 16, n = PICK.length, K = PP * PP
    const rX: Rect = { x: pad + 60, y: top + K * c + 50, c }, rE: Rect = { x: rX.x + K * c + 30, y: rX.y, c }, rW: Rect = { x: rE.x, y: rX.y - K * c - 26, c }
    const fin = eout(clamp(p / 0.06)), g = gemm(clamp((p - 0.1) / 0.4), n, 8, K, 'fast')
    PICK.forEach((q, i) => patchTile(q, pad + 8, rX.y + i * c + 0.5, 3.6, fin, hue(i)))
    mk.drawMat({ r: rX, vals: EMB.X, kind: 'row', alpha: fin, name: 'patches', shape: '6 × 16', real: '196 × 768', label: 'bottom', noText: true, labelW: K * c + 20 })
    mk.drawMat({ r: rW, vals: EMB.Wp, kind: 'w', alpha: fin, name: 'W_patch', shape: '16 × 8', real: '768 × 768', noText: true, labelW: 260 })
    mk.drawMat({ r: rE, vals: EMB.E, kind: 'row', alpha: fin, name: 'x', shape: '6 × 8', real: '196 × 768', label: 'bottom', reveal: g.rev, noText: true, labelW: 180 })
    mk.hit('e', rE, n, 8)
    const fs = mk.resolve({ e: { g, K } })
    if (fs) mk.gemmOverlay({ A: rX, Av: EMB.X, B: rW, Bv: EMB.Wp, C: rE, f: fs, names: ['x', 'patch', 'W_patch'], note: `Patch ${PICK[fs.i] + 1}’s grey levels (minus a constant) times column ${fs.j + 1} of the shared matrix. Toy: 16 values per patch and 8 outputs, where ViT has 768 and 768.` })
    else mk.formula = { segs: [['x = patches · W_patch + b', C.ink], ['     =     ', C.mute], ['Conv2d(3 → 768, kernel 16, stride 16)', C.ink2]], note: 'Every patch uses the same matrix, just as every text token uses the same embedding table. A convolution whose kernel is as large as its stride computes exactly this.' }
    // the real filters' principal components
    const ra = eout(clamp((p - 0.55) / 0.12))
    if (ra > 0) {
      const fx0 = rE.x + 8 * c + 70, ts = Math.min(40, (W - pad - fx0) / 7 - 6)
      title('real vit-b/16 filters', fx0, top + 14, ra)
      caption(`top 28 principal components, ${Math.round(explained.reduce((a, b) => a + b, 0) * 100)}% of the variance`, fx0, top + 32, ra, C.ink2, 'left')
      ctx.save(); ctx.globalAlpha = ra; ctx.imageSmoothingEnabled = false
      tiles.forEach((cv, i) => ctx.drawImage(cv, fx0 + (i % 7) * (ts + 6), top + 48 + Math.floor(i / 7) * (ts + 6), ts, ts))
      ctx.restore()
      caption('shown in their own colours: edges, stripes and colour contrasts,', fx0, top + 48 + 4 * (ts + 6) + 16, ra, C.mute, 'left')
      caption('like the first layer of a convolutional network', fx0, top + 48 + 4 * (ts + 6) + 32, ra, C.mute, 'left')
    }
  }

  /* ---------- scene 4: [CLS] and the real position similarities ---------- */
  function scenePositions(p: number) {
    const { H } = stage, avail = H - top - bot
    const mm = Math.floor((avail - 14) / GRID), cs = (mm - 3) / GRID, gx = pad + 10, gy = top + 10
    const fin = eout(clamp(p / 0.08))
    const sweep = p > 0.35 ? Math.min(GRID * GRID - 1, Math.floor(eio(clamp((p - 0.35) / 0.6)) * GRID * GRID)) : 0
    const f = mk.focus && (mk.focus.key === 'mm' || mk.focus.key === 'big') ? mk.focus : null
    const sel = f ? f.i * GRID + f.j : sweep
    const shade = (v: number, a: number) => (v >= 0 ? rgba(C.ink, (0.04 + 0.9 * v) * a) : rgba(C.neg, 0.25 * -v * a))
    // 14 × 14 small maps: map (r, c) shows how similar position (r, c) is to every position
    for (let q = 0; q < GRID * GRID; q++) {
      const a = fin * eout(clamp((p - 0.04 - (q / (GRID * GRID)) * 0.25) / 0.05))
      if (a <= 0) continue
      const ox = gx + (q % GRID) * mm, oy = gy + Math.floor(q / GRID) * mm
      for (let t = 0; t < GRID * GRID; t++) { ctx.fillStyle = shade(posSim(q, t), a); ctx.fillRect(ox + (t % GRID) * cs, oy + Math.floor(t / GRID) * cs, cs + 0.2, cs + 0.2) }
      if (q === sel) { ctx.strokeStyle = rgba(C.ink, a); ctx.lineWidth = 1.5; ctx.strokeRect(ox - 1, oy - 1, GRID * cs + 2, GRID * cs + 2) }
    }
    mk.hit('mm', { x: gx, y: gy, c: mm }, GRID, GRID)
    // the selected map, large
    const bs = 14, bx = gx + GRID * mm + 60, by = top + 110, ba = eout(clamp((p - 0.3) / 0.1))
    if (ba > 0) {
      const r0 = Math.floor(sel / GRID), c0 = sel % GRID
      title(`position row ${r0 + 1}, column ${c0 + 1}`, bx, by - 18, ba)
      for (let t = 0; t < GRID * GRID; t++) mk.paintCell(bx + (t % GRID) * bs, by + Math.floor(t / GRID) * bs, bs, posSim(sel, t), 1, null, 0.9, ba)
      ctx.strokeStyle = rgba(C.ink, ba); ctx.lineWidth = 2; ctx.strokeRect(bx + c0 * bs, by + r0 * bs, bs, bs)
      ctx.strokeStyle = rgba(C.ink, 0.22 * ba); ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, by + 0.5, GRID * bs - 1, GRID * bs - 1)
      mk.hit('big', { x: bx, y: by, c: bs }, GRID, GRID)
      caption('cosine similarity of its position vector', bx, by + GRID * bs + 22, ba, C.ink2, 'left')
      caption('to each of the 196 others; filled: similar', bx, by + GRID * bs + 38, ba, C.mute, 'left')
    }
    // the input sequence, [CLS] first
    const sa = eout(clamp((p - 0.1) / 0.1))
    if (sa > 0) {
      const sy = top + 34
      rr(bx, sy - 11, 56, 22, 5); ctx.fillStyle = rgba(C.ink, 0.85 * sa); ctx.fill()
      ctx.font = F.mono(11, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.bg, sa); ctx.fillText('[CLS]', bx + 28, sy + 0.5)
      for (let q = 0; q < 8; q++) patchTile(q, bx + 66 + q * 26, sy - 9, 4.5, sa, hue(q))
      caption('… 196', bx + 66 + 8 * 26 + 4, sy + 4, sa, C.mute, 'left')
      mathRun([['+ P', false], ['0', true], [', P', false], ['1', true], [', … P', false], ['196', true]], bx, sy + 36, sa, 16, C.ink2)
    }
    const hv = f?.key === 'big' ? f.i * GRID + f.j : null
    mk.formula = hv !== null
      ? { segs: [[`cos(P[${sel + 1}], P[${hv + 1}])`, C.ink], ['  =  ', C.mute], [fmtF(posSim(sel, hv)), C.ink]], note: `Positions ${sel + 1} and ${hv + 1} are ${Math.abs(Math.floor(sel / GRID) - Math.floor(hv / GRID))} rows and ${Math.abs((sel % GRID) - (hv % GRID))} columns apart. Real ViT-B/16 weights.` }
      : { segs: [['sim(i, j) = P_i · P_j / (‖P_i‖ ‖P_j‖)', C.ink]], note: 'Real ViT-B/16 position embeddings. Nothing told the model that position 15 sits under position 1; it learned it, because neighbouring patches look alike. Each small map is one position; its own spot is brightest.' }
  }

  /* ---------- scene 5: only [CLS] is classified ---------- */
  function sceneHead(p: number) {
    const { H } = stage, avail = H - top - bot
    const ys = [top + avail * 0.24, top + avail * 0.7], x0 = pad + 120, n = 10, dx = 44
    const as = [eout(clamp(p / 0.2)), eout(clamp((p - 0.2) / 0.2))]
    ys.forEach((y, r) => {
      const a = as[r]
      if (a <= 0) return
      k.rowName(r ? 'ViT' : 'GPT-2', r ? '197 outputs, 1 used' : 'every output used', pad, y - 6, a)
      for (let i = 0; i < n; i++) {
        const x = x0 + 40 + i * dx, isCls = r === 1 && i === 0
        if (r === 1 && i === 0) { rr(x - 22, y - 11, 44, 22, 5); ctx.fillStyle = rgba(C.ink, 0.85 * a); ctx.fill(); ctx.font = F.mono(10.5, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.bg, a); ctx.fillText('[CLS]', x, y + 0.5) }
        else { ctx.fillStyle = rgba(hue(i), 0.8 * a); ctx.fillRect(x - 12, y - 8, 24, 16) }
        const used = r === 0 || isCls, ua = eout(clamp((p - 0.35 - r * 0.15) / 0.15)) * a
        ctx.strokeStyle = rgba(used ? C.ink : C.faint, (used ? 0.55 : 0.8) * ua); ctx.lineWidth = 1
        if (!used) ctx.setLineDash([2, 3])
        ctx.beginPath(); ctx.moveTo(x, y + 12); ctx.lineTo(x, y + 30); ctx.stroke(); ctx.setLineDash([])
        if (used) { ctx.fillStyle = rgba(C.ink, 0.6 * ua); ctx.fillRect(x - 3, y + 30, 6, 6) }
      }
      caption('…', x0 + 40 + n * dx - 8, y + 4, a, C.mute, 'left')
      const hx = x0 + 40 + n * dx + 40, ha = eout(clamp((p - 0.5 - r * 0.15) / 0.15)) * a
      arrow([[hx - 20, y + 33], [hx + 20, y + 33]], ha)
      ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, ha)
      fillRich(r ? 'LN → W_head → 1,000 class scores' : 'ln_f → W_Eᵀ → 50,257 scores each', hx + 30, y + 33)
      caption(r ? 'one label for the whole image' : 'the next token at every position', hx + 30, y + 56, ha, C.ink2, 'left')
    })
    mk.formula = { segs: [['GPT-2  logits = LN(x) · W_Eᵀ, at every position', C.ink2], ['     ·     ', C.mute], ['ViT  logits = LN(x_CLS) · W_head', C.ink]], note: 'The patch outputs are still computed (every patch attends to every other in every layer) but only [CLS] is read. Many later ViTs average the patch outputs instead of using [CLS].' }
  }

  return { blocks: sceneBlocks, patches: scenePatches, embed: sceneEmbed, positions: scenePositions, head: sceneHead }
}
