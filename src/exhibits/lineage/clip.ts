import { F, rr, serifAt } from '../../core/draw'
import { raw } from '../../core/i18n'
import { fmt, fmtF, gemm, type Rect } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp } from '../../core/util'
import { clip, preview, softmax } from '../../lib/clip/data'
import type { Nav } from '../registry'
import { mountExhibit, type Env } from '../kit'

/*
 * CLIP as a diff against GPT-2: two encoders, one for images (a ViT) and one for text (a GPT-2-like
 * causal Transformer read at its end-of-text token), trained so that matching pairs point the same way
 * in a shared 512-dimensional space. Every number here is real CLIP ViT-B/32, run offline on four
 * simple drawn images and a few captions.
 */

const PHASES = [
  { id: 'towers', name: 'Two encoders', short: 'Towers', dur: 8 },
  { id: 'space', name: 'One shared space', short: 'Space', dur: 9 },
  { id: 'matrix', name: 'The contrastive matrix', short: 'I · Tᵀ', dur: 12 },
  { id: 'zeroshot', name: 'Zero-shot classification', short: 'Zero-shot', dur: 9 },
  { id: 'scale', name: 'Temperature', short: '× 100', dur: 8 },
]

const N = clip.images.length, SHOW = 8
const LABELS = clip.images.map((im) => im.label)
/** Each pair takes the palette hue nearest its shape's colour (red, blue, green, yellow), so labels never contradict the pictures. */
const PAIR_HUE = [6, 0, 4, 5]
const TOKS = LABELS.map((text, i) => ({ text, c: PAIR_HUE[i] }))
const shortT = (t: string) => t.replace(/^a photo of (a |something )/, '')
/** The part of each cosine that the 8 drawn dimensions do not account for (the other 504). */
const rest = (i: number, j: number) => clip.sim[i][j] - clip.images[i].head.reduce((s, v, k) => s + v * clip.captions[j].head[k], 0)
const TT = clip.captions.map((c) => c.head)
const TcolT = Array.from({ length: SHOW }, (_, k) => TT.map((h) => h[k])) // Tᵀ: 8 × 4

const COMPARE: Record<string, [string, string]> = {
  towers: ['GPT-2’s forward pass', 'anatomy'], space: ['GPT-2’s embeddings', 'anatomy/embedding'], matrix: ['GPT-2’s attention scores', 'anatomy/attention?phase=scores'],
  zeroshot: ['GPT-2’s softmax', 'anatomy/unembed?phase=softmax'], scale: ['GPT-2’s temperature', 'anatomy/unembed?phase=temp'],
}

const CAPS: Record<string, [string, string]> = {
  towers: ['CLIP has two encoders. The text one is shaped like GPT-2 (causal, 12 layers) but outputs one vector, read at the end-of-text token, instead of next-token scores. The image one is a Vision Transformer. Each vector is projected to 512 numbers and scaled to length 1. Click a label to jump.', 'image → ViT → 512 · text → Transformer → 512'],
  space: ['These are real CLIP embeddings of four drawn images and their captions. On their first principal component, all images sit on one side and all captions on the other (the known “modality gap”); on the second, the heart’s picture and caption sit at one end and the triangle’s at the other, while the square and the star, pictures and captions alike, fall close together in between.', 'length-1 vectors · dot product = cosine'],
  matrix: ['Training takes a batch of image–caption pairs and computes every image against every caption: one matrix product. The loss pushes the diagonal (the true pairs) up and everything else down, along rows and along columns. Real CLIP numbers; hover the cells.', 'S = I · Tᵀ · softmax both ways'],
  zeroshot: ['Because labels are just text, CLIP classifies without training for the task: write one caption per class, embed them, and pick the most similar. CLIP was never trained on these shapes, yet all eight answers here are right.', 'class = argmax cos(image, “a photo of a …”)'],
  scale: ['The raw cosines are close together (0.2 to 0.4). CLIP multiplies them by a learned scale, the inverse of a temperature, before the softmax. It starts at 14.3 and training pushes it to its cap of 100, which turns small gaps into confident answers.', 'softmax(100 · cos)'],
}

export function mountClip(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: '',
      eyebrow: 'Architectures · Vision & diffusion',
      title: 'CLIP',
      subtitle: 'CLIP ViT-B/32 · images and text in one space',
      specs: [
        { label: 'compared', value: 'CLIP ViT-B/32', real: 'GPT-2 small', realLabel: 'vs' },
        { label: 'text tower', value: '12 × 512 · causal', real: '12 × 768' },
        { label: 'image tower', value: 'ViT-B/32 · 12 × 768', real: 'none', realLabel: '' },
        { label: 'output', value: '1 vector of 512', real: '50,257 scores' },
        { label: 'context', value: '77 tokens', real: '1,024' },
        { label: 'params', value: '151M', real: '124M' },
      ],
    },
    size: [1040, 500],
    aria: 'CLIP compared with GPT-2: an image encoder and a text encoder map pictures and captions into one shared space, trained so that each image is most similar to its own caption; real CLIP similarities for four drawn shapes.',
    phases: PHASES, learn: 'clip', tokens: TOKS, compare: COMPARE, caps: CAPS,
    still: ['matrix', 11],
    hints: {
      towers: 'Click a dark label on the drawing to jump to that part.',
      matrix: 'Hover a cell of S to see how it is computed; click or tap to pin it.',
      zeroshot: 'Hover a cell to read its cosine and probability; click or tap to pin it.',
    },
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { pill, caption, title, lane, glass, arrow } = k
  const hue = (i: number): RGB => C.tok[PAIR_HUE[i % PAIR_HUE.length]]
  const pad = 36, top = 56, bot = 46
  const pics = clip.images.map((_, i) => preview(i))
  /** A drawn image, in its own colours (they are the model's input, not theme colours). */
  function pic(i: number, x: number, y: number, s: number, a: number, ring = false) {
    ctx.save(); ctx.globalAlpha = a; ctx.imageSmoothingEnabled = true; ctx.drawImage(pics[i], x, y, s, s); ctx.restore()
    ctx.strokeStyle = rgba(ring ? hue(i) : C.ink, (ring ? 1 : 0.25) * a); ctx.lineWidth = ring ? 1.6 : 1; ctx.strokeRect(x - 0.5, y - 0.5, s + 1, s + 1)
  }
  function capChip(j: number, x: number, y: number, a: number, align: 'left' | 'right' = 'left') {
    ctx.font = F.mono(11.5); const t = clip.captions[j].text, w = ctx.measureText(t).width + 14, x0 = align === 'left' ? x : x - w
    rr(x0, y - 11, w, 22, 5); ctx.fillStyle = rgba(hue(j), 0.14 * a); ctx.fill(); ctx.strokeStyle = rgba(hue(j), 0.6 * a); ctx.lineWidth = 1; ctx.stroke()
    ctx.fillStyle = rgba(C.ink, a); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(t, x0 + 7, y + 0.5)
    return w
  }

  /* ---------- scene 1: two towers ---------- */
  function sceneTowers(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const ys = [top + avail * 0.14, top + avail * 0.5, top + avail * 0.84], as = [0, 0.2, 0.35].map((t) => eout(clamp((p - t) / 0.25)))
    const x0 = pad + 290, pl = x0 + 130, out = pl + 150, sx = W - pad - 120, sy = (ys[1] + ys[2]) / 2
    ys.forEach((y, r) => {
      const a = as[r]
      if (a <= 0) return
      k.rowName(['GPT-2', 'Text', 'Image'][r], ['small · for comparison', 'CLIP · 63M', 'CLIP · 88M'][r], pad, y, a)
      if (r === 0) { ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText('The cat sat on the', pad + 120, y) }
      else if (r === 1) capChip(0, pad + 120, y, a)
      else pic(0, pad + 150, y - 22, 44, a)
      for (let i = 0; i < 5; i++) lane(x0, pl + 60, y + (i - 2) * 4, C.tok[i], a)
      glass(pl, y - 26, y + 26, r ? 0.45 : 0.2, { w: 8, d: 9 }, a)
      caption(r === 2 ? '× 12 · no mask' : '× 12 · causal', pl, y + 44, a, r ? C.ink2 : C.mute)
      const box = (x: number, wd: number, t: string, sub: string) => {
        rr(x - wd / 2, y - 15, wd, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.7 * a); ctx.lineWidth = 1; ctx.stroke()
        caption(t, x, y + 4, a, C.ink); caption(sub, x, y + 44, a)
      }
      if (r === 0) { box(out + 40, 150, 'next-token scores', '50,257 per position'); return }
      box(out + 10, 96, r === 1 ? '[EOT] · W_t' : '[CLS] · W_i', r === 1 ? 'one vector' : 'one vector')
      arrow([[out + 60, y], [sx - 70, lerp(y, sy, 0.8)]], a)
    })
    // the shared space: every vector has length 1
    const sa = eout(clamp((p - 0.55) / 0.2))
    if (sa > 0) {
      const R = 58
      ctx.strokeStyle = rgba(C.ink, 0.35 * sa); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(sx, sy, R, 0, 7); ctx.stroke()
      caption('shared space, 512 dims', sx, sy + R + 22, sa, C.ink2)
      caption('length 1', sx, sy + R + 38, sa, C.mute)
      const at = (ang: number, col: RGB) => { ctx.strokeStyle = rgba(col, sa); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + Math.cos(ang) * R, sy - Math.sin(ang) * R); ctx.stroke() }
      at(0.9, hue(0)); at(0.72, hue(0))
      ctx.strokeStyle = rgba(C.ink, 0.6 * sa); ctx.beginPath(); ctx.arc(sx, sy, 22, -0.9, -0.72); ctx.stroke()
    }
    const la = eout(clamp((p - 0.6) / 0.2))
    if (la > 0) {
      pill('space', sx - 40, sy - 84, 'space', la)
      pill('I · Tᵀ', sx + 30, sy - 84, 'matrix', la)
      pill('zero-shot', sx, sy + 124, 'zeroshot', la)
      pill('× 100', pl, ys[1] - 46, 'scale', la)
    }
    mk.formula = { segs: [['i = unit(W_i · ViT(image)[CLS])', C.ink2], ['     ·     ', C.mute], ['t = unit(W_t · Text(caption)[EOT])', C.ink2], ['     ·     ', C.mute], ['similarity = i · t', C.ink]], note: 'The text tower has GPT-2’s causal mask and a byte-level BPE like GPT-2’s (49,408 tokens, lower-cased) but only 77 positions; the whole caption is summed up in the vector at its end-of-text token.' }
  }

  /* ---------- scene 2: the shared space, real ---------- */
  function sceneSpace(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.08)), cs = 18, x0 = pad + 170
    title('first 8 of 512 dims, real', x0, top + 12, fin)
    const rowY = (r: number) => top + 36 + r * 42
    for (let r = 0; r < 2 * N; r++) {
      const a = fin * eout(clamp((p - 0.04 - r * 0.04) / 0.08)), isImg = r < N, j = r % N, y = rowY(r)
      if (a <= 0) continue
      if (isImg) pic(j, x0 - 46, y - 2, 30, a, true)
      else { ctx.font = F.mono(11.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(hue(j), a); ctx.fillText(clip.captions[j].text, x0 - 12, y + 13) }
      const head = isImg ? clip.images[j].head : clip.captions[j].head
      head.forEach((v, kk) => mk.paintCell(x0 + kk * cs, y + 4, cs, v, 0.08, hue(j), 0, a))
      caption('…', x0 + SHOW * cs + 8, y + 17, a, C.mute, 'left')
    }
    // the principal-component map
    const mx0 = x0 + SHOW * cs + 90, mx1 = W - pad - 20, my0 = top + 40, my1 = top + avail - 30
    const ys = clip.pca.map((q) => q[1]), lo = Math.min(...ys) - 0.04, hi = Math.max(...ys) + 0.04
    const X = (v: number) => lerp(mx0 + 150, mx1 - 60, (v + 0.6) / 1.2), Y = (v: number) => lerp(my1 - 20, my0 + 30, (v - lo) / (hi - lo))
    /** Pixel heights pushed apart to at least `gap`, in their order, so close points keep readable labels. */
    const spread = (v: number[], gap: number) => {
      const o = v.map((y, k) => [y, k]).sort((a, b) => a[0] - b[0]), out = [...v]
      for (let r = 1; r < o.length; r++) o[r][0] = Math.max(o[r][0], o[r - 1][0] + gap)
      const shift = (o[o.length - 1][0] - v[o[o.length - 1][1]]) / 2
      o.forEach(([y, k]) => { out[k] = y - shift })
      return out
    }
    const iY = spread(clip.pca.slice(0, N).map((q) => Y(q[1])), 32), tY = spread(clip.pca.slice(N).map((q) => Y(q[1])), 20)
    const ma = eout(clamp((p - 0.35) / 0.12))
    if (ma > 0) {
      title('two principal directions of the 8 vectors', mx0, top + 12, ma)
      ctx.strokeStyle = rgba(C.ink, 0.15 * ma); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(X(0), my0); ctx.lineTo(X(0), my1); ctx.moveTo(mx0, Y(0)); ctx.lineTo(mx1, Y(0)); ctx.stroke()
      caption('captions', X(-0.54), my1 + 18, ma, C.ink2); caption('images', X(0.54), my1 + 18, ma, C.ink2)
      // a dot at each vector's true place; its picture or caption beside it, moved apart where two are close
      for (let j = 0; j < N; j++) {
        const la = eout(clamp((p - 0.5 - j * 0.06) / 0.1)) * ma, [ix, iy] = clip.pca[j], [tx, ty] = clip.pca[N + j]
        ctx.setLineDash([3, 4]); ctx.strokeStyle = rgba(hue(j), 0.7 * la); ctx.beginPath(); ctx.moveTo(X(tx), Y(ty)); ctx.lineTo(X(ix), Y(iy)); ctx.stroke(); ctx.setLineDash([])
        ctx.strokeStyle = rgba(hue(j), 0.5 * ma); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(X(ix), Y(iy)); ctx.lineTo(X(ix) + 12, iY[j]); ctx.moveTo(X(tx), Y(ty)); ctx.lineTo(X(tx) - 10, tY[j]); ctx.stroke()
        for (const [x, y] of [[X(ix), Y(iy)], [X(tx), Y(ty)]]) { ctx.fillStyle = rgba(hue(j), ma); ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill() }
        pic(j, X(ix) + 14, iY[j] - 13, 26, ma, true)
        ctx.font = F.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(hue(j), ma); raw(() => ctx.fillText(clip.captions[j].text, X(tx) - 14, tY[j]))
      }
    }
    mk.formula = { segs: [['‖i‖ = ‖t‖ = 1', C.ink2], ['     ·     ', C.mute], ['i · t = cos(angle between them)', C.ink]], note: 'Real CLIP ViT-B/32 embeddings. The map is a projection of 512 dimensions onto the two directions along which these 8 vectors vary most; the horizontal one separates the two kinds of input, the vertical one the four items.' }
  }

  /* ---------- scene 3: the contrastive matrix ---------- */
  function sceneMatrix(p: number) {
    const c = 20, cS = 34, fin = eout(clamp(p / 0.06))
    const rS: Rect = { x: pad + 90 + SHOW * c + 34, y: top + 76 + SHOW * c + 26, c: cS }
    const rI: Rect = { x: pad + 90, y: rS.y + (cS - c) / 2, c }
    const rT: Rect = { x: rS.x + (cS - c) / 2, y: rS.y - SHOW * c - 26, c }
    const g = gemm(clamp((p - 0.08) / 0.4), N, N, SHOW, 'fast')
    // S has larger cells than I and Tᵀ: space the rows of I and the columns of Tᵀ to line up with it
    const Ivals = clip.images.map((im) => im.head)
    for (let i = 0; i < N; i++) {
      pic(i, pad + 40, rS.y + i * cS + 3, cS - 6, fin, true)
      mk.drawMat({ r: { x: rI.x, y: rS.y + i * cS + (cS - c) / 2, c }, vals: [Ivals[i]], kind: 'row', rowCols: [hue(i)], alpha: fin, name: i ? '' : 'I', shape: i ? '' : '4 × 8', real: i ? undefined : '4 × 512', label: i ? 'none' : 'top', noText: true, labelW: SHOW * c + 20 })
    }
    for (let j = 0; j < N; j++) mk.drawMat({ r: { x: rS.x + j * cS + (cS - c) / 2, y: rT.y, c }, vals: TcolT.map((row) => [row[j]]), kind: 'col', alpha: fin, name: j ? '' : 'Tᵀ', shape: j ? '' : '8 × 4', real: j ? undefined : '512 × 4', label: j ? 'none' : 'top', noText: true, labelW: 140 })
    for (let j = 0; j < N; j++) {
      ctx.save(); ctx.translate(rS.x + (j + 0.5) * cS + 8, rT.y - 30); ctx.rotate(-Math.PI / 5)
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(hue(j), fin); ctx.fillText(clip.captions[j].text, 0, 0); ctx.restore()
    }
    mk.drawMat({ r: rS, vals: clip.sim, kind: 'score', alpha: fin, name: 'S = cos', shape: '4 × 4', label: 'bottom', reveal: g.rev, labelW: 200 })
    mk.hit('s', rS, N, N)
    // the two softmaxes and the loss
    const sa = eout(clamp((p - 0.55) / 0.1)), L = clip.sim.map((r) => r.map((v) => v * clip.scale))
    const rowP = L.map(softmax), colP = L[0].map((_, j) => softmax(L.map((r) => r[j])))
    const rR: Rect = { x: rS.x + N * cS + 70, y: rS.y, c: cS }, rC: Rect = { x: rR.x + N * cS + 50, y: rS.y, c: cS }
    if (sa > 0) {
      arrow([[rS.x + N * cS + 12, rS.y + (N * cS) / 2], [rR.x - 12, rS.y + (N * cS) / 2]], sa)
      caption('× 100', (rS.x + N * cS + rR.x) / 2, rS.y + (N * cS) / 2 - 10, sa, C.ink2)
      mk.drawMat({ r: rR, vals: rowP, kind: 'attn', alpha: sa, name: 'rows', shape: 'image → text', label: 'bottom', paint: (x, y, cc, i, j, a) => mk.paintAttn(x, y, cc, rowP[i][j], hue(j), a), text: (i, j) => fmt(rowP[i][j]), labelW: 140 })
      mk.drawMat({ r: rC, vals: colP.map((_, i) => colP.map((col) => col[i])), kind: 'attn', alpha: sa, name: 'columns', shape: 'text → image', label: 'bottom', paint: (x, y, cc, i, j, a) => mk.paintAttn(x, y, cc, colP[j][i], hue(i), a), text: (i, j) => fmt(colP[j][i]), labelW: 160 })
    }
    const la = eout(clamp((p - 0.72) / 0.1))
    if (la > 0) {
      const ceRow = -rowP.reduce((s, r, i) => s + Math.log(r[i]), 0) / N, ceCol = -colP.reduce((s, col, j) => s + Math.log(col[j]), 0) / N
      ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, la)
      ctx.fillText(`loss = ½ (${fmtF(ceRow)} + ${fmtF(ceCol)}) = ${fmtF((ceRow + ceCol) / 2)}`, rR.x, rS.y - 40)
      caption('−log of each diagonal cell, averaged both ways', rR.x, rS.y - 22, la, C.mute, 'left')
      caption('CLIP trained with 32,768 pairs per batch:', rR.x, top + 60, la, C.ink2, 'left')
      caption('a 32,768 × 32,768 matrix every step', rR.x, top + 76, la, C.ink2, 'left')
    }
    const fs = mk.resolve({ s: { g, K: SHOW } })
    if (fs && mk.focus?.key !== 'x') mk.gemmOverlay({ A: { x: rI.x, y: rS.y + (cS - c) / 2, c }, Av: Ivals, B: { x: rS.x + (cS - c) / 2, y: rT.y, c }, Bv: TcolT, C: rS, f: fs, names: ['S', 'I', 'Tᵀ'], note: `${LABELS[fs.i]} image · “${clip.captions[fs.j].text}”. Only 8 of the 512 terms are drawn; the other 504 are added as one number. Real CLIP values.`, more: { n: 512 - SHOW, sum: rest }, fmtC: fmtF })
    else mk.formula = { segs: [['S = I · Tᵀ', C.ink], ['     ·     ', C.mute], ['loss = ½ (CE over rows + CE over columns), targets on the diagonal', C.ink2]], note: 'Each image should pick out its own caption among the batch, and each caption its own image. Real CLIP ViT-B/32 numbers for four drawn shapes.' }
  }

  /* ---------- scene 4: zero-shot classification ---------- */
  function sceneZeroshot(p: number) {
    const cs = 44, groups = [{ key: 'zs', name: 'a photo of a …', d: clip.shapes, x: pad + 110 }, { key: 'zc', name: 'a photo of something …', d: clip.colours, x: pad + 110 + N * cs + 190 }]
    const y0 = top + 96, fin = eout(clamp(p / 0.08))
    groups.forEach(({ key, name, d, x }, gi) => {
      const a = fin * eout(clamp((p - gi * 0.3) / 0.12))
      if (a <= 0) return
      const P = d.sim.map((r) => softmax(r.map((v) => v * clip.scale)))
      title(name, x, y0 - 70, a)
      d.texts.forEach((t, j) => { ctx.save(); ctx.translate(x + (j + 0.5) * cs + 6, y0 - 12); ctx.rotate(-Math.PI / 5); ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(shortT(t), 0, 0); ctx.restore() })
      for (let i = 0; i < N; i++) {
        pic(i, x - 44, y0 + i * cs + 5, cs - 10, a, true)
        const best = P[i].indexOf(Math.max(...P[i]))
        for (let j = 0; j < N; j++) {
          const rv = clamp((p - gi * 0.3 - 0.05 - i * 0.04) / 0.08)
          if (rv <= 0) continue
          const fa = mk.paintAttn(x + j * cs, y0 + i * cs, cs, P[i][j], C.ink, a * rv)
          mk.cellText(fmt(P[i][j]), x + j * cs, y0 + i * cs, cs, fa, a * rv)
          if (j === best) { ctx.strokeStyle = rgba(C.ink, a * rv); ctx.lineWidth = 2; ctx.strokeRect(x + j * cs + 1, y0 + i * cs + 1, cs - 2, cs - 2) }
        }
      }
      ctx.strokeStyle = rgba(C.ink, 0.22 * a); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y0 + 0.5, N * cs - 1, N * cs - 1)
      mk.hit(key, { x, y: y0, c: cs }, N, N)
      const right = P.filter((r, i) => r.indexOf(Math.max(...r)) === i).length
      caption(`${right} of 4 right`, x, y0 + N * cs + 26, a * eout(clamp((p - gi * 0.3 - 0.25) / 0.1)), C.ink2, 'left')
    })
    const f = mk.focus && (mk.focus.key === 'zs' || mk.focus.key === 'zc') ? mk.focus : null
    if (f) {
      const d = f.key === 'zs' ? clip.shapes : clip.colours, P = softmax(d.sim[f.i].map((v) => v * clip.scale))
      mk.formula = { segs: [[`${LABELS[f.i]} image · “${d.texts[f.j]}”`, C.ink], ['   cos ', C.mute], [fmtF(d.sim[f.i][f.j]), C.ink2], ['  →  p ', C.mute], [fmtF(P[f.j]), C.ink]], note: 'The probability is a softmax of 100 × cosine over the four prompts. Real CLIP ViT-B/32.' }
    } else mk.formula = { segs: [['p(class | image) = softmax over classes of 100 · cos(i, t_class)', C.ink]], note: 'The classes are written as captions and embedded once. Swapping in new classes needs no training, only new captions. Real CLIP ViT-B/32 on four drawn shapes.' }
  }

  /* ---------- scene 5: the learned temperature ---------- */
  function sceneScale(p: number) {
    const fin = eout(clamp(p / 0.08)), t = Math.exp(lerp(0, Math.log(clip.scale), eio(clamp((p - 0.12) / 0.6)))), row = clip.sim[0]
    const P = softmax(row.map((v) => v * t)), x0 = pad + 250, bw = 420, y0 = top + 60
    pic(0, pad + 20, y0 - 6, 60, fin, true)
    title('one image, four captions', pad + 20, y0 - 22, fin)
    row.forEach((v, j) => {
      const y = y0 + j * 44
      capChip(j, x0 - 16, y + 10, fin, 'right')
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, fin); ctx.fillText(`cos ${fmtF(v)}`, x0, y + 10)
      ctx.fillStyle = rgba(hue(j), 0.85 * fin); ctx.fillRect(x0 + 90, y + 2, Math.max(1.5, P[j] * bw), 16)
      ctx.fillStyle = rgba(C.ink, fin); ctx.fillText(fmt(P[j]), x0 + 98 + P[j] * bw, y + 10)
    })
    // the scale, on a log axis from 1 to 100
    const ax0 = x0 + 90, ax1 = ax0 + bw, ay = y0 + 4 * 44 + 40, X = (s: number) => lerp(ax0, ax1, Math.log(s) / Math.log(100))
    ctx.strokeStyle = rgba(C.ink, 0.3 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(ax0, ay); ctx.lineTo(ax1, ay); ctx.stroke()
    for (const s of [1, 3, 10, 14.3, 30, 100]) { ctx.beginPath(); ctx.moveTo(X(s), ay - 4); ctx.lineTo(X(s), ay + 4); ctx.stroke(); caption(String(s), X(s), ay + 20, fin, s === 14.3 ? C.ink2 : C.mute) }
    ctx.fillStyle = rgba(C.ink, fin); ctx.beginPath(); ctx.arc(X(t), ay, 5, 0, 7); ctx.fill()
    ctx.font = serifAt(18); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, fin); ctx.fillText(`scale ${t < 10 ? t.toFixed(1) : Math.round(t)}`, ax0, ay - 16)
    caption('14.3 = 1 / 0.07, the starting value · 100, the cap it reaches', ax0, ay + 44, fin, C.ink2, 'left')
    mk.formula = { segs: [['p = softmax(scale · cos)', C.ink], ['     ·     ', C.mute], [`scale ${t < 10 ? t.toFixed(1) : Math.round(t)}  →  p(“${clip.captions[0].text}”) = ${fmtF(P[0])}`, C.ink2]], note: 'The same as dividing by a temperature T = 1 / scale, as in GPT-2’s sampling, except that here it is learned during training. The cosines themselves never change on this slide.' }
  }

  return { towers: sceneTowers, space: sceneSpace, matrix: sceneMatrix, zeroshot: sceneZeroshot, scale: sceneScale }
}
