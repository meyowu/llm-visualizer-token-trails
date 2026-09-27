import { F, fillRich, rr, serifAt, type TokLike } from '../../core/draw'
import { fmt, fmtF, type Rect } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, gauss, lerp, rng } from '../../core/util'
import { mountExhibit, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * FlashAttention: exact attention computed in tiles that fit in on-chip SRAM, with an online softmax,
 * so the N × N score matrix never goes to HBM. The tiling runs real arithmetic at toy size (8 tokens,
 * d = 4, 4 × 4 tiles) and is checked against ordinary attention; memory and traffic use real sizes.
 */

const PHASES = [
  { id: 'memory', name: 'Fast and slow memory', short: 'SRAM · HBM', dur: 8 },
  { id: 'standard', name: 'Standard attention', short: 'Standard', dur: 10 },
  { id: 'online', name: 'An online softmax', short: 'Online softmax', dur: 11 },
  { id: 'tiles', name: 'Tile by tile', short: 'Tiles', dur: 14 },
  { id: 'io', name: 'Less traffic, no N × N', short: 'Savings', dur: 9 },
]

const N = 8, D = 4, B = 4, NB = N / B
const TOKS: TokLike[] = Array.from({ length: N }, (_, i) => ({ text: `t${i + 1}`, c: i }))
const { Q, K, V } = (() => { const r = rng(314), m = () => Array.from({ length: N }, () => Array.from({ length: D }, () => gauss(r))); return { Q: m(), K: m(), V: m() } })()
const scale = 1 / Math.sqrt(D)
const S = Q.map((q) => K.map((k) => q.reduce((s, v, d) => s + v * k[d], 0) * scale))
const softmax = (z: number[]) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), t = e.reduce((a, b) => a + b, 0); return e.map((v) => v / t) }
const P = S.map(softmax)
const O = P.map((p) => Array.from({ length: D }, (_, d) => p.reduce((s, w, j) => s + w * V[j][d], 0)))

/** FlashAttention's loop (outer: query blocks, inner: key/value blocks), recording the state after each tile. */
interface TileStep { i: number; j: number; m: number[]; l: number[]; acc: number[][]; mOld: number[]; tile: number[][] }
const STEPS: TileStep[] = []
const OF: number[][] = Array.from({ length: N }, () => Array(D).fill(0))
for (let i = 0; i < NB; i++) {
  let m = Array(B).fill(-Infinity), l = Array(B).fill(0), acc = Array.from({ length: B }, () => Array(D).fill(0))
  for (let j = 0; j < NB; j++) {
    const tile = Array.from({ length: B }, (_, r) => Array.from({ length: B }, (_, c) => S[i * B + r][j * B + c]))
    const mOld = m.slice(), mNew = m.map((v, r) => Math.max(v, ...tile[r]))
    const pt = tile.map((row, r) => row.map((s) => Math.exp(s - mNew[r])))
    l = l.map((v, r) => Math.exp(mOld[r] - mNew[r]) * v + pt[r].reduce((a, b) => a + b, 0))
    acc = acc.map((row, r) => row.map((v, d) => Math.exp(mOld[r] - mNew[r]) * v + pt[r].reduce((s, p, c) => s + p * V[j * B + c][d], 0)))
    m = mNew
    STEPS.push({ i, j, m: m.slice(), l: l.slice(), acc: acc.map((r) => r.slice()), mOld, tile })
  }
  for (let r = 0; r < B; r++) for (let d = 0; d < D; d++) OF[i * B + r][d] = acc[r][d] / l[r]
}
const MAXDIFF = Math.max(...O.flatMap((row, r) => row.map((v, d) => Math.abs(v - OF[r][d]))))

/** The online-softmax walk-through: the row of S whose maximum rises most in its second chunk, so the rescaling shows. */
const ROW = S.map((z, i) => [Math.max(...z.slice(B)) - Math.max(...z.slice(0, B)), i]).sort((a, b) => b[0] - a[0])[0][1]
const ON = (() => {
  const z = S[ROW], c1 = z.slice(0, B), c2 = z.slice(B)
  const m1 = Math.max(...c1), l1 = c1.reduce((s, v) => s + Math.exp(v - m1), 0)
  const m2 = Math.max(m1, ...c2), l2 = Math.exp(m1 - m2) * l1 + c2.reduce((s, v) => s + Math.exp(v - m2), 0)
  return { z, m1, l1, m2, l2, full: softmax(z) }
})()

/** HBM traffic per head, in elements: standard ≈ 4N² + 4Nd; FlashAttention with query blocks of 128 reads K and V N/128 times. */
const stdIO = (n: number, d = 64) => 4 * n * n + 4 * n * d
const flashIO = (n: number, d = 64, br = 128) => 2 * n * d * Math.ceil(n / br) + 2 * n * d

const COMPARE: Record<string, [string, string]> = {
  memory: ['decode’s memory limit', 'serving/kv-cache?phase=bound'], standard: ['GPT-2’s attention scores', 'anatomy/attention?phase=scores'], online: ['GPT-2’s softmax', 'anatomy/attention?phase=softmax'],
  tiles: ['GPT-2’s A · V', 'anatomy/attention?phase=av'], io: ['the KV cache’s size', 'serving/kv-cache?phase=size'],
}

const CAPS: Record<string, [string, string]> = {
  memory: ['A GPU has a little very fast memory on the chip (SRAM, next to the arithmetic units) and a lot of slower memory beside it (HBM). Moving data between them often takes longer than the arithmetic, so fast kernels keep work in SRAM.', 'SRAM 19 TB/s · HBM 1.5 TB/s'],
  standard: ['Ordinary attention runs as separate steps, each reading its inputs from HBM and writing its result back. Two of those results, the scores S and the weights P, are N × N: for long inputs they dominate both the traffic and the memory.', 'S, P: N × N written to HBM and read back'],
  online: ['Softmax needs a row’s maximum and sum before any weight is final. An online softmax keeps a running maximum m and sum ℓ: when a new chunk brings a larger maximum, everything so far is rescaled by e^(m_old − m_new). The result is exactly the ordinary softmax.', 'ℓ ← e^(m_old − m_new) ℓ + Σ e^(s − m_new)'],
  tiles: ['FlashAttention loads a block of queries and streams blocks of keys and values through SRAM. Each 4 × 4 tile of scores lives only on the chip; the running m, ℓ and output are rescaled as each tile arrives, and divided by ℓ at the end. Real arithmetic, checked against ordinary attention.', 'O_i = Σ_j e^(S_ij − m) V_j / ℓ, tile by tile'],
  io: ['The N × N matrices are never stored, so attention’s memory grows with N instead of N², and much less data crosses between HBM and the chip. The paper reports 2 to 4 times faster attention and the same exact result; the backward pass recomputes the tiles instead of storing them.', 'memory O(N) · traffic ~4× lower'],
}

export function mountFlashAttention(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'Hover a tile or a result cell to see its numbers; click or tap to pin it.',
      eyebrow: 'Serving · Kernels',
      title: 'FlashAttention',
      subtitle: 'exact attention, tiled in on-chip memory',
      specs: [
        { label: 'result', value: 'exact', real: 'same as standard', realLabel: '' },
        { label: 'memory', value: 'O(N)', real: 'standard O(N²)', realLabel: '' },
        { label: 'A100 SRAM', value: '192 KB × 108 SMs' },
        { label: 'A100 HBM', value: '40–80 GB' },
        { label: 'shown', value: 'toy: 8 tokens, d 4, tiles 4 × 4' },
      ],
    },
    size: [1040, 480],
    aria: 'FlashAttention: attention computed tile by tile in the GPU’s fast on-chip memory with an online softmax, so the large score matrix is never written to slow memory; the result is exactly the same as ordinary attention.',
    phases: PHASES, learn: 'flashattention', tokens: TOKS, compare: COMPARE, caps: CAPS,
    still: ['tiles', 13],
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { caption, title, arrow } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46
  function box(x: number, y: number, w: number, h: number, label: string, sub: string, a: number, strong = false) {
    rr(x, y, w, h, 8); ctx.fillStyle = rgba(C.ink, (strong ? 0.07 : 0.035) * a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, (strong ? 0.7 : 0.35) * a); ctx.lineWidth = 1; ctx.stroke()
    title(label, x + 12, y + 20, a); caption(sub, x + 12, y + 38, a, C.ink2, 'left')
  }

  /* ---------- scene 1: the memory hierarchy ---------- */
  function sceneMemory(p: number) {
    const { W } = stage
    const rows: [string, string, number, number, string][] = [['SRAM', 'on the chip, beside the arithmetic', 19000, 20, '20 MB'], ['HBM', 'the GPU’s main memory', 1500, 40000, '40 GB'], ['CPU DRAM', 'across the PCIe bus', 12.8, 1e6, '> 1 TB']]
    const x0 = pad + 250, bwMax = W - pad - x0 - 330, lmax = Math.log10(19000 / 5)
    title('bandwidth', x0, top + 18, 1); title('size', W - pad - 250, top + 18, 1)
    rows.forEach(([name, sub, gbps, mb, size], r) => {
      const a = eout(clamp((p - 0.04 - r * 0.14) / 0.12)), y = top + 50 + r * 92
      if (a <= 0) return
      ctx.font = serifAt(22); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(name, pad, y + 20)
      caption(sub, pad, y + 40, a, C.mute, 'left')
      const bw = Math.max(4, ((Math.log10(gbps / 5)) / lmax) * bwMax) * eio(clamp((p - 0.08 - r * 0.14) / 0.2))
      ctx.fillStyle = rgba(r === 0 ? C.tok[4] : C.ink, (r === 0 ? 0.9 : 0.6) * a); ctx.fillRect(x0, y + 4, bw, 22)
      ctx.font = F.mono(12); ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(gbps >= 1000 ? `${gbps / 1000} TB/s` : `${gbps} GB/s`, x0 + bw + 8, y + 20)
      const sw = Math.max(3, (Math.log10(mb) / 6) * 200)
      ctx.fillStyle = rgba(C.ink, 0.5 * a); ctx.fillRect(W - pad - 250, y + 4, sw, 22)
      ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(size, W - pad - 250 + sw + 8, y + 20)
    })
    const na = eout(clamp((p - 0.6) / 0.12))
    caption('bars on a log scale · numbers for an A100 40 GB, from the FlashAttention paper', x0, top + 50 + 3 * 92, na, C.mute, 'left')
    mk.formula = { segs: [['SRAM ≈ 13 × faster than HBM, but about 2,000 × smaller', C.ink]], note: 'The A100 has 192 KB of SRAM on each of its 108 streaming multiprocessors. A kernel that keeps its intermediate results there, instead of writing them to HBM between steps, can run several times faster doing the same arithmetic.' }
  }

  /* ---------- scene 2: standard attention's round trips ---------- */
  function sceneStandard(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const hb = { x: pad, y: top + avail * 0.56, w: W - 2 * pad, h: avail * 0.44 }, sr = { x: W / 2 - 150, y: top + 10, w: 300, h: avail * 0.34 }
    box(sr.x, sr.y, sr.w, sr.h, 'sram · compute', 'small and fast', 1, true)
    box(hb.x, hb.y, hb.w, hb.h, 'hbm', 'large and slow', 1)
    const mats: [string, number, boolean][] = [['Q', 0.08, false], ['K', 0.18, false], ['S', 0.34, true], ['P', 0.52, true], ['V', 0.7, false], ['O', 0.86, false]]
    const steps: [string, string[], string, string][] = [['S = QKᵀ', ['Q', 'K'], 'S', 'read Q, K · write S'], ['P = softmax(S)', ['S'], 'P', 'read S · write P'], ['O = PV', ['P', 'V'], 'O', 'read P, V · write O']]
    const n = 1024, d = 64, cost = [2 * n * d + n * n, 2 * n * n, n * n + n * d + n * d]
    const si = Math.min(2, Math.floor(clamp((p - 0.06) / 0.75) * 3)), sp = clamp((p - 0.06) / 0.75) * 3 - si
    const at = (name: string) => hb.x + mats.find((m) => m[0] === name)![1] * hb.w
    const my = hb.y + hb.h / 2 + 12
    for (const [name, , big] of mats) {
      const x = at(name), s = big ? 46 : 18, done = steps.findIndex((st) => st[2] === name)
      const shown = done < 0 || si > done || (si === done && sp > 0.6)
      rr(x - s / 2, my - (big ? 46 : 46) / 2, s, 46, 3); ctx.fillStyle = rgba(big ? C.neg : C.ink, (shown ? (big ? 0.5 : 0.35) : 0.05)); ctx.fill()
      ctx.font = serifAt(18); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, shown ? 1 : 0.35); ctx.fillText(name, x, my + 42)
      caption(big ? 'N × N' : 'N × d', x, my - 30, shown ? 1 : 0.35, big ? C.ink2 : C.mute)
    }
    const [op, ins, outM, label] = steps[si]
    ctx.font = serifAt(20); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, 1); ctx.fillText(op, sr.x + sr.w / 2, sr.y + sr.h / 2 + 12)
    for (const nm of ins) if (sp < 0.5) arrow([[at(nm), my - 50], [lerp(at(nm), sr.x + sr.w / 2, 0.6), sr.y + sr.h + 6]], clamp(sp * 3))
    if (sp >= 0.5) arrow([[sr.x + sr.w / 2, sr.y + sr.h + 6], [at(outM), my - 50]], clamp((sp - 0.5) * 3))
    caption(label, sr.x + sr.w + 20, sr.y + sr.h / 2 + 16, 1, C.ink2, 'left')
    const moved = cost.slice(0, si).reduce((a, b) => a + b, 0) + cost[si] * clamp(sp)
    ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, 1)
    ctx.fillText(`HBM traffic so far: ${(moved * 2 / 2 ** 20).toFixed(1)} MiB`, pad, sr.y + 20)
    caption('one head, N = 1,024, d = 64, 16-bit', pad, sr.y + 38, 1, C.mute, 'left')
    mk.formula = { segs: [['traffic ≈ 4N² + 4Nd elements', C.ink], ['     ·     ', C.mute], [`N = 1,024: ${((stdIO(n) * 2) / 2 ** 20).toFixed(1)} MiB per head per layer`, C.ink2]], note: 'The three steps are three separate kernels. S and P, N × N each, are written out and read back; the N × d matrices are small by comparison. The arithmetic itself is fast; the waiting is on HBM.' }
  }

  /* ---------- scene 3: the online softmax on one row ---------- */
  function sceneOnline(p: number) {
    const c = 44, x0 = pad + 140, y0 = top + 60, fin = eout(clamp(p / 0.06))
    const a1 = eout(clamp((p - 0.1) / 0.12)), a2 = eout(clamp((p - 0.4) / 0.12)), a3 = eout(clamp((p - 0.72) / 0.12))
    title(`one row of scores (row ${ROW + 1}), in two chunks`, x0, y0 - 18, fin)
    ON.z.forEach((v, j) => {
      const a = j < B ? a1 : a2
      mk.paintCell(x0 + j * c + (j >= B ? 16 : 0), y0, c, v, 2.5, hue(j), 0, Math.max(0.15, a))
      mk.cellText(fmt(v), x0 + j * c + (j >= B ? 16 : 0), y0, c, 0.3, Math.max(0.15, a))
    })
    const line = (y: number, a: number, parts: [string, RGB][]) => {
      if (a <= 0) return
      let x = x0
      ctx.font = F.mono(13); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
      for (const [t, col] of parts) { ctx.fillStyle = rgba(col, a); ctx.fillText(t, x, y); x += ctx.measureText(t).width }
    }
    line(y0 + c + 44, a1, [['after chunk 1:  ', C.mute], [`m = ${fmtF(ON.m1)}`, C.ink], ['   ', C.mute], [`ℓ = Σ e^(s − m) = ${fmtF(ON.l1)}`, C.ink]])
    line(y0 + c + 80, a2, [['after chunk 2:  ', C.mute], [`m = ${fmtF(ON.m2)}`, C.ink], ['   ', C.mute], [`ℓ = e^(${fmtF(ON.m1)} − ${fmtF(ON.m2)}) × ${fmtF(ON.l1)} + Σ e^(s − m) = ${fmtF(ON.l2)}`, C.ink]])
    line(y0 + c + 108, a2 * (ON.m2 > ON.m1 ? 1 : 0), [['                 the old sum is rescaled because the maximum went up', C.mute]])
    // the final weights, equal to the ordinary softmax
    if (a3 > 0) {
      const y = y0 + c + 150
      ctx.font = F.mono(12); ctx.fillStyle = rgba(C.mute, a3); ctx.textAlign = 'left'; ctx.fillText('e^(s − m) / ℓ', pad, y + c / 2 + 4)
      ON.z.forEach((v, j) => {
        const w = Math.exp(v - ON.m2) / ON.l2, x = x0 + j * c + (j >= B ? 16 : 0)
        const fa = mk.paintAttn(x, y, c, w, hue(j), a3); mk.cellText(fmt(w), x, y, c, fa, a3)
      })
      const diff = Math.max(...ON.full.map((w, j) => Math.abs(w - Math.exp(ON.z[j] - ON.m2) / ON.l2)))
      caption(diff < 1e-12 ? 'identical to the softmax of the whole row, computed at once' : `largest difference from the softmax of the whole row: ${diff.toExponential(0)}`, x0, y + c + 24, a3, C.ink2, 'left')
    }
    mk.formula = { segs: [['m_new = max(m_old, max of chunk)', C.ink2], ['   ·   ', C.mute], ['ℓ_new = e^(m_old − m_new) ℓ_old + Σ e^(s − m_new)', C.ink]], note: 'Subtracting the running maximum keeps every exponential at most 1, so nothing overflows; the rescaling corrects the earlier terms when the maximum changes. The same factor rescales the partial output, so the V rows can be summed as the chunks arrive.' }
  }

  /* ---------- scene 4: the tiles ---------- */
  function sceneTiles(p: number) {
    const { W } = stage
    const c = 18, fin = eout(clamp(p / 0.05))
    const rS: Rect = { x: pad + 60 + D * c + 24, y: top + D * c + 70, c }, rQ: Rect = { x: pad + 60, y: rS.y, c }, rKt: Rect = { x: rS.x, y: rS.y - D * c - 26, c }
    const rO: Rect = { x: rS.x + N * c + 40, y: rS.y, c }, rV: Rect = { x: rO.x, y: rS.y - N * c - 26, c }
    const sp = clamp((p - 0.06) / 0.8) * STEPS.length, si = Math.min(STEPS.length - 1, Math.floor(sp)), f = sp >= STEPS.length ? 1 : sp - si, st = STEPS[si]
    const done = p > 0.88
    const Kt = Array.from({ length: D }, (_, d) => K.map((r) => r[d]))
    mk.drawMat({ r: rQ, vals: Q, kind: 'row', alpha: fin, name: 'Q', shape: '8 × 4', real: 'N × 64', label: 'bottom', noText: true, labelW: D * c + 20 })
    mk.drawMat({ r: rKt, vals: Kt, kind: 'col', alpha: fin, name: 'Kᵀ', shape: '4 × 8', real: '64 × N', noText: true })
    mk.drawMat({ r: rV, vals: V, kind: 'row', alpha: fin, name: 'V', shape: '8 × 4', real: 'N × 64', noText: true, labelW: 120 })
    // S is never stored: a dashed outline, with only the current tile filled
    ctx.setLineDash([3, 3]); ctx.strokeStyle = rgba(C.ink, 0.35 * fin); ctx.lineWidth = 1; ctx.strokeRect(rS.x + 0.5, rS.y + 0.5, N * c - 1, N * c - 1); ctx.setLineDash([])
    caption('S: never stored', rS.x, rS.y + N * c + 22, fin, C.ink2, 'left')
    if (!done) {
      const ta = clamp(f / 0.25) * (1 - clamp((f - 0.85) / 0.15)) * fin
      st.tile.forEach((row, r) => row.forEach((v, cc) => mk.paintCell(rS.x + (st.j * B + cc) * c, rS.y + (st.i * B + r) * c, c, v, 2.5, null, 0.6, ta)))
      ctx.strokeStyle = rgba(C.ink, fin); ctx.lineWidth = 2
      ctx.strokeRect(rQ.x - 1, rQ.y + st.i * B * c - 1, D * c + 2, B * c + 2)
      ctx.strokeRect(rKt.x + st.j * B * c - 1, rKt.y - 1, B * c + 2, D * c + 2)
      ctx.strokeRect(rV.x - 1, rV.y + st.j * B * c - 1, D * c + 2, B * c + 2)
      ctx.strokeRect(rS.x + st.j * B * c, rS.y + st.i * B * c, B * c, B * c)
    }
    mk.hit('s', rS, N, N)
    // O: finished rows, and the running accumulator for the current block (divided by ℓ so far, to show it converging)
    const Ocur = O.map((row, r) => {
      const blk = Math.floor(r / B)
      if (done || blk < st.i) return OF[r]
      if (blk > st.i) return row.map(() => 0)
      const rr2 = r - st.i * B, use = f > 0.5 ? st : si > 0 && STEPS[si - 1].i === st.i ? STEPS[si - 1] : null
      return use ? use.acc[rr2].map((v) => v / use.l[rr2]) : row.map(() => 0)
    })
    mk.drawMat({ r: rO, vals: Ocur, kind: 'row', alpha: fin, name: 'O', shape: '8 × 4', real: 'N × 64', label: 'bottom', noText: true, labelW: 120, reveal: (r) => (done || Math.floor(r / B) < st.i || (Math.floor(r / B) === st.i && (f > 0.5 || (si > 0 && STEPS[si - 1].i === st.i))) ? 1 : 0) })
    mk.hit('o', rO, N, D)
    // SRAM: what is on the chip for this tile
    const sx = rO.x + D * c + 60, sw = W - pad - sx
    if (sw > 180) {
      box(sx, top + 10, sw, 250, 'sram, this tile', `query block ${st.i + 1}, key block ${st.j + 1}`, fin, true)
      const items: [string, string][] = [['Q_i, K_j, V_j', '3 × 4 × 4'], ['S_ij', '4 × 4'], ['m, ℓ', '2 × 4'], ['O_i', '4 × 4']]
      items.forEach(([nm, sz], n) => { ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, fin); fillRich(nm, sx + 14, top + 76 + n * 22); ctx.fillStyle = rgba(C.mute, fin); ctx.fillText(sz, sx + 130, top + 76 + n * 22) })
      ctx.font = F.mono(12); ctx.fillStyle = rgba(C.ink2, fin); ctx.fillText('m per row:', sx + 14, top + 178)
      st.m.forEach((v, r) => { ctx.fillStyle = rgba(hue(st.i * B + r), fin); ctx.fillText(fmt(v), sx + 14 + r * 44, top + 198) })
      ctx.fillStyle = rgba(C.ink2, fin); ctx.fillText('ℓ per row:', sx + 14, top + 222)
      st.l.forEach((v, r) => { ctx.fillStyle = rgba(hue(st.i * B + r), fin); ctx.fillText(fmt(v), sx + 14 + r * 44, top + 242) })
      caption(`tile ${Math.min(STEPS.length, si + 1)} of ${STEPS.length}`, sx, top + 286, fin, C.ink, 'left')
      if (done) { caption(`max |O − standard O| = ${MAXDIFF.toExponential(0)}`, sx, top + 306, fin, C.tok[4], 'left'); caption('the same result, exactly', sx, top + 322, fin, C.ink2, 'left') }
    }
    const fo = mk.focus
    if (fo?.key === 's') {
      const blk = Math.floor(fo.i / B) * NB + Math.floor(fo.j / B)
      mk.formula = { segs: [[`S[${fo.i + 1},${fo.j + 1}]`, C.ink], ['  =  q · k / 2  =  ', C.mute], [fmtF(S[fo.i][fo.j]), C.ink]], note: `Computed in tile ${blk + 1} inside SRAM, used to update m, ℓ and O for its row, then dropped. It is never written to HBM.` }
    } else if (fo?.key === 'o') mk.formula = { segs: [[`O[${fo.i + 1},${fo.j + 1}]`, C.ink], ['  tiled  ', C.mute], [fmtF(OF[fo.i][fo.j]), C.ink], ['   ·   standard  ', C.mute], [fmtF(O[fo.i][fo.j]), C.ink2]], note: 'Both computed here, in plain arithmetic: the tiled loop and ordinary attention (full S, softmax, then P · V).' }
    else mk.formula = { segs: [['for each tile:  m ← max(m, rowmax S_ij)', C.ink2], ['   ', C.mute], ['ℓ ← e^(m_old − m) ℓ + Σ e^(S_ij − m)', C.ink2], ['   ', C.mute], ['O ← e^(m_old − m) O + e^(S_ij − m) V_j', C.ink]], note: 'After the last key block, O is divided by ℓ. Toy sizes: blocks of 4; on an A100 FlashAttention-2 uses blocks of about 128 queries by 64 keys, sized to fill SRAM.' }
  }

  /* ---------- scene 5: savings ---------- */
  function sceneIo(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const cx0 = pad + 70, cx1 = W / 2 + 60, cy0 = top + 40, cy1 = top + avail - 40, fin = eout(clamp(p / 0.06))
    const ns = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    const X = (n: number) => lerp(cx0, cx1, Math.log2(n / 512) / 6), lo = Math.log10(0.1), hi = Math.log10(stdIO(32768) * 2 / 2 ** 20 * 1.3), Y = (mib: number) => lerp(cy1, cy0, (Math.log10(mib) - lo) / (hi - lo))
    title('hbm traffic per head, mib', cx0, cy0 - 16, fin)
    ctx.strokeStyle = rgba(C.ink, 0.25 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy1); ctx.lineTo(cx1, cy1); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, fin)
    for (const n of ns) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(n >= 1024 ? `${n / 1024}K` : String(n), X(n), cy1 + 6) }
    for (const v of [1, 10, 100, 1000]) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(v.toLocaleString('en-US'), cx0 - 6, Y(v)) }
    caption('sequence length N', (cx0 + cx1) / 2, cy1 + 32, fin)
    const g = eio(clamp((p - 0.08) / 0.4)) * (ns.length - 1)
    const curve = (fn: (n: number) => number, col: RGB, lab: string) => {
      ctx.strokeStyle = rgba(col, fin); ctx.lineWidth = 2; ctx.beginPath()
      for (let i = 0; i <= Math.floor(g); i++) { const n = ns[i], y = Y((fn(n) * 2) / 2 ** 20); if (i) ctx.lineTo(X(n), y); else ctx.moveTo(X(n), y) }
      ctx.stroke()
      if (g >= ns.length - 1) caption(`${lab} ${Math.round((fn(32768) * 2) / 2 ** 20).toLocaleString('en-US')} MiB`, cx1 + 8, Y((fn(32768) * 2) / 2 ** 20) + 4, fin, col, 'left')
    }
    curve(stdIO, C.ink, 'standard')
    curve(flashIO, C.tok[4], 'flash')
    // memory for the N × N matrix
    const ma = eout(clamp((p - 0.55) / 0.12)), mx = cx1 + 170
    if (ma > 0) {
      title('memory for s or p, one head', mx, cy0 - 16, ma)
      ;[2048, 8192, 32768, 131072].forEach((n, i) => {
        const y = cy0 + 20 + i * 36, bytes = n * n * 2
        ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink2, ma)
        ctx.fillText(`N = ${n >= 1024 ? n / 1024 + 'K' : n}`, mx, y)
        ctx.fillStyle = rgba(C.ink, ma); ctx.fillText(bytes >= 2 ** 30 ? `${bytes / 2 ** 30} GiB` : `${bytes / 2 ** 20} MiB`, mx + 110, y)
      })
      caption('FlashAttention: none, only m and ℓ (2 × N)', mx, cy0 + 20 + 4 * 36, ma, C.tok[4], 'left')
    }
    mk.formula = { segs: [['standard ≈ 4N² + 4Nd', C.ink2], ['     ·     ', C.mute], ['flash ≈ 2Nd · N / B_r + 2Nd  (B_r = 128)', C.ink]], note: 'A simple count of elements moved between HBM and SRAM for one head, d = 64, 16-bit. The exact counts depend on the GPU and kernel; the FlashAttention paper proves Θ(N² d² / M) accesses for SRAM size M, against Θ(N d + N²) for the standard method.' }
  }

  return { memory: sceneMemory, standard: sceneStandard, online: sceneOnline, tiles: sceneTiles, io: sceneIo }
}
