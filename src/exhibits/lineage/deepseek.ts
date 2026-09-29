import { F, chipW, drawChip, fillRich, rr, serifAt } from '../../core/draw'
import { fmt, fmtF, gemm, type Rect } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, gauss, lerp, rng } from '../../core/util'
import type { Nav } from '../registry'
import { mountExhibit, words, type Env } from '../kit'

/*
 * DeepSeek-V3 as a diff against GPT-2. Both halves of the block change: attention becomes multi-head
 * latent attention (MLA), which caches one small latent per token, and the MLP becomes DeepSeekMoE
 * (1 shared + 256 small routed experts, 8 per token), balanced by a per-expert bias instead of an extra
 * loss. Real sizes are DeepSeek-V3's; the drawings compute with toy vectors.
 */

const PHASES = [
  { id: 'blocks', name: 'GPT-2 vs DeepSeek-V3', short: 'Blocks', dur: 7 },
  { id: 'mla', name: 'Multi-head latent attention', short: 'MLA', dur: 12 },
  { id: 'cache', name: 'KV cache per token', short: 'KV cache', dur: 8 },
  { id: 'moe', name: 'Shared and routed experts', short: 'DeepSeekMoE', dur: 11 },
  { id: 'balance', name: 'Balancing with a bias', short: 'Bias', dur: 10 },
]

const TOKS = words(['The', 'cat', 'sat', 'on', 'the'])
const N = TOKS.length
/** Toy MLA: d_model 8, latent 2, 2 heads × d_head 4. */
const DM = 8, DC = 2, NH = 2, DH = 4
const MLA_SEED = 5

const dot = (a: number[], b: number[]) => a.reduce((s, v, k) => s + v * b[k], 0)
const matmul = (A: number[][], B: number[][]) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)))
const sig = (x: number) => 1 / (1 + Math.exp(-x))

const MLA = (() => {
  const r = rng(MLA_SEED), m = (a: number, b: number, s: number) => Array.from({ length: a }, () => Array.from({ length: b }, () => gauss(r) * s))
  const Hm = m(N, DM, 1), Wd = m(DM, DC, 0.45), Wk = m(DC, NH * DH, 0.8), Wv = m(DC, NH * DH, 0.8)
  const Cm = matmul(Hm, Wd)
  return { H: Hm, Wd, Wk, Wv, C: Cm, K: matmul(Cm, Wk), V: matmul(Cm, Wv) }
})()

/** Toy DeepSeekMoE routing for one token over 256 experts laid out 16 × 16. */
const NR = 256, TOPK = 8, GRID = 16
const MOE = (() => {
  const r = rng(909), u = Array.from({ length: 8 }, () => gauss(r))
  const s = Array.from({ length: NR }, () => sig(0.3 * dot(u, Array.from({ length: 8 }, () => gauss(r)))))
  const top = [...s.keys()].sort((a, b) => s[b] - s[a]).slice(0, TOPK)
  const z = top.reduce((a, e) => a + s[e], 0)
  return { s, top, w: top.map((e) => s[e] / z) }
})()
const choose = (n: number, k: number) => { let c = 1; for (let i = 0; i < k; i++) c = (c * (n - i)) / (i + 1); return c }

/** Toy bias balancing: 64 tokens, 16 experts, top 2; each step the bias moves toward an even load. */
const BT = 64, BE = 16, BK = 2, STEPS = 20, GAMMA = 0.01
const BAL = (() => {
  const r = rng(3)
  const U = Array.from({ length: BT }, () => Array.from({ length: 8 }, () => gauss(r)))
  const pop = Array.from({ length: 8 }, () => gauss(r))
  const Ec = Array.from({ length: BE }, (_, e) => Array.from({ length: 8 }, (_, k) => gauss(r) * 0.5 + (e % 5 === 0 ? 0.6 * pop[k] : 0)))
  const S = U.map((u) => Ec.map((c) => sig(0.5 * dot(u, c))))
  let b: number[] = Array(BE).fill(0)
  const hist: { load: number[]; b: number[] }[] = []
  for (let st = 0; st <= STEPS; st++) {
    const load = Array(BE).fill(0)
    for (const s of S) [...s.keys()].sort((x, y) => s[y] + b[y] - s[x] - b[x]).slice(0, BK).forEach((e) => load[e]++)
    hist.push({ load, b: [...b] })
    b = b.map((v, e) => v + GAMMA * Math.sign((BT * BK) / BE - load[e]))
  }
  return hist
})()

/** KV cache per token in KiB (BF16), at DeepSeek-V3's size: 128 heads × 128, 61 layers. */
const kib = (perLayer: number, layers = 61) => (perLayer * layers * 2) / 1024
const CACHE: [string, number, string][] = [
  ['MHA', kib(2 * 128 * 128), '128 key and 128 value heads'],
  ['GQA', kib(2 * 8 * 128), '8 key/value heads (as LLaMA 3)'],
  ['MQA', kib(2 * 128), '1 key/value head'],
  ['MLA', kib(512 + 64), 'latent 512 + RoPE key 64'],
  ['GPT-2 small', kib(2 * 12 * 64, 12), 'MHA, 12 layers of 768'],
]

const COMPARE: Record<string, [string, string]> = {
  blocks: ['GPT-2’s block', 'anatomy/layernorm?phase=stream'], mla: ['GPT-2’s K and V', 'anatomy/attention?phase=qkv'],
  cache: ['LLaMA’s GQA', 'lineage/llama?phase=gqa'], moe: ['Mixtral’s experts', 'lineage/mixtral?phase=dispatch'], balance: ['Mixtral’s balancing loss', 'lineage/mixtral?phase=balance'],
}

const CAPS: Record<string, [string, string]> = {
  blocks: ['DeepSeek-V3 changes both halves of GPT-2’s block. Attention becomes multi-head latent attention (MLA), which caches one small latent per token instead of every head’s keys and values. The MLP becomes DeepSeekMoE: one shared expert plus 256 small routed experts, 8 per token. Click a label to jump there.', '61 layers · 671B total · 37B per token'],
  mla: ['MLA squeezes each token’s vector into a latent c of 512 numbers (2 here) and caches only that. Every head’s keys and values are rebuilt from c by up-projections. Position goes into a separate 64-number RoPE key shared by all heads, which is cached too. Hover the cells.', 'c = h · W_DKV · K = c · W_UK · V = c · W_UV'],
  cache: ['Per token and layer, full multi-head attention would cache 2 × 128 heads × 128 numbers; MLA caches 512 + 64. Over 61 layers that is 3.8 MiB against 69 KiB per token, 57 times less, which is what makes a 128K-token context affordable.', '(512 + 64) × 61 layers × 2 bytes'],
  moe: ['DeepSeekMoE splits the experts into many small ones: 256 routed experts, 8 chosen per token, plus one shared expert every token uses. Scores are sigmoids, normalised over the chosen 8. More, smaller experts give far more combinations to specialise.', '1 shared + top 8 of 256 routed'],
  balance: ['To keep experts evenly used without an extra loss term, each expert has a bias that is added to its score only when choosing experts. After each step, an overloaded expert’s bias goes down and an idle one’s goes up. Shown with 16 toy experts and 64 tokens.', 'choose by s_i + b_i · weight by s_i'],
}

export function mountDeepseek(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: '',
      eyebrow: 'Architectures · Decoder-only',
      title: 'DeepSeek',
      subtitle: 'DeepSeek-V3 · latent attention and fine-grained experts',
      specs: [
        { label: 'compared', value: 'DeepSeek-V3', real: 'GPT-2 small', realLabel: 'vs' },
        { label: 'layers', value: '61', real: '12' },
        { label: 'd_model', value: '7,168', real: '768' },
        { label: 'attention', value: 'MLA · 128 heads', real: 'MHA · 12' },
        { label: 'experts', value: '1 shared + 8 of 256', real: '1 dense MLP' },
        { label: 'params', value: '671B · 37B active', real: '124M' },
        { label: 'context', value: '128K', real: '1,024' },
      ],
    },
    size: [1040, 480],
    aria: 'DeepSeek-V3 compared with GPT-2: multi-head latent attention caches one small latent per token instead of all keys and values, and each MLP becomes a shared expert plus 8 of 256 small routed experts, balanced by a per-expert bias.',
    phases: PHASES, learn: 'deepseek', tokens: TOKS, compare: COMPARE, caps: CAPS,
    still: ['mla', 11],
    hints: {
      blocks: 'Click a dark label on the drawing to jump to that part.',
      mla: 'Hover a cell of c, K or V to see how it is computed; click or tap to pin it.',
      moe: 'Hover an expert to read its affinity; click or tap to pin it.',
    },
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { pill, addNode, caption, title, lane, glass } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46
  const diamond = (x: number, y: number, r: number, a: number) => {
    ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath()
    ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.8 * a); ctx.lineWidth = 1.2; ctx.stroke()
  }
  /** The 16 × 16 grid of routed experts, filled by affinity; returns the cell size. */
  function grid(x: number, y: number, cs: number, a: number, on: (e: number) => number, sel: (e: number) => number, col: RGB) {
    for (let e = 0; e < NR; e++) {
      const gx = x + (e % GRID) * cs, gy = y + Math.floor(e / GRID) * cs
      ctx.fillStyle = rgba(C.ink, (0.04 + 0.3 * on(e)) * a); ctx.fillRect(gx + 0.5, gy + 0.5, cs - 1, cs - 1)
      const s = sel(e)
      if (s > 0) { ctx.fillStyle = rgba(col, 0.9 * s * a); ctx.fillRect(gx + 0.5, gy + 0.5, cs - 1, cs - 1) }
    }
  }

  /* ---------- scene 1: the blocks ---------- */
  function sceneBlocks(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const sx0 = pad + 130, sx1 = W - pad - 16, fx = (f: number) => lerp(sx0, sx1, f)
    const ys = [top + avail * 0.2, top + avail * 0.56], as = [eout(clamp(p / 0.3)), eout(clamp((p - 0.25) / 0.3))]
    const X = { pos: fx(0.04), n1: fx(0.17), at: fx(0.33), a1: fx(0.47), n2: fx(0.6), ml: fx(0.78), rt: fx(0.68), a2: fx(0.95) }
    ys.forEach((y, r) => {
      const a = as[r]
      if (a <= 0) return
      k.rowName(r ? 'DeepSeek-V3' : 'GPT-2', r ? '671B · 37B active' : 'small · 124M', pad, y, a)
      for (let i = 0; i < N; i++) lane(sx0, lerp(sx0, sx1, a), y + (i - (N - 1) / 2) * 4, hue(i), a)
      if (!r) {
        rr(X.pos - 26, y - 15, 52, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.7 * a); ctx.lineWidth = 1; ctx.stroke()
        caption('+ W_P', X.pos, y + 4, a, C.ink); caption('learned positions', X.pos, y + 44, a)
      } else caption('RoPE, in attention', X.pos, y + 44, a)
      ;[X.n1, X.n2].forEach((x) => { glass(x, y - 26, y + 26, 0.2, { w: 8, d: 9 }, a); caption(r ? 'RMSNorm' : 'LayerNorm', x, y + 44, a) })
      glass(X.at, y - 26, y + 26, r ? 0.55 : 0.2, { w: 8, d: 9 }, a)
      caption(r ? 'MLA' : 'MHA', X.at, y + 44, a, r ? C.ink2 : C.mute)
      addNode(X.a1, y, a); addNode(X.a2, y, a)
      if (!r) { glass(X.ml, y - 26, y + 26, 0.2, { w: 8, d: 9 }, a); caption('MLP · GELU', X.ml, y + 44, a); return }
      // DeepSeekMoE: a shared expert that always runs, and 8 of 256 routed experts
      const ey = y + 58, gx = X.ml - 10, cs = 5
      diamond(X.rt, y, 10, a)
      glass(X.rt + 36, ey - 14, ey + 14, 0.6, { w: 7, d: 6 }, a)
      caption('shared', X.rt + 36, ey + 34, a, C.ink2)
      grid(gx, ey - 40, cs, a, (e) => MOE.s[e], (e) => (MOE.top.includes(e) ? 1 : 0), hue(0))
      caption('256 routed, 8 run', gx + (GRID * cs) / 2, ey + 58, a, C.ink2)
      ctx.strokeStyle = rgba(hue(0), 0.8 * a); ctx.lineWidth = 1.5
      ctx.beginPath(); ctx.moveTo(X.rt, y + 10); ctx.bezierCurveTo(X.rt, ey, X.rt + 20, ey, X.rt + 28, ey); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(X.rt + 6, y + 6); ctx.bezierCurveTo(X.rt + 40, y + 16, gx - 30, ey - 20, gx - 4, ey - 20); ctx.stroke()
    })
    const la = eout(clamp((p - 0.55) / 0.25)), [yA, yB] = ys
    if (la > 0) {
      ctx.setLineDash([2, 4]); ctx.strokeStyle = rgba(C.ink, 0.35 * la); ctx.lineWidth = 1
      for (const x of [X.at, X.ml]) { ctx.beginPath(); ctx.moveTo(x, yA + 52); ctx.lineTo(x, yB - 58); ctx.stroke() }
      ctx.setLineDash([])
      pill('MLA', X.at - 30, yB - 44, 'mla', la)
      pill('KV cache', X.at + 44, yB - 44, 'cache', la)
      pill('8 of 256', X.ml - 34, yB - 44, 'moe', la)
      pill('bias', X.ml + 40, yB - 44, 'balance', la)
    }
    mk.formula = { segs: [['GPT-2  attn → MLP', C.ink2], ['     ·     ', C.mute], ['DeepSeek-V3  MLA → shared + top-8 routed experts', C.ink]], note: 'The first 3 of the 61 layers keep a dense MLP. V3 also trains with multi-token prediction: an extra module predicts the token after next, as a denser training signal.' }
  }

  /* ---------- scene 2: MLA, three GEMMs through a narrow latent ---------- */
  function sceneMla(p: number) {
    const { W } = stage
    const fin = eout(clamp(p / 0.06)), c = 22
    const rH: Rect = { x: pad + 70, y: top + 236, c }
    const rC: Rect = { x: rH.x + DM * c + 36, y: rH.y, c }, rK: Rect = { x: rC.x + DC * c + 44, y: rH.y, c }, rV: Rect = { x: rK.x + NH * DH * c + 36, y: rH.y, c }
    const rWd: Rect = { x: rC.x, y: rH.y - DM * c - 26, c }, rWk: Rect = { x: rK.x, y: rH.y - DC * c - 26, c }, rWv: Rect = { x: rV.x, y: rH.y - DC * c - 26, c }
    const g1 = gemm(clamp((p - 0.06) / 0.3), N, DC, DM, 'slow'), g2 = gemm(clamp((p - 0.4) / 0.18), N, NH * DH, DC, 'fast'), g3 = gemm(clamp((p - 0.6) / 0.18), N, NH * DH, DC, 'fast')
    TOKS.forEach((t, i) => drawChip(rH.x - 10 - chipW(t.text), rH.y + (i + 0.5) * c, t, fin, Math.min(20, c * 0.8)))
    mk.drawMat({ r: rH, vals: MLA.H, kind: 'row', alpha: fin, name: 'h', shape: '5 × 8', real: '5 × 7,168', labelW: DM * c + 30 })
    mk.drawMat({ r: rWd, vals: MLA.Wd, kind: 'w', alpha: fin, name: 'W_DKV', shape: '8 × 2', real: '7,168 × 512', labelW: 260 })
    mk.drawMat({ r: rC, vals: MLA.C, kind: 'row', alpha: fin, name: 'c', shape: '5 × 2', label: 'bottom', reveal: g1.rev, labelW: 90 })
    const ka = fin * eout(clamp((p - 0.36) / 0.06))
    mk.drawMat({ r: rWk, vals: MLA.Wk, kind: 'w', alpha: ka, name: 'W_UK', shape: '2 × 8', real: '512 × 128·128' })
    mk.drawMat({ r: rK, vals: MLA.K, kind: 'row', alpha: ka, name: 'K', shape: '5 × 2·4', real: '5 × 128·128', label: 'bottom', reveal: g2.rev })
    const va = fin * eout(clamp((p - 0.56) / 0.06))
    mk.drawMat({ r: rWv, vals: MLA.Wv, kind: 'w', alpha: va, name: 'W_UV', shape: '2 × 8', real: '512 × 128·128' })
    mk.drawMat({ r: rV, vals: MLA.V, kind: 'row', alpha: va, name: 'V', shape: '5 × 2·4', real: '5 × 128·128', label: 'bottom', reveal: g3.rev })
    // head boundaries inside K and V
    ctx.setLineDash([2, 3]); ctx.strokeStyle = rgba(C.ink, 0.45); ctx.lineWidth = 1
    for (const [r, a] of [[rK, ka], [rV, va]] as [Rect, number][]) { if (a <= 0) continue; ctx.globalAlpha = a; ctx.beginPath(); ctx.moveTo(r.x + DH * c, r.y - 4); ctx.lineTo(r.x + DH * c, r.y + N * c + 4); ctx.stroke(); ctx.globalAlpha = 1 }
    ctx.setLineDash([])
    mk.hit('c', rC, N, DC); mk.hit('k', rK, N, NH * DH); mk.hit('v', rV, N, NH * DH)
    // only the latent is cached
    const cb = eout(clamp((p - 0.8) / 0.08))
    if (cb > 0) {
      ctx.setLineDash([4, 3]); ctx.strokeStyle = rgba(C.ink, 0.9 * cb); ctx.lineWidth = 1.3
      rr(rC.x - 7, rC.y - 7, DC * c + 14, N * c + 14, 6); ctx.stroke(); ctx.setLineDash([])
      title('kv cache', rC.x - 7, rC.y + N * c + 46, cb)
      const tx = rV.x + NH * DH * c + 34, ty = top + 90
      if (tx + 150 < W - pad) {
        title('cached per token', tx, ty, cb)
        caption('MHA: K and V', tx, ty + 30, cb, C.mute, 'left')
        ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink2, cb)
        ctx.fillText('2 × 2 × 4 = 16', tx, ty + 50)
        caption('real, per layer: 32,768', tx, ty + 68, cb, C.mute, 'left')
        caption('MLA: c (+ RoPE key)', tx, ty + 104, cb, C.mute, 'left')
        ctx.font = F.mono(13, 500); ctx.fillStyle = rgba(C.ink, cb)
        ctx.fillText('2', tx, ty + 124)
        caption('real, per layer: 512 + 64 = 576', tx, ty + 142, cb, C.mute, 'left')
      }
    }
    const fs = mk.resolve({ c: { g: g1, K: DM }, k: { g: g2, K: DC }, v: { g: g3, K: DC } })
    if (fs?.key === 'c') mk.gemmOverlay({ A: rH, Av: MLA.H, B: rWd, Bv: MLA.Wd, C: rC, f: fs, names: ['c', 'h', 'W_DKV'], note: 'Down-projection: the token’s 7,168 numbers (8 here) squeezed into a 512-number latent (2 here). This is what the cache keeps.' })
    else if (fs?.key === 'k') mk.gemmOverlay({ A: rC, Av: MLA.C, B: rWk, Bv: MLA.Wk, C: rK, f: fs, names: ['K', 'c', 'W_UK'], note: `Up-projection: head ${Math.floor(fs.j / DH) + 1}’s key, rebuilt from the latent. At inference W_UK can be folded into the query projection, so the keys never need to be materialised.` })
    else if (fs?.key === 'v') mk.gemmOverlay({ A: rC, Av: MLA.C, B: rWv, Bv: MLA.Wv, C: rV, f: fs, names: ['V', 'c', 'W_UV'], note: `Up-projection: head ${Math.floor(fs.j / DH) + 1}’s value, rebuilt from the same latent. W_UV can likewise be folded into the output projection.` })
    else mk.formula = { segs: [['c = h · W_DKV', C.ink], ['   ·   ', C.mute], ['K = c · W_UK', C.ink2], ['   ·   ', C.mute], ['V = c · W_UV', C.ink2]], note: 'The latent is the bottleneck: every head’s keys and values are functions of the same 512 numbers per token. A separate 64-number key carries RoPE, since a rotation cannot pass through the up-projection.' }
  }

  /* ---------- scene 3: KV cache per token ---------- */
  function sceneCache(p: number) {
    const { W } = stage
    const x0 = pad + 170, w = W - pad - x0 - 170, max = CACHE[0][1], rowH = 58
    title('kv cache per token, bf16', x0, top + 18, eout(clamp(p / 0.06)))
    title('at 128K tokens', W - pad - 130, top + 18, eout(clamp(p / 0.06)))
    CACHE.forEach(([name, v, sub], r) => {
      const a = eout(clamp((p - 0.04 - r * 0.12) / 0.12)), y = top + 44 + r * rowH, g = eio(clamp((p - 0.06 - r * 0.12) / 0.2))
      if (a <= 0) return
      const isMla = name === 'MLA', isGpt = r === CACHE.length - 1
      ctx.font = serifAt(20); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(isGpt ? C.ink2 : C.ink, a); ctx.fillText(name, pad, y + 18)
      caption(sub, pad, y + 36, a, C.mute, 'left')
      const bw = Math.max(2, (v / max) * (w - 100) * g)
      ctx.fillStyle = rgba(isMla ? C.tok[4] : C.ink, (isMla ? 0.9 : isGpt ? 0.35 : 0.7) * a); ctx.fillRect(x0, y + 4, bw, 20)
      const label = v >= 1024 ? `${(v / 1024).toFixed(1)} MiB` : `${v.toFixed(1)} KiB`
      ctx.font = F.mono(12, isMla ? 600 : 400); ctx.fillStyle = rgba(isMla ? C.ink : C.ink2, a * g); ctx.fillText(label, x0 + bw + 8, y + 19)
      const at = isGpt ? `${((v * 1024) / 1024).toFixed(0)} MiB at 1,024` : (v * 131072) / 1024 / 1024 >= 100 ? `${Math.round((v * 131072) / 1024 / 1024)} GiB` : `${((v * 131072) / 1024 / 1024).toFixed(1)} GiB`
      ctx.font = F.mono(12); ctx.fillStyle = rgba(isMla ? C.ink : C.ink2, a * g); ctx.fillText(at, W - pad - 130, y + 19)
    })
    const ra = eout(clamp((p - 0.75) / 0.1))
    caption(`MLA caches ${Math.round(CACHE[0][1] / CACHE[3][1])}× less than MHA and about twice MQA, and the DeepSeek-V2 paper reports better quality than MHA`, x0, top + 44 + CACHE.length * rowH + 16, ra, C.ink2, 'left')
    mk.formula = { segs: [['MHA  2 × 128 × 128 × 61 × 2 B = 3.8 MiB', C.ink2], ['     ·     ', C.mute], ['MLA  (512 + 64) × 61 × 2 B = 68.6 KiB', C.ink]], note: 'The cache grows with every token of every sequence being served, so its size per token sets how long a context and how large a batch fit in GPU memory.' }
  }

  /* ---------- scene 4: DeepSeekMoE ---------- */
  function sceneMoe(p: number) {
    const { H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.06)), cs = 18, gx = pad + 250, gy = top + 40
    const ga = eout(clamp((p - 0.04) / 0.2)), pick = (e: number) => { const r = MOE.top.indexOf(e); return r < 0 ? 0 : clamp((p - 0.3 - r * 0.025) / 0.04) }
    drawChip(pad, gy + GRID * cs * 0.5, TOKS[1], fin, 22)
    diamond(pad + 70, gy + GRID * cs * 0.5, 11, fin)
    // the shared expert: always on
    glass(pad + 150, gy + 20, gy + 70, 0.6, { w: 9, d: 8 }, fin)
    caption('shared', pad + 150, gy + 96, fin, C.ink2); caption('always runs', pad + 150, gy + 112, fin)
    ctx.strokeStyle = rgba(hue(1), 0.8 * fin); ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(pad + 60, gy + GRID * cs * 0.5 - 8); ctx.bezierCurveTo(pad + 90, gy + 45, pad + 120, gy + 45, pad + 140, gy + 45); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(pad + 81, gy + GRID * cs * 0.5); ctx.lineTo(gx - 8, gy + GRID * cs * 0.5); ctx.stroke()
    title('256 routed experts, shaded by affinity', gx, gy - 14, fin)
    grid(gx, gy, cs, ga, (e) => MOE.s[e], pick, hue(1))
    MOE.top.forEach((e) => {
      const a = pick(e)
      if (a <= 0) return
      ctx.strokeStyle = rgba(C.ink, a); ctx.lineWidth = 1.5; ctx.strokeRect(gx + (e % GRID) * cs, gy + Math.floor(e / GRID) * cs, cs, cs)
    })
    mk.hit('e', { x: gx, y: gy, c: cs }, GRID, GRID)
    // the chosen 8, normalised
    const la = eout(clamp((p - 0.52) / 0.08)), lx = gx + GRID * cs + 50
    if (la > 0) {
      title('top 8, normalised', lx, gy - 14, la)
      MOE.top.forEach((e, r) => {
        const y = gy + 12 + r * 24, w = MOE.w[r]
        ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, la); ctx.fillText(`E${e + 1}`, lx, y)
        ctx.fillStyle = rgba(hue(1), 0.85 * la); ctx.fillRect(lx + 50, y - 6, w * 600, 12)
        ctx.fillStyle = rgba(C.ink, la); ctx.fillText(fmt(w), lx + 58 + w * 600, y)
      })
      caption(`σ scores ${fmt(MOE.s[MOE.top[7]])}–${fmt(MOE.s[MOE.top[0]])} / their sum: near 1/8 each`, lx, gy + 12 + TOPK * 24 + 4, la, C.mute, 'left')
    }
    const ca = eout(clamp((p - 0.72) / 0.1))
    if (ca > 0) {
      const y = top + avail - 30
      caption(`ways to pick the experts:  Mixtral C(8, 2) = 28  ·  DeepSeek-V3 C(256, 8) ≈ ${Math.round(choose(256, 8) / 1e12)} trillion`, gx, y, ca, C.ink2, 'left')
      caption('each routed expert: 3 × 7,168 × 2,048 weights, a quarter of a Mixtral expert', gx, y + 18, ca, C.mute, 'left')
    }
    const f = mk.hovered('e')
    if (f) {
      const e = f.i * GRID + f.j, r = MOE.top.indexOf(e)
      ctx.strokeStyle = rgba(C.ink, 1); ctx.lineWidth = 2; ctx.strokeRect(gx + f.j * cs - 1, gy + f.i * cs - 1, cs + 2, cs + 2)
      mk.formula = r >= 0
        ? { segs: [[`E${e + 1}`, C.ink], ['  s = σ(u · e)  =  ', C.mute], [fmtF(MOE.s[e]), C.ink2], ['  →  weight  ', C.mute], [fmtF(MOE.w[r]), C.ink]], note: `Rank ${r + 1} of 8 for ‘cat’: it runs, and its output counts ${Math.round(MOE.w[r] * 100)}%.` }
        : { segs: [[`E${e + 1}`, C.ink], ['  s = σ(u · e)  =  ', C.mute], [fmtF(MOE.s[e]), C.ink2]], note: 'Not in the top 8 for ‘cat’: this expert does not run for it.' }
    } else mk.formula = { segs: [['y = x + shared(x) + Σ over top 8:  g_i · E_i(x)', C.ink], ['     ·     ', C.mute], ['g_i = s_i / Σ s_j', C.ink2]], note: 'Toy affinities for one token (‘cat’). Each expert has a learned centroid e_i; the score is a sigmoid of its dot product with the token. The shared expert holds knowledge every token needs, so the routed ones can specialise.' }
  }

  /* ---------- scene 5: balancing with a bias ---------- */
  function sceneBalance(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.06)), step = Math.min(STEPS, Math.floor(eio(clamp((p - 0.08) / 0.8)) * (STEPS + 1)))
    const x0 = pad + 20, bw = 26, gap = 8, yb = top + avail - 70, hMax = avail - 150, even = (BT * BK) / BE, vMax = 24
    const cur = BAL[step]
    title(`step ${step} of ${STEPS}`, x0, top + 12, fin)
    for (let e = 0; e < BE; e++) {
      const x = x0 + e * (bw + gap), h = (cur.load[e] / vMax) * hMax, b = cur.b[e]
      ctx.fillStyle = rgba(cur.load[e] > even ? C.ink : C.ink2, (cur.load[e] > even ? 0.85 : 0.55) * fin); ctx.fillRect(x, yb - h, bw, h)
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, fin); ctx.fillText(String(e + 1), x + bw / 2, yb + 6)
      // the bias as a small signed bar under the axis
      // the bias as a small signed bar under the axis: filled up when positive, outlined down when negative
      const bh = (b / 0.08) * 20
      if (bh >= 0) { ctx.fillStyle = rgba(C.ink, 0.85 * fin); ctx.fillRect(x + 6, yb + 40 - bh, bw - 12, Math.max(1, bh)) }
      else { ctx.strokeStyle = rgba(C.neg, 0.9 * fin); ctx.lineWidth = 1; ctx.strokeRect(x + 6.5, yb + 40.5, bw - 13, -bh - 1) }
    }
    ctx.strokeStyle = rgba(C.ink, 0.25 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 - 6, yb + 0.5); ctx.lineTo(x0 + BE * (bw + gap), yb + 0.5); ctx.stroke()
    ctx.strokeStyle = rgba(C.ink, 0.12 * fin); ctx.beginPath(); ctx.moveTo(x0 - 6, yb + 40.5); ctx.lineTo(x0 + BE * (bw + gap), yb + 40.5); ctx.stroke()
    ctx.setLineDash([4, 4]); ctx.strokeStyle = rgba(C.ink2, 0.8 * fin); ctx.beginPath(); ctx.moveTo(x0 - 6, yb - (even / vMax) * hMax); ctx.lineTo(x0 + BE * (bw + gap), yb - (even / vMax) * hMax); ctx.stroke(); ctx.setLineDash([])
    caption(`even: ${even} tokens`, x0 + BE * (bw + gap) + 4, yb - (even / vMax) * hMax + 4, fin, C.ink2, 'left')
    caption('tokens per expert', x0, top + 32, fin, C.mute, 'left')
    caption('bias b_i', x0 + BE * (bw + gap) + 4, yb + 44, fin, C.mute, 'left')
    // spread over the steps
    const cx0 = W - pad - 250, cx1 = W - pad - 10, cy0 = top + 60, cy1 = top + 220, sMax = 24
    const spread = BAL.map((h) => Math.max(...h.load) - Math.min(...h.load))
    title('max − min load', cx0, cy0 - 18, fin)
    ctx.strokeStyle = rgba(C.ink, 0.25 * fin); ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy1); ctx.lineTo(cx1, cy1); ctx.stroke()
    ctx.strokeStyle = rgba(C.ink, 0.9 * fin); ctx.lineWidth = 1.6; ctx.beginPath()
    for (let s = 0; s <= step; s++) { const x = lerp(cx0, cx1, s / STEPS), y = lerp(cy1, cy0, spread[s] / sMax); if (s) ctx.lineTo(x, y); else ctx.moveTo(x, y) }
    ctx.stroke()
    ctx.fillStyle = rgba(C.ink, fin); ctx.beginPath(); ctx.arc(lerp(cx0, cx1, step / STEPS), lerp(cy1, cy0, spread[step] / sMax), 3, 0, 7); ctx.fill()
    ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, fin)
    ctx.fillText('0', cx0, cy1 + 6); ctx.fillText(String(STEPS), cx1, cy1 + 6)
    caption(`now ${spread[step]} (from ${spread[0]})`, cx0, cy1 + 36, fin, C.ink2, 'left')
    ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, fin)
    fillRich('b_i −= γ if overloaded, += γ if idle', cx0, cy1 + 70)
    fillRich('gate weights still use s_i alone', cx0, cy1 + 88)
    mk.formula = { segs: [['chosen = top-k of (s_i + b_i)', C.ink], ['     ·     ', C.mute], ['weight ∝ s_i', C.ink2], ['     ·     ', C.mute], [`b_i ± γ each step (γ = ${GAMMA} here, 0.001 in V3)`, C.ink2]], note: 'The bias changes which experts are chosen but not how much their outputs count, so balancing does not distort the training signal the way an auxiliary loss can. V3 keeps only a very small sequence-level balance loss as a safeguard.' }
  }

  return { blocks: sceneBlocks, mla: sceneMla, cache: sceneCache, moe: sceneMoe, balance: sceneBalance }
}
