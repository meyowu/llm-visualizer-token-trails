import { F, fillRich, rr, serifAt } from '../../core/draw'
import { select, type Spec } from '../../core/frame'
import { raw, t, tf } from '../../core/i18n'
import { getParams, setParams } from '../../core/link'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eout, lerp } from '../../core/util'
import { families, fmtBytes, fmtParams, fmtTokens, kindOf, kvBytes, modelById, type Model } from '../../lib/models/data'
import type { Nav } from '../registry'
import { mountExhibit, words, type Env } from '../kit'

/*
 * Any two decoder language models, side by side: what differs, layer by layer, the attention heads and what they
 * cache, the MLP or its experts, where the parameters are, and the KV cache as the context grows. Every number comes
 * from the model's own config.json and checkpoint (scripts/models-export.ts); nothing is drawn at toy size.
 */

const PHASES = [
  { id: 'diff', name: 'What differs', short: 'Diff', dur: 9 },
  { id: 'layers', name: 'Layer by layer', short: 'Layers', dur: 9 },
  { id: 'heads', name: 'Heads and the cache per token', short: 'Heads', dur: 9 },
  { id: 'mlp', name: 'The MLP: dense or experts', short: 'MLP', dur: 9 },
  { id: 'params', name: 'Where the parameters are', short: 'Params', dur: 8 },
  { id: 'kv', name: 'KV cache against context', short: 'Cache', dur: 9 },
]

const COMPARE: Record<string, [string, string]> = {
  diff: ['LLaMA, drawn as a diff to GPT-2', 'lineage/llama'],
  layers: ['Mamba’s fixed state', 'lineage/mamba?phase=cost'],
  heads: ['Grouped-query attention', 'lineage/llama?phase=gqa'],
  mlp: ['Shared and routed experts', 'lineage/deepseek?phase=moe'],
  params: ['Mixtral’s active parameters', 'lineage/mixtral?phase=params'],
  kv: ['How big the cache gets', 'serving/kv-cache?phase=size'],
}

const nf = (n: number) => n.toLocaleString('en-US')
/** Pieces of text, each translated on its own (they carry numbers), joined. */
const join = (pieces: (string | false | null | undefined)[]) => pieces.filter(Boolean).map((s) => t(s as string)).join(' · ')

/* ---------- describing a model in words ---------- */
const WORD: Record<string, string> = { F: 'full', S: 'sliding', C: 'chunked', M: 'latent', L: 'linear' }
/** The shortest repeating unit of the layer codes (a partial last repeat allowed). */
function period(codes: string): number {
  for (let p = 1; p <= codes.length; p++) {
    const n = Math.floor(codes.length / p) * p
    if (p === codes.length || (n >= 2 * p && codes.slice(0, n) === codes.slice(0, p).repeat(n / p))) return p
  }
  return codes.length
}
function layerWord(m: Model, code: string): string {
  const k = m.attn.kinds[code], kind = WORD[kindOf(code)]
  const w = k.window ? `${kind} (${nf(k.window)})` : kind
  return noRope(m, code) ? `${w}, no RoPE` : w
}
/** An attention layer without RoPE in a model that has it elsewhere (a linear layer never has positions to rotate). */
const noRope = (m: Model, code: string) => code === code.toLowerCase() && m.pos.kind === 'rope' && kindOf(code) !== 'L'
function patternText(m: Model): string[] {
  const codes = m.attn.codes, order = [...new Set(codes)]
  const out = order.map((c) => `${[...codes].filter((x) => x === c).length} ${layerWord(m, c)}`)
  if (order.length > 1) {
    const unit = codes.slice(0, period(codes))
    out.push(order.map((c) => [...unit].filter((x) => x === c).length).join(' : '))
  }
  return out
}
function attnText(m: Model): string[] {
  // Kimi Linear's MLA layers use no RoPE: the small extra key part is shared by the heads but not rotated
  if (m.mla) return [`MLA · ${m.heads} heads`, m.pos.kind === 'rope' ? `latent ${m.mla.latent} + RoPE key ${m.mla.rope}` : `latent ${m.mla.latent} + shared key ${m.mla.rope}`, m.indexer ? `DSA indexer, top ${nf(m.indexer.topk)}` : '']
  if (m.kvHeads === m.heads) return [`MHA · ${m.heads} heads × ${m.headDim}`]
  if (m.kvHeads === 1) return [`MQA · ${m.heads} heads × ${m.headDim}`]
  return [`GQA · ${m.heads} query, ${m.kvHeads} key/value heads`, `head size ${m.headDim}`]
}
function posText(m: Model): string[] {
  const p = m.pos
  if (p.kind === 'learned') return [`learned, ${nf(m.context)} positions`]
  if (p.kind === 'none') return ['none: no layer uses RoPE']
  const scaling: Record<string, string> = { yarn: 'YaRN for long context', llama3: 'Llama 3 scaling for long context', linear: 'linear scaling for long context' }
  return [`RoPE θ ${nf(p.theta)}`, p.partial < 0.99 ? `on ${Math.round(p.partial * 100)}% of each head` : '', p.localTheta ? `local layers θ ${nf(p.localTheta)}` : '',
    p.layers < attnLayers(m) ? `${attnLayers(m) - p.layers} attention layers without` : '', p.scaling ? scaling[p.scaling] ?? '' : '']
}
/** Layers that attend (all but the linear ones). */
const attnLayers = (m: Model) => [...m.attn.codes].filter((c) => kindOf(c) !== 'L').length
const normText = (m: Model) => [m.norm, m.sandwich ? 'before and after each sub-layer' : 'before each sub-layer', m.qkNorm ? 'QK-Norm' : '']
function mlpText(m: Model): string[] {
  const act = m.act
  if (!m.moe) return [`${act} · ${nf(m.denseWidth ?? 0)} wide`, `${((m.denseWidth ?? 0) / m.hidden).toFixed(1)}× d_model`]
  const e = m.moe, dense = e.dense ? (m.mlp.startsWith('D'.repeat(e.dense)) ? `first ${e.dense} layers dense` : 'every other layer dense') : ''
  return [`${act} · ${e.experts} experts × ${nf(e.width)}`, `top ${e.topK}${e.shared ? ` + ${e.shared} shared` : ''}`, dense]
}
function extrasText(m: Model): string[] {
  const x = [m.sinks && 'attention sinks', m.gatedAttn && 'gated attention', m.softcap && `logit soft-cap ${m.softcap.attn ?? '–'} / ${m.softcap.final ?? '–'}`,
    m.mtpLayers > 0 && 'multi-token prediction', m.qkvBias && 'QKV bias'].filter(Boolean) as string[]
  return x.length ? x : ['—']
}
/** Numbers per token the layers that keep every token store, over all of them. */
const growing = (m: Model) => [...m.attn.codes].reduce((s, c) => s + (/[FM]/i.test(c) ? m.attn.kinds[c].perToken : 0), 0)
/** How wide a token's MLP is: the dense width, or the experts it is sent to plus the shared ones. */
function tokenWidth(m: Model): number {
  if (!m.moe) return m.denseWidth ?? 0
  return (m.moe.topK + m.moe.shared) * m.moe.width
}

export function mountCompare(root: HTMLElement, nav: Nav): () => void {
  const q = getParams()
  const sel = { a: modelById(q.get('a') ?? 'gpt2').id, b: modelById(q.get('b') ?? 'deepseek-v3').id }
  const A = () => modelById(sel.a), B = () => modelById(sel.b)

  const specs = (): Spec[] => {
    const a = A(), b = B()
    return [
      { label: 'compared', value: a.name, real: b.name, realLabel: 'vs' },
      { label: 'layers', value: `${a.layers} · ${b.layers}` },
      { label: 'd_model', value: `${nf(a.hidden)} · ${nf(b.hidden)}` },
      { label: 'params', value: `${fmtParams(a.params.total)} · ${fmtParams(b.params.total)}` },
      { label: 'per token', value: `${fmtParams(a.params.active)} · ${fmtParams(b.params.active)}` },
      { label: 'context', value: `${fmtTokens(a.context)} · ${fmtTokens(b.context)}` },
    ]
  }
  const CAPS: Record<string, [string, string]> = {}
  function captions() {
    const a = A(), b = B(), n = Math.min(32768, a.context, b.context)
    CAPS.diff = [tf('{} and {}, row by row; the rows that differ are bright. Pick any two models above. Every number comes from the model’s own config and checkpoint on Hugging Face.', a.name, b.name), tf('{} of {} rows differ', diffRows().filter((r) => r[1] !== r[2]).length, diffRows().length)]
    CAPS.layers = [tf('One column per layer: its attention above, its MLP below. {}: {}. {}: {}.', a.name, join(patternText(a)), b.name, join(patternText(b))), tf('{} layers · {} layers', a.layers, b.layers)]
    CAPS.heads = ['Query heads above, the key and value heads they read below. Grouped-query attention shares one key/value head among several query heads; latent attention caches one small vector per token instead; a linear layer keeps a fixed state.', tf('per token: {} · {}', fmtBytes(2 * growing(a)), fmtBytes(2 * growing(b)))]
    CAPS.mlp = ['A dense MLP sends every token through one wide network. A mixture of experts stores many smaller ones and sends each token to a few, plus any shared expert every token uses. Lit cells are one token’s experts.', tf('width per token: {} · {}', nf(tokenWidth(a)), nf(tokenWidth(b)))]
    CAPS.params = [tf('{} stores {} parameters and a token runs through {}; {} stores {} and runs {}. Routed experts a token is not sent to (hatched) sit in memory unused.', a.name, fmtParams(a.params.total), fmtParams(a.params.active), b.name, fmtParams(b.params.total), fmtParams(b.params.active)), tf('{} · {} per token', fmtParams(a.params.active), fmtParams(b.params.active))]
    CAPS.kv = [tf('The cache for one sequence, in 16-bit, up to each model’s context length. At {} tokens {} holds {} and {} holds {}. Sliding-window and linear layers stop growing; full and latent layers grow with every token.', fmtTokens(n), a.name, fmtBytes(kvBytes(a, n)), b.name, fmtBytes(kvBytes(b, n))), 'Σ layers min(n, window) × per token, or a state']
  }
  /** The comparison table: label, and each model's value as English pieces. */
  function diffRows(): [string, string, string, string[], string[]][] {
    const a = A(), b = B()
    const rows: [string, string[], string[]][] = [
      ['layers', [`${a.layers} layers`, `d_model ${nf(a.hidden)}`], [`${b.layers} layers`, `d_model ${nf(b.hidden)}`]],
      ['attention', attnText(a), attnText(b)],
      ['layer types', patternText(a), patternText(b)],
      ['positions', posText(a), posText(b)],
      ['norm', normText(a), normText(b)],
      ['MLP', mlpText(a), mlpText(b)],
      ['vocabulary', [`${nf(a.vocab)} tokens`, a.tied ? 'tied to the output' : ''], [`${nf(b.vocab)} tokens`, b.tied ? 'tied to the output' : '']],
      ['context', [`${nf(a.context)} tokens`], [`${nf(b.context)} tokens`]],
      ['extras', extrasText(a), extrasText(b)],
    ]
    return rows.map(([l, x, y]) => [l, x.filter(Boolean).join(' · '), y.filter(Boolean).join(' · '), x, y])
  }
  captions()

  return mountExhibit(root, nav, {
    frame: {
      formulaHint: '',
      eyebrow: 'Architectures · Compare',
      title: 'Architecture diff',
      subtitle: 'any two models, read from their own checkpoints',
      specs: specs(),
    },
    size: [1040, 500],
    aria: 'Two language models compared from their own configs and checkpoints: what differs, the attention and MLP of every layer, the attention heads and what they cache, the experts, where the parameters are, and the KV cache as the context grows.',
    phases: PHASES, learn: 'compare', tokens: words([]), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['diff', 8],
    hints: {
      layers: 'Hover a layer to read it.',
      params: 'Hover a part of a bar to read it.',
      kv: 'Hover the chart to read both caches at that length.',
    },
    scenes: (env) => scenes(env, { A, B, sel, specs, captions, diffRows }),
  })
}

interface Ctl {
  A: () => Model; B: () => Model; sel: { a: string; b: string }
  specs: () => Spec[]; captions: () => void
  diffRows: () => [string, string, string, string[], string[]][]
}

function scenes({ stage, ctx, mk, k, frame, player }: Env, c: Ctl) {
  const { title } = k
  const pad = 36, top = 40, bot = 34
  const HUE: [RGB, RGB] = [C.tok[0], C.tok[3]]
  const both = () => [c.A(), c.B()] as const

  /* ---------- the two dropdowns and a swap ---------- */
  const groups = families()
  const refresh = () => { setParams({ a: c.sel.a, b: c.sel.b }); frame.setSpecs(c.specs()); c.captions() }
  const pickB = select(player.meta, 'B', groups, c.sel.b, (v) => { c.sel.b = v; refresh() })
  const swap = document.createElement('button')
  swap.type = 'button'; swap.className = 'cycle'; swap.textContent = '⇄'
  swap.setAttribute('aria-label', t('Swap the two models'))
  swap.addEventListener('click', () => { [c.sel.a, c.sel.b] = [c.sel.b, c.sel.a]; pickA.set(c.sel.a); pickB.set(c.sel.b); refresh() })
  player.meta.prepend(swap)
  const pickA = select(player.meta, 'A', groups, c.sel.a, (v) => { c.sel.a = v; refresh() })
  // the letters in front of the dropdowns take the models' colours
  player.meta.querySelectorAll<HTMLElement>('.pick span').forEach((lab, i) => { lab.style.color = `rgb(${HUE[i].join(',')})` })

  let mouse: [number, number] | null = null
  stage.canvas.addEventListener('pointermove', (e) => { mouse = stage.local(e) as [number, number] })
  stage.canvas.addEventListener('pointerleave', () => { mouse = null })
  const inside = (x: number, y: number, w: number, h: number) => !!mouse && mouse[0] >= x && mouse[0] <= x + w && mouse[1] >= y && mouse[1] <= y + h

  /** A model's name in the serif, in its colour, with a note under it. */
  function name(m: Model, i: number, x: number, y: number, a: number, sub?: string, inline = false) {
    ctx.font = serifAt(20); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(HUE[i], a)
    const w = raw(() => { ctx.fillText(m.name, x, y); return ctx.measureText(m.name).width })
    if (sub) { ctx.font = F.small; ctx.fillStyle = rgba(C.mute, a); raw(() => (inline ? fillRich(sub, x + w + 14, y) : fillRich(sub, x, y + 16))) }
  }
  /** Text already translated (or a name): drawn as is, shrunk to fit w. */
  function fit(s: string, x: number, y: number, w: number, px: number, col: RGB, a: number, weight = 400) {
    let size = px
    ctx.font = F.mono(size, weight)
    raw(() => {
      while (size > 10.5 && ctx.measureText(s).width > w) { size -= 0.5; ctx.font = F.mono(size, weight) }
      let out = s
      while (out.length > 4 && ctx.measureText(out).width > w) out = out.slice(0, -2) + '…'
      ctx.fillStyle = rgba(col, a); ctx.textAlign = 'left'; ctx.fillText(out, x, y)
    })
  }
  function hatch(x: number, y: number, w: number, h: number, col: RGB, a: number, step = 5) {
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip()
    ctx.strokeStyle = rgba(col, 0.55 * a); ctx.lineWidth = 1
    for (let d = -h; d < w; d += step) { ctx.beginPath(); ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); ctx.stroke() }
    ctx.restore()
  }

  /* ---------- 1: what differs ---------- */
  function sceneDiff(p: number) {
    const { W, H } = stage, [a, b] = both()
    const xa = pad + 150, colW = (W - pad - xa - 28) / 2, xb = xa + colW + 28
    name(a, 0, xa, top + 14, eout(clamp(p / 0.1)), `${a.org} · ${a.released}`)
    name(b, 1, xb, top + 14, eout(clamp(p / 0.1)), `${b.org} · ${b.released}`)
    const rows = c.diffRows()
    const y0 = top + 58, rowH = (H - bot - y0) / rows.length
    let differ = 0
    rows.forEach(([label, va, vb, pa, pb], r) => {
      const al = eout(clamp((p - 0.06 - r * 0.045) / 0.12)), y = y0 + r * rowH + rowH / 2 + 4
      if (al <= 0) return
      const d = va !== vb
      if (d) differ++
      ctx.strokeStyle = rgba(C.faint, 0.5 * al); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(pad, y0 + (r + 1) * rowH); ctx.lineTo(W - pad, y0 + (r + 1) * rowH); ctx.stroke()
      title(label, pad + 16, y, al)
      if (d) { ctx.font = F.mono(12, 600); ctx.fillStyle = rgba(C.ink, al); ctx.textAlign = 'left'; ctx.fillText('≠', pad, y) }
      fit(join(pa), xa, y, colW, 12, d ? C.ink : C.mute, al, d ? 500 : 400)
      fit(join(pb), xb, y, colW, 12, d ? C.ink : C.mute, al, d ? 500 : 400)
    })
    const pub = (m: Model) => `${fmtParams(m.params.total)}${m.moe ? ` · ${fmtParams(m.params.active)} ${t('per token')}` : ''}`
    mk.formula = {
      segs: [[tf('{} of {} rows differ', differ, rows.length), C.ink], ['     ', C.mute], [a.name, HUE[0]], [`  ${pub(a)}`, C.ink2], ['   ·   ', C.mute], [b.name, HUE[1]], [`  ${pub(b)}`, C.ink2]],
      note: tf('Parameters counted from every tensor in the checkpoint (published: {} and {}). Active counts include the embeddings; authors differ on that.', `${a.published.total}B`, `${b.published.total}B`),
    }
  }

  /* ---------- 2: layer by layer ---------- */
  function attnCell(code: string, m: Model, x: number, y: number, w: number, h: number, col: RGB, a: number) {
    const kind = kindOf(code)
    ctx.lineWidth = 1
    if (kind === 'F') { ctx.fillStyle = rgba(col, 0.85 * a); ctx.fillRect(x, y, w, h) }
    else if (kind === 'M') { ctx.fillStyle = rgba(col, 0.85 * a); ctx.fillRect(x, y + h * 0.55, w, h * 0.45); ctx.strokeStyle = rgba(col, 0.7 * a); ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1) }
    else if (kind === 'S' || kind === 'C') {
      ctx.fillStyle = rgba(col, 0.85 * a); ctx.fillRect(x, y + h * 0.75, w, h * 0.25)
      if (kind === 'C') ctx.setLineDash([2, 2])
      ctx.strokeStyle = rgba(col, 0.6 * a); ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); ctx.setLineDash([])
    } else { ctx.strokeStyle = rgba(col, 0.7 * a); ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); ctx.fillStyle = rgba(col, 0.85 * a); ctx.fillRect(x + 1, y + h / 2 - 1, w - 2, 2) }
    if (noRope(m, code)) { ctx.strokeStyle = rgba(C.ink, 0.8 * a); ctx.beginPath(); ctx.arc(x + w / 2, y - 5, Math.min(2.5, w / 2), 0, 7); ctx.stroke() }
  }
  const KIND_NOTE: Record<string, string> = {
    F: 'attends to every earlier token and keeps its K and V',
    S: 'attends to the last tokens only, a sliding window',
    C: 'attends within its chunk of the text only',
    M: 'caches one compressed latent per token (MLA)',
    L: 'linear attention: a fixed-size state, whatever the length',
  }
  function kindLine(m: Model, code: string): string {
    const k = m.attn.kinds[code], kind = kindOf(code)
    if (kind === 'L') return t(`state ${nf(k.state ?? 0)} numbers`)
    return t(`${nf(k.perToken)} numbers per token`) + (k.window ? ` · ${t(`window ${nf(k.window)}`)}` : '')
  }
  function sceneLayers(p: number) {
    const { W, H } = stage, ms = both()
    const xs = pad + 170, maxL = Math.max(ms[0].layers, ms[1].layers), cw = Math.min(18, (W - pad - xs) / maxL), gap = cw > 7 ? 2 : 1
    const rowH = (H - top - bot - 50) / 2
    let hover: { m: Model; i: number; r: number } | null = null
    ms.forEach((m, r) => {
      const y0 = top + 26 + r * rowH, ha = 34, ym = y0 + ha + 10, hm = 12
      name(m, r, pad, y0 - 14, eout(clamp(p / 0.1)), tf('{} layers', m.layers), true)
      ctx.font = F.small; ctx.fillStyle = rgba(C.mute, eout(clamp(p / 0.1))); ctx.textAlign = 'right'
      ctx.fillText('attention', xs - 10, y0 + ha / 2 + 4); ctx.fillText('MLP', xs - 10, ym + hm / 2 + 4)
      for (let i = 0; i < m.layers; i++) {
        const a = eout(clamp((p * 1.4 - (i / maxL) * 0.6) / 0.2)), x = xs + i * cw
        if (a <= 0) continue
        const code = m.attn.codes[i], on = inside(x, y0 - 8, cw, ha + hm + 20)
        if (on) hover = { m, i, r }
        attnCell(code, m, x, y0, cw - gap, ha, HUE[r], a * (on ? 1 : 0.9))
        if (m.mlp[i] === 'E') { ctx.strokeStyle = rgba(HUE[r], 0.7 * a); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, ym + 0.5, cw - gap - 1, hm - 1); hatch(x, ym, cw - gap, hm, HUE[r], a, 3) }
        else { ctx.fillStyle = rgba(HUE[r], 0.5 * a); ctx.fillRect(x, ym, cw - gap, hm) }
        if (on) { ctx.strokeStyle = rgba(C.ink, a); ctx.lineWidth = 1.5; ctx.strokeRect(x - 1, y0 - 1, cw - gap + 2, ym + hm - y0 + 2) }
      }
    })
    // the legend: every kind either model has
    const la = eout(clamp((p - 0.4) / 0.15)), present = new Set([...ms[0].attn.codes, ...ms[1].attn.codes].map(kindOf))
    let lx = xs, ly = H - bot - 8
    for (const kd of ['F', 'S', 'C', 'M', 'L'].filter((x) => present.has(x as 'F'))) {
      attnCell(kd, ms[0], lx, ly - 11, 9, 14, C.ink2, la)
      ctx.font = F.small; ctx.fillStyle = rgba(C.mute, la); ctx.textAlign = 'left'; ctx.fillText(WORD[kd], lx + 14, ly)
      lx += 30 + ctx.measureText(t(WORD[kd])).width
    }
    ctx.fillStyle = rgba(C.ink2, 0.5 * la); ctx.fillRect(lx, ly - 9, 9, 9); ctx.font = F.small; ctx.fillStyle = rgba(C.mute, la); ctx.fillText('dense MLP', lx + 14, ly); lx += 30 + ctx.measureText(t('dense MLP')).width
    ctx.strokeStyle = rgba(C.ink2, 0.7 * la); ctx.strokeRect(lx + 0.5, ly - 8.5, 8, 8); hatch(lx, ly - 9, 9, 9, C.ink2, la, 3); ctx.fillStyle = rgba(C.mute, la); ctx.fillText('experts', lx + 14, ly)
    if (ms.some((m) => [...m.attn.codes].some((cd) => noRope(m, cd)))) { lx += 30 + ctx.measureText(t('experts')).width; ctx.strokeStyle = rgba(C.ink, 0.8 * la); ctx.beginPath(); ctx.arc(lx + 4, ly - 4, 2.5, 0, 7); ctx.stroke(); ctx.fillText('no RoPE', lx + 14, ly) }

    const h = hover as { m: Model; i: number; r: number } | null
    if (h) {
      const code = h.m.attn.codes[h.i], kind = kindOf(code)
      mk.formula = {
        segs: [[h.m.name, HUE[h.r]], [`  ${t(`layer ${h.i + 1} of ${h.m.layers}`)}`, C.ink], ['  ·  ', C.mute], [t(layerWord(h.m, code)), C.ink], ['  ', C.mute], [kindLine(h.m, code), C.ink2], ['  ·  ', C.mute],
          [h.m.mlp[h.i] === 'E' && h.m.moe ? t(`${h.m.moe.experts} experts, top ${h.m.moe.topK}`) : t(`dense MLP ${nf(h.m.denseWidth ?? 0)}`), C.ink2]],
        note: KIND_NOTE[kind],
      }
    } else {
      mk.formula = { segs: [[ms[0].name, HUE[0]], [`  ${join(patternText(ms[0]))}`, C.ink2], ['     ', C.mute], [ms[1].name, HUE[1]], [`  ${join(patternText(ms[1]))}`, C.ink2]], note: 'How much of a column is filled is what the layer keeps per token: every token (full), a compressed latent, only a window, or a fixed state (outlined).' }
    }
  }

  /* ---------- 3: heads and the cache per token ---------- */
  function sceneHeads(p: number) {
    const { W, H } = stage, ms = both()
    const xs = pad + 170, xr = W - pad - 190, rowH = (H - top - bot) / 2
    ms.forEach((m, r) => {
      const y0 = top + 20 + r * rowH, a0 = eout(clamp(p / 0.12))
      name(m, r, pad, y0, a0, join(attnText(m)), true)
      const n = m.heads, span = xr - xs - 30, s = Math.max(2, Math.min(12, span / n - 2)), step = span / n
      const xq = (j: number) => xs + j * step + (step - s) / 2
      const yq = y0 + 22, yk = y0 + rowH - 80
      const aq = eout(clamp((p - 0.05) / 0.2)), al = eout(clamp((p - 0.25) / 0.25)), ak = eout(clamp((p - 0.35) / 0.2))
      ctx.font = F.small; ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.mute, aq)
      ctx.fillText('query heads', xs - 10, yq + s / 2 + 4)
      for (let j = 0; j < n; j++) { ctx.fillStyle = rgba(HUE[r], 0.85 * aq); ctx.fillRect(xq(j), yq, s, s) }
      if (m.mla) {
        // one latent per token that every head's keys and values are rebuilt from, plus a small RoPE key
        const lw = span * 0.62, rw = span * 0.62 * (m.mla.rope / m.mla.latent), lx = xs + (span - lw - rw - 8) / 2
        ctx.strokeStyle = rgba(HUE[r], 0.35 * al); ctx.lineWidth = 1
        for (let j = 0; j < n; j += Math.max(1, Math.floor(n / 24))) { ctx.beginPath(); ctx.moveTo(xq(j) + s / 2, yq + s); ctx.lineTo(lerp(lx, lx + lw, j / (n - 1)), yk); ctx.stroke() }
        ctx.fillStyle = rgba(HUE[r], 0.85 * ak); ctx.fillRect(lx, yk, lw, 14); ctx.fillRect(lx + lw + 8, yk, rw, 14)
        ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink2, ak)
        ctx.fillText(`latent ${m.mla.latent}`, lx, yk + 30); ctx.fillText(m.pos.kind === 'rope' ? `RoPE ${m.mla.rope}` : `shared key ${m.mla.rope}`, lx + lw + 8, yk + 30)
        ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.mute, ak); ctx.fillText('cached', xs - 10, yk + 11)
      } else {
        const g = n / m.kvHeads, kvs = Math.min(s * 1.4, 14)
        const xk = (j: number) => (xq(j * g) + xq(j * g + g - 1) + s) / 2
        ctx.strokeStyle = rgba(HUE[r], 0.35 * al); ctx.lineWidth = 1
        for (let j = 0; j < n; j++) { ctx.beginPath(); ctx.moveTo(xq(j) + s / 2, yq + s); ctx.lineTo(xk(Math.floor(j / g)), yk); ctx.stroke() }
        for (let j = 0; j < m.kvHeads; j++) { ctx.fillStyle = rgba(HUE[r], 0.85 * ak); ctx.fillRect(xk(j) - kvs / 2, yk, kvs, kvs) }
        ctx.font = F.small; ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.mute, ak); ctx.fillText('key/value heads', xs - 10, yk + kvs / 2 + 4)
      }
      // what one layer keeps per token, and every layer together
      const ar = eout(clamp((p - 0.5) / 0.2)), full = [...new Set(m.attn.codes)].find((cd) => /[FSCM]/i.test(cd))
      const lin = [...m.attn.codes].filter((cd) => kindOf(cd) === 'L').length
      title('per token', xr + 10, y0 + 22, ar)
      ctx.font = F.mono(13, 600); ctx.textAlign = 'left'; ctx.fillStyle = rgba(HUE[r], ar)
      ctx.fillText(fmtBytes(2 * growing(m)), xr + 10, y0 + 46)
      ctx.font = F.small; ctx.fillStyle = rgba(C.mute, ar)
      if (full) ctx.fillText(`${nf(m.attn.kinds[full].perToken)} numbers a layer`, xr + 10, y0 + 64)
      if (lin) ctx.fillText(`+ ${lin} linear layers, fixed`, xr + 10, y0 + 80)
      const sw = [...new Set(m.attn.codes)].filter((cd) => m.attn.kinds[cd].window)
      if (sw.length) ctx.fillText(`window layers stop at ${nf(m.attn.kinds[sw[0]].window!)}`, xr + 10, y0 + (lin ? 96 : 80))
    })
    mk.formula = {
      segs: [[ms[0].name, HUE[0]], [`  ${fmtBytes(2 * growing(ms[0]))} ${t('per token')}`, C.ink], ['     ', C.mute], [ms[1].name, HUE[1]], [`  ${fmtBytes(2 * growing(ms[1]))} ${t('per token')}`, C.ink]],
      note: 'In 16-bit, over the layers that keep every token: 2 × key/value heads × head size per layer, or the latent and its RoPE key. Sliding-window layers stop at their window; linear layers keep a fixed state instead.',
    }
  }

  /* ---------- 4: the MLP ---------- */
  /** A token's experts: fixed per (model, token), so the drawing is a function of time. */
  function chosen(m: Model, tok: number): Set<number> {
    const e = m.moe!, out = new Set<number>()
    let h = 2166136261 ^ tok * 16777619 ^ m.layers
    while (out.size < e.topK) { h = Math.imul(h ^ (h >>> 15), 2246822519); h ^= h >>> 13; out.add(Math.abs(h) % e.experts) }
    return out
  }
  function sceneMlp(p: number) {
    const { W, H } = stage, ms = both()
    const xs = pad + 170, xr = W - pad - 190, rowH = (H - top - bot) / 2, tok = Math.min(4, Math.floor(p * 5))
    const widest = Math.max(...ms.map((m) => m.denseWidth ?? 0), ...ms.map((m) => (m.moe ? m.moe.width : 0)))
    ms.forEach((m, r) => {
      const y0 = top + 20 + r * rowH, a0 = eout(clamp(p / 0.12))
      name(m, r, pad, y0, a0, join(mlpText(m)), true)
      const ar = eout(clamp((p - 0.1) / 0.2))
      if (!m.moe) {
        const bw = (xr - xs - 30) * ((m.denseWidth ?? 0) / widest)
        rr(xs, y0 + 30, bw, 26, 5); ctx.fillStyle = rgba(HUE[r], 0.75 * ar); ctx.fill()
        ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink2, ar); ctx.fillText(`${nf(m.denseWidth ?? 0)} wide, every token`, xs, y0 + 74)
      } else {
        const e = m.moe, cols = Math.min(64, e.experts), rows = Math.ceil(e.experts / cols)
        const cell = Math.min(14, (xr - xs - 90) / cols), g = cell > 6 ? 2 : 1, xe = xs + (e.shared ? 70 : 0)
        const on = chosen(m, tok)
        for (let s = 0; s < e.shared; s++) { rr(xs + s * 30, y0 + 26, 26, 26, 4); ctx.fillStyle = rgba(HUE[r], 0.85 * ar); ctx.fill() }
        if (e.shared) { ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, ar); ctx.fillText('shared', xs, y0 + 68) }
        for (let i = 0; i < e.experts; i++) {
          const x = xe + (i % cols) * cell, y = y0 + 26 + Math.floor(i / cols) * cell, lit = on.has(i)
          if (lit) { ctx.fillStyle = rgba(HUE[r], 0.95 * ar); ctx.fillRect(x, y, cell - g, cell - g) }
          else { ctx.strokeStyle = rgba(HUE[r], 0.4 * ar); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cell - g - 1, cell - g - 1) }
        }
        ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink2, ar)
        ctx.fillText(`${e.experts} experts × ${nf(e.width)} · this token: ${e.topK}${e.shared ? ` + ${e.shared} shared` : ''}`, xe, y0 + 26 + rows * cell + 16)
        if (e.dense) { ctx.fillStyle = rgba(C.mute, ar); ctx.fillText(`${e.dense} dense layers of ${nf(m.denseWidth ?? 0)}`, xe, y0 + 26 + rows * cell + 32) }
      }
      const aw = eout(clamp((p - 0.35) / 0.2))
      title('width per token', xr + 10, y0 + 22, aw)
      ctx.font = F.mono(13, 600); ctx.textAlign = 'left'; ctx.fillStyle = rgba(HUE[r], aw); ctx.fillText(nf(tokenWidth(m)), xr + 10, y0 + 46)
      ctx.font = F.small; ctx.fillStyle = rgba(C.mute, aw)
      ctx.fillText(m.moe ? `${m.moe.topK + m.moe.shared} × ${nf(m.moe.width)}` : `${((m.denseWidth ?? 0) / m.hidden).toFixed(1)}× d_model`, xr + 10, y0 + 64)
      if (m.moe) ctx.fillText(`of ${nf((m.moe.experts + m.moe.shared) * m.moe.width)} stored`, xr + 10, y0 + 80)
    })
    mk.formula = {
      segs: [[ms[0].name, HUE[0]], [`  ${join(mlpText(ms[0]))}`, C.ink2], ['     ', C.mute], [ms[1].name, HUE[1]], [`  ${join(mlpText(ms[1]))}`, C.ink2]],
      note: 'The lit experts change with each token; which ones a real router picks depends on the token’s vector. Here five tokens pass, with made-up choices.',
    }
  }

  /* ---------- 5: where the parameters are ---------- */
  const PARTS: [string, (m: Model) => number, number, boolean?][] = [
    ['embeddings', (m) => m.params.embed + m.params.head + m.params.pos, 0.3],
    ['attention', (m) => m.params.attn + m.params.indexer, 0.9],
    ['dense MLP', (m) => m.params.mlp, 0.6],
    ['shared experts', (m) => m.params.shared, 0.75],
    ['routed, this token', (m) => (m.moe ? m.params.routed * (m.moe.topK / m.moe.experts) : 0), 0.55],
    ['routed, idle', (m) => (m.moe ? m.params.routed * (1 - m.moe.topK / m.moe.experts) : 0), 0, true],
    ['norms and router', (m) => m.params.norm + m.params.router, 0.15],
  ]
  function sceneParams(p: number) {
    const { W, H } = stage, ms = both()
    const xs = pad + 170, bw = W - pad - xs
    // an absolute scale first: both totals and what a token uses, on a log axis
    const lx0 = xs, lx1 = W - pad, ly = top + 20, lo = 7.5, hi = 12.3
    const X = (n: number) => lerp(lx0, lx1, (Math.log10(n) - lo) / (hi - lo))
    const a0 = eout(clamp(p / 0.15))
    ctx.strokeStyle = rgba(C.faint, a0); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(lx0, ly); ctx.lineTo(lx1, ly); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.fillStyle = rgba(C.mute, a0)
    for (const [n, s] of [[1e8, '100M'], [1e9, '1B'], [1e10, '10B'], [1e11, '100B'], [1e12, '1T']] as [number, string][]) { ctx.fillRect(X(n) - 0.5, ly - 3, 1, 6); raw(() => ctx.fillText(s, X(n), ly + 16)) }
    ctx.font = F.small; ctx.textAlign = 'right'; ctx.fillText('parameters (log)', xs - 10, ly + 4)
    ms.forEach((m, r) => {
      const at = eout(clamp((p - 0.05 - r * 0.05) / 0.2))
      ctx.fillStyle = rgba(HUE[r], at); ctx.beginPath(); ctx.arc(X(m.params.total), ly, 5, 0, 7); ctx.fill()
      ctx.strokeStyle = rgba(HUE[r], at); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(m.params.active), ly, 5, 0, 7); ctx.stroke()
    })
    ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, a0); ctx.fillText('filled: stored · ring: per token', lx0, ly - 12)

    const rowH = (H - ly - 76 - bot) / 2
    let hover: { m: Model; r: number; part: number } | null = null
    ms.forEach((m, r) => {
      const y0 = ly + 66 + r * rowH, a1 = eout(clamp((p - 0.15 - r * 0.08) / 0.2))
      name(m, r, pad, y0 - 12, a1, `${fmtParams(m.params.total)} · ${fmtParams(m.params.active)} ${t('per token')}`, true)
      const grow = eout(clamp((p - 0.2 - r * 0.08) / 0.35))
      let x = xs
      PARTS.forEach(([label, f, alpha, idle], i) => {
        const n = f(m), w = bw * (n / m.params.total) * grow
        if (w <= 0.2) return
        if (idle) { ctx.strokeStyle = rgba(HUE[r], 0.5 * a1); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y0 + 0.5, w - 1, 27); hatch(x, y0, w, 28, HUE[r], a1) }
        else { ctx.fillStyle = rgba(HUE[r], alpha * a1); ctx.fillRect(x, y0, w, 28) }
        if (inside(x, y0, w, 28)) { hover = { m, r, part: i }; ctx.strokeStyle = rgba(C.ink, a1); ctx.lineWidth = 1.5; ctx.strokeRect(x, y0 - 1, w, 30) }
        if (w > 70 && grow > 0.95) { ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink, a1); ctx.fillText(label, x + 6, y0 + 46) }
        x += w
      })
    })
    // the legend: every part either model has
    const lg = eout(clamp((p - 0.5) / 0.15))
    let lx = xs
    PARTS.forEach(([label, f, alpha, idle]) => {
      if (!ms.some((m) => f(m) > 0)) return
      if (idle) { ctx.strokeStyle = rgba(C.ink2, 0.5 * lg); ctx.lineWidth = 1; ctx.strokeRect(lx + 0.5, H - bot - 16.5, 11, 11); hatch(lx, H - bot - 17, 12, 12, C.ink2, lg, 4) }
      else { ctx.fillStyle = rgba(C.ink2, alpha * lg); ctx.fillRect(lx, H - bot - 17, 12, 12) }
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, lg); ctx.fillText(label, lx + 17, H - bot - 7)
      lx += 34 + ctx.measureText(t(label)).width
    })

    const h = hover as { m: Model; r: number; part: number } | null
    if (h) {
      const [label, f] = PARTS[h.part], n = f(h.m)
      mk.formula = { segs: [[h.m.name, HUE[h.r]], [`  ${t(label)}  `, C.ink], [fmtParams(n), C.ink], [`  ·  ${(100 * n / h.m.params.total).toFixed(1)}%`, C.ink2]], note: 'Counted from the checkpoint’s tensors; buffers, quantization scales, vision encoders and multi-token-prediction layers are left out.' }
    } else {
      mk.formula = { segs: [[ms[0].name, HUE[0]], [`  ${fmtParams(ms[0].params.total)} · ${fmtParams(ms[0].params.active)} ${t('per token')}`, C.ink2], ['     ', C.mute], [ms[1].name, HUE[1]], [`  ${fmtParams(ms[1].params.total)} · ${fmtParams(ms[1].params.active)} ${t('per token')}`, C.ink2]], note: 'Per token counts every weight a token passes through, the embeddings included; published figures often leave the embeddings out.' }
    }
  }

  /* ---------- 6: the KV cache against context ---------- */
  function sceneKv(p: number) {
    const { W, H } = stage, ms = both()
    const x0 = pad + 80, x1 = W - pad - 190, y0 = top + 16, y1 = H - bot - 30
    const nMax = Math.max(131072, ...ms.map((m) => m.context)), nMin = 128
    const lo = Math.log2(nMin), hiN = Math.log2(nMax)
    const X = (n: number) => lerp(x0, x1, (Math.log2(n) - lo) / (hiN - lo))
    const top2 = Math.max(...ms.map((m) => kvBytes(m, m.context))), bot2 = Math.min(...ms.map((m) => kvBytes(m, nMin)))
    const ylo = Math.floor(Math.log2(bot2)) - 1, yhi = Math.ceil(Math.log2(top2)) + 1
    const Y = (b: number) => lerp(y1, y0, (Math.log2(b) - ylo) / (yhi - ylo))
    const a0 = eout(clamp(p / 0.12))
    ctx.lineWidth = 1
    // grid: powers of 1024 (and ×32 between) on y, powers of 2 in thousands on x
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, a0)
    for (let e = Math.ceil(ylo); e <= yhi; e++) {
      if (e % 5 !== 0) continue
      ctx.strokeStyle = rgba(C.faint, 0.45 * a0); ctx.beginPath(); ctx.moveTo(x0, Y(2 ** e)); ctx.lineTo(x1, Y(2 ** e)); ctx.stroke()
      ctx.textAlign = 'right'; raw(() => ctx.fillText(fmtBytes(2 ** e), x0 - 8, Y(2 ** e) + 4))
    }
    for (let e = Math.ceil(lo); e <= hiN; e += 2) {
      ctx.strokeStyle = rgba(C.faint, 0.3 * a0); ctx.beginPath(); ctx.moveTo(X(2 ** e), y0); ctx.lineTo(X(2 ** e), y1); ctx.stroke()
      ctx.textAlign = 'center'; raw(() => ctx.fillText(fmtTokens(2 ** e), X(2 ** e), y1 + 16))
    }
    ctx.textAlign = 'left'; ctx.fillText('tokens in the sequence (log)', x0, y1 + 30)
    ctx.save(); ctx.translate(pad + 8, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('KV cache, 16-bit (log)', 0, 0); ctx.restore()

    const grow = eout(clamp((p - 0.1) / 0.6))
    ms.forEach((m, r) => {
      const nEnd = Math.min(m.context, 2 ** lerp(lo, hiN, grow)), steps = 90
      ctx.strokeStyle = rgba(HUE[r], a0); ctx.lineWidth = 2; ctx.beginPath()
      for (let s = 0; s <= steps; s++) {
        const n = 2 ** lerp(lo, Math.log2(Math.max(nMin, nEnd)), s / steps)
        const x = X(n), y = Y(kvBytes(m, n))
        if (s) ctx.lineTo(x, y); else ctx.moveTo(x, y)
      }
      ctx.stroke()
      if (nEnd >= m.context) {
        const x = X(m.context), y = Y(kvBytes(m, m.context))
        ctx.fillStyle = rgba(HUE[r], a0); ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill()
      }
      // the label at the right: name, and the cache at its full context
      const la = eout(clamp((p - 0.6) / 0.15)), ly = y0 + 20 + r * 64
      name(m, r, x1 + 24, ly, la, `${fmtBytes(kvBytes(m, m.context))} ${tf('at {} tokens', fmtTokens(m.context))}`)
    })
    // hover: both caches at one length
    let n = 32768
    if (mouse && mouse[0] >= x0 && mouse[0] <= x1 && mouse[1] >= y0 && mouse[1] <= y1) n = Math.round(2 ** lerp(lo, hiN, (mouse[0] - x0) / (x1 - x0)))
    const xn = X(n)
    ctx.strokeStyle = rgba(C.ink, 0.35 * a0); ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(xn, y0); ctx.lineTo(xn, y1); ctx.stroke(); ctx.setLineDash([])
    ms.forEach((m, r) => { if (n <= m.context) { ctx.fillStyle = rgba(HUE[r], a0); ctx.beginPath(); ctx.arc(xn, Y(kvBytes(m, n)), 3, 0, 7); ctx.fill() } })
    const at = (m: Model) => (n <= m.context ? fmtBytes(kvBytes(m, n)) : t('beyond its context'))
    mk.formula = { segs: [[tf('{} tokens', fmtTokens(n)), C.ink], ['     ', C.mute], [ms[0].name, HUE[0]], [`  ${at(ms[0])}`, C.ink], ['     ', C.mute], [ms[1].name, HUE[1]], [`  ${at(ms[1])}`, C.ink]], note: 'One sequence. A server holds this for every sequence in its batch, on top of the weights.' }
  }

  return { diff: sceneDiff, layers: sceneLayers, heads: sceneHeads, mlp: sceneMlp, params: sceneParams, kv: sceneKv }
}
