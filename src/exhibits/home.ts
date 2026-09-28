import { F, chipW, drawChip, plate, rr, serifAt, spaced, tokText, useCtx, upper } from '../core/draw'
import { lang, raw, setLang, t } from '../core/i18n'
import { lastPlace } from '../core/progress'
import { Stage, runLoop } from '../core/stage'
import { C, blend, mixc, rgba, type RGB } from '../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../core/util'
import { mixing, nextDist, presets } from '../lib/gpt2/data'
import { pageName, type Nav } from './registry'

/*
 * The landing page: one animation, full screen, of what the whole site is about. A prompt splits into tokens, their
 * lanes run through GPT-2 small's 12 blocks and take on each other's colours (by the real attention of that run),
 * the last lane becomes the real next-token distribution, and the top token is appended. Then the name and the way
 * in. Everything is a function of the page's own clock T; a click, a key or Skip jumps to the end.
 */

// the moments of the animation, in seconds
const TYPE: [number, number] = [0.2, 1.3], SPLIT: [number, number] = [1.6, 2.6], LANES: [number, number] = [2.6, 6.2]
const NEXT: [number, number] = [6.2, 8.0], APPEND: [number, number] = [8.0, 9.4], TITLE = 9.4, END = 10.8
const BLOCKS = 12
const span = (T: number, [a, b]: [number, number]) => clamp((T - a) / (b - a))

/** Played once per visit: coming back (the logo, a language switch) opens at the end, with Replay. */
let played = false

export function mountHome(root: HTMLElement, nav: Nav): () => void {
  root.classList.add('home')
  const last = lastPlace()
  const resume = last && last.route !== 'start' && last.route !== 'home' ? last : null
  root.innerHTML = `
    <div class="home-stage"></div>
    <button class="home-lang" type="button">${lang === 'zh' ? 'English' : '中文'}</button>
    <button class="home-skip" type="button">${t('Skip ›')}</button>
    <section class="home-cta" aria-live="polite">
      <div class="home-mark" aria-hidden="true"></div>
      <h1>Token Trails</h1>
      <p class="home-tag">${t('Follow the tokens through AI systems')}</p>
      <p class="home-line">${t('That is all a language model does: split the text into tokens, mix them through its layers, pick the next one, and go again. This site follows those tokens, with the numbers of real models.')}</p>
      <div class="home-go">
        <button class="st-go home-start" type="button">${t('Start the tour →')}</button>
        ${resume ? `<button class="st-go st-resume home-resume" type="button">${t('Resume')}: ${t(pageName(resume.route) ?? resume.route.split('/').pop() ?? '')} ›</button>` : ''}
        <button class="home-replay" type="button">${t('↺ Replay')}</button>
      </div>
    </section>`
  const q = <E extends HTMLElement>(s: string) => root.querySelector(s) as E
  // the rail's logo, reused
  const mark = document.querySelector('.brand svg')
  if (mark) q('.home-mark').appendChild(mark.cloneNode(true))

  const stage = new Stage(q('.home-stage'), 320, 420, 'A prompt splits into tokens; their lanes run through GPT-2’s 12 blocks and mix their colours by the real attention of the run; the last lane becomes the real next-token probabilities, and the most likely token is appended.')
  const ctx = stage.ctx

  // the real run: GPT-2 small on "The cat sat on the"
  const pass = presets().find((p) => p.key === 'cat')!.passes[0]
  const texts = pass.texts, N = texts.length
  const mix = mixing(pass.att)
  const dist = nextDist(pass.next, 1, 5)
  const top = dist.rows[0]
  const hue = (i: number): RGB => C.tok[i % 7]
  /** Lane i's colour after block l: the tokens it has taken in, blended by the real attention. */
  const laneCol = (i: number, l: number): RGB => (l < 0 ? hue(i) : blend(texts.map((_, k) => hue(k)), mix[l][i]))

  let T = played || reducedMotion() ? END : 0
  const cta = q('.home-cta'), skip = q<HTMLButtonElement>('.home-skip')
  let keyed = false
  const toEnd = () => { if (T < END) T = END; played = true }
  const replay = () => { T = 0; keyed = false; cta.classList.remove('on'); skip.hidden = false }

  function draw() {
    useCtx(ctx)
    stage.begin()
    const { W, H } = stage
    const wide = W >= 760
    // wide: tokens, blocks and next-token bars in one row, centred; narrow: the bars go under the blocks
    const Wc = wide ? Math.min(W - 64, 1180) : W - 32, left = (W - Wc) / 2
    const chip = wide ? F.mono(14, 500) : F.chip, chipH = wide ? 30 : 24
    const gap = wide ? clamp(H * 0.07, 30, 50) : clamp(H * 0.055, 26, 38)
    const cy = wide ? H * 0.52 : H * 0.36
    const y = (i: number) => cy + (i - (N - 1) / 2) * gap
    const xChip = left, rowH = wide ? 32 : 24
    const bw = wide ? Math.min(330, Wc * 0.27) : Wc
    const bars = wide ? { x: left + Wc - bw, y: cy - 2.5 * rowH, w: bw } : { x: left, y: y(N) + gap * 2.6, w: bw }
    const x0 = xChip + Math.max(...texts.map((s) => chipW(s, chip))) + 14, xe = wide ? bars.x - clamp(Wc * 0.08, 60, 110) : W - 18
    const px = (l: number) => lerp(x0 + (xe - x0) * 0.1, xe - (xe - x0) * 0.08, l / (BLOCKS - 1))
    const ap = eio(span(T, APPEND))
    // once the name comes up, the machinery steps back
    const dim = lerp(1, 0.16, eout(clamp((T - TITLE) / 0.8)))

    /* 1. the prompt, typed, then split into token chips */
    const sent = texts.join(''), size = wide ? 52 : 30, sy = cy
    ctx.font = serifAt(size)
    const widths = raw(() => texts.map((s) => ctx.measureText(s).width)), full = widths.reduce((a, b) => a + b, 0)
    const sx0 = W / 2 - full / 2
    const typed = Math.round(eout(span(T, TYPE)) * sent.length), split = eio(span(T, SPLIT))
    let off = 0, at = 0
    texts.forEach((s, i) => {
      const lead = raw(() => ctx.measureText(s.match(/^\s*/)![0]).width)
      const wx = sx0 + off + lead
      // the part of this word typed so far
      const shown = s.slice(0, Math.max(0, typed - at)).trimStart()
      off += widths[i]; at += s.length
      if (split <= 0) {
        ctx.font = serifAt(size); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, dim)
        raw(() => ctx.fillText(shown, wx, sy))
      } else if (split < 1) {
        // the word shrinks into its chip on the way to its lane
        const x = lerp(wx, xChip, split), yy = lerp(sy, y(i), split)
        ctx.font = serifAt(size * lerp(1, 0.5, split)); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
        ctx.fillStyle = rgba(mixc(C.ink, hue(i), split), (1 - split) * dim)
        raw(() => ctx.fillText(s.trimStart(), x, yy))
        drawChip(x, yy, { text: s, c: i }, eout(split) * dim, chipH, false, chip)
      } else drawChip(xChip, y(i), { text: s, c: i }, dim, chipH, false, chip)
    })
    if (T >= SPLIT[1] - 0.2) label('tokens', xChip, y(0) - gap * 0.9, eout(clamp((T - SPLIT[1] + 0.2) / 0.4)) * dim, 'left')

    /* 2. the lanes, through the 12 blocks, taking in each other's colours */
    const la = span(T, [LANES[0] - 0.2, LANES[0] + 0.3])
    if (la > 0) {
      const yt = y(0) - gap * 0.9, yb = lerp(y(N - 1), y(N), ap) + gap * 0.9
      const heads = texts.map((_, i) => lerp(x0, xe, eio(clamp((T - LANES[0] - i * 0.08) / (LANES[1] - LANES[0] - 0.5)))))
      for (let l = 0; l < BLOCKS; l++) {
        const act = Math.max(0, ...heads.map((h) => 1 - Math.abs(h - px(l)) / 50)) * 0.9
        ctx.globalAlpha = la * dim; plate(px(l), yt, yb, act, { w: 7, d: 8 }); ctx.globalAlpha = 1
      }
      const stops = [x0, ...Array.from({ length: BLOCKS }, (_, l) => px(l)), xe]
      texts.forEach((_, i) => {
        ctx.lineWidth = 2.2; ctx.lineCap = 'round'
        for (let s = 0; s + 1 < stops.length; s++) {
          const a = stops[s], b = Math.min(stops[s + 1], heads[i])
          if (b <= a) break
          ctx.strokeStyle = rgba(laneCol(i, s - 1), 0.9 * dim)
          ctx.beginPath(); ctx.moveTo(a, y(i)); ctx.lineTo(b, y(i)); ctx.stroke()
        }
      })
      label(`${BLOCKS} blocks`, (px(0) + px(BLOCKS - 1)) / 2, yt - 16, la * dim, 'center')
      const ca = eout(clamp((T - LANES[0] - 1.2) / 0.5)) * dim
      if (ca > 0) { ctx.font = F.small; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, ca); ctx.fillText('attention mixes the lanes', (px(0) + px(BLOCKS - 1)) / 2, yb + 26) }
    }

    /* 3. the last lane becomes the next-token distribution */
    const na = span(T, NEXT)
    if (na > 0) {
      const ye = y(N - 1), labW = wide ? 100 : 86
      const col = laneCol(N - 1, BLOCKS - 1)
      ctx.strokeStyle = rgba(col, 0.9 * dim); ctx.lineWidth = 2.2
      // it runs into the top row: the token that is picked and appended
      const fy = bars.y + rowH / 2, f = eout(clamp(na * 3)), path: [number, number][] = []
      if (wide) {
        const bx = bars.x - 12, mx = (xe + bx) / 2
        for (let s = 0; s <= 32; s++) {
          const u = s / 32, v = 1 - u
          path.push([v * v * v * xe + 3 * v * v * u * mx + 3 * v * u * u * mx + u * u * u * bx, v * v * v * ye + 3 * v * v * u * ye + 3 * v * u * u * fy + u * u * u * fy])
        }
      } else path.push([xe, ye], [xe, fy], [xe - 8, fy])
      ctx.beginPath(); ctx.moveTo(xe, ye)
      polyTo(path, f)
      ctx.stroke()
      label('next token', bars.x, bars.y - 16, eout(clamp(na * 3)) * dim, 'left')
      dist.rows.forEach((r, k) => {
        const g = eout(clamp((na - 0.25 - k * 0.06) / 0.35)), yy = bars.y + k * rowH + rowH / 2
        if (g <= 0) return
        const on = k === 0 && T >= NEXT[0] + 1.2
        raw(() => tokText(bars.x, yy, r.text, g * dim * (on || k > 0 ? 1 : 0.85), on ? hue(N) : null, F.mono(wide ? 14 : 12)))
        const w = (bars.w - labW - (wide ? 44 : 70)) * (r.p / top.p) * g
        rr(bars.x + labW, yy - 5, Math.max(2, w), 10, 3)
        ctx.fillStyle = rgba(on ? hue(N) : C.ink2, (on ? 0.9 : 0.45) * dim); ctx.fill()
        ctx.font = F.mono(wide ? 12.5 : 11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(on ? C.ink : C.mute, g * dim)
        raw(() => ctx.fillText(`${(r.p * 100).toFixed(r.p < 0.1 ? 1 : 0)}%`, bars.x + labW + w + 8, yy))
      })
    }

    /* 4. the top token is appended, and the model would run again */
    if (ap > 0) {
      // it arcs under the lanes to take its place as the next input
      const fx = lerp(bars.x, xChip, ap), fy = lerp(bars.y + rowH / 2, y(N), ap) + Math.sin(Math.PI * ap) * gap * (wide ? 3 : 1.2)
      drawChip(fx, fy, { text: top.text, c: N }, dim, chipH, ap < 1, chip)
      // the text so far, above it all
      const ta = eout(clamp((ap - 0.5) / 0.5)) * dim, ty = wide ? Math.max(70, y(0) - gap * 2.4) : Math.max(56, y(0) - gap * 2.2)
      if (ta > 0) {
        ctx.font = serifAt(wide ? 34 : 22); ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left'
        const a = sent, b = top.text
        const wa = raw(() => ctx.measureText(a).width), wb = raw(() => ctx.measureText(b).width), tx = W / 2 - (wa + wb) / 2
        ctx.fillStyle = rgba(C.ink, ta); raw(() => ctx.fillText(a, tx, ty))
        ctx.fillStyle = rgba(hue(N), ta); raw(() => ctx.fillText(b, tx + wa, ty))
      }
    }

    // the way in
    const shown = T >= TITLE + 0.3
    if (cta.classList.contains('on') !== shown) {
      cta.classList.toggle('on', shown)
      // a keyboard reader lands on the way in (Enter starts the tour)
      const f = document.activeElement
      if (shown && (keyed || f === skip)) q('.home-start').focus({ preventScroll: true })
    }
    if (skip.hidden !== (T >= END)) skip.hidden = T >= END
  }

  /** Line along a polyline up to fraction f of its length. */
  function polyTo(pts: [number, number][], f: number) {
    const seg = pts.slice(1).map((q, i) => Math.hypot(q[0] - pts[i][0], q[1] - pts[i][1])), total = seg.reduce((a, b) => a + b, 0)
    let left = total * f
    for (let i = 0; i < seg.length && left > 0; i++) {
      const u = Math.min(1, left / (seg[i] || 1)), [x0, y0] = pts[i], [x1, y1] = pts[i + 1]
      ctx.lineTo(lerp(x0, x1, u), lerp(y0, y1, u)); left -= seg[i]
    }
  }

  function label(s: string, x: number, y: number, a: number, align: CanvasTextAlign) {
    if (a <= 0) return
    ctx.font = F.label; spaced(true); ctx.textAlign = align; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
    ctx.fillText(upper(s), x, y); spaced(false)
  }

  // skipping: a click on the drawing, any key but Tab, or the Skip button
  stage.canvas.addEventListener('click', toEnd)
  skip.addEventListener('click', toEnd)
  const onKey = (e: KeyboardEvent) => { if (T < END && e.key !== 'Tab' && !e.metaKey && !e.ctrlKey && !e.altKey) { keyed = true; toEnd() } }
  window.addEventListener('keydown', onKey)
  q('.home-start').addEventListener('click', () => nav('start'))
  if (resume) q('.home-resume').addEventListener('click', () => nav(`${resume.route}?phase=${resume.phase}`))
  q('.home-replay').addEventListener('click', replay)
  q('.home-lang').addEventListener('click', () => { played = true; setLang(lang === 'zh' ? 'en' : 'zh') })

  const stop = runLoop((dt) => {
    if (T < END) { T = Math.min(END, T + dt); if (T >= END) played = true }
    draw()
  }, () => T >= END)
  return () => { stop(); stage.destroy(); window.removeEventListener('keydown', onKey) }
}
