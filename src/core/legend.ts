/*
 * The legend of the site's visual language, shared by the start page and the "How to read the
 * pictures" panel under every exhibit.
 */

/** The site's visual language: [small SVG drawn with the site's tokens, what it means]. */
const LEGEND: [string, string][] = [
  ['<svg viewBox="0 0 64 20"><rect x="1" y="3" width="26" height="14" rx="4" fill="none" stroke="var(--t0)"/><rect x="35" y="3" width="26" height="14" rx="4" fill="none" stroke="var(--t2)"/></svg>', 'Each token keeps its own colour everywhere.'],
  ['<svg viewBox="0 0 64 20"><path d="M2 10h30" stroke="var(--t1)" stroke-width="2"/><path d="M32 10h30" stroke="var(--ink2)" stroke-width="2"/></svg>', 'A lane is one token’s vector flowing through the model; its colour blends as it takes in other tokens.'],
  ['<svg viewBox="0 0 64 20"><path d="M26 17l8-5V1l-8 5z" fill="none" stroke="var(--ink2)"/><path d="M40 17l8-5V1l-8 5z" fill="none" stroke="var(--mute)"/></svg>', 'A glass plate is a layer the lanes pass through.'],
  ['<svg viewBox="0 0 64 20"><rect x="14" y="4" width="11" height="11" fill="var(--ink2)"/><rect x="39" y="4.5" width="10" height="10" fill="none" stroke="var(--ink2)"/></svg>', 'In a matrix, a filled cell is positive and an outlined cell is negative.'],
  ['<svg viewBox="0 0 64 20"><rect x="1" y="7" width="14" height="6" fill="none" stroke="var(--mute)"/><rect x="24" y="1" width="6" height="18" fill="none" stroke="var(--mute)"/><rect x="24" y="7" width="6" height="6" fill="var(--ink2)"/><path d="M15 10h9" stroke="var(--faint)" stroke-dasharray="2 2"/></svg>', 'In a matrix product C = A · B, row i of A (left) meets column j of B (above) at cell (i, j) of C.'],
  ['<svg viewBox="0 0 64 20"><circle cx="32" cy="10" r="6" fill="none" stroke="var(--ink2)"/><path d="M29 10h6M32 7v6" stroke="var(--ink2)"/></svg>', '⊕ adds a layer’s output back onto the lane (the residual stream).'],
  ['<svg viewBox="0 0 64 20"><text x="32" y="15" text-anchor="middle" fill="var(--ink)" font-size="14" font-family="var(--mono)">↗</text></svg>', 'Parts marked ↗ open a detail view. Hover or tap any matrix cell to see its formula; click to pin it.'],
]

export const legendList = () =>
  `<ul class="legend-list">${LEGEND.map(([svg, t]) => `<li><span class="sw" aria-hidden="true">${svg}</span><span>${t}</span></li>`).join('')}</ul>`
