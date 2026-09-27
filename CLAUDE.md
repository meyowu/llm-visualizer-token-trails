# Token Trails

A portfolio of interactive, animated visualizations of AI concepts, built step by step (Transformer first, then architecture variants, inference engines and agents). Vite + TypeScript with no UI framework; every exhibit draws on a 2D canvas.

**Language:** everything in the repo is English: code, comments, UI copy (captions, canvas labels, aria labels), docs and commit messages. The user chats in Chinese; answer them in Chinese, but never write Chinese into the repo.

## Commands

- `npm run dev`: dev server on :5173.
- `npm run typecheck`: `tsc --noEmit` (TypeScript 7, strict, `noUnusedLocals`/`noUnusedParameters`).
- `npm run build`: typecheck, then a production build to `dist/`.
- `npm run build:artifact`: a single-file page in `dist-artifact/index.html` (via `vite-plugin-singlefile` + `scripts/artifact.mjs`, which strips the doctype/html/head/body wrappers). Publish that file to the existing preview Artifact by its `url` rather than creating a new one.

There are no tests. To verify a change: typecheck, then open the page and step through the phases (← / → jump between phases, Space plays/pauses). Check the console for errors.

## Layout

```
src/main.ts                 rail nav from the registry, hash router, zoom transition between views
src/styles.css              design tokens (dark-first, light via prefers-color-scheme / data-theme) + frame styles
src/core/stage.ts           Stage: DPR-aware canvas (min size, scrolls horizontally when narrow), runLoop()
src/core/player.ts          Player: phases, t, play/pause, speed, scrubbable timeline, keyboard shortcuts
src/core/frame.ts           createFrame(): header/specs, stage host, caption line, controls; toggle()
src/core/draw.ts            primitives: chips, plate(), bracketLabel(), mathName()/mathRun(), fonts F
src/core/theme.ts           canvas palette C (read from CSS tokens), rgba/mixc/blend/pop
src/exhibits/registry.ts    categories + exhibits; an item is live when it has `route` and `mount`
src/exhibits/transformer/
  model.ts                  GPT-2 ids, canned next-token distribution, toy attention block, overview pass data
  overview.ts               forward pass, tokenizer → sampling
  attention.ts              attention detail view, every GEMM animated cell by cell
```

## Adding an exhibit or detail view

1. Write `mountX(root, nav): () => void`: `createFrame` → `new Stage` → `new Player(PHASES, frame.controls)` → `runLoop` that ticks, draws, updates the timeline UI and sets the caption. Return a destroy that stops the loop and calls `player.destroy()` and `stage.destroy()`.
2. Register it in `registry.ts` (`route` + `mount`). A route one level deeper than the current one (`transformer/mlp`) opens with a zoom-in.
3. To open a detail view from the overview, hit-test its plate in `overview.ts` (see `onAttnPlate`), highlight it on hover with `drawOpenHint`, and call `nav(route, {x: e.clientX, y: e.clientY})`. Mention the click in that phase's caption.

## Conventions

- **Everything is a function of time.** Drawing reads `player.t` / `player.prog(id)` and keeps no animation state of its own, so the timeline scrubs both ways. Hover state is the only exception.
- **Phases** have a `name` (caption title) and an optional `short` (timeline label, e.g. `QKᵀ`).
- **Detail-view pattern** (follow `attention.ts`):
  - One `sceneX(p)` per group of phases.
  - Each scene has a layout computed in `geom()`, laid out so that a GEMM's A row i lines up with C row i and B column j lines up with C column j (A left, B above, C at the intersection). Transitions lerp matrix rects between scene layouts.
  - `gemm(p, m, n, K, 'slow'|'fast')` schedules cells. `resolve()` picks the hovered cell or the animated one. `gemmOverlay()` draws the row/column highlights and the dashed guides, and sets the formula line at the bottom.
  - Register hoverable result matrices in `hits` every frame.
- **Toy vs real scale.** Compute real arithmetic at toy size (`TOY` in `model.ts`: d_model 8, d_head 4, 2 heads). Always show the GPT-2 small shape next to it (`real` on matrices, `N × 768` etc., and `value / real` in the header specs).
- **Visual language is fixed.** The user approved it; don't redesign it.
  - Colors come only from `C` / the CSS tokens. Never hard-code hex in drawing code, and check both themes.
  - Each token keeps its hue (`--t0…--t6`, token i → `i % 7`) everywhere. Information mixing between tokens is shown by blending hues, and the blend must match the overview's lanes (`mixStep` / `laneCols`).
  - Layers are tilted glass plates. Matrices are slabs with a 5px depth face. A filled cell is positive, an outlined cell is negative.
  - Fonts: Newsreader italic for names and math, Geist for UI text, Geist Mono for labels and numbers. Group labels are uppercase mono over thin brackets.
  - Keep canvas text minimal, using real units and terms of art (GPT-2 ids, Ġ, 768 → 3072, FLOPs). Explanations go in the caption line: one or two plain sentences per phase. The canvas formula line carries a short note for the focused cell.
- Seeds are chosen for readable patterns (`WEIGHT_SEED` in `model.ts`). Changing a seed changes every number shown, so re-check the attention patterns if you touch it.

## Gotchas

- The browser pane's `preview_start` with the `dev` config has failed to serve before. If :5173 doesn't answer, run `npx vite --port 5174` in the background and navigate there. A hidden pane throttles rAF, so take a fresh screenshot before judging a frozen frame.
- Unicode subscripts don't render in Newsreader; use `mathRun()` for math with subscripts.
- The repo is private: github.com/meyowu/token-trails. Commit only when asked, and branch off `main` for larger changes.
