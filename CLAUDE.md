# Token Trails

A portfolio of interactive, animated visualizations of AI concepts, built step by step (Transformer first, then architecture variants, inference engines and agents). Vite + TypeScript with no UI framework; every exhibit draws on a 2D canvas.

**Language:** the repo is English: code, identifiers, comments, docs, commit messages, and the source UI copy (captions, canvas labels, aria labels). The site itself is bilingual with an EN / 中文 switch; the only Chinese in the repo is translated UI strings, kept in locale files and keyed to the English source, which stays the reference. The user chats in Chinese; answer them in Chinese.

## Commands

- `npm run dev`: dev server on :5173.
- `npm run typecheck`: `tsc --noEmit` (TypeScript 7, strict, `noUnusedLocals`/`noUnusedParameters`).
- `npm run build`: typecheck, then a production build to `dist/`.
- `npm run build:artifact`: a single-file page in `dist-artifact/index.html` (via `vite-plugin-singlefile` + `scripts/artifact.mjs`, which strips the doctype/html/head/body wrappers). Publish that file to the existing preview Artifact by its `url` rather than creating a new one.

There are no tests. To verify a change: typecheck, then open the page and step through the phases (← / → jump between phases, Space plays/pauses). Check the console for errors.

## Layout

```
src/main.ts                 rail nav, hash router (#/route?phase=id), tour order with previous/next (Shift+←/→), zoom transitions
src/styles.css              design tokens (dark-first, light via prefers-color-scheme / data-theme) + frame styles
src/core/stage.ts           Stage: DPR-aware canvas with a min size; scaled to fit a narrower host (phones get a full-size toggle);
                            runLoop(step, idle) skips frames while idle and nothing was poke()d
src/core/player.ts          Player: phases, t, play/pause, step buttons, Step/Auto pacing, speed, timeline, keys, All steps list
                            (player.describe), ?phase= in the URL; openAtPhase() for links
src/core/prefs.ts           pref.get/set: reader preferences in localStorage (speed, pacing, temperature, strategy)
src/core/frame.ts           createFrame(): header/specs, stage host, formula strip, caption line, controls; toggle(), stepper(), rich()
src/core/draw.ts            primitives: chips, plate(), bracketLabel(), mathName()/mathRun(), fonts F
src/core/matrix.ts          MatrixKit: drawMat (slabs), gemm() schedules, gemmOverlay, hover hits; formula goes to frame.setFormula
src/core/theme.ts           canvas palette C (read from CSS tokens), rgba/mixc/blend/pop
src/exhibits/start.ts       landing page (#/start, the default): live next-token demo, what a Transformer is, the path, legend
src/exhibits/registry.ts    categories (Anatomy, Lineage, Training, Serving, Agents) → entries: exhibits or sub-headings;
                            an exhibit may have `children` (its steps); live when it has `route` and `mount`
src/lib/gpt2/
  bpe.ts                    GPT-2 byte-level BPE (pre-split, merges by rank, ids), symbolText(); no imports, so Node can load it
  data.ts                   decodes src/data/gpt2.json: presets, nextDist() at any T, headKind(), mixing() (lane colours)
src/data/gpt2.json          real GPT-2 small activations for 3 prompts × 3 greedy passes (made by scripts/gpt2-export.ts)
scripts/gpt2-export.ts      offline GPT-2 small forward pass in plain TS (Node 23+); weights in ~/.cache/token-trails/gpt2
src/exhibits/transformer/
  model.ts                  toy model: prompt ids, toy attention + MLP blocks, LayerNorm params, laneMix()
  overview.ts               forward pass with real GPT-2 numbers, tokenizer → greedy pick; plates open the detail views
  tokenizer.ts              tokenizer detail view: pre-split, bytes, BPE merges by rank, ids (id = 256 + merge rank)
  embedding.ts              embedding detail view: onehot · W_E as a lookup, + W_P, into the stream
  layernorm.ts              pre-LN stream schematic, then ln_1 as dots on number lines (μ, σ, γ, β)
  attention.ts              attention detail view, every GEMM animated cell by cell
  mlp.ts                    MLP detail view: up-projection, GELU curve, down-projection, residual
  unembed.ts                ln_f, logits = x · W_Eᵀ (tied), temperature, softmax, sampling strategies; real GPT-2 numbers
src/exhibits/lineage/
  llama.ts                  LLaMA 3 vs GPT-2: blocks, RoPE, RMSNorm, SwiGLU, GQA (real numbers are LLaMA 3 8B)
```

## Adding an exhibit or detail view

1. Write `mountX(root, nav): () => void`: `createFrame` (with `formula: true` for a detail view) → `new Stage` → `new Player(PHASES, frame.controls)` → `new MatrixKit(stage, tokens, frame.setFormula)` → set `player.describe` (caption per phase, for All steps) → `runLoop(step, () => !player.playing)` that ticks, draws, updates the timeline UI and sets the caption. Only set an initial `player.t` when it is still 0 (a `?phase=` link may have placed it). Return a destroy that stops the loop and calls `player.destroy()` and `stage.destroy()`.
2. Register it in `registry.ts` (`route` + `mount`), and add it to `TOUR` in `main.ts` if it belongs to the reading order. Steps of an exhibit go in its `children` with a deeper route (`anatomy/mlp`), which opens with a zoom-in. Routes are hash-based (`#/anatomy/attention`, `#/lineage/llama`); old ones keep working through `ALIASES`.
3. To open a detail view from the overview, add its plate to `plateAt` / `PLATE_ROUTES` in `overview.ts` (hover highlight and `drawOpenHint` follow). Mention the click in that phase's caption.

## Conventions

- **Everything is a function of time.** Drawing reads `player.t` / `player.prog(id)` and keeps no animation state of its own, so the timeline scrubs both ways. Hover state is the only exception.
- **Phases** have a `name` (caption title) and an optional `short` (timeline label, e.g. `QKᵀ`).
- **Detail-view pattern** (follow `attention.ts`):
  - One `sceneX(p)` per group of phases.
  - Each scene has a layout computed in `geom()`, laid out so that a GEMM's A row i lines up with C row i and B column j lines up with C column j (A left, B above, C at the intersection). Transitions lerp matrix rects between scene layouts.
  - Use `MatrixKit` from `core/matrix.ts`: `gemm(p, m, n, K, 'slow'|'fast')` schedules cells, `mk.resolve()` picks the hovered cell or the animated one, `mk.gemmOverlay()` draws the highlights and guides and sets the formula line. It supports a transposed B (`bT`, used for W_projᵀ in `mlp.ts`) and bias vectors.
  - Call `mk.begin()` each frame and register hoverable result matrices with `mk.hit()`.
- **Real vs toy numbers.** Never show a made-up number as GPT-2's. The overview and Unembed use a real GPT-2 small run (`src/lib/gpt2/data.ts`); when only part of a real vector fits, say how much is drawn (`8 drawn`, `dims 1–16 of 768`). Detail views that animate every GEMM compute real arithmetic at toy size (`TOY` in `model.ts`: d_model 8, d_head 4, 2 heads; `TOY_FF` 32), say `shown: toy` in the specs, and show the GPT-2 small shape next to each matrix (`real`, `N × 768`, `value / real`).
- **Changing the real data.** Edit the prompts or fields in `scripts/gpt2-export.ts`, run `node scripts/gpt2-export.ts` (needs model.safetensors, merges.txt, vocab.json from huggingface.co/openai-community/gpt2 in ~/.cache/token-trails/gpt2), and commit the regenerated `src/data/gpt2.json`. Keep it small: it is bundled into the page.
- **Visual language is fixed.** The user approved it; don't redesign it.
  - Colors come only from `C` / the CSS tokens. Never hard-code hex in drawing code, and check both themes.
  - Each token keeps its hue (`--t0…--t6`, token i → `i % 7`) everywhere. Information mixing between tokens is shown by blending hues: the overview uses real attention (`mixing()` in `data.ts`, first-token sinks count as no-ops), the toy detail views use `laneMix()` in `model.ts`, averaged over heads.
  - Layers are tilted glass plates. Matrices are slabs with a 5px depth face. A filled cell is positive, an outlined cell is negative.
  - Fonts: EB Garamond italic for names and math (`serifAt()`, scaled up 10%), Geist for UI text, JetBrains Mono for labels and numbers (`MONO`, never under 10.5px on canvas). Group labels are uppercase mono over thin brackets. Use the constants in `core/draw.ts`, never a hard-coded font string.
  - Keep canvas text minimal, using real units and terms of art (GPT-2 ids, Ġ, 768 → 3072, FLOPs). Explanations go in the caption line: one or two plain sentences per phase. The formula strip under the stage (HTML, via `mk.formula` / `frame.setFormula`) carries the focused cell's arithmetic and a short note.
- Seeds are chosen for readable patterns (`WEIGHT_SEED` in `model.ts`). Changing a seed changes every number shown, so re-check the attention patterns if you touch it.

## Gotchas

- The browser pane is unreliable for checking drawings: screenshots of an emulated viewport can come out blank or half-scaled, and a hidden pane stops rAF and ResizeObserver (document.visibilityState is "hidden"), so sizes go stale. To inspect a frame, POST `canvas.toDataURL()` to a small local server and read the PNG.
- The browser pane's `preview_start` with the `dev` config has failed to serve before. If :5173 doesn't answer, run `npx vite --port 5174` in the background and navigate there. A hidden pane throttles rAF, so take a fresh screenshot before judging a frozen frame.
- Don't rely on Unicode subscript characters in the serif; use `mathRun()` or `fillRich()` for math with subscripts.
- The repo is private: github.com/meyowu/token-trails.

## Shipping

When a piece of work is finished and verified, ship it without asking: commit on `main`, push, then `npm run build:artifact` and republish `dist-artifact/index.html` to the existing preview Artifact (https://claude.ai/artifact/RbScAmBiuDJhyQ9xvgG9vz) by its `url`.
