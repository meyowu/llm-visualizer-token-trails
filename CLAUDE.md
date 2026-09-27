# Latent Atlas

Interactive, animated visualizations of AI concepts. Vite + TypeScript, no UI framework; every exhibit draws on a 2D canvas.

## Conventions

- **Everything is a function of time.** Each exhibit owns a `Player` (phases with durations). Drawing reads `player.t` / `player.prog(id)` and must not keep animation state of its own, so scrubbing backwards always works.
- **Visual language is fixed** — keep it for new exhibits:
  - Dark-first palette from the CSS tokens in `src/styles.css`. Canvas colors come from `C` in `core/theme.ts`; never hard-code hex in drawing code.
  - Each token has a hue (`--t0…--t6`, token i → `i % 7`) and stays that hue everywhere. Mixed information is shown by blending hues.
  - Layers are tilted glass plates (`plate()`), matrices are slabs with a 5px depth face. Filled cell = positive, outlined cell = negative.
  - Fonts: Newsreader italic for names and math, Geist for UI, Geist Mono for labels and numbers. Uppercase mono group labels sit over thin brackets.
  - Minimal text on the canvas: real units and terms of art (GPT-2 ids, Ġ, 768 → 3072). Explanations go in the caption line, in Chinese; technical labels stay in English.
- **Toy vs real scale.** Detail views compute real arithmetic at toy size (`TOY` in `model.ts`) and always show the GPT-2 small size next to it.
- Routes are hash-based (`#/transformer/attention`). A deeper route opens with a zoom-in from the click point.

## Commands

- `npm run dev`, `npm run build` (runs `tsc --noEmit` first), `npm run build:artifact` (single-file page for a claude.ai Artifact).
