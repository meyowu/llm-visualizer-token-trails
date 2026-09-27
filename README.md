# Token Trails

Follow the tokens through AI systems: animated, explorable walk-throughs of the Transformer, inference engines and agent workflows.

## What's here

- **Transformer / Forward pass**: one full forward pass of GPT-2 small, from tokenization to sampling at the LM head. Scrub the timeline, switch attention heads, and drag the temperature.
- **Transformer / Attention**: every matrix product in one attention head, computed with real arithmetic at toy scale (d_model 8, d_head 4) and animated cell by cell: X·W_Q/K/V → Q·Kᵀ → ÷√d → mask → softmax → A·V → concat·W_O → residual add. Hover any result cell to see which row and column produced it.

On the overview, click a part marked ↗ (currently the attn plate) to zoom into its detail view.

## Development

```bash
npm install
npm run dev              # http://localhost:5173
npm run build            # typecheck + production build to dist/
npm run build:artifact   # single-file build to dist-artifact/index.html, for publishing as a claude.ai Artifact
```

## Structure

```
src/
  main.ts                 sidebar navigation, hash router, zoom transitions between views
  styles.css              design tokens (colors, fonts) and page frame styles
  core/
    stage.ts              DPR-aware canvas and rAF loop
    player.ts             phase timeline: play, scrub, speed, keyboard shortcuts
    frame.ts              exhibit page frame: header, specs, caption, controls
    draw.ts               shared drawing primitives: token chips, glass plates, labels, math text
    theme.ts              canvas palette read from the CSS tokens; follows light/dark changes
  exhibits/
    registry.ts           categories and exhibits; an exhibit goes live once it has a route
    transformer/
      model.ts            toy model: GPT-2 token ids, real attention arithmetic, sampling distribution
      overview.ts         forward-pass overview
      attention.ts        attention detail view
```

To add an exhibit, write a `mount(root, nav) => destroy` function under `exhibits/` and give its entry in `registry.ts` a `route` and `mount`.
