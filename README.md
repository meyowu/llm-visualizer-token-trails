# Token Trails

Follow the tokens through AI systems: animated, explorable walk-throughs of the Transformer, inference engines and agent workflows.

## What's here

- **Transformer / Forward pass**: one full forward pass of GPT-2 small, from tokenization to sampling at the LM head. Scrub the timeline, switch attention heads, and drag the temperature.
- **Transformer / Tokenizer**: GPT-2's byte-level BPE on the prompt: regex pre-split, bytes (space → Ġ), merge rules fired in rank order, and each token's id (256 + merge rank) placed on the vocabulary.
- **Transformer / Embedding**: ids as one-hot rows times W_E (a GEMM that is really a row lookup, on a log-scaled vocabulary axis), plus the first N rows of W_P, giving the residual stream.
- **Transformer / Attention**: every matrix product in one attention head, computed with real arithmetic at toy scale (d_model 8, d_head 4) and animated cell by cell: X·W_Q/K/V → Q·Kᵀ → ÷√d → mask → softmax → A·V → concat·W_O → residual add. Hover any result cell to see which row and column produced it.
- **Transformer / MLP**: the feed-forward block at toy scale (d_model 8 → d_ff 32, the same 4× as GPT-2): X·W_fc + b → GELU (each activation plotted on the curve) → ·W_proj + b → residual add.

On the overview, click a part marked ↗ (the tokens, the embedding strips, the attn plate, the mlp plate) to zoom into its detail view.

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
    matrix.ts             matrices as 2.5D slabs, GEMM cell schedules, hover and the formula line
    theme.ts              canvas palette read from the CSS tokens; follows light/dark changes
  exhibits/
    registry.ts           categories and exhibits; an exhibit goes live once it has a route
    transformer/
      model.ts            toy model: GPT-2 token ids, real attention and MLP arithmetic, sampling distribution
      overview.ts         forward-pass overview
      tokenizer.ts        tokenizer detail view
      embedding.ts        embedding detail view
      attention.ts        attention detail view
      mlp.ts              MLP detail view
```

To add an exhibit, write a `mount(root, nav) => destroy` function under `exhibits/` and give its entry in `registry.ts` a `route` and `mount`.
