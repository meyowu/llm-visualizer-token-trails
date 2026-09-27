# Token Trails

Follow the tokens through AI systems: animated, explorable walk-throughs of the Transformer, inference engines and agent workflows.

The atlas follows a model's life in five parts: **Anatomy** (GPT-2 taken apart), **Lineage** (other architectures as changes to GPT-2), **Training**, **Serving** and **Agents**.

## What's here

- **Start here**: what a language model does (a live next-token example with real GPT-2 numbers), what a Transformer is, the path through the Anatomy chapters, and how to read the pictures.
- **Anatomy / Forward pass**: one full forward pass of GPT-2 small with the numbers of a real run, for three prompts, each continued greedily for three passes: real token ids, embeddings, all 144 attention heads (with a label for what each head does), block 1's MLP neurons, the logit lens after every block, and the real next-token distribution.
  - **Tokenizer**: GPT-2's byte-level BPE on the prompt: regex pre-split, bytes (space → Ġ), merge rules fired in rank order, and each token's id (256 + merge rank) placed on the vocabulary.
  - **Embedding**: ids as one-hot rows times W_E (a GEMM that is really a row lookup, on a log-scaled vocabulary axis), plus the first N rows of W_P, giving the residual stream.
  - **LayerNorm & Residual**: the pre-LN residual stream through one block (sub-layers read a normalised copy and add back), then ln_1 on the real toy numbers: each token's features as dots, centred by μ, scaled by σ, then γ ⊙ x̂ + β.
  - **Attention**: every matrix product in one attention head, computed at toy scale (d_model 8, d_head 4) and animated cell by cell: X·W_Q/K/V → Q·Kᵀ → ÷√d → mask → softmax → A·V → concat·W_O → residual add. Hover any result cell to see which row and column produced it.
  - **MLP**: the feed-forward block at toy scale (d_model 8 → d_ff 32, the same 4× as GPT-2): X·W_fc + b → GELU (each activation plotted on the curve) → ·W_proj + b → residual add.

  - **Unembed & Sampling**: real GPT-2 numbers: ln_f on the last position, logits as x · W_Eᵀ (tied to the embedding; 8 of the 768 dimensions drawn, each logit the full sum), the whole 50,257-token distribution at any temperature (top 256 exact, the rest summarised), softmax, and sampling strategies (sample, greedy, top-k, top-p) drawn as an inverse CDF.
- **Lineage / LLaMA**: LLaMA 3 as a diff against GPT-2: the two blocks side by side, then RoPE (rotating q/k pairs by position, and why only the offset matters), RMSNorm vs LayerNorm, the gated SwiGLU MLP, and MHA vs GQA vs MQA with real KV-cache sizes.

Every page pauses at the end of each step so there is time to read (switch to Auto to play straight through); ← and → move between steps. On the overview, click a part marked ↗ (the tokens, the embedding strips, the attn, mlp, ln_f and W_U plates, the next-token bars) to zoom into its detail view.

## Development

```bash
npm install
npm run dev              # http://localhost:5173
npm run build            # typecheck + production build to dist/
npm run build:artifact   # single-file build to dist-artifact/index.html, for publishing as a claude.ai Artifact
node scripts/gpt2-export.ts  # regenerate src/data/gpt2.json from GPT-2 small weights (see the script header)
```

The browser never runs a model. `scripts/gpt2-export.ts` runs GPT-2 small once, offline, in plain TypeScript, and exports the slices the pages draw to `src/data/gpt2.json`. The detail views that animate every matrix product use a toy model at d_model 8 and say so.

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
  lib/gpt2/
    bpe.ts                GPT-2's byte-level BPE tokenizer
    data.ts               decodes the exported GPT-2 run: distributions at any T, head kinds, lane colours
  data/gpt2.json          real GPT-2 small activations for the overview and Unembed
  exhibits/
    start.ts              landing page
    registry.ts           categories, sub-headings and exhibits (with child steps); an exhibit goes live once it has a route
    transformer/
      model.ts            toy model for the detail views: attention and MLP arithmetic at d_model 8
      overview.ts         forward-pass overview
      tokenizer.ts        tokenizer detail view
      embedding.ts        embedding detail view
      layernorm.ts        layernorm & residual detail view
      attention.ts        attention detail view
      mlp.ts              MLP detail view
      unembed.ts          unembed & sampling detail view
    lineage/
      llama.ts            LLaMA 3 compared with GPT-2
scripts/
  gpt2-export.ts          offline GPT-2 small forward pass that writes src/data/gpt2.json
```

To add an exhibit, write a `mount(root, nav) => destroy` function under `exhibits/` and give its entry in `registry.ts` a `route` and `mount`.
