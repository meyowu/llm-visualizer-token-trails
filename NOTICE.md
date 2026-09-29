# Third-party material

Token Trails includes or is derived from the following. Each keeps its own license.

## Fonts

Bundled as woff2 files from the [@fontsource](https://fontsource.org) packages, all under the
[SIL Open Font License 1.1](https://openfontlicense.org):

| Font | Authors |
| --- | --- |
| EB Garamond | Georg Duffner, Octavio Pardo |
| Geist | Vercel |
| JetBrains Mono | JetBrains |

## GPT-2's tokenizer

`src/lib/gpt2/merges.txt` is GPT-2's list of BPE merge rules, published by OpenAI with GPT-2
([openai-community/gpt2](https://huggingface.co/openai-community/gpt2), MIT License).

## Numbers from public models

The files in `src/data` hold numbers computed offline from public checkpoints by the scripts in `scripts/`:
activations, probabilities, attention weights, gradients, statistics of weights, token counts, and generated text.
No model weights are included. Each model is under its own license:

| Data | Models | License |
| --- | --- | --- |
| `gpt2.json`, `backprop.json`, `quant.json`, `finetune.json` | [GPT-2 small](https://huggingface.co/openai-community/gpt2) | MIT |
| `scaling.json` | GPT-2 small, [medium](https://huggingface.co/openai-community/gpt2-medium), [large](https://huggingface.co/openai-community/gpt2-large) | MIT |
| `spec.json` | GPT-2 small and [distilgpt2](https://huggingface.co/distilbert/distilgpt2) | MIT, Apache-2.0 |
| `bert.json` | [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | Apache-2.0 |
| `t5.json` | [t5-small](https://huggingface.co/google-t5/t5-small) | Apache-2.0 |
| `vit.json` | [ViT-B/16](https://huggingface.co/google/vit-base-patch16-224) | Apache-2.0 |
| `clip.json` | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | MIT |
| `dit.json` | [DiT-XL/2](https://huggingface.co/facebook/DiT-XL-2-256) | CC BY-NC 4.0 |
| `mamba.json` | [Mamba-130m](https://huggingface.co/state-spaces/mamba-130m-hf) | Apache-2.0 |
| `icl.json` | GPT-2 small and [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | MIT, Apache-2.0 |
| `react.json`, `tools.json`, `multi.json` | Qwen3-1.7B | Apache-2.0 |
| `sft.json`, `lora.json` | Qwen3-1.7B and [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | Apache-2.0 |
| `rag.json` | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and Qwen3-1.7B | Apache-2.0 |
| `models.json` | the configurations and tensor shapes of 29 public checkpoints (no weights), listed in `scripts/models-export.ts` | each model's own license |

Model names and the published figures quoted on the pages (parameter counts, context lengths and other facts from
papers and model cards) belong to their authors; the papers are linked from each page's references.
