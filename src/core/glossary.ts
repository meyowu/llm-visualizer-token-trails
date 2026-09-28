/*
 * Terms of art, defined once. Captions mark the first mention of each with a dotted underline
 * and its definition as a tooltip; the Glossary page lists them all.
 */

export interface Term {
  term: string
  /** Lower-case spellings to mark in captions; empty for glossary-only entries. */
  match: string[]
  def: string
}

export const TERMS: Term[] = [
  { term: 'adaLN-Zero', match: ['adaln-zero', 'adaln'], def: 'DiT’s conditioning: the timestep and class set each block’s LayerNorm scale and shift and a gate on each sub-layer’s output; the gates start at zero, so blocks start as the identity.' },
  { term: 'Attention', match: [], def: 'Lets each position mix in information from itself and earlier positions, weighted by how well its query matches their keys.' },
  { term: 'Attention head', match: [], def: 'One of several attention computations run side by side, each on its own slice of the vector (64 of 768 numbers in GPT-2), free to track a different relation.' },
  { term: 'Attention sink', match: ['attention sink'], def: 'Many heads park spare attention on the first token, whose value adds almost nothing; the overview colours treat that attention as a no-op.' },
  { term: 'Base model', match: ['base model'], def: 'A model after pretraining only: it continues text in the style of its training data and has not been tuned to follow instructions.' },
  { term: 'BPE', match: ['bpe'], def: 'Byte-pair encoding: a tokenizer that starts from bytes and repeatedly merges the most frequent adjacent pair seen in training text. GPT-2 learned 50,000 merges this way.' },
  { term: 'Causal mask', match: ['causal mask'], def: 'Hides later positions from each position, so the model cannot peek at the tokens it is learning to predict.' },
  { term: 'Chat template', match: ['chat template'], def: 'The text format a chat model was tuned on: special tokens such as <|im_start|> open each message and name who wrote it, and the model’s turn begins where the template stops.' },
  { term: 'Chunked prefill', match: ['chunked prefill'], def: 'Splitting a long prompt’s prefill into pieces that ride along with other requests’ decode steps, so no single step becomes slow.' },
  { term: 'Classifier-free guidance', match: ['classifier-free guidance'], def: 'At sampling time, running a diffusion model with and without the condition and moving further in the direction the condition adds, for images that match it more strongly.' },
  { term: 'Compute-bound and memory-bound', match: ['compute-bound', 'memory-bound'], def: 'Whether a GPU step is limited by how fast it can do arithmetic or by how fast it can read memory; a step doing few operations per byte read (like decoding one token) waits on memory.' },
  { term: 'Continuous batching', match: ['continuous batching', 'iteration-level'], def: 'Rescheduling a serving batch after every decode step: finished requests leave at once and waiting ones take their place, instead of waiting for the whole batch to finish.' },
  { term: 'Contrastive learning', match: ['contrastive'], def: 'Training by comparison: matching pairs (an image and its caption) are pulled together and every mismatched pair in the batch is pushed apart.' },
  { term: 'Copy-on-write', match: ['copy-on-write'], def: 'Sharing memory between users until one of them writes to it; only then is a private copy made.' },
  { term: 'Cross-attention', match: ['cross-attention'], def: 'Attention whose queries come from one sequence and keys and values from another: in a translation model, the decoder (target) reads the encoder’s output (source).' },
  { term: 'Encoder and decoder', match: ['encoder', 'decoder'], def: 'In the 2017 Transformer, the encoder reads the whole source sentence at once; the decoder writes the output one token at a time, under a causal mask, reading the encoder’s output through cross-attention.' },
  { term: 'Decoder-only', match: ['decoder-only'], def: 'A Transformer that reads left to right under a causal mask and predicts the next token. GPT-2 and LLaMA are decoder-only.' },
  { term: 'Encoder-only', match: ['encoder-only'], def: 'A Transformer with no causal mask that reads a whole text at once and outputs a vector per token, for understanding rather than generating text. BERT is encoder-only.' },
  { term: 'Diffusion model', match: ['diffusion'], def: 'A model trained to remove noise from data; generating means starting from pure noise and denoising step by step.' },
  { term: 'Embedding', match: ['embedding matrix'], def: 'A learned vector of numbers (768 in GPT-2) that stands for a token or a position; tokens used in similar ways end up with similar vectors.' },
  { term: 'Few-shot prompting', match: ['few-shot'], def: 'Putting a few worked examples of a task in the prompt before the real question, so the model continues the pattern.' },
  { term: 'FlashAttention', match: ['flashattention'], def: 'An exact attention kernel that works through tiles of queries, keys and values in the GPU’s on-chip SRAM, with an online softmax, so the N × N score matrix is never written to HBM.' },
  { term: 'FLOPs', match: ['flops'], def: 'Floating-point operations. One multiply-add counts as two.' },
  { term: 'Fine-tuning', match: ['fine-tuned', 'fine-tuning'], def: 'Training a pretrained model a little further, on a smaller dataset for one task, usually with a small new output layer on top.' },
  { term: 'GELU', match: ['gelu'], def: 'A smooth nonlinearity: large positive inputs pass almost unchanged, small ones are damped and negative ones are pushed close to 0.' },
  { term: 'GEMM', match: ['gemm'], def: 'General matrix multiply, C = A · B: the operation almost all of a Transformer’s compute goes into.' },
  { term: 'GQA', match: ['gqa'], def: 'Grouped-query attention: several query heads share one key/value head, which shrinks the KV cache.' },
  { term: 'Greedy decoding', match: ['greedy'], def: 'Always picking the most likely next token.' },
  { term: 'In-context learning', match: ['in-context learning'], def: 'A model doing a task it is shown in its prompt, by examples or instructions, with no change to its weights.' },
  { term: 'Induction head', match: ['induction head'], def: 'A head that finds an earlier copy of the current token and attends to the token that came after it, so a repeated pattern can be continued.' },
  { term: 'Instruction tuning', match: ['instruction-tuned', 'instruction tuning'], def: 'Fine-tuning a pretrained model on instructions paired with good answers, written in a chat template, so that it answers requests instead of continuing the text.' },
  { term: 'KV cache', match: ['kv cache'], def: 'While generating, the keys and values of past tokens are stored so each new token only computes its own. It grows with every token and every layer.' },
  { term: 'BatchNorm', match: [], def: 'Normalises each feature across the examples in a batch (common in vision). Transformers use LayerNorm instead, which normalises each token on its own.' },
  { term: 'Patch embedding', match: ['patch embedding'], def: 'ViT’s replacement for a vocabulary lookup: each 16 × 16 image patch is flattened and multiplied by one shared matrix, the same as a convolution with a 16 × 16 kernel and stride 16.' },
  { term: 'Online softmax', match: ['online softmax'], def: 'Computing a softmax in chunks with a running maximum and a running sum, rescaling earlier terms when the maximum grows; the result equals the ordinary softmax.' },
  { term: 'PagedAttention', match: ['pagedattention', 'block table', 'block tables'], def: 'vLLM’s way of storing the KV cache: in fixed-size blocks handed out on demand and found through a per-request block table, like virtual-memory pages.' },
  { term: 'Perplexity', match: ['perplexity'], def: 'e raised to the average next-token loss: roughly, how many tokens the model is choosing between at each step. Lower is better.' },
  { term: 'Post-LN and pre-LN', match: ['pre-ln', 'post-ln'], def: 'Where LayerNorm sits: after each residual add (post-LN, the 2017 Transformer) or on the copy each sub-layer reads (pre-LN, GPT-2 and later), which trains more stably.' },
  { term: 'LayerNorm', match: ['layernorm'], def: 'Rescales each token’s vector to mean 0 and spread 1, then applies a learned scale (γ) and shift (β) per feature.' },
  { term: 'Logit lens', match: ['logit lens'], def: 'Reading a middle layer through the final LayerNorm and the unembedding, to see what the model would predict if it stopped there.' },
  { term: 'Logits', match: ['logits', 'logit'], def: 'Raw, unnormalised scores, one per vocabulary token; softmax turns them into probabilities.' },
  { term: 'Masked language modelling', match: ['masked language modelling', 'masked language model', 'mlm'], def: 'BERT’s training task: some tokens are hidden and the model predicts them from the words on both sides.' },
  { term: 'Mixture of experts (MoE)', match: ['mixture of experts', 'moe'], def: 'A layer with several expert MLPs and a router; each token runs through only the few experts the router picks, so the model can store many more parameters than it uses per token.' },
  { term: 'Router', match: ['router'], def: 'In a mixture of experts, a small matrix that scores every expert for a token; the top-scoring experts run, weighted by a softmax over their scores.' },
  { term: 'Multi-head latent attention (MLA)', match: ['mla', 'multi-head latent attention'], def: 'DeepSeek’s attention: each token’s keys and values are compressed into one small latent vector, which is all the KV cache stores; every head’s keys and values are rebuilt from it.' },
  { term: 'Multi-token prediction', match: ['multi-token prediction'], def: 'Training a model to also predict tokens further ahead (the one after next), as an extra, denser training signal.' },
  { term: 'MLP', match: ['mlp'], def: 'Multi-layer perceptron: two matrix products with a nonlinearity (GELU) between them, applied to each token on its own.' },
  { term: 'x · W and W x', match: [], def: 'Two ways to write the same product. Papers often write W x (a column vector); code and this site write x · W (x @ W), with one row per token.' },
  { term: 'Prefill and decode', match: ['prefill', 'decode step', 'decode steps'], def: 'The two phases of generation: prefill runs the whole prompt in one pass and fills the KV cache; each decode step then runs a single new token.' },
  { term: 'Quantization', match: ['quantization', 'quantized', 'quantizes'], def: 'Storing numbers in fewer bits, for example weights as 8- or 4-bit integers with a scale, to make a model smaller and faster to serve at a small cost in accuracy.' },
  { term: 'Query, key, value', match: [], def: 'Three vectors made from each token for attention: what it is looking for, what it offers, and what it passes on when chosen.' },
  { term: 'Residual stream', match: ['residual stream'], def: 'The per-token vector that runs through the whole model. Every layer reads it and adds its output back instead of replacing it.' },
  { term: 'Relative position bias', match: ['relative position bias', 'relative position buckets'], def: 'T5’s way of giving attention a sense of order: each query–key pair is sorted by distance into a bucket, and a learned number per bucket and head is added to its score.' },
  { term: 'RMSNorm', match: ['rmsnorm'], def: 'LayerNorm without the mean: divides by the root mean square, then applies a learned scale.' },
  { term: 'RNN', match: ['rnn', 'rnns', 'recurrent neural networks'], def: 'Recurrent neural network: reads a sequence one token at a time, updating a hidden state h[t] = f(h[t−1], x[t]). Each step waits for the previous one, so the steps cannot run in parallel.' },
  { term: 'RoPE', match: ['rope'], def: 'Rotary position embedding: query and key pairs are rotated by angles that grow with position, so the position part of a score depends only on the offset between tokens.' },
  { term: 'Speculative decoding', match: ['speculative decoding', 'draft model'], def: 'Generating with a small draft model that guesses several tokens and a large model that checks them all in one pass, keeping the agreeing prefix; the output is the same as the large model’s.' },
  { term: 'SRAM and HBM', match: ['sram', 'hbm'], def: 'A GPU’s two main memories: SRAM is small and very fast, on the chip beside the arithmetic units; HBM is large (tens of GB) and slower, next to the chip.' },
  { term: 'Selective scan', match: ['selective scan', 'selectivity'], def: 'Mamba’s state-space step, in which the step size Δ and the matrices B and C depend on the current token, so the model chooses per token what to write into its state and what to keep.' },
  { term: 'Sentinel token', match: ['sentinel', 'sentinels', 'sentinel tokens'], def: 'In T5’s span corruption, a placeholder token (<extra_id_0>, <extra_id_1>, …) that marks where a span was dropped; the target lists each sentinel followed by its span.' },
  { term: 'Span corruption', match: ['span corruption'], def: 'T5’s pretraining task: spans of the input are replaced by sentinel tokens, and the model writes out only the dropped spans.' },
  { term: 'Softmax', match: ['softmax'], def: 'Turns a list of scores into positive weights that sum to 1: the exp of each score divided by the sum of all the exps.' },
  { term: 'State-space model (SSM)', match: ['state-space model', 'state-space', 'ssm'], def: 'A sequence layer that carries a fixed-size state from token to token, h_t = Ā h_t−1 + B̄ x_t, and reads its output from it, y_t = C h_t. Work grows linearly with length.' },
  { term: 'SwiGLU', match: ['swiglu'], def: 'An MLP in which one projection, passed through SiLU, gates another element by element.' },
  { term: 'Teacher forcing', match: ['teacher forcing'], def: 'In training, the decoder is fed the correct previous tokens rather than its own guesses, so every position can be trained at once under the causal mask.' },
  { term: 'Zero-shot', match: ['zero-shot'], def: 'Doing a task without any training examples for it, such as classifying images by comparing them with a caption written for each class.' },
  { term: 'Vision Transformer (ViT)', match: ['vision transformer', 'vit'], def: 'A Transformer encoder whose tokens are image patches plus a [CLS] token; the [CLS] output is classified.' },
  { term: 'WordPiece', match: ['wordpiece'], def: 'BERT’s tokenizer: words are split greedily into the longest pieces in its vocabulary; ## marks a piece that continues a word.' },
  { term: 'Temperature', match: ['temperature'], def: 'Logits are divided by T before softmax: below 1 sharpens the distribution, above 1 flattens it, and T → 0 becomes greedy.' },
  { term: 'Text to text', match: ['text in, text out', 'text-to-text'], def: 'T5’s framing: every task, from translation to classification, is a string in and a string out, named by a prefix such as “summarize:”.' },
  { term: 'Token', match: [], def: 'A piece of text the model reads as one unit (a word, part of a word, a byte or punctuation), each with an id in the vocabulary.' },
  { term: 'Top-k and top-p', match: ['top-k', 'top-p'], def: 'Sampling that first keeps only the k likeliest tokens, or the smallest set whose probabilities add up to p, then renormalises.' },
  { term: 'Training', match: [], def: 'Adjusting every weight, over many steps, so the model gives a higher probability to the actual next token across a large body of text. The Training chapter is coming.' },
]

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
const PATTERNS = TERMS.flatMap((t) => t.match.map((m) => ({ t, re: new RegExp(`(?<![\\w-])${m.replace(/[-]/g, '\\-')}(?![\\w-])`, 'gi') })))
/** Whether index i of s falls inside a placeholder already inserted for another term. */
const inside = (s: string, i: number) => s.lastIndexOf('\u0000', i) > s.lastIndexOf('\u0002', i)

/** Mark the first mention of each glossary term in rich() HTML; tags are left untouched. */
export function withTerms(html: string): string {
  const done = new Set<Term>()
  return html.split(/(<[^>]+>)/).map((part) => {
    if (part.startsWith('<')) return part
    let out = part
    for (const { t, re } of PATTERNS) {
      if (done.has(t)) continue
      re.lastIndex = 0
      let m: RegExpExecArray | null
      while ((m = re.exec(out)) && inside(out, m.index)) { /* skip matches inside another term */ }
      if (!m) continue
      done.add(t)
      out = out.slice(0, m.index) + `\u0000${t.term}\u0001${m[0]}\u0002` + out.slice(m.index + m[0].length)
    }
    // placeholders keep one term from matching inside another's markup
    return out.replace(/\u0000([^\u0001]*)\u0001([^\u0002]*)\u0002/g, (_, term: string, text: string) => `<abbr class="term" title="${esc(TERMS.find((t) => t.term === term)!.def)}">${text}</abbr>`)
  }).join('')
}
