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
  { term: 'Attention', match: [], def: 'Lets each position mix in information from itself and earlier positions, weighted by how well its query matches their keys.' },
  { term: 'Attention head', match: [], def: 'One of several attention computations run side by side, each on its own slice of the vector (64 of 768 numbers in GPT-2), free to track a different relation.' },
  { term: 'Attention sink', match: ['attention sink'], def: 'Many heads park spare attention on the first token, whose value adds almost nothing; the overview colours treat that attention as a no-op.' },
  { term: 'BPE', match: ['bpe'], def: 'Byte-pair encoding: a tokenizer that starts from bytes and repeatedly merges the most frequent adjacent pair seen in training text. GPT-2 learned 50,000 merges this way.' },
  { term: 'Causal mask', match: ['causal mask'], def: 'Hides later positions from each position, so the model cannot peek at the tokens it is learning to predict.' },
  { term: 'Cross-attention', match: ['cross-attention'], def: 'Attention whose queries come from one sequence and keys and values from another: in a translation model, the decoder (target) reads the encoder’s output (source).' },
  { term: 'Encoder and decoder', match: ['encoder', 'decoder'], def: 'In the 2017 Transformer, the encoder reads the whole source sentence at once; the decoder writes the output one token at a time, under a causal mask, reading the encoder’s output through cross-attention.' },
  { term: 'Decoder-only', match: ['decoder-only'], def: 'A Transformer that reads left to right under a causal mask and predicts the next token. GPT-2 and LLaMA are decoder-only.' },
  { term: 'Embedding', match: ['embedding matrix'], def: 'A learned vector of numbers (768 in GPT-2) that stands for a token or a position; tokens used in similar ways end up with similar vectors.' },
  { term: 'FLOPs', match: ['flops'], def: 'Floating-point operations. One multiply-add counts as two.' },
  { term: 'GELU', match: ['gelu'], def: 'A smooth nonlinearity: large positive inputs pass almost unchanged, small ones are damped and negative ones are pushed close to 0.' },
  { term: 'GEMM', match: ['gemm'], def: 'General matrix multiply, C = A · B: the operation almost all of a Transformer’s compute goes into.' },
  { term: 'GQA', match: ['gqa'], def: 'Grouped-query attention: several query heads share one key/value head, which shrinks the KV cache.' },
  { term: 'Greedy decoding', match: ['greedy'], def: 'Always picking the most likely next token.' },
  { term: 'Induction head', match: ['induction head'], def: 'A head that finds an earlier copy of the current token and attends to the token that came after it, so a repeated pattern can be continued.' },
  { term: 'KV cache', match: ['kv cache'], def: 'While generating, the keys and values of past tokens are stored so each new token only computes its own. It grows with every token and every layer.' },
  { term: 'BatchNorm', match: [], def: 'Normalises each feature across the examples in a batch (common in vision). Transformers use LayerNorm instead, which normalises each token on its own.' },
  { term: 'Post-LN and pre-LN', match: ['pre-ln', 'post-ln'], def: 'Where LayerNorm sits: after each residual add (post-LN, the 2017 Transformer) or on the copy each sub-layer reads (pre-LN, GPT-2 and later), which trains more stably.' },
  { term: 'LayerNorm', match: ['layernorm'], def: 'Rescales each token’s vector to mean 0 and spread 1, then applies a learned scale (γ) and shift (β) per feature.' },
  { term: 'Logit lens', match: ['logit lens'], def: 'Reading a middle layer through the final LayerNorm and the unembedding, to see what the model would predict if it stopped there.' },
  { term: 'Logits', match: ['logits', 'logit'], def: 'Raw, unnormalised scores, one per vocabulary token; softmax turns them into probabilities.' },
  { term: 'Mixture of experts (MoE)', match: ['mixture of experts', 'moe'], def: 'A layer with several expert MLPs and a router; each token runs through only the few experts the router picks, so the model can store many more parameters than it uses per token.' },
  { term: 'Router', match: ['router'], def: 'In a mixture of experts, a small matrix that scores every expert for a token; the top-scoring experts run, weighted by a softmax over their scores.' },
  { term: 'MLP', match: ['mlp'], def: 'Multi-layer perceptron: two matrix products with a nonlinearity (GELU) between them, applied to each token on its own.' },
  { term: 'x · W and W x', match: [], def: 'Two ways to write the same product. Papers often write W x (a column vector); code and this site write x · W (x @ W), with one row per token.' },
  { term: 'Query, key, value', match: [], def: 'Three vectors made from each token for attention: what it is looking for, what it offers, and what it passes on when chosen.' },
  { term: 'Residual stream', match: ['residual stream'], def: 'The per-token vector that runs through the whole model. Every layer reads it and adds its output back instead of replacing it.' },
  { term: 'RMSNorm', match: ['rmsnorm'], def: 'LayerNorm without the mean: divides by the root mean square, then applies a learned scale.' },
  { term: 'RNN', match: ['rnn', 'rnns', 'recurrent neural networks'], def: 'Recurrent neural network: reads a sequence one token at a time, updating a hidden state h[t] = f(h[t−1], x[t]). Each step waits for the previous one, so the steps cannot run in parallel.' },
  { term: 'RoPE', match: ['rope'], def: 'Rotary position embedding: query and key pairs are rotated by angles that grow with position, so the position part of a score depends only on the offset between tokens.' },
  { term: 'Softmax', match: ['softmax'], def: 'Turns a list of scores into positive weights that sum to 1: the exp of each score divided by the sum of all the exps.' },
  { term: 'SwiGLU', match: ['swiglu'], def: 'An MLP in which one projection, passed through SiLU, gates another element by element.' },
  { term: 'Teacher forcing', match: ['teacher forcing'], def: 'In training, the decoder is fed the correct previous tokens rather than its own guesses, so every position can be trained at once under the causal mask.' },
  { term: 'Temperature', match: ['temperature'], def: 'Logits are divided by T before softmax: below 1 sharpens the distribution, above 1 flattens it, and T → 0 becomes greedy.' },
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
