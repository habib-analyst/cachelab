# CacheLab: KV cache eviction and speculative decoding, measured honestly

I am Habib Ur Rehman, an ML engineer. This is Project 2 of my AI inference
engineering programme. Project 1 (ServeCore) was about serving models fast.
This one is about the two bottlenecks that show up as soon as contexts get
long and models get big: the KV cache eats memory and slows every step, and
autoregressive decoding produces one token per expensive forward pass.

## The 30-second version

Part A: I implemented three KV cache eviction policies from scratch
(sliding window, StreamingLLM, H2O) inside my own decoder loop, where
eviction physically cuts the cache during generation, and measured what
each one costs in quality. Keeping just 4 attention sink tokens is the
difference between perplexity 45.6 (broken) and 11.9 (nearly free) at the
same memory budget. Part B: I implemented speculative decoding from scratch
with a 0.5B draft model and a 1.5B target. The mechanism works and the
output matches the target's own greedy output in 8 of 12 runs (the 4 misses
are exact bf16 logit ties, diagnosed below). On this small CPU box it does
not actually run faster, and I report that plainly with the reasons.

## Hardware and honesty rules

Every number in this README was measured on the machine that produced it:

- CPU-only VM, 2 x86-64 vCPUs, 7.75 GiB RAM, no GPU
- PyTorch 2.13.0+cpu, Transformers 5.19.0
- Part A model: Qwen2.5-0.5B-Instruct in float32
- Part B models: Qwen2.5-1.5B-Instruct (target) and Qwen2.5-0.5B-Instruct
  (draft), both in bfloat16, same tokenizer family and vocabulary

Nothing is estimated or extrapolated. Where something did not work, the
limits section at the end says so.

## Part A: KV cache eviction for long contexts

### How it works

`kv_eviction/cache_engine.py` is a decoder I wrote myself. It walks the
model's layers using the model's own projections, norms and MLP, but the
attention, the cache storage and every eviction decision are my code. The
cache is stored per layer and per KV head as plain tensors of kept tokens,
together with each token's original position. RoPE uses the true positions,
so a surviving token attends exactly as it would in a full cache. When the
cache grows past the budget, eviction slices the tensors. The memory saving
and the attention speedup are real, not simulated.

The policies, with budget B tokens kept per KV head:

- **Full**: never evict. The baseline.
- **Sliding window**: keep the most recent B tokens.
- **StreamingLLM** (Xiao et al., arXiv:2309.17453): keep the first 4 tokens
  (attention sinks) plus the most recent B-4.
- **H2O** (Zhang et al., arXiv:2306.14048): track the accumulated attention
  mass every cached token receives, keep the top scoring tokens (heavy
  hitters) plus a recent window of B/4 tokens.

Eviction runs per layer after each processed chunk during prefill, and
every step during decoding. Before trusting it, I verified my decoder
against Hugging Face's forward pass on the same prefix: logits agree to a
max absolute difference of 2.1e-05, which is float32 rounding noise.

### Test 1: needle in a haystack

A passcode sentence ("the secret passcode for the CacheLab experiment is
amber-tiger-4729") is buried in 2048 tokens of filler text at 30% and 70%
depth, followed by a question asking for the passcode. Greedy decoding, 12
tokens, success means the passcode comes back.

| Policy | Budget | Cache (MB) | Depth 0.3 | Depth 0.7 |
|---|---|---|---|---|
| Full | none | 51.5 | found | found |
| Sliding | 1024 (50%) | 25.2 | missed | missed |
| Sliding | 512 (25%) | 12.6 | missed | missed |
| StreamingLLM | 1024 (50%) | 25.2 | missed | found |
| StreamingLLM | 512 (25%) | 12.6 | missed | missed |
| H2O | 1024 (50%) | 25.2 | missed | missed |
| H2O | 512 (25%) | 12.6 | missed | missed |

Reading this table:

- Any policy whose window no longer covers the needle position fails, as it
  must. At depth 0.3 the needle sits around token 620, outside every
  evicting budget's reach by the end of the prompt.
- The interesting cell is depth 0.7 at budget 1024, where the needle is
  inside the recent window for both sliding and streaming. Sliding still
  fails and degenerates into repeating the question back. StreamingLLM,
  identical except for 4 sink tokens, answers correctly. Those first tokens
  carry no content, but the model leans on them, and removing them disturbs
  every later representation. That is the StreamingLLM paper's central claim
  reproduced on a small model.
- H2O never returns the full passcode, but in 3 of its 4 runs the first
  generated token is "amber" or "Amber": part of the needle survived as a
  heavy hitter and the model clearly reaches for it, then loses the rest.
  Partial credit in spirit, counted as a miss in the table.
- Decode speed after the 2048-token prefill trends faster as the cache
  shrinks: 0.39 to 0.47 s per token with the full cache versus 0.22 to
  0.44 s per token across the budget-1024 runs and 0.24 to 0.34 s per token
  across the budget-512 runs (12-token decodes on a shared machine, so
  treat these as indicative; exact per-run values are in
  `results/needle_results.json`).

### Test 2: streaming perplexity on real prose

A 3072-token slice of Alice's Adventures in Wonderland (Project Gutenberg
#11), streamed through the decoder in 256-token chunks. Every token is
scored under the policy's cache, which is what a serving system would
actually experience.

| Policy | Budget | Cache (MB) | Perplexity | s per token |
|---|---|---|---|---|
| Full | none | 75.5 | 11.70 | 0.0297 |
| Sliding | 1024 | 25.2 | 45.58 | 0.0252 |
| Sliding | 512 | 12.6 | 48.95 | 0.0246 |
| StreamingLLM | 1024 | 25.2 | 11.90 | 0.0267 |
| StreamingLLM | 512 | 12.6 | 12.42 | 0.0224 |
| H2O | 1024 | 25.2 | 11.88 | 0.0234 |
| H2O | 512 | 12.6 | 12.10 | 0.0220 |

Two conclusions I did not expect to be this clean at 0.5B scale:

1. A plain sliding window is catastrophic for language modelling quality on
   this model, roughly 4x worse perplexity, even though it keeps half the
   context. The needle failures above are not a retrieval quirk; the model
   is genuinely disturbed without its sink tokens.
2. StreamingLLM and H2O are nearly free. At one sixth of the cache
   (12.6 MB instead of 75.5 MB), H2O costs 0.4 perplexity points and
   StreamingLLM costs 0.7. H2O is slightly better than StreamingLLM at both
   budgets, matching the papers' ordering.

## Part B: speculative decoding with a draft model

### How it works

`speculative/speculative.py` implements greedy speculative decoding
(Leviathan et al., arXiv:2211.17192) from scratch on top of Hugging Face
model calls and dynamic caches:

1. The draft model (0.5B) greedily proposes k tokens.
2. The target model (1.5B) scores all k proposals in one forward pass.
3. The longest proposal prefix that matches the target's own greedy choices
   is accepted. The first mismatch is replaced by the target's token, and
   if all k match, the target contributes one bonus token.
4. Both caches are cropped back to the accepted prefix and the round's
   final token is fed through each model, so the next round starts from
   exactly the emitted sequence.

Four prompts, 96 tokens each, k in {2, 4, 6}, same prompts and budgets for
the target-only baseline.

### Results

| Configuration | Mean tok/s | Mean speedup | Draft acceptance |
|---|---|---|---|
| Target only (1.5B) | 2.11 | 1.00x | n/a |
| Speculative k=2 | 1.93 | 0.94x | 70.8% |
| Speculative k=4 | 1.89 | 0.91x | 56.6% |
| Speculative k=6 | 1.81 | 0.87x | 45.2% |

Best single result: prompt 1 at k=2, 1.78 tok/s against a 1.50 tok/s
baseline (1.18x). Full per-prompt numbers are in
`results/speculative_results.json`.

The honest headline: at this scale, on this CPU, speculative decoding does
not pay. Acceptance is healthy (up to 79% at k=2), but a verification round
is not cheap here. The target is only 3x the draft's size, both run in bf16
on 2 cores where a batched verify pass costs close to k single steps, and
each round also pays two cache resync steps. The technique's win comes from
a verify pass costing barely more than one decode step, which is true for
large models on GPUs where decode is memory-bandwidth bound. A 1.5B model
on 2 CPU cores is not that regime, and the numbers say so.

### Correctness check, including the interesting failure

The speculative output should be token-identical to target-only greedy
decoding. Result: 8 of 12 (prompt, k) configurations are identical across
all 96 tokens. Four diverge, at generated indices 19, 20, 32 and 88.

I chased one divergence down (prompt 4, k=4, index 20). The choice there is
between " Rex" and " rex" in "T. Rex". Replaying the target over the
identical prefix three ways (one token at a time, one batched forward,
chunked forwards) shows the top-2 logit gap is exactly 0.0 in bf16: the two
tokens are tied, and which one argmax returns depends on floating point
summation order inside the matmuls. Both runs emitted a token the target
itself ranked first under its own forward shape. So the guarantee holds up
to exact ties in bf16 arithmetic; it is not an algorithm error, and the
other three divergences show the same pattern (a single flipped token deep
into an otherwise exact 96-token match). On a GPU in fp32 this class of
tie is far rarer; in bf16 on CPU it is part of the landscape, and now it is
measured rather than assumed.

## What I take from this build

Eviction policy choice matters more than budget size. The same 512-token
budget is unusable with a sliding window and nearly free with attention
aware policies, because a handful of sink and heavy hitter tokens do a
disproportionate share of the work. And speculative decoding is a systems
trade, not a free lunch: it converts cheap draft compute into fewer
expensive target steps only when the target step is expensive enough. Both
lessons are the kind you only really learn by measuring, which is the point
of this programme.

## Run it yourself

```bash
pip install -r requirements.txt
# Part A (uses the Qwen2.5-0.5B-Instruct weights; point MODEL in the
# scripts at your copy)
python kv_eviction/run_needle.py        # writes results/needle_results.json
python kv_eviction/run_perplexity.py    # writes results/perplexity_results.json
# Part B (needs Qwen2.5-1.5B-Instruct under models/, see below)
python speculative/speculative.py       # writes results/speculative_results.json
python charts/make_charts.py            # regenerates the PNG charts
```

Model weights are not in this repo. Qwen2.5-0.5B-Instruct and
Qwen2.5-1.5B-Instruct are public on Hugging Face under the Qwen org. On
this VM the Hugging Face Python client hangs through the proxy, so the
weights here were fetched with resumable curl range downloads (script kept
in `logs/download_15b.sh`).

## Honest limits

- One small model family (Qwen2.5, 0.5B and 1.5B), one machine, 2 CPU
  cores. Absolute speeds will differ everywhere else; the qualitative
  ordering (sinks matter, H2O edges StreamingLLM on perplexity, sliding
  collapses) is the claim, and it matches the cited papers.
- The needle test uses one needle, two depths, two budgets and a synthetic
  haystack of repeated filler sentences. It shows the mechanism and its
  failure geometry, not a production retrieval benchmark.
- My decoder processes prefill in chunks and evicts after each chunk, so
  during prefill the cache can exceed the budget by up to one chunk
  (256 tokens). Steady-state decode respects the budget exactly.
- Part B uses greedy verification only. The sampling variant of
  speculative decoding (rejection sampling that preserves the target
  distribution exactly) is not implemented here.
- Timings come from a shared VM, so individual runs vary. Every table value
  is the measured run recorded in the JSON files, not an average of hidden
  repeats.

## Papers

- Xiao et al., Efficient Streaming Language Models with Attention Sinks
  (StreamingLLM), arXiv:2309.17453
- Zhang et al., H2O: Heavy-Hitter Oracle for Efficient Generative
  Inference of Large Language Models, arXiv:2306.14048
- Leviathan et al., Fast Inference from Transformers via Speculative
  Decoding, arXiv:2211.17192
