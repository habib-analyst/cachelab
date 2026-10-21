"""Speculative decoding, implemented from scratch (Leviathan et al.,
arXiv:2211.17192), greedy variant.

A small draft model (Qwen2.5-0.5B-Instruct) proposes k tokens ahead. The
target model (Qwen2.5-1.5B-Instruct) checks all k proposals in ONE forward
pass. The longest prefix of proposals that matches the target's own greedy
choices is accepted; the first mismatch is replaced by the target's token,
and if every proposal matches, the target also gives one bonus token for
free. Both KV caches are then cropped back to the accepted prefix and the
final token of the round is fed through each model so the next round starts
from exactly the emitted sequence.

Because every emitted token is either a proposal the target agreed with or
the target's own greedy choice, the output stream is the target's greedy
stream. In exact arithmetic the two are identical; this script verifies
that token by token against a target-only greedy baseline and reports any
divergence honestly (bf16 rounding in a batched verify pass can flip a
near-tied argmax).

The baseline and the speculative run use the same target weights, same
dtype, same prompts and the same token budget.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFT_PATH = "/home/hatch/workspace/goals/ai-inference-engineering-buildout/projects/servecore/models/Qwen2.5-0.5B-Instruct"
TARGET_PATH = os.path.join(BASE, "models", "Qwen2.5-1.5B-Instruct")
OUT = os.path.join(BASE, "results", "speculative_results.json")

PROMPTS = [
    "The history of the Roman Empire began when",
    "To make a good omelette, you first need to",
    "The mitochondria is known as the powerhouse of the cell because",
    "In a shocking turn of events, scientists announced yesterday that",
]
MAX_NEW = 96
EOS_ID = 151645  # <|im_end|>, generation_config eos for both models


@torch.no_grad()
def prefill(model, prompt_ids):
    cache = DynamicCache()
    out = model(input_ids=torch.tensor([prompt_ids]), past_key_values=cache,
                use_cache=True)
    return cache, out.logits[0, -1]


@torch.no_grad()
def step(model, cache, token):
    out = model(input_ids=torch.tensor([[token]]), past_key_values=cache,
                use_cache=True)
    return out.logits[0, -1]


@torch.no_grad()
def target_only(target, prompt_ids, max_new):
    cache, logits = prefill(target, prompt_ids)
    gen = []
    t0 = time.perf_counter()
    while len(gen) < max_new:
        nxt = int(logits.argmax())
        gen.append(nxt)
        if nxt == EOS_ID or len(gen) >= max_new:
            break
        logits = step(target, cache, nxt)
    return gen, time.perf_counter() - t0


@torch.no_grad()
def speculative(draft, target, prompt_ids, max_new, k):
    dcache, d_logits = prefill(draft, prompt_ids)
    tcache, t_logits = prefill(target, prompt_ids)
    gen = []
    proposed = 0
    accepted = 0
    rounds = 0
    t0 = time.perf_counter()
    while len(gen) < max_new:
        rounds += 1
        d_len_before = dcache.get_seq_length()
        t_len_before = tcache.get_seq_length()

        # 1) draft proposes k tokens greedily
        props = []
        dl = d_logits
        for i in range(k):
            x = int(dl.argmax())
            props.append(x)
            if i < k - 1:
                dl = step(draft, dcache, x)
        proposed += k

        # 2) target verifies all k in one forward pass
        out = target(input_ids=torch.tensor([props]), past_key_values=tcache,
                     use_cache=True)
        preds = [int(t_logits.argmax())] + \
                [int(out.logits[0, i].argmax()) for i in range(k)]

        # 3) accept the longest matching prefix
        a = 0
        while a < k and props[a] == preds[a]:
            a += 1
        accepted += a
        z = preds[a]  # corrected token (a<k) or bonus token (a==k)
        emitted = props[:a] + [z]

        # stop at EOS: everything after the first EOS is not part of the
        # target's greedy stream
        stop = False
        if EOS_ID in emitted:
            emitted = emitted[:emitted.index(EOS_ID) + 1]
            stop = True
        gen.extend(emitted)
        if stop or len(gen) >= max_new:
            break

        # 4) resync caches to exactly the emitted prefix
        # target cache currently holds prefix + props; keep prefix + props[:a],
        # then feed z so its KV and next logits are real.
        tcache.crop(t_len_before + a)
        t_logits = step(target, tcache, z)
        # draft cache holds prefix + props[:k-1]; crop to prefix + props[:a]
        # (when a == k the last proposal was never fed: feed it first).
        if a == k:
            dl = step(draft, dcache, props[k - 1])
            dcache.crop(d_len_before + k)
        else:
            dcache.crop(d_len_before + a)
        d_logits = step(draft, dcache, z)

    dt = time.perf_counter() - t0
    stats = {"rounds": rounds, "draft_proposed": proposed,
             "draft_accepted": accepted,
             "acceptance_rate": round(accepted / max(1, proposed), 4)}
    return gen, dt, stats


def main():
    tok = AutoTokenizer.from_pretrained(TARGET_PATH)
    print("loading draft (bf16)...", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_PATH, dtype=torch.bfloat16)
    draft.eval()
    print("loading target (bf16)...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET_PATH, dtype=torch.bfloat16)
    target.eval()
    import resource
    print("RSS GB after loads:", round(resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), flush=True)

    results = {"task": "speculative_decoding_greedy",
               "draft": "Qwen2.5-0.5B-Instruct", "target": "Qwen2.5-1.5B-Instruct",
               "dtype": "bfloat16", "max_new": MAX_NEW, "prompts": []}

    for prompt in PROMPTS:
        ids = tok(prompt, add_special_tokens=False).input_ids
        entry = {"prompt": prompt, "prompt_tokens": len(ids)}

        base_ids, base_t = target_only(target, ids, MAX_NEW)
        entry["baseline"] = {"tokens": len(base_ids), "time_s": round(base_t, 3),
                             "tok_per_s": round(len(base_ids) / base_t, 2)}

        for k in (4, 2, 6):
            spec_ids, spec_t, st = speculative(draft, target, ids, MAX_NEW, k)
            trimmed = spec_ids[:len(base_ids)]
            first_diff = next((i for i, (x, y) in enumerate(
                zip(trimmed, base_ids)) if x != y), None)
            identical = trimmed == base_ids
            entry[f"spec_k{k}"] = {
                "tokens": len(spec_ids), "time_s": round(spec_t, 3),
                "tok_per_s": round(len(spec_ids) / spec_t, 2),
                "speedup_vs_baseline": round(
                    (len(spec_ids) / spec_t) / (len(base_ids) / base_t), 3),
                "identical_to_baseline": identical,
                "first_divergence_index": first_diff,
                **st}
            print(f"k={k} prompt={prompt[:30]!r} spec {len(spec_ids)} tok "
                  f"{entry[f'spec_k{k}']['tok_per_s']} tok/s vs base "
                  f"{entry['baseline']['tok_per_s']} tok/s, "
                  f"identical={identical}, accept={st['acceptance_rate']}",
                  flush=True)
        entry["baseline_text"] = tok.decode(base_ids)
        results["prompts"].append(entry)
        json.dump(results, open(OUT, "w"), indent=2)
    print("SPEC DONE", flush=True)


if __name__ == "__main__":
    main()
