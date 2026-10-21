"""Streaming perplexity eval: how much does an evicting cache hurt the
model's next-token predictions on real long prose?

A 3072-token slice of Alice's Adventures in Wonderland (Project Gutenberg
#11) is streamed through the evicting decoder in 256-token chunks. Every
token's log-probability is scored from the logits produced under the
policy's cache, exactly the conditions a serving system would face.
"""

import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from cache_engine import EvictingDecoder  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/hatch/workspace/goals/ai-inference-engineering-buildout/projects/servecore/models/Qwen2.5-0.5B-Instruct"
OUT = os.path.join(BASE, "results", "perplexity_results.json")
N_TOKENS = 3072
CHUNK = 256
BUDGETS = [1024, 512]
POLICIES = ["sliding", "streaming", "h2o"]


def load_doc_ids(tok):
    text = open(os.path.join(BASE, "data_alice.txt"), encoding="utf-8").read()
    start = text.find("CHAPTER I.")
    text = text[start:]
    ids = tok(text, add_special_tokens=False).input_ids
    return ids[:N_TOKENS]


@torch.no_grad()
def streaming_ppl(model, ids, policy, budget):
    dec = EvictingDecoder(model, policy=policy, budget=budget,
                          sinks=4, recent_frac=0.25)
    total_nll = 0.0
    count = 0
    prev_logits = None
    t0 = time.perf_counter()
    for i in range(0, len(ids), CHUNK):
        part = torch.tensor(ids[i:i + CHUNK], dtype=torch.long)
        logits = dec.process(part)  # [L, vocab]
        # logits that predict this chunk's tokens: previous chunk's last
        # logits, then this chunk's logits shifted by one.
        preds = logits[:-1] if prev_logits is None else torch.cat(
            [prev_logits, logits[:-1]], dim=0)
        targets = part if prev_logits is None else part
        if prev_logits is None:
            # first chunk: token 0 has no prediction; score tokens 1..
            lp = F.log_softmax(preds.float(), dim=-1)
            total_nll += -float(lp.gather(-1, targets[1:].unsqueeze(-1)).sum())
            count += len(targets) - 1
        else:
            lp = F.log_softmax(preds.float(), dim=-1)
            total_nll += -float(lp.gather(-1, targets.unsqueeze(-1)).sum())
            count += len(targets)
        prev_logits = logits[-1:]
    dt = time.perf_counter() - t0
    return {"ppl": round(math.exp(total_nll / count), 3),
            "tokens_scored": count, "time_s": round(dt, 1),
            "s_per_token": round(dt / len(ids), 4),
            "kept_tokens_max": dec.max_kept,
            "cache_mb_final": round(dec.cache_bytes() / 1e6, 2)}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    ids = load_doc_ids(tok)
    print(f"doc tokens: {len(ids)}", flush=True)

    results = {"task": "streaming_perplexity", "model": "Qwen2.5-0.5B-Instruct",
               "dtype": "float32", "doc": "Alice's Adventures in Wonderland (Gutenberg #11)",
               "n_tokens": len(ids), "runs": []}
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    done = {(r["policy"], r["budget"]) for r in results["runs"]}

    configs = [("full", None)] + [(p, b) for p in POLICIES for b in BUDGETS]
    for policy, budget in configs:
        if (policy, budget) in done:
            continue
        r = streaming_ppl(model, ids, policy, budget)
        r = {"policy": policy, "budget": budget, **r}
        results["runs"].append(r)
        json.dump(results, open(OUT, "w"), indent=2)
        print(f"{policy} budget={budget}: ppl={r['ppl']} time={r['time_s']}s "
              f"cache={r['cache_mb_final']}MB", flush=True)
    print("PPL DONE", flush=True)


if __name__ == "__main__":
    main()
