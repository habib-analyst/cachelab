"""Needle-in-a-haystack eval for KV cache eviction policies.

For each policy/budget/depth: build a ~2048-token haystack with the needle
sentence at the given depth, prefill with the evicting decoder, greedily
generate up to 12 tokens, and check whether the passcode comes back.
Results are appended to results/needle_results.json after every run.
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from cache_engine import EvictingDecoder  # noqa: E402
from haystack import build_prompt  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/hatch/workspace/goals/ai-inference-engineering-buildout/projects/servecore/models/Qwen2.5-0.5B-Instruct"
OUT = os.path.join(BASE, "results", "needle_results.json")

CONTEXT_TOKENS = 2048
DEPTHS = [0.3, 0.7]
BUDGETS = [1024, 512]
POLICIES = ["sliding", "streaming", "h2o"]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()

    results = {"task": "needle_in_haystack", "model": "Qwen2.5-0.5B-Instruct",
               "dtype": "float32", "context_tokens": CONTEXT_TOKENS, "runs": []}
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    done = {(r["policy"], r["budget"], r["depth"]) for r in results["runs"]}

    configs = [("full", None)] + [(p, b) for p in POLICIES for b in BUDGETS]
    for policy, budget in configs:
        for depth in DEPTHS:
            if (policy, budget, depth) in done:
                continue
            _, ids = build_prompt(tok, CONTEXT_TOKENS, depth)
            dec = EvictingDecoder(model, policy=policy, budget=budget,
                                  sinks=4, recent_frac=0.25)
            out_ids, stats = dec.generate(ids, max_new_tokens=12, chunk=256)
            text = tok.decode(out_ids)
            run = {"policy": policy, "budget": budget, "depth": depth,
                   "prompt_tokens": len(ids), "found": "4729" in text,
                   "generated": text, **stats}
            run["cache_mb_final"] = round(stats["cache_bytes_final"] / 1e6, 2)
            results["runs"].append(run)
            json.dump(results, open(OUT, "w"), indent=2)
            print(f"{policy} budget={budget} depth={depth}: found={run['found']} "
                  f"prefill={stats['prefill_s']}s decode/tok={stats['decode_s_per_token']}s "
                  f"kept={stats['kept_tokens_final']} text={text!r}", flush=True)
    print("NEEDLE DONE", flush=True)


if __name__ == "__main__":
    main()
