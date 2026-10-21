"""Charts for CacheLab, drawn only from the measured result JSONs."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "charts")
POLICIES = ["sliding", "streaming", "h2o"]
COLORS = {"full": "#444444", "sliding": "#1f77b4", "streaming": "#2ca02c",
          "h2o": "#d62728"}


def needle_charts():
    data = json.load(open(os.path.join(RES, "needle_results.json")))
    runs = data["runs"]

    # accuracy per config (over depths)
    configs = []
    for r in runs:
        key = (r["policy"], r["budget"])
        if key not in configs:
            configs.append(key)
    acc = {}
    for key in configs:
        rs = [r for r in runs if (r["policy"], r["budget"]) == key]
        acc[key] = sum(r["found"] for r in rs) / len(rs)

    labels = ["full cache" if b is None else f"{p}\n{b}" for p, b in configs]
    vals = [acc[k] for k in configs]
    cols = [COLORS[p] for p, _ in configs]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, vals, color=cols)
    ax.set_ylabel("needle retrieval accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Needle retrieval at 2048-token context, by cache policy and budget")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "needle_accuracy.png"), dpi=130)
    plt.close(fig)

    # decode latency per config (mean over depths)
    lat = {}
    for key in configs:
        rs = [r for r in runs if (r["policy"], r["budget"]) == key]
        lat[key] = sum(r["decode_s_per_token"] for r in rs) / len(rs)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, [lat[k] for k in configs], color=cols)
    ax.set_ylabel("decode seconds per token")
    ax.set_title("Decode latency after a 2048-token prefill (lower is faster)")
    for i, k in enumerate(configs):
        ax.text(i, lat[k] + 0.012, f"{lat[k]:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "decode_latency.png"), dpi=130)
    plt.close(fig)


def ppl_chart():
    data = json.load(open(os.path.join(RES, "perplexity_results.json")))
    runs = {(r["policy"], r["budget"]): r for r in data["runs"]}
    full = runs[("full", None)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.scatter([full["cache_mb_final"]], [full["ppl"]], s=90,
               color=COLORS["full"], label="full cache", zorder=3)
    for p in POLICIES:
        xs, ys = [], []
        for b in (512, 1024):
            r = runs[(p, b)]
            xs.append(r["cache_mb_final"])
            ys.append(r["ppl"])
        ax.plot(xs, ys, "o-", color=COLORS[p], label=p)
    ax.set_xlabel("KV cache size (MB, final)")
    ax.set_ylabel("perplexity (Alice in Wonderland, 3072 tokens)")
    ax.set_title("Perplexity cost of eviction: quality vs cache memory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ppl_vs_cache.png"), dpi=130)
    plt.close(fig)


def spec_chart():
    data = json.load(open(os.path.join(RES, "speculative_results.json")))
    prompts = data["prompts"]
    base = sum(p["baseline"]["tok_per_s"] for p in prompts) / len(prompts)
    ks = [2, 4, 6]
    rates = []
    accs = []
    for k in ks:
        rates.append(sum(p[f"spec_k{k}"]["tok_per_s"] for p in prompts) / len(prompts))
        accs.append(sum(p[f"spec_k{k}"]["acceptance_rate"] for p in prompts) / len(prompts))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(["target only"] + [f"speculative k={k}" for k in ks],
                  [base] + rates, color=["#444444", "#1f77b4", "#2ca02c", "#d62728"])
    ax.set_ylabel("tokens per second (mean over 4 prompts)")
    ax.set_title("Speculative decoding speed, Qwen2.5-1.5B target + 0.5B draft, CPU bf16")
    for bar, v in zip(bars, [base] + rates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.1f}",
                ha="center")
    for i, a in enumerate(accs):
        ax.text(i + 1, 0.4, f"accept {a:.0%}", ha="center", color="white",
                fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speculative_speed.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    needle_charts()
    ppl_chart()
    spec_chart()
    print("charts done")
