"""
From-scratch decoder with a real, physically-evicted KV cache.

I wrote this instead of using a library cache because the whole point of
CacheLab is to control what stays in the cache. The forward pass walks the
model's own layers (embeddings, norms, projections, MLP) but the attention
itself, the cache storage, and every eviction decision are implemented here.

Cache layout (per layer, per KV head):
    K, V     : [n_kept, head_dim] tensors, rows are individual past tokens
    pos      : [n_kept] original sequence positions of those rows
    score    : [n_kept] accumulated attention mass (only used by H2O)

Kept rows keep their original positions, and RoPE is applied with the true
position of every token, so a surviving token attends exactly as it would in
a full cache. Eviction physically slices the K/V tensors, so memory use and
attention cost really drop; nothing is simulated.

Policies (budget B = max kept tokens per KV head):
    full      : never evict (baseline)
    sliding   : keep the most recent B tokens
    streaming : keep the first S tokens (attention sinks) + most recent B-S
    h2o       : keep the top (B - R) tokens by accumulated attention score
                plus the most recent R tokens (Heavy Hitter Oracle,
                Zhang et al., arXiv:2306.14048). StreamingLLM is Xiao et al.,
                arXiv:2309.17453.

Eviction runs per layer right after each processed chunk, so during chunked
prefill the cache can overshoot the budget by at most one chunk, exactly
like chunked-prefill serving systems. During single-token decoding the
budget is enforced every step.
"""

import math
import time

import torch
import torch.nn.functional as F


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class EvictingDecoder:
    def __init__(self, model, policy="full", budget=None, sinks=4, recent_frac=0.25):
        assert policy in ("full", "sliding", "streaming", "h2o")
        self.model = model
        self.cfg = model.config
        self.policy = policy
        self.budget = budget  # None means unbounded (full)
        self.sinks = sinks
        self.recent_frac = recent_frac
        self.n_layers = self.cfg.num_hidden_layers
        self.n_q = self.cfg.num_attention_heads
        self.n_kv = self.cfg.num_key_value_heads
        self.group = self.n_q // self.n_kv
        self.head_dim = self.cfg.hidden_size // self.cfg.num_attention_heads
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.next_pos = 0
        # per layer: dict of per-kv-head tensors
        self.cache = [
            {"K": [None] * self.n_kv, "V": [None] * self.n_kv,
             "pos": [None] * self.n_kv, "score": [None] * self.n_kv}
            for _ in range(self.n_layers)
        ]
        self.max_kept = 0  # largest cache size (tokens) seen, for memory maths

    def kept_tokens(self):
        c = self.cache[0]
        if c["K"][0] is None:
            return 0
        return int(c["K"][0].shape[0])

    def cache_bytes(self):
        """Exact bytes of the current KV cache (fp32 K and V)."""
        total = 0
        for layer_cache in self.cache:
            for h in range(self.n_kv):
                k = layer_cache["K"][h]
                if k is not None:
                    total += (k.numel() + layer_cache["V"][h].numel()) * k.element_size()
        return total

    # -------------------------------------------------------------- eviction
    def _keep_indices(self, layer_cache, h):
        """Return indices to keep for head h of one layer, per the policy."""
        pos = layer_cache["pos"][h]
        n = pos.shape[0]
        B = self.budget
        if B is None or n <= B:
            return None  # keep everything
        recent = max(1, int(round(B * self.recent_frac)))
        if self.policy == "sliding":
            return torch.arange(n - B, n)
        if self.policy == "streaming":
            s = min(self.sinks, B)
            idx = torch.cat([torch.arange(s), torch.arange(n - (B - s), n)])
            return idx
        if self.policy == "h2o":
            score = layer_cache["score"][h]
            recent_idx = torch.arange(n - recent, n)
            n_heavy = B - recent
            if n_heavy <= 0:
                return recent_idx
            cand_score = score[: n - recent]
            heavy_idx = torch.topk(cand_score, k=min(n_heavy, cand_score.shape[0])).indices
            idx = torch.cat([heavy_idx, recent_idx])
            return idx.sort().values
        return None

    def _evict_layer(self, li):
        layer_cache = self.cache[li]
        for h in range(self.n_kv):
            idx = self._keep_indices(layer_cache, h)
            if idx is None:
                continue
            layer_cache["K"][h] = layer_cache["K"][h][idx].contiguous()
            layer_cache["V"][h] = layer_cache["V"][h][idx].contiguous()
            layer_cache["pos"][h] = layer_cache["pos"][h][idx].contiguous()
            layer_cache["score"][h] = layer_cache["score"][h][idx].contiguous()

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def process(self, input_ids):
        """Run a chunk of tokens through the model, updating the cache.

        input_ids: LongTensor [L] (batch size 1 only).
        Returns logits [L, vocab] for every position in the chunk.
        """
        model = self.model
        backbone = model.model
        L = int(input_ids.shape[0])
        positions = torch.arange(self.next_pos, self.next_pos + L)
        hidden = backbone.embed_tokens(input_ids.unsqueeze(0))[0]  # [L, H]

        for li, layer in enumerate(backbone.layers):
            attn = layer.self_attn
            lc = self.cache[li]

            x = layer.input_layernorm(hidden)
            q = attn.q_proj(x).view(L, self.n_q, self.head_dim)
            k_new = attn.k_proj(x).view(L, self.n_kv, self.head_dim)
            v_new = attn.v_proj(x).view(L, self.n_kv, self.head_dim)

            cos, sin = backbone.rotary_emb(x.unsqueeze(0), positions.unsqueeze(0))
            cos = cos[0]  # [L, head_dim]
            sin = sin[0]
            q = q * cos.unsqueeze(1) + _rotate_half(q) * sin.unsqueeze(1)
            k_new = k_new * cos.unsqueeze(1) + _rotate_half(k_new) * sin.unsqueeze(1)

            # append new K/V to this layer's cache (per KV head)
            for h in range(self.n_kv):
                if lc["K"][h] is None:
                    lc["K"][h] = k_new[:, h, :]
                    lc["V"][h] = v_new[:, h, :]
                    lc["pos"][h] = positions.clone()
                    lc["score"][h] = torch.zeros(L)
                else:
                    lc["K"][h] = torch.cat([lc["K"][h], k_new[:, h, :]], dim=0)
                    lc["V"][h] = torch.cat([lc["V"][h], v_new[:, h, :]], dim=0)
                    lc["pos"][h] = torch.cat([lc["pos"][h], positions], dim=0)
                    lc["score"][h] = torch.cat(
                        [lc["score"][h], torch.zeros(L)], dim=0)

            # attention: each KV head serves a group of Q heads
            attn_out = torch.empty(L, self.n_q, self.head_dim)
            scale = 1.0 / math.sqrt(self.head_dim)
            for h in range(self.n_kv):
                K = lc["K"][h]                       # [n, d]
                V = lc["V"][h]                       # [n, d]
                cpos = lc["pos"][h]                  # [n]
                Qg = q[:, h * self.group:(h + 1) * self.group, :]  # [L, g, d]
                scores = torch.einsum("lgd,nd->lgn", Qg, K) * scale
                mask = cpos.unsqueeze(0).unsqueeze(0) > positions.view(L, 1, 1)
                scores = scores.masked_fill(mask, float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)
                if self.policy == "h2o":
                    # accumulated attention mass per cached token
                    lc["score"][h] += probs.sum(dim=(0, 1))
                attn_out[:, h * self.group:(h + 1) * self.group, :] = torch.einsum(
                    "lgn,nd->lgd", probs, V)

            hidden = hidden + attn.o_proj(attn_out.reshape(L, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

            self._evict_layer(li)
            self.max_kept = max(self.max_kept, self.kept_tokens())

        hidden = backbone.norm(hidden)
        logits = model.lm_head(hidden)
        self.next_pos += L
        return logits

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=16, chunk=256, eos_id=None):
        """Prefill the prompt (chunked, evicting), then greedy-decode.

        Returns (generated_ids, stats dict with timings and cache size).
        """
        self.reset()
        t0 = time.perf_counter()
        logits = None
        for i in range(0, len(prompt_ids), chunk):
            part = torch.tensor(prompt_ids[i:i + chunk], dtype=torch.long)
            logits = self.process(part)
        t_prefill = time.perf_counter() - t0

        out = []
        t0 = time.perf_counter()
        nxt = int(logits[-1].argmax())
        for _ in range(max_new_tokens):
            out.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
            logits = self.process(torch.tensor([nxt], dtype=torch.long))
            nxt = int(logits[-1].argmax())
        t_decode = time.perf_counter() - t0

        stats = {
            "prefill_s": round(t_prefill, 3),
            "decode_s": round(t_decode, 3),
            "decode_tokens": len(out),
            "decode_s_per_token": round(t_decode / max(1, len(out)), 4),
            "kept_tokens_final": self.kept_tokens(),
            "kept_tokens_max": self.max_kept,
            "cache_bytes_final": self.cache_bytes(),
        }
        return out, stats
