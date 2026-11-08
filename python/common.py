"""Shared pieces for the SmolLM2-135M profiling pass.

The float reference forward pass is written out by hand (no HF modules) so
that every intermediate site can be observed, and so that the same dataflow
can later be mirrored by the integer engine.  Weights are stored as bf16 in
the checkpoint; widening bf16 to float32 is exact, so the float32 baseline
uses precisely the checkpointed values.
"""
import hashlib
import json
import math
import os

import numpy as np
import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "results")

REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SEQ_LEN = 512
N_CAL = 64       # calibration windows: 0..63
N_VAL = 64       # validation windows: 64..127


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        return json.load(f)


def setup_determinism():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)


def load_weights():
    cfg = load_config()
    f = safe_open(os.path.join(MODEL_DIR, "model.safetensors"), "pt")
    g = lambda k: f.get_tensor(k).to(torch.float32).contiguous()
    W = {"embed": g("model.embed_tokens.weight"), "norm": g("model.norm.weight"), "layers": []}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        W["layers"].append({
            "ln1": g(p + "input_layernorm.weight"),
            "ln2": g(p + "post_attention_layernorm.weight"),
            "q": g(p + "self_attn.q_proj.weight"),
            "k": g(p + "self_attn.k_proj.weight"),
            "v": g(p + "self_attn.v_proj.weight"),
            "o": g(p + "self_attn.o_proj.weight"),
            "gate": g(p + "mlp.gate_proj.weight"),
            "up": g(p + "mlp.up_proj.weight"),
            "down": g(p + "mlp.down_proj.weight"),
        })
    return W


def load_token_windows():
    """Tokenize the pinned wikitext-2 raw test split and cut non-overlapping
    windows.  Returns (windows int64 [N, SEQ_LEN], info dict)."""
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    rows = pq.read_table(os.path.join(DATA_DIR, "wikitext2_test.parquet")).column("text").to_pylist()
    text = "\n\n".join(rows)
    ids = tok.encode(text, add_special_tokens=False).ids
    n_win = len(ids) // SEQ_LEN
    arr = np.asarray(ids[: n_win * SEQ_LEN], dtype=np.int64).reshape(n_win, SEQ_LEN)
    info = {
        "rows": len(rows),
        "text_chars": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "total_tokens": len(ids),
        "n_windows": int(n_win),
        "dropped_tail_tokens": len(ids) - n_win * SEQ_LEN,
        "window_start_offsets": [int(i * SEQ_LEN) for i in range(n_win)],
        "all_windows_ids_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }
    return arr, info


def rope_tables(T, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(T, dtype=torch.float32)
    fr = torch.outer(pos, inv_freq)
    emb = torch.cat([fr, fr], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class NullRec:
    def chan(self, *a, **k): pass
    def flat(self, *a, **k): pass
    def heads(self, *a, **k): pass


def forward(W, cfg, ids, rec=None, return_logits=True, embed_lookup=None, head_matrix=None):
    """Float32 reference forward for one sequence (batch size one).

    embed_lookup / head_matrix allow the tied matrix to be replaced
    separately for the input lookup and the output head (experiments only).
    """
    rec = rec or NullRec()
    D = cfg["hidden_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = D // nh
    grp = nh // nkv
    eps = cfg["rms_norm_eps"]
    T = ids.shape[0]
    cos, sin = rope_tables(T, hd, cfg["rope_theta"])
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scale = 1.0 / math.sqrt(hd)

    E_in = W["embed"] if embed_lookup is None else embed_lookup
    E_out = W["embed"] if head_matrix is None else head_matrix

    x = E_in[ids]
    rec.chan("embed_out", -1, x)

    def rms(x, w, tag, l):
        ms = x.pow(2).mean(-1)
        rec.flat(tag + "_ms", l, ms)
        inv = torch.rsqrt(ms + eps)
        rec.flat(tag + "_inv_rms", l, inv)
        n = x * inv[:, None]
        rec.chan(tag + "_normed", l, n)
        y = n * w
        rec.chan(tag + "_out", l, y)
        return y

    for l, L in enumerate(W["layers"]):
        rec.chan("resid_in", l, x)
        h = rms(x, L["ln1"], "norm1", l)
        q = h @ L["q"].T
        k = h @ L["k"].T
        v = h @ L["v"].T
        rec.chan("q_proj", l, q)
        rec.chan("k_proj", l, k)
        rec.chan("v_proj", l, v)
        q = q.view(T, nh, hd).transpose(0, 1)       # [nh,T,hd]
        k = k.view(T, nkv, hd).transpose(0, 1)      # [nkv,T,hd]
        v = v.view(T, nkv, hd).transpose(0, 1)
        q = q * cos[None] + rotate_half(q) * sin[None]
        k = k * cos[None] + rotate_half(k) * sin[None]
        rec.chan("q_rope", l, q.transpose(0, 1).reshape(T, D))
        rec.chan("k_rope", l, k.transpose(0, 1).reshape(T, nkv * hd))
        k_e = k.repeat_interleave(grp, dim=0)
        v_e = v.repeat_interleave(grp, dim=0)
        s = (q @ k_e.transpose(1, 2)) * scale        # [nh,T,T], all entries finite pre-mask
        sv = s[:, causal]                            # valid (unmasked) entries only, [nh,Nv]
        rec.heads("attn_score", l, sv)
        s_m = s.masked_fill(~causal, float("-inf"))
        rmax = s_m.amax(-1, keepdim=True)
        shifted = s_m - rmax
        rec.heads("attn_score_shifted", l, shifted[:, causal])
        rec.flat("attn_rowmax", l, rmax.reshape(-1))
        e = torch.exp(shifted)                       # masked -> exp(-inf) = 0
        esum = e.sum(-1, keepdim=True)
        rec.flat("attn_expsum", l, esum.reshape(-1))
        p = e / esum
        rec.heads("attn_prob", l, p[:, causal])
        rec.flat("attn_sink_mass", l, p[:, :, 0].reshape(-1))
        ctx = (p @ v_e).transpose(0, 1).reshape(T, D)
        rec.chan("attn_ctx", l, ctx)
        o = ctx @ L["o"].T
        rec.chan("o_proj", l, o)
        x = x + o
        rec.chan("resid_mid", l, x)
        h2 = rms(x, L["ln2"], "norm2", l)
        g = h2 @ L["gate"].T
        u = h2 @ L["up"].T
        rec.chan("gate_proj", l, g)
        rec.chan("up_proj", l, u)
        sg = torch.nn.functional.silu(g)
        rec.chan("silu_gate", l, sg)
        gu = sg * u
        rec.chan("mlp_gated", l, gu)
        d = gu @ L["down"].T
        rec.chan("down_proj", l, d)
        x = x + d
        rec.chan("resid_out", l, x)

    y = rms(x, W["norm"], "final", len(W["layers"]))
    if not return_logits:
        return y
    logits = y @ E_out.T
    rec.flat("logits", len(W["layers"]), logits.reshape(-1))
    return logits


def nll_sum(logits, ids):
    """Sum of next-token negative log-likelihood (nats) within one window."""
    lp = torch.log_softmax(logits[:-1].double(), dim=-1)
    return -lp.gather(1, ids[1:, None]).sum().item(), ids.shape[0] - 1
