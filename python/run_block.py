"""Single-block check: quantize the float model's residual input at layer L, run the integer
block and the float block on the same input, compare every checkpoint in units of its step."""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import intref
from intref import Model, dequant

class Stop(Exception): pass
class Grab:
    def __init__(self, layer): self.layer = layer; self.x = None
    def chan(self, site, l, t):
        if site == "resid_in" and l == self.layer:
            self.x = t.clone(); raise Stop
    def flat(self, *a, **k): pass
    def heads(self, *a, **k): pass

def rope64(T, hd, theta):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    fr = torch.outer(torch.arange(T, dtype=torch.float64), inv)
    emb = torch.cat([fr, fr], -1)
    return emb.cos(), emb.sin()

def deq_layer(M, l):
    Ly = M.layers[l]; L = {}
    for name in ("q", "k", "v", "o", "gate", "up", "down"):
        q, m, n = Ly[name]
        L[name] = torch.from_numpy(q * (m * 2.0 ** (-n.astype(np.float64)))[:, None])
    for name in ("ln1", "ln2"):
        wq, f = Ly[name]
        L[name] = torch.from_numpy(wq.astype(np.float64) * 2.0 ** (-f))
    return L

def float_block(L, cfg, l, x):
    T, D = x.shape; nh, nkv, hd = 9, 3, 64
    eps = cfg["rms_norm_eps"]; out = {"resid_in": x}
    cos, sin = rope64(T, hd, cfg["rope_theta"])
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    h = rms(x, L["ln1"]); out["norm1_out"] = h
    q = h @ L["q"].T; k = h @ L["k"].T; v = h @ L["v"].T
    out.update(q_proj=q, k_proj=k, v_proj=v)
    q = q.view(T, nh, hd).transpose(0, 1); k = k.view(T, nkv, hd).transpose(0, 1); v = v.view(T, nkv, hd).transpose(0, 1)
    q = q * cos[None] + rotate_half(q) * sin[None]; k = k * cos[None] + rotate_half(k) * sin[None]
    out["q_rope"] = q.transpose(0, 1).reshape(T, D); out["k_rope"] = k.transpose(0, 1).reshape(T, nkv * hd)
    ke = k.repeat_interleave(3, 0); ve = v.repeat_interleave(3, 0)
    s = (q @ ke.transpose(1, 2)) / 8.0
    out["attn_score"] = torch.where(causal, s, torch.zeros(()))
    p = torch.softmax(s.masked_fill(~causal, float("-inf")), -1)
    out["attn_prob"] = p
    ctx = (p @ ve).transpose(0, 1).reshape(T, D); out["attn_ctx"] = ctx
    o = ctx @ L["o"].T; out["o_res"] = o
    x = x + o; out["resid_mid"] = x
    h2 = rms(x, L["ln2"]); out["norm2_out"] = h2
    g = h2 @ L["gate"].T; u = h2 @ L["up"].T
    out.update(gate_proj=g, up_proj=u)
    sg = torch.nn.functional.silu(g); out["silu_gate"] = sg
    gu = sg * u; out["mlp_gated"] = gu
    d = gu @ L["down"].T; out["down_res"] = d
    out["resid_out"] = x + d
    return out

def main():
    layers = [int(a) for a in sys.argv[1].split(",")]
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    wide = (sys.argv[4] != "narrow") if len(sys.argv) > 4 else True
    setup_determinism()
    cfg = load_config(); W = load_weights()
    wins, _ = load_token_windows()
    ids = torch.from_numpy(wins[win][:T])
    t0 = time.time()
    M = Model(wide=wide)
    print("model ready", round(time.time() - t0, 1), "s", flush=True)
    report = {}
    for l in layers:
        g = Grab(l)
        try:
            with torch.no_grad(): forward(W, cfg, ids, rec=g)
        except Stop: pass
        x_float = g.x.double().numpy()
        x_int = np.clip(np.rint(x_float * 2 ** 15), -2 ** 31, 2 ** 31 - 1).astype(np.int64)
        tr = {}
        t1 = time.time()
        M.block(l, x_int, tr)
        xin_f = torch.from_numpy(dequant(x_int, 15))
        with torch.no_grad(): ref = float_block(deq_layer(M, l), cfg, l, xin_f)
        res = {}
        for name, f in tr["_f"].items():
            q = tr[name]
            if name == "attn_prob":
                a = dequant(q, 16)
            else:
                a = dequant(q, f)
            b = ref[name].double().numpy()
            if name in ("attn_score",):
                mask = np.tril(np.ones((T, T), bool))[None]
                a = np.where(mask, a, 0.0); b = np.where(mask, b, 0.0)
            err = np.abs(a - b)
            step = 2.0 ** (-f)
            res[name] = {"f": f, "max_err": float(err.max()), "max_err_steps": float(err.max() / step),
                         "rms_err": float(np.sqrt((err ** 2).mean())), "ref_rms": float(np.sqrt((b ** 2).mean())),
                         "rel_rms_err": float(np.sqrt((err ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-30)),
                         "saturated": int((np.abs(q) >= (2 ** 31 - 1 if abs(q).max() > 40000 else 32767)).sum())}
        report[l] = res
        print(f"layer {l} ({time.time()-t1:.1f}s)")
        for name, r in res.items():
            print(f"  {name:11s} f={r['f']:3d} max {r['max_err']:.3e} ({r['max_err_steps']:9.2f} steps)  rel rms {r['rel_rms_err']:.2e}")
    json.dump(report, open(os.path.join(OUT_DIR, f"block_check_{'wide' if wide else 'narrow'}.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
