"""Activation profile of the float32 reference forward pass.

Usage: profile_acts.py TAG START END
Windows START..END-1 (512 tokens each) are run one at a time (batch size one).

Per (site, layer) the profile keeps
  hist      exact counts of |x| in bins that are 32 linear sub-bins per octave
            (octave edges at exact powers of two, so exceedance counts at 2^k
            are exact); zero values are counted separately
  tokhist   same histogram for the per-token abs-max over channels (chan sites)
  ch_max    per-channel abs-max (chan and heads sites)
  ch_sumsq  per-channel sum of squares in float64
  scalars   n, zeros, signed min/max, underflow/overflow bin clamps
Masked attention entries are never passed to the recorder.
"""
import hashlib, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

LO, HI, SUB = -30, 20, 32
NB = (HI - LO) * SUB


def bin_counts(a):
    """a: 1-D float32 tensor of non-negative values. Returns (zeros, hist[NB], under, over)."""
    zeros = int((a == 0).sum())
    nz = a[a > 0]
    if nz.numel() == 0:
        return zeros, torch.zeros(NB, dtype=torch.int64), 0, 0
    m, e = torch.frexp(nz)                       # nz = m * 2^e, m in [0.5, 1)
    idx = (e.to(torch.int64) - 1 - LO) * SUB + ((m * 2 - 1) * SUB).floor().to(torch.int64)
    under = int((idx < 0).sum())
    over = int((idx >= NB).sum())
    idx = idx.clamp(0, NB - 1)
    return zeros, torch.bincount(idx, minlength=NB), under, over


class Prof:
    def __init__(self):
        self.S = {}

    def _slot(self, site, l):
        key = f"{site}|{l}"
        if key not in self.S:
            self.S[key] = {"hist": np.zeros(NB, np.int64), "n": 0, "zeros": 0, "under": 0, "over": 0,
                           "smin": np.inf, "smax": -np.inf}
        return self.S[key]

    def _add_hist(self, s, a, field="hist"):
        z, h, u, o = bin_counts(a)
        s[field] += h.numpy()
        if field == "hist":
            s["zeros"] += z; s["under"] += u; s["over"] += o
        else:
            s["tok_zeros"] = s.get("tok_zeros", 0) + z

    def _scalars(self, s, t):
        assert torch.isfinite(t).all(), "non-finite value in recorded tensor"
        s["n"] += t.numel()
        s["smin"] = min(s["smin"], float(t.min()))
        s["smax"] = max(s["smax"], float(t.max()))

    def flat(self, site, l, t):
        s = self._slot(site, l)
        t = t.reshape(-1)
        self._scalars(s, t)
        self._add_hist(s, t.abs())

    def chan(self, site, l, t):
        # t: [T, C]
        s = self._slot(site, l)
        self._scalars(s, t)
        a = t.abs()
        self._add_hist(s, a.reshape(-1))
        if "tokhist" not in s:
            s["tokhist"] = np.zeros(NB, np.int64)
        self._add_hist(s, a.amax(1), "tokhist")
        cm = a.amax(0).numpy()
        ss = t.double().pow(2).sum(0).numpy()
        s["ch_max"] = cm if "ch_max" not in s else np.maximum(s["ch_max"], cm)
        s["ch_sumsq"] = ss if "ch_sumsq" not in s else s["ch_sumsq"] + ss
        s["ntok"] = s.get("ntok", 0) + t.shape[0]

    def heads(self, site, l, t):
        # t: [nh, Nv] valid entries only; channel = head
        s = self._slot(site, l)
        self._scalars(s, t)
        a = t.abs()
        self._add_hist(s, a.reshape(-1))
        cm = a.amax(1).numpy()
        ss = t.double().pow(2).sum(1).numpy()
        s["ch_max"] = cm if "ch_max" not in s else np.maximum(s["ch_max"], cm)
        s["ch_sumsq"] = ss if "ch_sumsq" not in s else s["ch_sumsq"] + ss


def flatten_profile(P, nll, ntok_nll, ids_sha):
    out = {}
    for key, s in P.S.items():
        for f, v in s.items():
            out[f"{key}|{f}"] = np.asarray(v)
    out["nll_per_window"] = np.asarray(nll, dtype=np.float64)
    out["nll_tokens_per_window"] = np.asarray(ntok_nll, dtype=np.int64)
    h = hashlib.sha256()
    for k in sorted(out):
        h.update(k.encode()); h.update(np.ascontiguousarray(out[k]).tobytes())
    h.update(ids_sha.encode())
    return out, h.hexdigest()


def main():
    tag, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    setup_determinism()
    cfg = load_config()
    W = load_weights()
    wins, info = load_token_windows()
    wins = wins[a:b]
    ids_sha = hashlib.sha256(wins.tobytes()).hexdigest()
    P = Prof()
    nll, nt = [], []
    t0 = time.time()
    for i in range(wins.shape[0]):
        ids = torch.from_numpy(wins[i])
        with torch.no_grad():
            logits = forward(W, cfg, ids, rec=P)
        n, c = nll_sum(logits, ids)
        nll.append(n); nt.append(c)
        if i % 8 == 0:
            print(f"{tag} window {a+i} elapsed {time.time()-t0:.0f}s mean nll {sum(nll)/sum(nt):.4f}", flush=True)
    out, digest = flatten_profile(P, nll, nt, ids_sha)
    np.savez(os.path.join(OUT_DIR, f"profile_{tag}.npz"), **out)
    meta = {"tag": tag, "windows": [a, b - 1], "ids_sha256": ids_sha, "profile_digest": digest,
            "float_nll_per_token": sum(nll) / sum(nt), "float_perplexity": float(np.exp(sum(nll) / sum(nt))),
            "tokens_scored": int(sum(nt)), "sites": len(P.S), "seconds": time.time() - t0}
    json.dump(meta, open(os.path.join(OUT_DIR, f"profile_{tag}.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
