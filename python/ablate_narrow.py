"""Three-way ablation of the narrow comparison mode.  Usage: ablate_narrow.py START END
All variants use the same windows, the same INT8 weights and gains, and the same arithmetic except for the storage
width of the named sites.  Every variant is compared with the same float reference that carries the dequantized
INT8 weights (KL and top-1), and negative log-likelihood is reported for all of them.

Variants (wide = flagged site-layers stored as INT32; narrow = INT16 per tensor):
  control_wide          flagged wide, residual wide (reproduces the earlier wide run)
  residual_only         flagged wide, residual forced to its INT16 grid at every block input and output
  flagged_only          flagged sites narrow, residual wide
  joint                 flagged sites narrow, residual narrow (reproduces the earlier narrow run)
  flagged_gated_only    only the mlp_gated flagged sites narrow, residual wide
  flagged_ctx_silu_only only the attn_ctx and silu_gate flagged sites narrow, residual wide
Of the 118 flagged site-layers, 81 are residual sites (resid_in, resid_mid, resid_out) and 7 are down_proj sites; in
this engine neither group has a stored activation format, so the sites that actually change width are the 26
mlp_gated, 2 attn_ctx and 2 silu_gate entries."""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from intref import Model, dequant

a, b = int(sys.argv[1]), int(sys.argv[2])
setup_determinism()
torch.set_num_threads(1)
cfg = load_config(); W = load_weights()
wins, _ = load_token_windows()


class AblModel(Model):
    narrow_sites = None          # None: every flagged site narrows when wide is False

    def fmt(self, site, l):
        e = self.prop[f"{site}|{l}"]
        flagged = e["class"].startswith("flagged")
        k = e["k"]
        if flagged:
            keep_wide = self.wide or (self.narrow_sites is not None and site not in self.narrow_sites)
            if keep_wide:
                return 32, 31 - k
        return 16, 15 - k


M = AblModel(wide=True)
Wd = {"embed": torch.from_numpy((M.emb_q * (M.emb_m * 2.0 ** (-M.emb_n.astype(np.float64)))[:, None])).float(), "layers": []}
for l, Ly in enumerate(M.layers):
    d = {}
    for name in ("q", "k", "v", "o", "gate", "up", "down"):
        q, m, n = Ly[name]
        d[name] = torch.from_numpy(q * (m * 2.0 ** (-n.astype(np.float64)))[:, None]).float()
    for name in ("ln1", "ln2"):
        wq, f = Ly[name]; d[name] = torch.from_numpy(wq.astype(np.float64) * 2.0 ** (-f)).float()
    Wd["layers"].append(d)
wq, f = M.final_gain; Wd["norm"] = torch.from_numpy(wq.astype(np.float64) * 2.0 ** (-f)).float()

VARIANTS = [
    ("control_wide",          True,  None,                          False),
    ("residual_only",         True,  None,                          True),
    ("flagged_only",          False, None,                          False),
    ("joint",                 False, None,                          True),
    ("flagged_gated_only",    False, {"mlp_gated"},                 False),
    ("flagged_ctx_silu_only", False, {"attn_ctx", "silu_gate"},     False),
]

def cmp(base, other):
    lb = torch.log_softmax(base.double(), -1); lo = torch.log_softmax(other.double(), -1)
    return {"kl": (lb.exp() * (lb - lo)).sum(-1).mean().item(), "top1": (base.argmax(-1) == other.argmax(-1)).float().mean().item()}

rows = {v[0]: [] for v in VARIANTS}
ref_rows = []
for i in range(a, b):
    ids_np = wins[i]; ids = torch.from_numpy(ids_np)
    with torch.no_grad():
        base = forward(W, cfg, ids)
        wonly = forward(Wd, cfg, ids)
    nb, c = nll_sum(base, ids); nw, _ = nll_sum(wonly, ids)
    ref_rows.append({"window": i, "tokens": c, "nll_base": nb, "nll_weightonly": nw, "weightonly_vs_base": cmp(base, wonly)})
    for name, wide, sites, nres in VARIANTS:
        M.wide = wide; M.narrow_sites = sites
        t1 = time.time()
        lg = torch.from_numpy(dequant(M.forward(ids_np, narrow_residual=nres), 16)).float()
        ni, _ = nll_sum(lg, ids)
        r = {"window": i, "tokens": c, "nll_int": ni, "vs_weightonly": cmp(wonly, lg), "vs_base": cmp(base, lg)}
        rows[name].append(r)
        print(f"w{i} {name:22s} nll {ni/c:.4f} (wonly {nw/c:.4f}) KL/wonly {r['vs_weightonly']['kl']:.2e} top1/wonly {r['vs_weightonly']['top1']:.4f} ({time.time()-t1:.0f}s)", flush=True)

tok = sum(r["tokens"] for r in ref_rows)
summ = {"windows": [a, b - 1], "tokens": tok,
        "nll_per_token": {"float32_baseline": sum(r["nll_base"] for r in ref_rows) / tok,
                          "int8_weight_float_reference": sum(r["nll_weightonly"] for r in ref_rows) / tok},
        "weightonly_vs_base": {m: float(np.mean([r["weightonly_vs_base"][m] for r in ref_rows])) for m in ("kl", "top1")},
        "variants": {}}
for name, *_ in VARIANTS:
    R = rows[name]
    kl = [r["vs_weightonly"]["kl"] for r in R]; t1 = [r["vs_weightonly"]["top1"] for r in R]
    summ["variants"][name] = {
        "nll_per_token": sum(r["nll_int"] for r in R) / tok,
        "kl_vs_weightonly": float(np.mean(kl)), "kl_vs_weightonly_window_min": float(min(kl)), "kl_vs_weightonly_window_max": float(max(kl)),
        "top1_vs_weightonly": float(np.mean(t1)),
        "kl_vs_base": float(np.mean([r["vs_base"]["kl"] for r in R])), "top1_vs_base": float(np.mean([r["vs_base"]["top1"] for r in R]))}
json.dump({"summary": summ, "reference_rows": ref_rows, "rows": rows}, open(os.path.join(OUT_DIR, "ablation_narrow.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
