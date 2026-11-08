"""Layer-restricted and channel-restricted residual narrowing.  Usage: ablate_layers.py START END
Tests whether the loss of the per-tensor INT16 residual grid comes from the sink-token layers (12-28, where channel 507
holds a value near 20,500) or from the rest of the network.  Flagged sites are wide in every variant, so only the
residual grid changes.  The grid is applied at block input and output (resid_in, resid_out) of the listed layers only.
Variants:
  control_wide        residual wide everywhere (reproduces the earlier control)
  resid_all           per-tensor INT16 grid at all 30 layers (reproduces the earlier residual-only run)
  resid_sink_layers   per-tensor grid at layers 12-28 only
  resid_other_layers  per-tensor grid at layers 0-11 and 29 only
  resid_sink_split    all 30 layers; channel 507 gets its own exponent and the other 575 channels share one set from
                      the calibration maximum of the remaining channels (two groups, same one-spare-bit rule)"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from intref import Model, dequant, rs, sat, F_RES

a, b = int(sys.argv[1]), int(sys.argv[2])
setup_determinism(); torch.set_num_threads(1)
cfg = load_config(); W = load_weights()
wins, _ = load_token_windows()
cal = np.load(os.path.join(OUT_DIR, "profile_cal.npz"), allow_pickle=True)
SINK = range(12, 29); CH = 507


class LModel(Model):
    layers_on = None      # set of layers whose grid is applied; None = all
    split = False

    def _grid(self, x, site, l):
        if self.layers_on is not None and l not in self.layers_on:
            return x
        if not self.split:
            return super()._grid(x, site, l)
        cm = np.asarray(cal[f"{site}|{l}|ch_max"], dtype=np.float64).copy()
        k_sink = int(np.floor(np.log2(max(cm[CH], 2.0 ** -40)))) + 2
        cm_rest = cm.copy(); cm_rest[CH] = 0.0
        k_rest = int(np.floor(np.log2(max(cm_rest.max(), 2.0 ** -40)))) + 2
        s = np.full(x.shape[-1], max(F_RES - (15 - k_rest), 0), dtype=np.int64)
        s[CH] = max(F_RES - (15 - k_sink), 0)
        s = s[None, :]
        half = np.where(s > 0, np.int64(1) << np.maximum(s - 1, 0), 0)
        y = np.where(s > 0, (x + half) >> s, x)
        return sat(y, 16) << s


M = LModel(wide=True)
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

OTHER = set(range(30)) - set(SINK)
VARIANTS = [("control_wide", False, None, False), ("resid_all", True, None, False),
            ("resid_sink_layers", True, set(SINK), False), ("resid_other_layers", True, OTHER, False),
            ("resid_sink_split", True, None, True)]

def cmp(base, other):
    lb = torch.log_softmax(base.double(), -1); lo = torch.log_softmax(other.double(), -1)
    return {"kl": (lb.exp() * (lb - lo)).sum(-1).mean().item(), "top1": (base.argmax(-1) == other.argmax(-1)).float().mean().item()}

rows = {v[0]: [] for v in VARIANTS}; ref = []
for i in range(a, b):
    ids_np = wins[i]; ids = torch.from_numpy(ids_np)
    with torch.no_grad():
        base = forward(W, cfg, ids); wonly = forward(Wd, cfg, ids)
    nb, c = nll_sum(base, ids); nw, _ = nll_sum(wonly, ids)
    ref.append({"window": i, "tokens": c, "nll_base": nb, "nll_weightonly": nw})
    for name, nres, lon, split in VARIANTS:
        M.layers_on = lon; M.split = split
        t1 = time.time()
        lg = torch.from_numpy(dequant(M.forward(ids_np, narrow_residual=nres), 16)).float()
        ni, _ = nll_sum(lg, ids)
        r = {"window": i, "tokens": c, "nll_int": ni, "vs_weightonly": cmp(wonly, lg)}
        rows[name].append(r)
        print(f"w{i} {name:20s} nll {ni/c:.4f} (wonly {nw/c:.4f}) KL {r['vs_weightonly']['kl']:.2e} top1 {r['vs_weightonly']['top1']:.4f} ({time.time()-t1:.0f}s)", flush=True)

tok = sum(r["tokens"] for r in ref)
summ = {"windows": [a, b - 1], "tokens": tok, "nll_int8_weight_float_reference": sum(r["nll_weightonly"] for r in ref) / tok, "variants": {}}
for name, *_ in VARIANTS:
    R = rows[name]
    summ["variants"][name] = {"nll_per_token": sum(r["nll_int"] for r in R) / tok,
                              "kl_vs_weightonly": float(np.mean([r["vs_weightonly"]["kl"] for r in R])),
                              "top1_vs_weightonly": float(np.mean([r["vs_weightonly"]["top1"] for r in R]))}
json.dump({"summary": summ, "reference_rows": ref, "rows": rows}, open(os.path.join(OUT_DIR, "ablation_layers.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
