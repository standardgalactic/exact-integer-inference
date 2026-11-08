"""Per-channel INT16 residual, evaluated as a storage grid.  Usage: ablate_perchannel.py START END
The residual stream is forced onto an INT16 grid with one power-of-two exponent per channel at every block input and
output (the same sites and the same simulation as the per-tensor narrow mode), with exponent f_c = 15 - k_c and
k_c = floor(log2(calibration absolute maximum of channel c)) + 2, the same spare-bit rule as the per-tensor scheme.
The engine keeps its INT32 residual arithmetic and exact normalization; only the stored value between blocks is
rounded.  Variants: residual_perchannel (flagged sites wide) and joint_perchannel (flagged sites narrow)."""
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


class PCModel(Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.pc_s = {}
        for site in ("resid_in", "resid_out"):
            for l in range(self.L):
                cm = np.asarray(cal[f"{site}|{l}|ch_max"], dtype=np.float64)
                k = np.floor(np.log2(np.maximum(cm, 2.0 ** -40))).astype(np.int64) + 2
                self.pc_s[(site, l)] = np.maximum(F_RES - (15 - k), 0)     # right-shift per channel, >= 0

    def _grid(self, x, site, l):
        s = self.pc_s[(site, l)][None, :]
        half = np.where(s > 0, np.int64(1) << np.maximum(s - 1, 0), 0)
        y = np.where(s > 0, (x + half) >> s, x)
        return sat(y, 16) << s


M = PCModel(wide=True)
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

VARIANTS = [("residual_perchannel", True), ("joint_perchannel", False)]

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
    for name, wide in VARIANTS:
        M.wide = wide
        t1 = time.time()
        lg = torch.from_numpy(dequant(M.forward(ids_np, narrow_residual=True), 16)).float()
        ni, _ = nll_sum(lg, ids)
        r = {"window": i, "tokens": c, "nll_int": ni, "vs_weightonly": cmp(wonly, lg), "vs_base": cmp(base, lg)}
        rows[name].append(r)
        print(f"w{i} {name:20s} nll {ni/c:.4f} (wonly {nw/c:.4f}) KL/wonly {r['vs_weightonly']['kl']:.2e} top1/wonly {r['vs_weightonly']['top1']:.4f} ({time.time()-t1:.0f}s)", flush=True)

tok = sum(r["tokens"] for r in ref)
summ = {"windows": [a, b - 1], "tokens": tok,
        "nll_per_token": {"float32_baseline": sum(r["nll_base"] for r in ref) / tok,
                          "int8_weight_float_reference": sum(r["nll_weightonly"] for r in ref) / tok}, "variants": {}}
for name, _ in VARIANTS:
    R = rows[name]
    kl = [r["vs_weightonly"]["kl"] for r in R]; t1 = [r["vs_weightonly"]["top1"] for r in R]
    summ["variants"][name] = {"nll_per_token": sum(r["nll_int"] for r in R) / tok,
        "kl_vs_weightonly": float(np.mean(kl)), "kl_vs_weightonly_window_min": float(min(kl)), "kl_vs_weightonly_window_max": float(max(kl)),
        "top1_vs_weightonly": float(np.mean(t1)),
        "kl_vs_base": float(np.mean([r["vs_base"]["kl"] for r in R])), "top1_vs_base": float(np.mean([r["vs_base"]["top1"] for r in R]))}
json.dump({"summary": summ, "reference_rows": ref, "rows": rows}, open(os.path.join(OUT_DIR, "ablation_perchannel.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
