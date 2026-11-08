"""Full-model comparison.  Usage: run_model.py MODE START END   (MODE = wide | narrow)
Reports, per window set: NLL of float32 baseline, float with dequantized INT8 weights (weight-only effect),
and the integer engine; plus KL and top-1 agreement of each against the float32 baseline and of the integer
engine against the weight-only float reference (execution arithmetic effect)."""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from intref import Model, dequant

mode, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
setup_determinism()
torch.set_num_threads(1)
cfg = load_config(); W = load_weights()
wins, _ = load_token_windows()
M = Model(wide=(mode == "wide"))

# float model carrying exactly the dequantized INT8 weights and INT16 gains
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

def cmp(base, other):
    lb = torch.log_softmax(base.double(), -1); lo = torch.log_softmax(other.double(), -1)
    return {"kl": (lb.exp() * (lb - lo)).sum(-1).mean().item(), "top1": (base.argmax(-1) == other.argmax(-1)).float().mean().item()}

rows = []
t0 = time.time()
for i in range(a, b):
    ids_np = wins[i]; ids = torch.from_numpy(ids_np)
    with torch.no_grad():
        base = forward(W, cfg, ids)
        wonly = forward(Wd, cfg, ids)
    t1 = time.time()
    lg_q = M.forward(ids_np, narrow_residual=(mode == "narrow"))
    lg = torch.from_numpy(dequant(lg_q, 16)).float()
    nb, c = nll_sum(base, ids); nw, _ = nll_sum(wonly, ids); ni, _ = nll_sum(lg, ids)
    r = {"window": i, "tokens": c, "nll_base": nb, "nll_weightonly": nw, "nll_int": ni,
         "int_vs_base": cmp(base, lg), "weightonly_vs_base": cmp(base, wonly), "int_vs_weightonly": cmp(wonly, lg),
         "int_seconds": time.time() - t1}
    rows.append(r)
    print(f"{mode} window {i}: nll base {nb/c:.4f} wonly {nw/c:.4f} int {ni/c:.4f} | KL int/base {r['int_vs_base']['kl']:.2e} wonly/base {r['weightonly_vs_base']['kl']:.2e} int/wonly {r['int_vs_weightonly']['kl']:.2e} | top1 int/wonly {r['int_vs_weightonly']['top1']:.4f} ({r['int_seconds']:.0f}s)", flush=True)
tot = {k: sum(r[k] for r in rows) for k in ("tokens", "nll_base", "nll_weightonly", "nll_int")}
summ = {"mode": mode, "windows": [a, b - 1], "tokens": tot["tokens"],
        "nll_per_token": {k: tot[k] / tot["tokens"] for k in ("nll_base", "nll_weightonly", "nll_int")}}
for key in ("int_vs_base", "weightonly_vs_base", "int_vs_weightonly"):
    summ[key] = {m: float(np.mean([r[key][m] for r in rows])) for m in ("kl", "top1")}
json.dump({"summary": summ, "rows": rows}, open(os.path.join(OUT_DIR, f"model_run_{mode}.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
