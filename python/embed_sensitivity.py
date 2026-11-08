"""Tied embedding matrix: lookup role and output-head role measured separately.

The tied matrix keeps one stored representation per scheme (one int8 matrix and
its scales); what differs between rows of the report is which role sees the
quantized values.  Quantizing two independent copies would be a different model
and is not done here.

Usage: embed_sensitivity.py START END
"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def quant_int8(E, mode):
    if mode == "per_tensor":
        s = E.abs().max() / 127.0
        q = torch.clamp(torch.round(E / s), -127, 127)
        return q * s, {"scale_count": 1}
    if mode == "per_row":
        s = E.abs().amax(1, keepdim=True) / 127.0
        s = torch.where(s == 0, torch.ones_like(s), s)
        q = torch.clamp(torch.round(E / s), -127, 127)
        return q * s, {"scale_count": E.shape[0]}
    raise ValueError(mode)


def quant_act(y, f):
    step = 2.0 ** (-f)
    q = torch.clamp(torch.round(y / step), -32768, 32767)
    clipped = int(((y / step).round().abs() > 32767).sum())
    return q * step, clipped


def compare(base, other, ids):
    lb = torch.log_softmax(base.double(), -1)
    lo = torch.log_softmax(other.double(), -1)
    kl = (lb.exp() * (lb - lo)).sum(-1).mean().item()
    nb, c = nll_sum(base, ids)
    no, _ = nll_sum(other, ids)
    return {"kl_base_to_q": kl, "d_nll_sum": no - nb, "tokens": c,
            "top1_agree": (base.argmax(-1) == other.argmax(-1)).float().mean().item(),
            "max_abs_logit_err": (base - other).abs().max().item(),
            "mean_abs_logit_err": (base - other).abs().mean().item()}


def main():
    a, b = int(sys.argv[1]), int(sys.argv[2])
    setup_determinism()
    cfg = load_config()
    W = load_weights()
    E = W["embed"]
    wins, _ = load_token_windows()
    wins = wins[a:b]

    wstats = {
        "shape": list(E.shape),
        "global_max_abs": E.abs().max().item(),
        "rms": E.pow(2).mean().sqrt().item(),
        "row_max_abs_percentiles": {str(p): float(np.percentile(E.abs().amax(1).numpy(), p)) for p in (0, 1, 10, 50, 90, 99, 100)},
        "rows_all_zero_under_per_tensor_int8": int((E.abs().amax(1) < (E.abs().max() / 127.0) / 2).sum()),
        "fraction_of_weights_rounding_to_zero_per_tensor_int8": float((E.abs() < (E.abs().max() / 127.0) / 2).float().mean()),
        "fraction_of_weights_rounding_to_zero_per_row_int8": float(((E.abs() < (E.abs().amax(1, keepdim=True) / 127.0) / 2)).float().mean()),
    }
    variants = {m: quant_int8(E, m)[0] for m in ("per_tensor", "per_row")}
    wstats["per_tensor_vs_per_row_weight_rmse"] = {m: (variants[m] - E).pow(2).mean().sqrt().item() for m in variants}

    configs = {"lookup_only_per_tensor": ("per_tensor", None), "lookup_only_per_row": ("per_row", None),
               "head_only_per_tensor": (None, "per_tensor"), "head_only_per_row": (None, "per_row"),
               "both_per_tensor": ("per_tensor", "per_tensor"), "both_per_row": ("per_row", "per_row")}
    agg = {k: [] for k in configs}
    act_agg = {f: [] for f in range(4, 14)}
    t0 = time.time()
    for i in range(wins.shape[0]):
        ids = torch.from_numpy(wins[i])
        with torch.no_grad():
            y = forward(W, cfg, ids, return_logits=False)
            base = y @ E.T
            for name, (lk, hd) in configs.items():
                if lk is None:                       # head role only: reuse the float hidden state
                    lg = y @ variants[hd].T
                else:
                    yy = forward(W, cfg, ids, return_logits=False, embed_lookup=variants[lk])
                    lg = yy @ (variants[hd] if hd else E).T
                agg[name].append(compare(base, lg, ids))
            for f in act_agg:                         # final hidden state stored as INT16, head in float
                yq, clipped = quant_act(y, f)
                r = compare(base, yq @ E.T, ids)
                r["clipped_values"] = clipped
                act_agg[f].append(r)
        print(f"window {a+i} {time.time()-t0:.0f}s", flush=True)

    def mean_of(lst):
        out = {}
        for k in lst[0]:
            vals = [d[k] for d in lst]
            out[k] = float(np.sum(vals)) if k in ("d_nll_sum", "tokens", "clipped_values") else float(np.mean(vals))
        out["d_nll_per_token"] = out["d_nll_sum"] / out["tokens"]
        return out

    report = {"windows": [a, b - 1], "weight_stats": wstats,
              "role_experiments": {k: mean_of(v) for k, v in agg.items()},
              "final_hidden_int16_fractional_bits": {str(f): mean_of(v) for f, v in act_agg.items()}}
    json.dump(report, open(os.path.join(OUT_DIR, "embedding_sensitivity.json"), "w"), indent=2)
    print(json.dumps(report["role_experiments"], indent=1))


if __name__ == "__main__":
    main()
