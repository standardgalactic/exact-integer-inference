"""Follow-up measurements on the calibration profile that bear on the format choice."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR
from profile_acts import LO, HI, SUB, NB
from summarize import load, absmax, exceed_count, below_count, k_for, hist_percentile

cal, _ = load("cal")
val, _ = load("val")
out = {}

# 1. channel starvation under one per-tensor INT16 scale (calibration max, one spare bit)
star = {}
for (site, l), s in cal.items():
    if "ch_sumsq" not in s or "ntok" not in s:
        continue
    k = k_for(absmax(s), True)
    step = 2.0 ** (k - 15)
    rms = np.sqrt(s["ch_sumsq"] / int(s["ntok"]))
    rms = rms[rms > 0]
    ratio = rms / step
    star.setdefault(site, []).append({"layer": l, "min_channel_rms_in_steps": float(ratio.min()),
                                      "p5_channel_rms_in_steps": float(np.percentile(ratio, 5)),
                                      "channels_below_8_steps": int((ratio < 8).sum()), "channels": int(ratio.size)})
out["channel_starvation_per_tensor_int16"] = {
    site: {"worst_layer_by_min": min(v, key=lambda d: d["min_channel_rms_in_steps"]),
           "layers_with_any_channel_below_8_steps": [d["layer"] for d in v if d["channels_below_8_steps"] > 0],
           "max_channels_below_8_steps": max(d["channels_below_8_steps"] for d in v)}
    for site, v in star.items()}

# 2. residual stream storage options, layers 12..28 where the massive token is present
res = []
for l in range(30):
    s = cal[("resid_in", l)]
    n = int(s["n"]); ntok = int(s["ntok"])
    cm = s["ch_max"]
    rms = np.sqrt(s["ch_sumsq"] / ntok)
    order = np.argsort(cm)[::-1]
    top = int(order[0])
    k_t = k_for(absmax(s), True)
    others = np.delete(cm, top)
    k_o = k_for(float(others.max()), True)
    p50 = hist_percentile(s["hist"], int(s["zeros"]), n, 50.0)
    p99 = hist_percentile(s["hist"], int(s["zeros"]), n, 99.0)
    kc = np.floor(np.log2(cm)).astype(int) + 2
    stepc = 2.0 ** (kc - 15.0)
    res.append({"layer": l, "top_channel": top, "absmax": absmax(s), "max_excluding_top_channel": float(others.max()),
                "int16_per_tensor": {"k": k_t, "step": 2.0 ** (k_t - 15), "p50_in_steps": p50 / 2.0 ** (k_t - 15), "p99_in_steps": p99 / 2.0 ** (k_t - 15)},
                "int16_per_tensor_without_top_channel": {"k": k_o, "step": 2.0 ** (k_o - 15), "p50_in_steps": p50 / 2.0 ** (k_o - 15)},
                "int32_f15_per_tensor": {"range": 2.0 ** 16, "step": 2.0 ** -15, "p50_in_steps": p50 / 2.0 ** -15,
                                         "covers_absmax_with_one_spare_bit": bool(absmax(s) * 2 < 2 ** 16)},
                "int16_per_channel": {"min_channel_rms_in_steps": float((rms / stepc)[rms > 0].min()),
                                      "p5_channel_rms_in_steps": float(np.percentile((rms / stepc)[rms > 0], 5))}})
out["residual_options"] = res

# 3. attention: score tails and probability underflow
att = []
rows_per_layer = 9 * 512 * 64
for l in range(30):
    sh = cal[("attn_score_shifted", l)]
    pr = cal[("attn_prob", l)]
    sk = cal[("attn_sink_mass", l)]
    n_sh, n_pr = int(sh["n"]), int(pr["n"])
    d = {"layer": l,
         "frac_shifted_score_abs_ge_8": exceed_count(sh["hist"], 3) / n_sh,
         "frac_shifted_score_abs_ge_16": exceed_count(sh["hist"], 4) / n_sh,
         "frac_shifted_score_abs_ge_32": exceed_count(sh["hist"], 5) / n_sh,
         "max_abs_shifted_score": absmax(sh),
         "sink_mass_p50": hist_percentile(sk["hist"], int(sk["zeros"]), int(sk["n"]), 50.0),
         "sink_mass_p90": hist_percentile(sk["hist"], int(sk["zeros"]), int(sk["n"]), 90.0)}
    for name, half_exp in (("q0_16", -17), ("q0_15", -16), ("q0_12", -13)):
        thr_bin = (half_exp - LO) * SUB
        h = pr["hist"]
        idx = np.arange(NB)
        octv, sub = np.divmod(idx, SUB)
        mid = 2.0 ** (LO + octv) * (1 + (sub + 0.5) / SUB)
        mass = float((h[:thr_bin] * mid[:thr_bin]).sum()) / rows_per_layer
        frac = below_count(h, int(pr["zeros"]), half_exp) / n_pr
        d[name] = {"frac_entries_rounding_to_zero": frac, "mean_row_mass_rounded_away": mass}
    att.append(d)
out["attention"] = att

# 4. normalization statistic domain
nm = {}
for site in ("norm1_ms", "norm2_ms", "final_ms", "norm1_inv_rms", "norm2_inv_rms", "final_inv_rms"):
    lo = min(float(cal[(site, l)]["smin"]) for (st, l) in cal if st == site)
    hi = max(float(cal[(site, l)]["smax"]) for (st, l) in cal if st == site)
    nm[site] = {"min": lo, "max": hi, "dynamic_range_octaves": math.log2(hi / lo)}
out["normalization_domain"] = nm

# 5. mask bookkeeping
out["masking"] = {"window_tokens": 512, "heads": 9,
                  "masked_entries_per_head_per_window": 512 * 511 // 2,
                  "valid_entries_per_head_per_window": 512 * 513 // 2,
                  "masked_fraction": (512 * 511 / 2) / (512 * 512),
                  "float_reference_handling": "masked entries set to -inf before softmax, never recorded in any range statistic",
                  "valid_entry_count_check": int(cal[("attn_score", 0)]["n"]) == 9 * (512 * 513 // 2) * 64}
json.dump(out, open(os.path.join(OUT_DIR, "extra_analysis.json"), "w"), indent=1)
print("ok")
