"""Per (site, layer) scaling proposal derived from profile_summary.json.

Default format: INT16, one static per-tensor exponent, k = floor(log2(cal absmax)) + 2
(one spare bit).  A site-layer is flagged when the median |x| spans fewer than 64
steps under that format.  The 64-step threshold is a screening rule, not a derived
bound; the real test is held-out perplexity once the integer engine exists.
"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR

R = json.load(open(os.path.join(OUT_DIR, "profile_summary.json")))
SPECIAL = {"attn_prob": "uint16_q0_16", "attn_sink_mass": "derived_not_stored", "attn_rowmax": "int16_per_tensor",
           "attn_expsum": "int32_fixed_point", "norm1_ms": "wide_fixed_point_with_normalized_rsqrt",
           "norm2_ms": "wide_fixed_point_with_normalized_rsqrt", "final_ms": "wide_fixed_point_with_normalized_rsqrt",
           "norm1_inv_rms": "int32_fixed_point", "norm2_inv_rms": "int32_fixed_point", "final_inv_rms": "int32_fixed_point",
           "logits": "int32_or_int16_per_tensor_head_output"}
out, counts = {}, collections.Counter()
for r in R:
    p = r["policy"]["headroom1"]
    step = p["step"]
    p50_steps = r["abs_p50.0"] / step if step else float("nan")
    cls = SPECIAL.get(r["site"])
    if cls is None:
        cls = "int16_per_tensor" if p50_steps >= 64 else "flagged_int16_per_tensor_resolves_median_poorly"
    entry = {"class": cls, "k": p["k"], "f": p["f"], "step": step, "median_abs_in_steps": p50_steps,
             "cal_absmax": r["absmax_cal"], "val_absmax": r["absmax_val"],
             "val_clipped_values": p["val_clipped"], "val_clip_rate": p["val_clip_rate"],
             "val_zero_round_rate_of_nonzero": p["val_round_to_zero_rate_of_nonzero"]}
    if cls.startswith("flagged"):
        entry["candidates_to_measure"] = ["int32 per-tensor with f chosen from k", "int16 per-channel power-of-two exponents"]
    out[f"{r['site']}|{r['layer']}"] = entry
    counts[cls] += 1
json.dump({"format_rule": "value = q * 2^-f, q in [-32768, 32767], f = 15 - k, k = floor(log2(cal absmax)) + 2",
           "class_counts": dict(counts), "sites": out}, open(os.path.join(OUT_DIR, "scaling_proposal.json"), "w"), indent=1)
print(dict(counts))
print("total val clipped values under the default rule:", sum(e["val_clipped_values"] for e in out.values()))
