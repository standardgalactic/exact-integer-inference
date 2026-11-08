"""Turn profile_cal / profile_val into tables and a scaling proposal.

Format convention for stored INT16 activations: value = q * 2^-f with q in
[-32768, 32767].  A site with integer-bit exponent k (range bound 2^k) has
f = 15 - k.  A value saturates when |x| >= 2^k (the sliver between 32767.5
and 32768 steps is ignored here; the exact verification pass counts it).

Policies compared:
  headroom1   smallest k with 2^k > 2 * calibration abs-max (one spare bit)
  headroom0   smallest k with 2^k > calibration abs-max (no spare bit)
  p99999      smallest k with calibration exceedance fraction at 2^k <= 1e-5
              (saturating the rare tail on purpose)
Clip rates are measured on the validation windows, which never influenced k.
"""
import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR
from profile_acts import LO, HI, SUB, NB

PCTS = [50.0, 90.0, 99.0, 99.9, 99.99]


def load(tag):
    z = np.load(os.path.join(OUT_DIR, f"profile_{tag}.npz"))
    sites = {}
    for key in z.files:
        if key.startswith("nll"):
            continue
        site, layer, field = key.rsplit("|", 2)
        sites.setdefault((site, int(layer)), {})[field] = z[key]
    return sites, z


def hist_percentile(hist, zeros, n, q):
    """Value below which fraction q of all entries (zeros included) lie; linear inside the bin."""
    target = q / 100.0 * n
    if target <= zeros:
        return 0.0
    cum = zeros + np.cumsum(hist)
    b = int(np.searchsorted(cum, target))
    b = min(b, NB - 1)
    prev = zeros + (cum[b - 1] - zeros if b > 0 else 0)
    inbin = hist[b]
    frac = (target - prev) / inbin if inbin else 0.0
    octv, sub = divmod(b, SUB)
    lo = 2.0 ** (LO + octv) * (1 + sub / SUB)
    return lo + frac * 2.0 ** (LO + octv) / SUB


def exceed_count(hist, k):
    """Exact count of |x| >= 2^k (octave edge)."""
    b = (k - LO) * SUB
    b = max(0, min(NB, b))
    return int(hist[b:].sum())


def below_count(hist, zeros, k):
    """Exact count of |x| < 2^k, zeros included."""
    b = (k - LO) * SUB
    b = max(0, min(NB, b))
    return int(zeros + hist[:b].sum())


def absmax(s):
    return max(abs(float(s["smin"])), abs(float(s["smax"])))


def k_for(maxabs, extra_bit):
    k = math.floor(math.log2(maxabs)) + 1 if maxabs > 0 else -20
    return k + (1 if extra_bit else 0)


def main():
    cal, _ = load("cal")
    val, _ = load("val")
    rows, prop = [], {}
    for (site, l), s in sorted(cal.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        v = val[(site, l)]
        n, nv = int(s["n"]), int(v["n"])
        mx_c, mx_v = absmax(s), absmax(v)
        r = {"site": site, "layer": l, "n_cal": n, "absmax_cal": mx_c, "absmax_val": mx_v,
             "val_over_cal_max": mx_v / mx_c if mx_c else float("nan"),
             "smin_cal": float(s["smin"]), "smax_cal": float(s["smax"]),
             "zero_fraction": float(s["zeros"]) / n}
        for p in PCTS:
            r[f"abs_p{p}"] = hist_percentile(s["hist"], int(s["zeros"]), n, p)
        if "ch_max" in s:
            cm = s["ch_max"]
            r["n_channels"] = int(cm.size)
            r["chan_max_median"] = float(np.median(cm))
            r["chan_max_over_median"] = float(cm.max() / np.median(cm)) if np.median(cm) > 0 else float("nan")
            r["top_channel"] = int(np.argmax(cm))
            if "ch_sumsq" in s and "ntok" in s:
                r["rms"] = float(math.sqrt(s["ch_sumsq"].sum() / (int(s["ntok"]) * cm.size)))
        if "tokhist" in s:
            r["tok_absmax_p50"] = hist_percentile(s["tokhist"], int(s.get("tok_zeros", 0)), int(s["ntok"]), 50.0)
            r["tok_absmax_p99"] = hist_percentile(s["tokhist"], int(s.get("tok_zeros", 0)), int(s["ntok"]), 99.0)
        r["underflow_bin_clamps"] = int(s["under"]); r["overflow_bin_clamps"] = int(s["over"])
        # policies
        p99 = r["abs_p99.0"]
        pol = {}
        k_a, k_b = k_for(mx_c, True), k_for(mx_c, False)
        k_c = None
        for k in range(LO, HI + 1):
            if exceed_count(s["hist"], k) / n <= 1e-5:
                k_c = k; break
        for name, k in (("headroom1", k_a), ("headroom0", k_b), ("p99999", k_c)):
            f = 15 - k
            half = k - 16                                  # half step = 2^(k-16)
            pol[name] = {
                "k": k, "f": f, "step": 2.0 ** (k - 15),
                "val_clipped": exceed_count(v["hist"], k), "val_clip_rate": exceed_count(v["hist"], k) / nv,
                "cal_clip_rate": exceed_count(s["hist"], k) / n,
                "val_round_to_zero_rate_of_nonzero": (below_count(v["hist"], 0, half)) / max(1, nv - int(v["zeros"])),
                "p99_in_steps": p99 / 2.0 ** (k - 15) if p99 > 0 else float("nan"),
            }
        r["policy"] = pol
        rows.append(r)
        prop[f"{site}|{l}"] = {name: {"k": p["k"], "f": p["f"]} for name, p in pol.items()}
    json.dump(rows, open(os.path.join(OUT_DIR, "profile_summary.json"), "w"), indent=1)
    json.dump(prop, open(os.path.join(OUT_DIR, "scaling_candidates.json"), "w"), indent=1)

    cols = ["site", "layer", "absmax_cal", "absmax_val", "val_over_cal_max", "abs_p50.0", "abs_p99.0", "abs_p99.9", "abs_p99.99",
            "chan_max_over_median", "top_channel", "rms", "zero_fraction"]
    with open(os.path.join(OUT_DIR, "profile_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ["k_h1", "f_h1", "val_clip_h1", "k_h0", "val_clip_h0", "k_p5", "val_clip_p5", "p99_steps_h1", "r2z_val_h1"])
        for r in rows:
            p = r["policy"]
            w.writerow([r.get(c, "") for c in cols] + [p["headroom1"]["k"], p["headroom1"]["f"], p["headroom1"]["val_clipped"],
                       p["headroom0"]["k"], p["headroom0"]["val_clipped"], p["p99999"]["k"], p["p99999"]["val_clipped"],
                       p["headroom1"]["p99_in_steps"], p["headroom1"]["val_round_to_zero_rate_of_nonzero"]])
    print("rows", len(rows))


if __name__ == "__main__":
    main()
