"""Export the quantized model and expected integer states for the C reference.

Usage: export_vectors.py OUTDIR WINDOW NTOKENS
Writes OUTDIR/model.bin (sequential little-endian arrays, order documented in intref.c) and
OUTDIR/ids.i64, exp_embed.i32, exp_resid_out.i32 [L,T,D], exp_logits.i32 [T,V], plus manifest.json with hashes.
"""
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import intref
from intref import Model, silu_table, exp_table

outdir, win, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(outdir, exist_ok=True)
wins, _ = load_token_windows()
ids = wins[win][:T].astype(np.int64)
M = Model(wide=True)
D = 576

def w(f, a, dt):
    f.write(np.ascontiguousarray(a, dtype=dt).tobytes())

with open(os.path.join(outdir, "model.bin"), "wb") as f:
    w(f, M.emb_q, np.int8); w(f, M.emb_m, np.int64); w(f, M.emb_n, np.int64)
    w(f, M.final_gain[0], np.int64); w(f, [M.final_gain[1]], np.int64)
    f_y = 15 - M.prop["final_out|30"]["k"]
    w(f, [f_y], np.int64)
    w(f, exp_table(), np.int64)
    C, S = M.rope
    w(f, C[:512], np.int64); w(f, S[:512], np.int64)
    for l in range(M.L):
        Ly = M.layers[l]
        names = ["norm1_out", "q_proj", "k_proj", "v_proj", "q_rope", "k_rope", "attn_ctx", "norm2_out", "gate_proj", "up_proj", "silu_gate", "mlp_gated"]
        fm = []
        for nme in names:
            bits, fr = M.fmt(nme, l); fm += [bits, fr]
        w(f, fm, np.int64)
        w(f, Ly["ln1"][0], np.int64); w(f, [Ly["ln1"][1]], np.int64)
        w(f, Ly["ln2"][0], np.int64); w(f, [Ly["ln2"][1]], np.int64)
        for nme in ("q", "k", "v", "o", "gate", "up", "down"):
            q, m, n = Ly[nme]
            w(f, q, np.int8); w(f, m, np.int64); w(f, n, np.int64)
        bits_g, f_g = M.fmt("gate_proj", l); bits_s, f_s = M.fmt("silu_gate", l)
        w(f, silu_table(f_g, f_s, bits_s), np.int64)

# expected states
x = M.embed(ids)
exp_embed = x.copy()
res = []
for l in range(M.L):
    x = M.block(l, x)
    res.append(x.copy())
bits_y, f_y = 16, 15 - M.prop["final_out|30"]["k"]
y = M.rmsnorm(x, M.final_gain, f_y, bits_y)
acc = intref.matmul_exact(y, M.emb_q, 1 << 15)
logits = intref.rs(acc * M.emb_m[None, :], M.emb_n + f_y - 16)
assert np.abs(logits).max() < 2 ** 31 and np.abs(np.array(res)).max() < 2 ** 31
for name, arr, dt in (("ids.i64", ids, np.int64), ("exp_embed.i32", exp_embed, np.int32),
                      ("exp_resid_out.i32", np.array(res), np.int32), ("exp_logits.i32", logits, np.int32)):
    arr.astype(dt).tofile(os.path.join(outdir, name))
man = {"window": win, "tokens": T, "files": {}}
for fn in sorted(os.listdir(outdir)):
    man["files"][fn] = {"bytes": os.path.getsize(os.path.join(outdir, fn)), "sha256": sha256_file(os.path.join(outdir, fn))}
json.dump(man, open(os.path.join(outdir, "manifest.json"), "w"), indent=1)
print("exported", {k: v["bytes"] for k, v in man["files"].items()})
