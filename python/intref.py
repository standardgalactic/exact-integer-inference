"""Integer reference semantics for SmolLM2-135M (numpy prototype).

Every value that crosses a step boundary is an integer.  Matrix products
accumulate exactly: operands are widened to float64 for BLAS, which is exact
because every partial sum stays below 2^53 (asserted), then converted back to
int64.  Information is discarded only at the points named below.

Conventions
  RS(x, n)   rounding right shift, (x + 2^(n-1)) >> n with arithmetic shift,
             i.e. round half toward +infinity; n <= 0 is an exact left shift.
  SAT(x, b)  clip to the signed b-bit range [-2^(b-1), 2^(b-1) - 1].
  value = q * 2^-f for every stored tensor; f can be negative.
  Exact tables are rounded half up from 50-digit mpmath values.

Formats (see FMT below)
  residual            INT32, f = 15
  INT16 sites         f = 15 - k from scaling_proposal.json (k = calibration exponent + spare bit)
  flagged sites       INT32, f = 31 - k   (wide mode) or INT16 (narrow mode, for comparison)
  attention scores    INT16, f = 7; exponent table indexed by (rowmax - score) in 1/128 units
  weights             INT8 per output row; row multiplier m in [2^13, 2^14], shift n: scale = m * 2^-n
  RoPE tables         Q30
  probabilities       Q0.16 held in int64 (value 65536 means 1.0)
"""
import json, math, os, sys
import numpy as np
import mpmath as mp

mp.mp.dps = 50
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, load_weights, OUT_DIR

F_RES = 15
F_SCORE = 7
DMAX = 12 * 128                   # exponent table covers rowmax - score in [0, 12) real units
EXP_BITS = 30
EPS_S = 6184753                   # round(1e-5 * 576 * 2^30), norm epsilon in squared-residual units
R_ISQRT = 40                      # U = isqrt(2^(2R) // T)
I64 = np.int64


def rs(x, n):
    if np.isscalar(n):
        n = int(n)
        if n <= 0:
            return x << (-n)
        return (x + (I64(1) << (n - 1))) >> n
    n = np.asarray(n, dtype=I64)
    assert (n >= 1).all(), "per-element shift must be >= 1"
    return (x + (I64(1) << (n - 1))) >> n


def sat(x, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(x, lo, hi)


def matmul_exact(A, Wf, bound_a, wmax=127):
    """A int64 [T,K], Wf float64 [N,K] of integer values -> exact int64 [T,N] = A @ W.T"""
    K = A.shape[1]
    assert bound_a * wmax * K < 2.0 ** 53, "float64 accumulation would not be exact"
    out = A.astype(np.float64) @ Wf.T
    return np.rint(out).astype(I64)


# ---------------------------------------------------------------- tables

def _round_half_up(v):
    return int(mp.floor(v + mp.mpf(1) / 2))


_exp_table = None
def exp_table():
    global _exp_table
    if _exp_table is None:
        _exp_table = np.array([_round_half_up(mp.mpf(2) ** EXP_BITS * mp.exp(-mp.mpf(d) / 128)) for d in range(DMAX + 1)], dtype=I64)
    return _exp_table


_silu_cache = {}
def silu_table(f_in, f_out, out_bits):
    key = (f_in, f_out, out_bits)
    if key not in _silu_cache:
        qs = np.arange(-32768, 32768)
        vals = np.empty(65536, dtype=I64)
        sc_in = mp.mpf(2) ** (-f_in)
        sc_out = mp.mpf(2) ** f_out
        for i, q in enumerate(qs):
            x = int(q) * sc_in
            y = x / (1 + mp.exp(-x))
            vals[i] = _round_half_up(y * sc_out)
        _silu_cache[key] = sat(vals, out_bits)
    return _silu_cache[key]


_rope_cache = {}
def rope_tables(T, theta=100000, head_dim=64):
    if T not in _rope_cache:
        half = head_dim // 2
        C = np.empty((T, half), dtype=I64); S = np.empty((T, half), dtype=I64)
        inv = [mp.mpf(theta) ** (-mp.mpf(2 * j) / head_dim) for j in range(half)]
        for p in range(T):
            for j in range(half):
                a = p * inv[j]
                C[p, j] = _round_half_up(mp.cos(a) * 2 ** 30)
                S[p, j] = _round_half_up(mp.sin(a) * 2 ** 30)
        _rope_cache[T] = (C, S)
    return _rope_cache[T]


# ---------------------------------------------------------------- weights

def quant_rows(W):
    """W float32 [N,K] -> (q int8 as float64 [N,K], m int64 [N], n int64 [N]); scale = m * 2^-n."""
    W = W.astype(np.float64)
    s = np.abs(W).max(1) / 127.0
    s = np.where(s == 0, 1.0, s)
    n = (13 - np.floor(np.log2(s))).astype(I64)
    m = np.rint(s * 2.0 ** n).astype(I64)
    assert m.min() >= 2 ** 13 and m.max() <= 2 ** 14
    sw = m * 2.0 ** (-n.astype(np.float64))
    q = np.clip(np.rint(W / sw[:, None]), -127, 127)
    return q, m, n


def quant_vec_int16(w):
    w = w.astype(np.float64)
    f = 14 - int(math.floor(math.log2(np.abs(w).max())))
    q = np.clip(np.rint(w * 2.0 ** f), -32768, 32767).astype(I64)
    return q, f


class Model:
    def __init__(self, wide=True, proposal_path=None):
        self.cfg = load_config()
        W = load_weights()
        self.wide = wide
        prop = json.load(open(proposal_path or os.path.join(OUT_DIR, "scaling_proposal.json")))["sites"]
        self.prop = prop
        self.L = self.cfg["num_hidden_layers"]
        self.emb_q, self.emb_m, self.emb_n = quant_rows(W["embed"].numpy())
        self.layers = []
        for l, Ly in enumerate(W["layers"]):
            d = {}
            for name in ("q", "k", "v", "o", "gate", "up", "down"):
                d[name] = quant_rows(Ly[name].numpy())
            d["ln1"] = quant_vec_int16(Ly["ln1"].numpy())
            d["ln2"] = quant_vec_int16(Ly["ln2"].numpy())
            self.layers.append(d)
        self.final_gain = quant_vec_int16(W["norm"].numpy())
        self.rope = rope_tables(512)

    # format lookup: returns (bits, f, k, flagged)
    def fmt(self, site, l):
        e = self.prop[f"{site}|{l}"]
        flagged = e["class"].startswith("flagged")
        k = e["k"]
        if flagged and self.wide:
            return 32, 31 - k
        return 16, 15 - k

    # ------------------------------------------------------------ steps
    def embed(self, ids):
        q = self.emb_q[ids].astype(I64)
        x = q * self.emb_m[ids][:, None]
        return rs(x, (self.emb_n[ids] - F_RES)[:, None])

    def rmsnorm(self, x, gain, f_out, bits_out, trace=None, tag=""):
        T, D = x.shape
        hi = x >> 16
        lo = x & 0xFFFF
        A = (hi * hi).sum(1); B = (hi * lo).sum(1); C = (lo * lo).sum(1)
        U = np.empty(T, dtype=I64)
        for t in range(T):
            S = (int(A[t]) << 32) + (int(B[t]) << 17) + int(C[t])
            Tt = S + EPS_S
            U[t] = math.isqrt((1 << (2 * R_ISQRT)) // Tt)
        z = rs(x * U[:, None], 16)                     # |x U| <= 2^40, so |z| <= 2^24
        wq, fw = gain
        acc = z * wq[None, :] * 3                      # 24 = 3 * 2^3, the 2^3 sits in the shift
        out = sat(rs(acc, 21 + fw - f_out), bits_out)
        if trace is not None:
            trace[tag + "_U"] = U; trace[tag + "_z"] = z; trace[tag + "_out"] = out
        return out

    def linear(self, a, bound_a, W, f_in, f_out, bits_out):
        q, m, n = W
        acc = matmul_exact(a, q, bound_a)
        assert np.abs(acc).max() * int(m.max()) < 2 ** 63
        return sat(rs(acc * m[None, :], n + f_in - f_out), bits_out), acc

    def block(self, l, x, trace=None):
        cfg = self.cfg
        T, D = x.shape
        nh, nkv, hd = 9, 3, 64
        Ly = self.layers[l]
        tr = trace if trace is not None else None
        if tr is not None:
            tr["resid_in"] = x.copy()
            tr["_f"] = {"resid_in": F_RES, "o_res": F_RES, "resid_mid": F_RES, "down_res": F_RES, "resid_out": F_RES,
                        "attn_score": F_SCORE, "attn_prob": 16}

        bits_h, f_h = self.fmt("norm1_out", l)
        h = self.rmsnorm(x, Ly["ln1"], f_h, bits_h, tr, "norm1")
        bh = 1 << (bits_h - 1)
        bits_q, f_q = self.fmt("q_proj", l); bits_k, f_k = self.fmt("k_proj", l); bits_v, f_v = self.fmt("v_proj", l)
        q, _ = self.linear(h, bh, Ly["q"], f_h, f_q, bits_q)
        k, _ = self.linear(h, bh, Ly["k"], f_h, f_k, bits_k)
        v, _ = self.linear(h, bh, Ly["v"], f_h, f_v, bits_v)
        if tr is not None: tr.update(q_proj=q, k_proj=k, v_proj=v)

        # RoPE (rotate_half): out[d] = a[d] c - a[d+32] s ; out[d+32] = a[d+32] c + a[d] s
        C, S = self.rope[0][:T], self.rope[1][:T]
        bits_qr, f_qr = self.fmt("q_rope", l); bits_kr, f_kr = self.fmt("k_rope", l)
        def rope(a, nheads, f_in, f_o, bits_o):
            a = a.reshape(T, nheads, hd)
            a1, a2 = a[:, :, :32], a[:, :, 32:]
            c, s = C[:, None, :], S[:, None, :]
            o1 = a1 * c - a2 * s
            o2 = a2 * c + a1 * s
            o = np.concatenate([o1, o2], axis=2)
            return sat(rs(o, 30 + f_in - f_o), bits_o)
        qr = rope(q, nh, f_q, f_qr, bits_qr)       # [T, nh, hd]
        kr = rope(k, nkv, f_k, f_kr, bits_kr)      # [T, nkv, hd]
        if tr is not None: tr.update(q_rope=qr.reshape(T, -1), k_rope=kr.reshape(T, -1))

        vv = v.reshape(T, nkv, hd)
        bq, bk, bv = 1 << (bits_qr - 1), 1 << (bits_kr - 1), 1 << (bits_v - 1)
        shift_s = f_qr + f_kr + 3 - F_SCORE
        assert shift_s >= 1
        E = exp_table()
        ctx = np.empty((T, nh, hd), dtype=I64)
        causal = np.tril(np.ones((T, T), dtype=bool))
        sc_all, p_all = [], []
        bits_c, f_c = self.fmt("attn_ctx", l)
        for hh in range(nh):
            g = hh // 3
            acc = matmul_exact(qr[:, hh, :], kr[:, g, :].astype(np.float64), bq, wmax=bk)
            s = sat(rs(acc, shift_s), 16)                                  # [T,T], f=7
            s_m = np.where(causal, s, -(1 << 40))
            rowmax = s_m.max(1, keepdims=True)
            d = rowmax - s
            valid = causal & (d <= DMAX)
            e = np.where(valid, E[np.clip(d, 0, DMAX)], 0)
            Z = e.sum(1, keepdims=True)
            P = (e * 65536 + Z // 2) // Z                                  # Q0.16
            assert 65536 * bv * T < 2 ** 53
            acc2 = np.rint(P.astype(np.float64) @ vv[:, g, :].astype(np.float64)).astype(I64)
            ctx[:, hh, :] = sat(rs(acc2, 16 + f_v - f_c), bits_c)
            if tr is not None:
                sc_all.append(np.where(causal, s, 0)); p_all.append(P)
        if tr is not None:
            tr["attn_score"] = np.stack(sc_all); tr["attn_prob"] = np.stack(p_all); tr["attn_ctx"] = ctx.reshape(T, -1)
        ctx = ctx.reshape(T, D)

        # o_proj lands directly in the residual format
        q_o, m_o, n_o = Ly["o"]
        acc = matmul_exact(ctx, q_o, 1 << (bits_c - 1))
        assert np.abs(acc).max() * int(m_o.max()) < 2 ** 63
        o_res = rs(acc * m_o[None, :], n_o + f_c - F_RES)
        x = sat(x + o_res, 32)
        if tr is not None: tr["o_res"] = o_res; tr["resid_mid"] = x.copy()

        bits_h2, f_h2 = self.fmt("norm2_out", l)
        h2 = self.rmsnorm(x, Ly["ln2"], f_h2, bits_h2, tr, "norm2")
        bh2 = 1 << (bits_h2 - 1)
        bits_g, f_g = self.fmt("gate_proj", l); bits_u, f_u = self.fmt("up_proj", l)
        g_, _ = self.linear(h2, bh2, Ly["gate"], f_h2, f_g, bits_g)
        u_, _ = self.linear(h2, bh2, Ly["up"], f_h2, f_u, bits_u)
        bits_s, f_s = self.fmt("silu_gate", l)
        tab = silu_table(f_g, f_s, bits_s)
        sg = tab[g_ + 32768]
        bits_m, f_m0 = self.fmt("mlp_gated", l)
        f_m = min(f_m0, f_s + f_u) if bits_m == 32 else f_m0
        gated = sat(rs(sg * u_, f_s + f_u - f_m), bits_m)
        if tr is not None: tr.update(gate_proj=g_, up_proj=u_, silu_gate=sg, mlp_gated=gated)
        q_d, m_d, n_d = Ly["down"]
        acc = matmul_exact(gated, q_d, 1 << (bits_m - 1))
        assert np.abs(acc).max() * int(m_d.max()) < 2 ** 63
        d_res = rs(acc * m_d[None, :], n_d + f_m - F_RES)
        x = sat(x + d_res, 32)
        if tr is not None:
            tr["down_res"] = d_res; tr["resid_out"] = x.copy()
            tr["_f"].update(norm1_out=f_h, q_proj=f_q, k_proj=f_k, v_proj=f_v, q_rope=f_qr, k_rope=f_kr, attn_ctx=f_c,
                            norm2_out=f_h2, gate_proj=f_g, up_proj=f_u, silu_gate=f_s, mlp_gated=f_m)
        return x

    def forward(self, ids, narrow_residual=False):
        x = self.embed(ids)
        for l in range(self.L):
            if narrow_residual:
                x = self._grid(x, "resid_in", l)
            x = self.block(l, x)
            if narrow_residual:
                x = self._grid(x, "resid_out", l)
        bits_y, f_y = 16, 15 - self.prop["final_out|30"]["k"]
        y = self.rmsnorm(x, self.final_gain, f_y, bits_y)
        q, m, n = self.emb_q, self.emb_m, self.emb_n
        acc = matmul_exact(y, q, 1 << 15)
        logits = rs(acc * m[None, :], n + f_y - 16)             # INT, f = 16
        return logits

    def _grid(self, x, site, l):
        """Narrow comparison mode: force the residual onto its INT16 per-tensor grid."""
        k = self.prop[f"{site}|{l}"]["k"]
        f = 15 - k
        s = F_RES - f
        return sat(rs(x, s), 16) << s


def dequant(q, f):
    return q.astype(np.float64) * 2.0 ** (-f)
