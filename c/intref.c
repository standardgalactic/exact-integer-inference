/*
 * Scalar C reference for the SmolLM2-135M integer semantics (see INTEGER_REFERENCE_NOTES.md).
 *
 * Usage: intref VECTORDIR
 *   reads VECTORDIR/model.bin, ids.i64 and the expected states exported by export_vectors.py,
 *   recomputes every state with plain integer arithmetic and reports exact mismatches.
 *
 * model.bin layout (little endian):
 *   emb_q int8[V*D], emb_m i64[V], emb_n i64[V], final_gain_q i64[D], final_gain_f i64, f_y i64,
 *   exp_table i64[DMAX+1], rope_cos i64[512*32], rope_sin i64[512*32],
 *   per layer: fmt i64[24] (bits,f pairs: norm1_out q_proj k_proj v_proj q_rope k_rope attn_ctx norm2_out
 *              gate_proj up_proj silu_gate mlp_gated), ln1_q i64[D], ln1_f i64, ln2_q i64[D], ln2_f i64,
 *              then q,k,v,o,gate,up,down each as int8[N*K], m i64[N], n i64[N], then silu table i64[65536].
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int64_t i64;
typedef __int128 i128;
typedef unsigned __int128 u128;

enum { V = 49152, D = 576, DFF = 1536, NH = 9, NKV = 3, HD = 64, NL = 30, DMAX = 12 * 128, ROPE_POS = 512 };
static const i64 EPS_S = 6184753;
static const int F_RES = 15, F_SCORE = 7, R_ISQRT = 40;

static inline i64 rs(i64 x, int n) {
    if (n <= 0) return x * ((i64)1 << (-n));
    return (x + ((i64)1 << (n - 1))) >> n;       /* arithmetic shift on signed values (gcc) */
}
static inline i64 sat(i64 x, int bits) {
    if (bits >= 64) return x;
    i64 hi = ((i64)1 << (bits - 1)) - 1, lo = -((i64)1 << (bits - 1));
    return x < lo ? lo : (x > hi ? hi : x);
}
static u128 isqrt128(u128 v) {
    u128 res = 0, bit = (u128)1 << 126;
    while (bit > v) bit >>= 2;
    while (bit) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else res >>= 1;
        bit >>= 2;
    }
    return res;
}

typedef struct { int8_t *q; i64 *m, *n; int N, K; } Lin;
typedef struct {
    i64 fm[24];
    i64 ln1[D], ln1f, ln2[D], ln2f;
    Lin w[7];                                   /* q k v o gate up down */
    i64 *silu;
} Layer;

static i64 *emb_m, *emb_n, fg_q[D], fg_f, f_y, *exp_tab, *rope_c, *rope_s;
static int8_t *emb_q;
static Layer L[NL];

static void *xread(FILE *f, size_t bytes) {
    void *p = malloc(bytes);
    if (!p || fread(p, 1, bytes, f) != bytes) { fprintf(stderr, "short read\n"); exit(2); }
    return p;
}
static void xreadto(FILE *f, void *dst, size_t bytes) {
    if (fread(dst, 1, bytes, f) != bytes) { fprintf(stderr, "short read\n"); exit(2); }
}

static void load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    emb_q = xread(f, (size_t)V * D);
    emb_m = xread(f, V * 8); emb_n = xread(f, V * 8);
    xreadto(f, fg_q, D * 8); xreadto(f, &fg_f, 8); xreadto(f, &f_y, 8);
    exp_tab = xread(f, (DMAX + 1) * 8);
    rope_c = xread(f, ROPE_POS * 32 * 8); rope_s = xread(f, ROPE_POS * 32 * 8);
    static const int Ns[7] = {576, 192, 192, 576, 1536, 1536, 576}, Ks[7] = {576, 576, 576, 576, 576, 576, 1536};
    for (int l = 0; l < NL; l++) {
        xreadto(f, L[l].fm, 24 * 8);
        xreadto(f, L[l].ln1, D * 8); xreadto(f, &L[l].ln1f, 8);
        xreadto(f, L[l].ln2, D * 8); xreadto(f, &L[l].ln2f, 8);
        for (int i = 0; i < 7; i++) {
            L[l].w[i].N = Ns[i]; L[l].w[i].K = Ks[i];
            L[l].w[i].q = xread(f, (size_t)Ns[i] * Ks[i]);
            L[l].w[i].m = xread(f, Ns[i] * 8); L[l].w[i].n = xread(f, Ns[i] * 8);
        }
        L[l].silu = xread(f, 65536 * 8);
    }
    fclose(f);
}

static void rmsnorm(const i64 *x, int T, const i64 *gq, int fw, int f_out, int bits_out, i64 *out) {
    for (int t = 0; t < T; t++) {
        const i64 *r = x + (size_t)t * D;
        i64 A = 0, B = 0, C = 0;
        for (int c = 0; c < D; c++) {
            i64 hi = r[c] >> 16, lo = r[c] & 0xFFFF;
            A += hi * hi; B += hi * lo; C += lo * lo;
        }
        i128 S = ((i128)A * ((i128)1 << 32)) + ((i128)B * ((i128)1 << 17)) + (i128)C;
        u128 Tt = (u128)(S + EPS_S);
        u128 q = ((u128)1 << (2 * R_ISQRT)) / Tt;
        i64 U = (i64)isqrt128(q);
        for (int c = 0; c < D; c++) {
            i64 z = rs(r[c] * U, 16);
            i64 acc = z * gq[c] * 3;
            out[(size_t)t * D + c] = sat(rs(acc, 21 + fw - f_out), bits_out);
        }
    }
}

static void linear(const i64 *a, int T, const Lin *w, int f_in, int f_out, int bits_out, i64 *out) {
    for (int t = 0; t < T; t++) {
        const i64 *ar = a + (size_t)t * w->K;
        for (int n = 0; n < w->N; n++) {
            const int8_t *wr = w->q + (size_t)n * w->K;
            i64 acc = 0;
            for (int k = 0; k < w->K; k++) acc += ar[k] * (i64)wr[k];
            out[(size_t)t * w->N + n] = sat(rs(acc * w->m[n], (int)(w->n[n] + f_in - f_out)), bits_out);
        }
    }
}

static void block(int l, i64 *x, int T) {
    Layer *Ly = &L[l];
    const i64 *fm = Ly->fm;
    int bits_h = fm[0], f_h = fm[1], bits_q = fm[2], f_q = fm[3], bits_k = fm[4], f_k = fm[5], bits_v = fm[6], f_v = fm[7];
    int bits_qr = fm[8], f_qr = fm[9], bits_kr = fm[10], f_kr = fm[11], bits_c = fm[12], f_c = fm[13];
    int bits_h2 = fm[14], f_h2 = fm[15], bits_g = fm[16], f_g = fm[17], bits_u = fm[18], f_u = fm[19];
    int bits_s = fm[20], f_s = fm[21], bits_m = fm[22], f_m0 = fm[23];
    (void)f_g; (void)bits_u;
    size_t TD = (size_t)T * D;
    i64 *h = malloc(TD * 8), *q = malloc(TD * 8), *k = malloc((size_t)T * 192 * 8), *v = malloc((size_t)T * 192 * 8);
    i64 *qr = malloc(TD * 8), *kr = malloc((size_t)T * 192 * 8), *ctx = malloc(TD * 8), *tmp = malloc(TD * 8);

    rmsnorm(x, T, Ly->ln1, (int)Ly->ln1f, f_h, bits_h, h);
    linear(h, T, &Ly->w[0], f_h, f_q, bits_q, q);
    linear(h, T, &Ly->w[1], f_h, f_k, bits_k, k);
    linear(h, T, &Ly->w[2], f_h, f_v, bits_v, v);

    for (int t = 0; t < T; t++) {
        const i64 *C = rope_c + (size_t)t * 32, *S = rope_s + (size_t)t * 32;
        for (int hh = 0; hh < NH; hh++) {
            const i64 *a = q + (size_t)t * D + hh * HD; i64 *o = qr + (size_t)t * D + hh * HD;
            for (int d = 0; d < 32; d++) {
                i64 o1 = a[d] * C[d] - a[d + 32] * S[d], o2 = a[d + 32] * C[d] + a[d] * S[d];
                o[d] = sat(rs(o1, 30 + f_q - f_qr), bits_qr); o[d + 32] = sat(rs(o2, 30 + f_q - f_qr), bits_qr);
            }
        }
        for (int g = 0; g < NKV; g++) {
            const i64 *a = k + (size_t)t * 192 + g * HD; i64 *o = kr + (size_t)t * 192 + g * HD;
            for (int d = 0; d < 32; d++) {
                i64 o1 = a[d] * C[d] - a[d + 32] * S[d], o2 = a[d + 32] * C[d] + a[d] * S[d];
                o[d] = sat(rs(o1, 30 + f_k - f_kr), bits_kr); o[d + 32] = sat(rs(o2, 30 + f_k - f_kr), bits_kr);
            }
        }
    }

    int shift_s = f_qr + f_kr + 3 - F_SCORE;
    i64 *s = malloc((size_t)T * 8), *P = malloc((size_t)T * 8);
    for (int hh = 0; hh < NH; hh++) {
        int g = hh / 3;
        for (int i = 0; i < T; i++) {
            const i64 *qi = qr + (size_t)i * D + hh * HD;
            i64 rowmax = -((i64)1 << 40);
            for (int j = 0; j <= i; j++) {
                const i64 *kj = kr + (size_t)j * 192 + g * HD;
                i64 acc = 0;
                for (int d = 0; d < HD; d++) acc += qi[d] * kj[d];
                s[j] = sat(rs(acc, shift_s), 16);
                if (s[j] > rowmax) rowmax = s[j];
            }
            i64 Z = 0;
            for (int j = 0; j <= i; j++) {
                i64 d = rowmax - s[j];
                P[j] = d <= DMAX ? exp_tab[d] : 0;
                Z += P[j];
            }
            for (int j = 0; j <= i; j++) P[j] = (P[j] * 65536 + Z / 2) / Z;
            for (int d = 0; d < HD; d++) {
                i64 acc = 0;
                for (int j = 0; j <= i; j++) acc += P[j] * v[(size_t)j * 192 + g * HD + d];
                ctx[(size_t)i * D + hh * HD + d] = sat(rs(acc, 16 + f_v - f_c), bits_c);
            }
        }
    }
    /* o_proj lands directly in the residual format */
    linear(ctx, T, &Ly->w[3], f_c, F_RES, 64, tmp);
    for (size_t i = 0; i < TD; i++) x[i] = sat(x[i] + tmp[i], 32);

    rmsnorm(x, T, Ly->ln2, (int)Ly->ln2f, f_h2, bits_h2, h);
    i64 *g_ = malloc((size_t)T * DFF * 8), *u_ = malloc((size_t)T * DFF * 8), *gated = malloc((size_t)T * DFF * 8);
    linear(h, T, &Ly->w[4], f_h2, (int)fm[17], bits_g, g_);
    linear(h, T, &Ly->w[5], f_h2, f_u, (int)fm[18], u_);
    int f_m = (bits_m == 32) ? (f_m0 < f_s + f_u ? f_m0 : f_s + f_u) : f_m0;
    for (size_t i = 0; i < (size_t)T * DFF; i++) {
        i64 sg = Ly->silu[g_[i] + 32768];
        (void)bits_s;
        gated[i] = sat(rs(sg * u_[i], f_s + f_u - f_m), bits_m);
    }
    linear(gated, T, &Ly->w[6], f_m, F_RES, 64, tmp);
    for (size_t i = 0; i < TD; i++) x[i] = sat(x[i] + tmp[i], 32);
    free(h); free(q); free(k); free(v); free(qr); free(kr); free(ctx); free(tmp); free(s); free(P); free(g_); free(u_); free(gated);
}

static size_t compare32(const char *name, const i64 *got, const int32_t *exp, size_t n) {
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < n; i++) if (got[i] != (i64)exp[i]) { if (!bad) first = i; bad++; }
    if (bad) printf("  MISMATCH %-12s %zu of %zu, first at %zu (got %lld expected %d)\n", name, bad, n, first, (long long)got[first], exp[first]);
    return bad;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: intref VECTORDIR\n"); return 2; }
    char path[1024];
    snprintf(path, sizeof path, "%s/model.bin", argc > 2 ? argv[2] : argv[1]); load_model(path);
    snprintf(path, sizeof path, "%s/ids.i64", argv[1]);
    FILE *f = fopen(path, "rb"); fseek(f, 0, SEEK_END); int T = (int)(ftell(f) / 8); fseek(f, 0, SEEK_SET);
    i64 *ids = xread(f, (size_t)T * 8); fclose(f);
    snprintf(path, sizeof path, "%s/exp_embed.i32", argv[1]); f = fopen(path, "rb"); int32_t *exp_embed = xread(f, (size_t)T * D * 4); fclose(f);
    snprintf(path, sizeof path, "%s/exp_resid_out.i32", argv[1]); f = fopen(path, "rb"); int32_t *exp_res = xread(f, (size_t)NL * T * D * 4); fclose(f);
    snprintf(path, sizeof path, "%s/exp_logits.i32", argv[1]); f = fopen(path, "rb"); int32_t *exp_log = xread(f, (size_t)T * V * 4); fclose(f);

    clock_t t0 = clock();
    size_t TD = (size_t)T * D, total_bad = 0;
    i64 *x = malloc(TD * 8);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < D; c++) {
            i64 id = ids[t];
            i64 xv = (i64)emb_q[(size_t)id * D + c] * emb_m[id];
            x[(size_t)t * D + c] = rs(xv, (int)(emb_n[id] - F_RES));
        }
    total_bad += compare32("embed", x, exp_embed, TD);
    for (int l = 0; l < NL; l++) {
        block(l, x, T);
        size_t bad = compare32("resid_out", x, exp_res + (size_t)l * TD, TD);
        printf("layer %2d: %s\n", l, bad ? "DIFFERS" : "exact");
        total_bad += bad;
        if (bad) break;
    }
    if (!total_bad) {
        i64 *y = malloc(TD * 8);
        rmsnorm(x, T, fg_q, (int)fg_f, (int)f_y, 16, y);
        i64 *logits = malloc((size_t)T * V * 8);
        for (int t = 0; t < T; t++)
            for (int n = 0; n < V; n++) {
                const int8_t *wr = emb_q + (size_t)n * D;
                i64 acc = 0;
                for (int k = 0; k < D; k++) acc += y[(size_t)t * D + k] * (i64)wr[k];
                logits[(size_t)t * V + n] = rs(acc * emb_m[n], (int)(emb_n[n] + f_y - 16));
            }
        total_bad += compare32("logits", logits, exp_log, (size_t)T * V);
        printf("logits: %s\n", total_bad ? "DIFFERS" : "exact");
    }
    printf("%s (%d tokens, %.1f s)\n", total_bad ? "FAIL" : "ALL STATES EXACT", T, (double)(clock() - t0) / CLOCKS_PER_SEC);
    return total_bad ? 1 : 0;
}
