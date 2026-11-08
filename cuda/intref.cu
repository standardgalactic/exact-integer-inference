/*
 * CUDA implementation of the SmolLM2-135M integer semantics.
 * Same model.bin and expected-state files as c/intref.c; every layer residual and the logits are
 * compared exactly against the exported vectors.
 *
 * Integer addition is associative, so any reduction order gives identical results; the kernels are
 * written for clarity, not speed.
 *
 * Build:  nvcc -O2 -arch=native -o intref_cuda intref.cu
 * Run:    ./intref_cuda ../vectors_w64_t128
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int64_t i64;
typedef __int128 i128;
typedef unsigned __int128 u128;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(3); } } while (0)

enum { V = 49152, D = 576, DFF = 1536, NH = 9, NKV = 3, HD = 64, NL = 30, DMAX = 12 * 128, ROPE_POS = 512 };
#define EPS_S 6184753LL
#define F_RES 15
#define F_SCORE 7
#define R_ISQRT 40

__device__ __forceinline__ i64 rs(i64 x, int n) {
    if (n <= 0) return x * ((i64)1 << (-n));
    return (x + ((i64)1 << (n - 1))) >> n;
}
__device__ __forceinline__ i64 sat(i64 x, int bits) {
    if (bits >= 64) return x;
    i64 hi = ((i64)1 << (bits - 1)) - 1, lo = -((i64)1 << (bits - 1));
    return x < lo ? lo : (x > hi ? hi : x);
}
__device__ u128 isqrt128(u128 v) {
    u128 res = 0, bit = (u128)1 << 126;
    while (bit > v) bit >>= 2;
    while (bit) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else res >>= 1;
        bit >>= 2;
    }
    return res;
}
/* floor(2^80 / t) for 2^22 <= t < 2^73 using only shifts, compares and subtracts (no 128-bit division) */
__device__ static u128 div_2_80(u128 t) {
    u128 rem = 1, q = 0;
    for (int i = 0; i < 80; i++) {
        rem <<= 1; q <<= 1;
        if (rem >= t) { rem -= t; q |= 1; }
    }
    return q;
}
__device__ i64 bsum(i64 *red, i64 v) {
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    i64 r = red[0]; __syncthreads(); return r;
}
__device__ i64 bmax(i64 *red, i64 v) {
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s && red[threadIdx.x + s] > red[threadIdx.x]) red[threadIdx.x] = red[threadIdx.x + s]; __syncthreads(); }
    i64 r = red[0]; __syncthreads(); return r;
}

__global__ void k_embed(const i64 *ids, const int8_t *eq, const i64 *em, const i64 *en, i64 *x, int T) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * D) return;
    int t = (int)(i / D), c = (int)(i % D);
    i64 id = ids[t];
    x[i] = rs((i64)eq[(size_t)id * D + c] * em[id], (int)(en[id] - F_RES));
}

__global__ void k_rmsnorm(const i64 *x, const i64 *gq, int fw, int f_out, int bits_out, i64 *out) {
    int t = blockIdx.x;
    const i64 *r = x + (size_t)t * D;
    __shared__ i64 red[256];
    __shared__ i64 Ush;
    i64 A = 0, B = 0, C = 0;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        i64 hi = r[c] >> 16, lo = r[c] & 0xFFFF;
        A += hi * hi; B += hi * lo; C += lo * lo;
    }
    i64 a = bsum(red, A), b = bsum(red, B), cc = bsum(red, C);
    if (threadIdx.x == 0) {
        u128 S = ((u128)(i128)a << 32) + ((u128)(i128)b << 17) + (u128)(i128)cc;
        u128 Tt = S + (u128)EPS_S;
        Ush = (i64)isqrt128(div_2_80(Tt));
    }
    __syncthreads();
    i64 U = Ush;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        i64 z = rs(r[c] * U, 16);
        out[(size_t)t * D + c] = sat(rs(z * gq[c] * 3, 21 + fw - f_out), bits_out);
    }
}

__global__ void k_linear(const i64 *a, const int8_t *W, const i64 *m, const i64 *nsh, int N, int K, int f_in, int f_out, int bits_out, i64 *out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= N) return;
    const i64 *ar = a + (size_t)t * K;
    const int8_t *wr = W + (size_t)n * K;
    i64 acc = 0;
    for (int k = 0; k < K; k++) acc += ar[k] * (i64)wr[k];
    out[(size_t)t * N + n] = sat(rs(acc * m[n], (int)(nsh[n] + f_in - f_out)), bits_out);
}

/* RoPE on q (nheads 9) and k (nheads 3) in one launch: index space T x 12 x 32 */
__global__ void k_rope(const i64 *q, const i64 *k, i64 *qr, i64 *kr, const i64 *C, const i64 *S, int T,
                       int f_q, int f_qr, int bits_qr, int f_k, int f_kr, int bits_kr) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * 12 * 32) return;
    int d = (int)(i % 32); size_t r = i / 32; int h = (int)(r % 12); int t = (int)(r / 12);
    const i64 *a; i64 *o; int f_in, f_o, bits;
    if (h < NH) { a = q + (size_t)t * D + h * HD; o = qr + (size_t)t * D + h * HD; f_in = f_q; f_o = f_qr; bits = bits_qr; }
    else { int g = h - NH; a = k + (size_t)t * 192 + g * HD; o = kr + (size_t)t * 192 + g * HD; f_in = f_k; f_o = f_kr; bits = bits_kr; }
    i64 c = C[(size_t)t * 32 + d], s = S[(size_t)t * 32 + d];
    i64 o1 = a[d] * c - a[d + 32] * s, o2 = a[d + 32] * c + a[d] * s;
    o[d] = sat(rs(o1, 30 + f_in - f_o), bits); o[d + 32] = sat(rs(o2, 30 + f_in - f_o), bits);
}

__global__ void k_attn(const i64 *qr, const i64 *kr, const i64 *v, const i64 *exp_tab, i64 *ctx, int T,
                       int shift_s, int f_v, int f_c, int bits_c) {
    extern __shared__ i64 sm[];
    i64 *s = sm, *P = sm + T, *red = sm + 2 * T;
    int i = blockIdx.x, hh = blockIdx.y, g = hh / 3;
    const i64 *qi = qr + (size_t)i * D + hh * HD;
    i64 lmax = -((i64)1 << 40);
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        const i64 *kj = kr + (size_t)j * 192 + g * HD;
        i64 acc = 0;
        for (int d = 0; d < HD; d++) acc += qi[d] * kj[d];
        s[j] = sat(rs(acc, shift_s), 16);
        if (s[j] > lmax) lmax = s[j];
    }
    i64 rowmax = bmax(red, lmax);
    i64 lz = 0;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        i64 d = rowmax - s[j];
        P[j] = d <= DMAX ? exp_tab[d] : 0;
        lz += P[j];
    }
    i64 Z = bsum(red, lz);
    for (int j = threadIdx.x; j <= i; j += blockDim.x) P[j] = (P[j] * 65536 + Z / 2) / Z;
    __syncthreads();
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        i64 acc = 0;
        for (int j = 0; j <= i; j++) acc += P[j] * v[(size_t)j * 192 + g * HD + d];
        ctx[(size_t)i * D + hh * HD + d] = sat(rs(acc, 16 + f_v - f_c), bits_c);
    }
}

__global__ void k_addsat(i64 *x, const i64 *y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = sat(x[i] + y[i], 32);
}

__global__ void k_gated(const i64 *g, const i64 *u, const i64 *silu, int f_s, int f_u, int f_m, int bits_m, i64 *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sat(rs(silu[g[i] + 32768] * u[i], f_s + f_u - f_m), bits_m);
}

/* ------------------------------------------------------------------ host side */
typedef struct { int8_t *q; i64 *m, *n; int N, K; } Lin;
typedef struct { i64 fm[24]; i64 ln1f, ln2f; i64 *ln1, *ln2; Lin w[7]; i64 *silu; } Layer;

static void *xread(FILE *f, size_t bytes) {
    void *p = malloc(bytes);
    if (!p || fread(p, 1, bytes, f) != bytes) { fprintf(stderr, "short read\n"); exit(2); }
    return p;
}
static void *to_dev(FILE *f, size_t bytes) {
    void *h = xread(f, bytes), *d;
    CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); free(h);
    return d;
}

static int8_t *emb_q; static i64 *emb_m, *emb_n, *fg_q, fg_f, f_y, *exp_tab, *rope_c, *rope_s;
static Layer L[NL];

static void load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    emb_q = (int8_t *)to_dev(f, (size_t)V * D);
    emb_m = (i64 *)to_dev(f, V * 8); emb_n = (i64 *)to_dev(f, V * 8);
    fg_q = (i64 *)to_dev(f, D * 8);
    if (fread(&fg_f, 8, 1, f) != 1 || fread(&f_y, 8, 1, f) != 1) exit(2);
    exp_tab = (i64 *)to_dev(f, (DMAX + 1) * 8);
    rope_c = (i64 *)to_dev(f, ROPE_POS * 32 * 8); rope_s = (i64 *)to_dev(f, ROPE_POS * 32 * 8);
    static const int Ns[7] = {576, 192, 192, 576, 1536, 1536, 576}, Ks[7] = {576, 576, 576, 576, 576, 576, 1536};
    for (int l = 0; l < NL; l++) {
        if (fread(L[l].fm, 8, 24, f) != 24) exit(2);
        L[l].ln1 = (i64 *)to_dev(f, D * 8); if (fread(&L[l].ln1f, 8, 1, f) != 1) exit(2);
        L[l].ln2 = (i64 *)to_dev(f, D * 8); if (fread(&L[l].ln2f, 8, 1, f) != 1) exit(2);
        for (int i = 0; i < 7; i++) {
            L[l].w[i].N = Ns[i]; L[l].w[i].K = Ks[i];
            L[l].w[i].q = (int8_t *)to_dev(f, (size_t)Ns[i] * Ks[i]);
            L[l].w[i].m = (i64 *)to_dev(f, Ns[i] * 8); L[l].w[i].n = (i64 *)to_dev(f, Ns[i] * 8);
        }
        L[l].silu = (i64 *)to_dev(f, 65536 * 8);
    }
    fclose(f);
}

static i64 *dalloc(size_t n) { i64 *p; CK(cudaMalloc(&p, n * 8)); return p; }
static void linear(const i64 *a, int T, const Lin *w, int f_in, int f_out, int bits_out, i64 *out) {
    dim3 grid((w->N + 127) / 128, T);
    k_linear<<<grid, 128>>>(a, w->q, w->m, w->n, w->N, w->K, f_in, f_out, bits_out, out);
    CK(cudaGetLastError());
}
static void rmsnorm(const i64 *x, int T, const i64 *gq, int fw, int f_out, int bits_out, i64 *out) {
    k_rmsnorm<<<T, 256>>>(x, gq, fw, f_out, bits_out, out); CK(cudaGetLastError());
}

static void block(int l, i64 *x, int T, i64 **buf) {
    Layer *Ly = &L[l]; const i64 *fm = Ly->fm;
    int bits_h = fm[0], f_h = fm[1], bits_q = fm[2], f_q = fm[3], bits_k = fm[4], f_k = fm[5], bits_v = fm[6], f_v = fm[7];
    int bits_qr = fm[8], f_qr = fm[9], bits_kr = fm[10], f_kr = fm[11], bits_c = fm[12], f_c = fm[13];
    int bits_h2 = fm[14], f_h2 = fm[15], bits_g = fm[16], f_g = fm[17], bits_u = fm[18], f_u = fm[19];
    int f_s = fm[21], bits_m = fm[22], f_m0 = fm[23];
    (void)f_g;
    i64 *h = buf[0], *q = buf[1], *k = buf[2], *v = buf[3], *qr = buf[4], *kr = buf[5], *ctx = buf[6], *tmp = buf[7];
    i64 *g_ = buf[8], *u_ = buf[9], *gated = buf[10];
    size_t TD = (size_t)T * D;
    rmsnorm(x, T, Ly->ln1, (int)Ly->ln1f, f_h, bits_h, h);
    linear(h, T, &Ly->w[0], f_h, f_q, bits_q, q);
    linear(h, T, &Ly->w[1], f_h, f_k, bits_k, k);
    linear(h, T, &Ly->w[2], f_h, f_v, bits_v, v);
    k_rope<<<(unsigned)(((size_t)T * 12 * 32 + 255) / 256), 256>>>(q, k, qr, kr, rope_c, rope_s, T, f_q, f_qr, bits_qr, f_k, f_kr, bits_kr);
    CK(cudaGetLastError());
    int shift_s = f_qr + f_kr + 3 - F_SCORE;
    size_t shm = ((size_t)2 * T + 128) * 8;
    k_attn<<<dim3(T, NH), 128, shm>>>(qr, kr, v, exp_tab, ctx, T, shift_s, f_v, f_c, bits_c);
    CK(cudaGetLastError());
    linear(ctx, T, &Ly->w[3], f_c, F_RES, 64, tmp);
    k_addsat<<<(unsigned)((TD + 255) / 256), 256>>>(x, tmp, TD);
    rmsnorm(x, T, Ly->ln2, (int)Ly->ln2f, f_h2, bits_h2, h);
    linear(h, T, &Ly->w[4], f_h2, (int)fm[17], bits_g, g_);
    linear(h, T, &Ly->w[5], f_h2, f_u, bits_u, u_);
    int f_m = (bits_m == 32) ? (f_m0 < f_s + f_u ? f_m0 : f_s + f_u) : f_m0;
    size_t TF = (size_t)T * DFF;
    k_gated<<<(unsigned)((TF + 255) / 256), 256>>>(g_, u_, Ly->silu, f_s, f_u, f_m, bits_m, gated, TF);
    CK(cudaGetLastError());
    linear(gated, T, &Ly->w[6], f_m, F_RES, 64, tmp);
    k_addsat<<<(unsigned)((TD + 255) / 256), 256>>>(x, tmp, TD);
    CK(cudaGetLastError());
}

static size_t compare32(const char *name, const i64 *dev, const int32_t *exp, size_t n) {
    i64 *h = (i64 *)malloc(n * 8);
    CK(cudaMemcpy(h, dev, n * 8, cudaMemcpyDeviceToHost));
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < n; i++) if (h[i] != (i64)exp[i]) { if (!bad) first = i; bad++; }
    if (bad) printf("  MISMATCH %-10s %zu of %zu, first at %zu (got %lld expected %d)\n", name, bad, n, first, (long long)h[first], exp[first]);
    free(h);
    return bad;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: intref_cuda VECTORDIR [MODELDIR]\n"); return 2; }
    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    if (!ndev) { fprintf(stderr, "no CUDA device\n"); return 2; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d, %.1f GiB)\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1073741824.0);
    char path[1024];
    snprintf(path, sizeof path, "%s/model.bin", argc > 2 ? argv[2] : argv[1]); load_model(path);
    snprintf(path, sizeof path, "%s/ids.i64", argv[1]);
    FILE *f = fopen(path, "rb"); fseek(f, 0, SEEK_END); int T = (int)(ftell(f) / 8); fseek(f, 0, SEEK_SET);
    i64 *ids_h = (i64 *)xread(f, (size_t)T * 8); fclose(f);
    if (T > ROPE_POS) { fprintf(stderr, "T > %d\n", ROPE_POS); return 2; }
    snprintf(path, sizeof path, "%s/exp_embed.i32", argv[1]); f = fopen(path, "rb"); int32_t *exp_embed = (int32_t *)xread(f, (size_t)T * D * 4); fclose(f);
    snprintf(path, sizeof path, "%s/exp_resid_out.i32", argv[1]); f = fopen(path, "rb"); int32_t *exp_res = (int32_t *)xread(f, (size_t)NL * T * D * 4); fclose(f);
    snprintf(path, sizeof path, "%s/exp_logits.i32", argv[1]); f = fopen(path, "rb"); int32_t *exp_log = (int32_t *)xread(f, (size_t)T * V * 4); fclose(f);

    size_t TD = (size_t)T * D, TF = (size_t)T * DFF;
    i64 *ids; CK(cudaMalloc(&ids, (size_t)T * 8)); CK(cudaMemcpy(ids, ids_h, (size_t)T * 8, cudaMemcpyHostToDevice));
    i64 *x = dalloc(TD);
    i64 *buf[11] = {dalloc(TD), dalloc(TD), dalloc((size_t)T * 192), dalloc((size_t)T * 192), dalloc(TD), dalloc((size_t)T * 192),
                    dalloc(TD), dalloc(TD > TF ? TD : TF), dalloc(TF), dalloc(TF), dalloc(TF)};
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);

    size_t total_bad = 0;
    k_embed<<<(unsigned)((TD + 255) / 256), 256>>>(ids, emb_q, emb_m, emb_n, x, T); CK(cudaGetLastError());
    total_bad += compare32("embed", x, exp_embed, TD);
    for (int l = 0; l < NL && !total_bad; l++) {
        block(l, x, T, buf);
        size_t bad = compare32("resid_out", x, exp_res + (size_t)l * TD, TD);
        printf("layer %2d: %s\n", l, bad ? "DIFFERS" : "exact");
        total_bad += bad;
    }
    if (!total_bad) {
        i64 *y = dalloc(TD), *logits = dalloc((size_t)T * V);
        rmsnorm(x, T, fg_q, (int)fg_f, (int)f_y, 16, y);
        Lin head = {emb_q, emb_m, emb_n, V, D};
        linear(y, T, &head, (int)f_y, 16, 64, logits);
        total_bad += compare32("logits", logits, exp_log, (size_t)T * V);
        printf("logits: %s\n", total_bad ? "DIFFERS" : "exact");
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    printf("%s (%d tokens, %.2f s on device incl. comparisons)\n", total_bad ? "FAIL" : "ALL STATES EXACT", T, ms / 1000.0);
    return total_bad ? 1 : 0;
}
