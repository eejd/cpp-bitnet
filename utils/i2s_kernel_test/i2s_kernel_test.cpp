// I2_S kernel correctness test (no model required).
//
// Unlike utils/test_gemm_kernel.sh (a timing benchmark on random data that never
// checks a result), every test here compares the kernel output against an
// independent scalar reference that decodes the *documented* I2_S layout:
//
//   * a row of K weights (K % 128 == 0) is stored as K/128 blocks of 32 bytes;
//   * weight k = B*128 + c*32 + gp  (B block, c in 0..3, gp in 0..31) lives in
//     byte B*32+gp at bit position 6-2c;
//   * code 0 / 1 / 2 encodes ternary -1 / 0 / +1;
//   * the f32 weight scale is stored after the packed data (nbytes = n/4 + 32).
//
// The kernel contract (see ggml-cpu.c / repack.cpp post-processing,
// `(tmp - act_sum) * ws / act_scale`) is that the integer kernels return
//   sum_k code_k * y_k      with code in {0,1,2}  (i.e. the +1 offset is NOT removed)
// and the caller subtracts sum_k y_k.
//
// Note: gemv/gemm are tested through the public entry points that ggml-cpu.c and repack.cpp call
// (ggml_gemv_i2_i8_s with 1 activation column, ggml_gemm_i2_i8_s with 4 columns); the raw
// ggml_vec_dot_i2_i8_s has two incompatible calling conventions in this tree, so it is not
// tested directly.
//
// Build variants (see run.sh):
//   default : links libggml-cpu, tests the code that ships in llama-cli
//             (quants.c vec_dot, ggml-cpu-i2s.c gemv/gemm, repack.cpp mul_mat path)
//   -DTEST_MAD : links src/ggml-bitnet-mad.cpp + ggml-cpu-i2s.c only (the kernels
//             that upstream PRs #586 / #633 patch); L1 tests only, no ggml graph.

#include <cassert>
#include <cstdarg>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml-quants.h"
#include "ggml-cpu-i2s.h"

#ifndef TEST_MAD
#include "ggml.h"
#include "ggml-cpu.h"
#endif

static int g_fail = 0, g_pass = 0;
static std::string g_json;

static void report(bool ok, const char * name, const char * detail) {
    (ok ? g_pass : g_fail)++;
    printf("%s  %-58s %s\n", ok ? "PASS" : "FAIL", name, detail);
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s{\"name\":\"%s\",\"pass\":%s,\"detail\":\"%s\"}", g_json.empty() ? "" : ",", name, ok ? "true" : "false", detail);
    g_json += buf;
}

// ---------------------------------------------------------------- reference

// canonical pack: ternary t[m*K+k] in {-1,0,1} -> packed bytes (K % 128 == 0)
static std::vector<uint8_t> pack_canonical(const std::vector<int8_t> & t, int M, int K) {
    std::vector<uint8_t> out((size_t) M * K / 4, 0);
    for (int m = 0; m < M; m++) {
        for (int B = 0; B < K / 128; B++) {
            for (int gp = 0; gp < 32; gp++) {
                auto code = [&](int c) { return (uint8_t)(t[(size_t) m * K + B * 128 + c * 32 + gp] + 1); };
                out[(size_t) m * (K / 4) + B * 32 + gp] = (code(0) << 6) | (code(1) << 4) | (code(2) << 2) | code(3);
            }
        }
    }
    return out;
}

static std::vector<int8_t> random_ternary(std::mt19937 & rng, size_t n) {
    std::vector<int8_t> t(n);
    std::uniform_int_distribution<int> d(-1, 1);
    for (auto & v : t) v = (int8_t) d(rng);
    return t;
}

static std::vector<int8_t> random_int8(std::mt19937 & rng, size_t n) {
    std::vector<int8_t> y(n);
    std::uniform_int_distribution<int> d(-128, 127);
    for (auto & v : y) v = (int8_t) d(rng);
    return y;
}

// sum_k (t_k + 1) * y_k   -- the kernel contract
static int32_t ref_code_dot(const int8_t * t, const int8_t * y, int K) {
    int32_t s = 0;
    for (int k = 0; k < K; k++) s += (int32_t)(t[k] + 1) * (int32_t) y[k];
    return s;
}

struct Case { int K, M, N; };

static const Case kCases[] = {
    {128, 4, 1},   {128, 8, 4},   {256, 8, 4},   {256, 12, 5},  {384, 16, 4},
    {2048, 32, 1}, {2048, 64, 4}, {2048, 36, 7}, {2560, 64, 4}, {2560, 68, 9},
    {4224, 8, 4},  {6912, 16, 4}, {6912, 20, 3},
};

static std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char * f, ...) {
    char b[512];
    va_list a; va_start(a, f); vsnprintf(b, sizeof(b), f, a); va_end(a);
    return b;
}

// --------------------------------------------------------------- L0 layout

#ifdef TEST_MAD
// libggml-base (needed for ggml_row_size / ggml_abort) references dequantize_row_i2_s, which
// ggml-bitnet-mad.cpp does not define (it lives in ggml-cpu/quants.c). Provide the documented
// canonical decode so the link succeeds; the L0 dequantize test is skipped for this variant.
void dequantize_row_i2_s(const uint8_t * x, float * y, int64_t n, const float i2_scale) {
    for (int64_t k = 0; k < n; k++) {
        int64_t B = k / 128, c = (k % 128) / 32, gp = k % 32;
        int code = (x[B * 32 + gp] >> (6 - 2 * c)) & 3;
        y[k] = i2_scale * (code == 0 ? -1.f : code == 2 ? 1.f : 0.f);
    }
}
#endif

static void test_dequantize_layout() {
#ifdef TEST_MAD
    return;
#endif
    std::mt19937 rng(1);
    for (int K : {128, 256, 2560}) {
        const int M = 3;
        auto t = random_ternary(rng, (size_t) M * K);
        auto packed = pack_canonical(t, M, K);
        std::vector<float> y((size_t) M * K);
        const float scale = 0.37f;
        long bad = 0; long first = -1;
        for (int m = 0; m < M; m++) {
            dequantize_row_i2_s(packed.data() + (size_t) m * K / 4, y.data() + (size_t) m * K, K, scale);
            for (int k = 0; k < K; k++) {
                if (y[(size_t) m * K + k] != scale * t[(size_t) m * K + k]) { if (!bad) first = (long) m * K + k; bad++; }
            }
        }
        report(bad == 0, fmt("L0 dequantize_row_i2_s canonical layout K=%d", K).c_str(),
               bad ? fmt("%ld/%d mismatches, first at %ld", bad, M * K, first).c_str() : "exact");
    }
}

// quantize_i2_s: for already-ternary floats (+-s, 0) the packed bytes must equal the canonical packing.
static void test_quantize_layout() {
    std::mt19937 rng(2);
    for (int K : {128, 256, 2560}) {
        const int M = 4;
        auto t = random_ternary(rng, (size_t) M * K);
        t[0] = 1; // quantize_i2_s takes the scale from the first non-zero |value|
        std::vector<float> w((size_t) M * K);
        const float s = 0.25f;
        for (size_t i = 0; i < w.size(); i++) w[i] = s * t[i];
        std::vector<uint8_t> q((size_t) M * K / 4 + 64, 0);
        quantize_i2_s(w.data(), q.data(), M, K, nullptr);
        auto ref = pack_canonical(t, M, K);
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < ref.size(); i++) if (q[i] != ref[i]) { if (!bad) first = i; bad++; }
        float sc; memcpy(&sc, q.data() + ref.size(), 4);
        bool ok = bad == 0 && sc == s;
        report(ok, fmt("L0 quantize_i2_s emits canonical layout K=%d", K).c_str(),
               ok ? "exact" : fmt("%zu/%zu bytes differ (first %zu), stored scale %g (want %g)", bad, ref.size(), first, sc, s).c_str());
    }
}

// --------------------------------------------------------------- L1 kernels

static void test_gemv() {
    std::mt19937 rng(4);
    for (auto c : kCases) {
        auto t = random_ternary(rng, (size_t) c.M * c.K);
        auto packed = pack_canonical(t, c.M, c.K);
        auto y = random_int8(rng, c.K);
        std::vector<float> s(c.M, -12345.f);
        ggml_gemv_i2_i8_s(c.K, s.data(), c.M, packed.data(), y.data(), 1, c.M);
        int bad = 0; double maxerr = 0;
        for (int m = 0; m < c.M; m++) {
            double e = fabs((double) s[m] - ref_code_dot(&t[(size_t) m * c.K], y.data(), c.K));
            if (e != 0) bad++;
            maxerr = fmax(maxerr, e);
        }
        report(bad == 0, fmt("L1 ggml_gemv_i2_i8_s K=%d M=%d N=1", c.K, c.M).c_str(),
               fmt("%d/%d outputs wrong, max|err|=%.0f", bad, c.M, maxerr).c_str());
    }
}

static void test_gemm() {
    std::mt19937 rng(5);
    for (auto c : kCases) {
        if (c.M % 4) continue; // callers align weight rows to 4
        for (int ncols : {4}) {  // ggml-cpu.c / repack.cpp call gemm with 4 activation columns
            auto t = random_ternary(rng, (size_t) c.M * c.K);
            auto packed = pack_canonical(t, c.M, c.K);
            auto y = random_int8(rng, (size_t) ncols * c.K);
            std::vector<float> s((size_t) ncols * c.M, -12345.f);
            ggml_gemm_i2_i8_s(c.K, s.data(), c.M, packed.data(), y.data(), ncols, c.M);
            int bad = 0; double maxerr = 0;
            for (int col = 0; col < ncols; col++) for (int m = 0; m < c.M; m++) {
                double e = fabs((double) s[(size_t) col * c.M + m] - ref_code_dot(&t[(size_t) m * c.K], &y[(size_t) col * c.K], c.K));
                if (e != 0) bad++;
                maxerr = fmax(maxerr, e);
            }
            report(bad == 0, fmt("L1 ggml_gemm_i2_i8_s K=%d M=%d cols=%d", c.K, c.M, ncols).c_str(),
                   fmt("%d/%d outputs wrong, max|err|=%.0f", bad, ncols * c.M, maxerr).c_str());
        }
    }
}

#ifndef TEST_MAD
// --------------------------------------------------------------- L2 mul_mat

static void emulate_quantize_row_i8(const float * x, int K, std::vector<int8_t> & q, float & scale, int32_t & sum) {
    float amax = 0.f;
    for (int i = 0; i < K; i++) amax = fmaxf(amax, fabsf(x[i]));
    scale = amax > 0.f ? 127.f / amax : 0.f;
    q.resize(K); sum = 0;
    for (int i = 0; i < K; i++) {
        int v = (int) roundf(x[i] * scale);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        q[i] = (int8_t) v; sum += v;
    }
}

static void test_mul_mat(int nth) {
    std::mt19937 rng(6 + nth);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto c : kCases) {
        const float ws = 0.0123f;
        auto t = random_ternary(rng, (size_t) c.M * c.K);
        auto packed = pack_canonical(t, c.M, c.K);

        std::vector<float> x((size_t) c.N * c.K);
        for (auto & v : x) v = nd(rng);
        for (int n = 0; n < c.N; n++) x[(size_t) n * c.K + (n * 7) % c.K] *= 20.f; // activation outlier

        struct ggml_init_params ip = { 64u * 1024 * 1024 + (size_t) c.M * c.K, nullptr, false };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_I2_S, c.K, c.M);
        struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.N);
        struct ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, Y);
        assert(ggml_nbytes(W) == (size_t) c.M * c.K / 4 + 32);
        memset(W->data, 0, ggml_nbytes(W));
        memcpy(W->data, packed.data(), packed.size());
        memcpy((uint8_t *) W->data + packed.size(), &ws, sizeof(float));
        memcpy(X->data, x.data(), x.size() * sizeof(float));
        ggml_graph_compute_with_ctx(ctx, gf, nth);
        const float * out = (const float *) Y->data;

        // reference 1: exact emulation of the documented pipeline (int8 activation quantization)
        // reference 2: float W @ x (no activation quantization) -- loose, catches garbage
        double max_rel_emul = 0, l2num = 0, l2den = 0;
        for (int n = 0; n < c.N; n++) {
            std::vector<int8_t> q; float sc; int32_t sum;
            emulate_quantize_row_i8(&x[(size_t) n * c.K], c.K, q, sc, sum);
            for (int m = 0; m < c.M; m++) {
                long long dot = 0; double fdot = 0;
                for (int k = 0; k < c.K; k++) {
                    dot += (long long) t[(size_t) m * c.K + k] * q[k];
                    fdot += (double) t[(size_t) m * c.K + k] * x[(size_t) n * c.K + k];
                }
                double emu = (double) dot * ws / sc;
                double ref = fdot * ws;
                double got = out[(size_t) n * c.M + m];
                max_rel_emul = fmax(max_rel_emul, fabs(got - emu) / (fabs(emu) + 1e-3 * fabs(ws) * sqrt((double) c.K)));
                l2num += (got - ref) * (got - ref);
                l2den += ref * ref;
            }
        }
        double rel_l2 = sqrt(l2num / (l2den + 1e-30));
        // tolerances (chosen before the first run): emulation must agree to float rounding (1e-4 relative);
        // vs. float reference the int8 activation quantization gives ~0.5% rel-L2, garbage gives >= ~100%.
        bool ok = max_rel_emul < 1e-4 && rel_l2 < 0.03;
        report(ok, fmt("L2 mul_mat I2_S K=%d M=%d N=%d threads=%d", c.K, c.M, c.N, nth).c_str(),
               fmt("max rel err vs emulation=%.3g, rel-L2 vs float W@x=%.3g", max_rel_emul, rel_l2).c_str());
        ggml_free(ctx);
    }
}
#endif

int main(int argc, char ** argv) {
    const char * json_path = nullptr;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];

    printf("compile-time features: ");
#if defined(__ARM_NEON)
    printf("NEON ");
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    printf("DOTPROD ");
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    printf("I8MM ");
#endif
#if defined(__ARM_FEATURE_SVE)
    printf("SVE ");
#endif
#if defined(__ARM_FEATURE_SME)
    printf("SME ");
#endif
#if defined(__AVX2__)
    printf("AVX2 ");
#endif
#ifdef TEST_MAD
    printf("| variant: src/ggml-bitnet-mad.cpp\n");
#else
    printf("| variant: linked libggml-cpu\n");
    ggml_cpu_init();
#endif

    test_dequantize_layout();
    test_quantize_layout();
    test_gemv();
    test_gemm();
#ifndef TEST_MAD
    test_mul_mat(1);
    test_mul_mat(4);
#endif

    printf("\nSUMMARY: %d passed, %d failed\n", g_pass, g_fail);
    if (json_path) {
        FILE * f = fopen(json_path, "w");
        if (f) { fprintf(f, "{\"passed\":%d,\"failed\":%d,\"tests\":[%s]}\n", g_pass, g_fail, g_json.c_str()); fclose(f); }
    }
    return g_fail ? 1 : 0;
}
