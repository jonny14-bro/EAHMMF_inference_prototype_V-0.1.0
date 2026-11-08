// qwen3moe.cpp - standalone CPU inference engine for Qwen3-MoE GGUF models.
// Target: Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf (arch "qwen3moe").
//
//   * GGUF v3 parser (mmap, zero-copy weights)
//   * byte-level BPE tokenizer with the Qwen2 pre-tokenizer (hand-written, exact Unicode L/N tables)
//   * dequant kernels: F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q5_K Q6_K
//   * transformer: GQA + QK-norm + NeoX RoPE, 128-expert top-8 MoE (softmax -> top-k -> renorm)
//   * batched prefill (weights are dequantized once per row and reused across the batch)
//   * sampler: repetition penalty, top-k, top-p, min-p, temperature, greedy
//   * ChatML chat mode with persistent KV cache
//
// Build: make        Run: ./qwen3moe -m model.gguf -i
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "eahmmf.h"
#include "unicode_tables.h"

#define MAXB 64  // max tokens per forward batch

[[noreturn]] static void die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

// ============================================================================
// fp16 / dequantization
// ============================================================================
static float g_f16[65536];
static float f16_slow(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
}
static inline float rd_f16(const uint8_t* p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return g_f16[h];
}

typedef void (*dq_fn)(const uint8_t*, float*);
struct TypeInfo {
    int blck = 0;
    size_t bytes = 0;
    dq_fn dq = nullptr;
    const char* name = "?";
};
static TypeInfo g_types[40];

static void dq_f32(const uint8_t* p, float* y) { memcpy(y, p, 128); }
static void dq_f16(const uint8_t* p, float* y) {
    for (int i = 0; i < 32; i++) y[i] = rd_f16(p + 2 * i);
}
static void dq_bf16(const uint8_t* p, float* y) {
    for (int i = 0; i < 32; i++) {
        uint16_t h;
        memcpy(&h, p + 2 * i, 2);
        uint32_t u = (uint32_t)h << 16;
        memcpy(&y[i], &u, 4);
    }
}
static void dq_q4_0(const uint8_t* b, float* y) {
    float d = rd_f16(b);
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; j++) {
        y[j] = ((qs[j] & 15) - 8) * d;
        y[j + 16] = ((qs[j] >> 4) - 8) * d;
    }
}
static void dq_q4_1(const uint8_t* b, float* y) {
    float d = rd_f16(b), m = rd_f16(b + 2);
    const uint8_t* qs = b + 4;
    for (int j = 0; j < 16; j++) {
        y[j] = (qs[j] & 15) * d + m;
        y[j + 16] = (qs[j] >> 4) * d + m;
    }
}
static void dq_q5_0(const uint8_t* b, float* y) {
    float d = rd_f16(b);
    uint32_t qh;
    memcpy(&qh, b + 2, 4);
    const uint8_t* qs = b + 6;
    for (int j = 0; j < 16; j++) {
        int xh0 = ((qh >> j) & 1) << 4, xh1 = ((qh >> (j + 16)) & 1) << 4;
        y[j] = (((qs[j] & 15) | xh0) - 16) * d;
        y[j + 16] = (((qs[j] >> 4) | xh1) - 16) * d;
    }
}
static void dq_q5_1(const uint8_t* b, float* y) {
    float d = rd_f16(b), m = rd_f16(b + 2);
    uint32_t qh;
    memcpy(&qh, b + 4, 4);
    const uint8_t* qs = b + 8;
    for (int j = 0; j < 16; j++) {
        int xh0 = ((qh >> j) & 1) << 4, xh1 = ((qh >> (j + 16)) & 1) << 4;
        y[j] = ((qs[j] & 15) | xh0) * d + m;
        y[j + 16] = ((qs[j] >> 4) | xh1) * d + m;
    }
}
static void dq_q8_0(const uint8_t* b, float* y) {
    float d = rd_f16(b);
    const int8_t* q = (const int8_t*)(b + 2);
    for (int i = 0; i < 32; i++) y[i] = d * q[i];
}
static inline void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
static void dq_q4_k(const uint8_t* b, float* y) {
    float d = rd_f16(b), mn = rd_f16(b + 2);
    const uint8_t* sc = b + 4;
    const uint8_t* q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is + 0, sc, &s, &m);
        float d1 = d * s, m1 = mn * m;
        scale_min_k4(is + 1, sc, &s, &m);
        float d2 = d * s, m2 = mn * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}
static void dq_q5_k(const uint8_t* b, float* y) {
    float d = rd_f16(b), mn = rd_f16(b + 2);
    const uint8_t* sc = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is + 0, sc, &s, &m);
        float d1 = d * s, m1 = mn * m;
        scale_min_k4(is + 1, sc, &s, &m);
        float d2 = d * s, m2 = mn * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}
static void dq_q6_k(const uint8_t* b, float* y) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = (const int8_t*)(b + 192);
    float d = rd_f16(b + 208);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static void init_types() {
    for (int i = 0; i < 65536; i++) g_f16[i] = f16_slow((uint16_t)i);
    auto set = [](int id, const char* n, int blck, size_t bytes, dq_fn f) { g_types[id] = {blck, bytes, f, n}; };
    set(0, "F32", 32, 128, dq_f32);
    set(1, "F16", 32, 64, dq_f16);
    set(2, "Q4_0", 32, 18, dq_q4_0);
    set(3, "Q4_1", 32, 20, dq_q4_1);
    set(6, "Q5_0", 32, 22, dq_q5_0);
    set(7, "Q5_1", 32, 24, dq_q5_1);
    set(8, "Q8_0", 32, 34, dq_q8_0);
    set(12, "Q4_K", 256, 144, dq_q4_k);
    set(13, "Q5_K", 256, 176, dq_q5_k);
    set(14, "Q6_K", 256, 210, dq_q6_k);
    set(30, "BF16", 32, 64, dq_bf16);
}

// ============================================================================
// GGUF
// ============================================================================
struct KV {
    int64_t i = 0;
    double f = 0;
    std::string s;
    bool is_arr = false;
    uint32_t elem_type = 0;
    std::vector<std::string> as;
    std::vector<int64_t> ai;
    std::vector<double> af;
};
struct Tensor {
    std::string name;
    uint32_t type = 0;
    int nd = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    uint64_t off = 0;
    uint64_t file_off = 0;  // absolute offset in the GGUF file
    const uint8_t* data = nullptr;
    size_t row_bytes = 0;
    int64_t nelem() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};
struct Reader {
    const uint8_t *p, *end;
    template <class T> T get() {
        if (p + sizeof(T) > end) die("GGUF truncated");
        T v;
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (p + n > end) die("GGUF truncated (string)");
        std::string s((const char*)p, n);
        p += n;
        return s;
    }
};
static void read_scalar(Reader& r, uint32_t t, KV& v) {
    switch (t) {
        case 0: v.i = r.get<uint8_t>(); break;
        case 1: v.i = r.get<int8_t>(); break;
        case 2: v.i = r.get<uint16_t>(); break;
        case 3: v.i = r.get<int16_t>(); break;
        case 4: v.i = r.get<uint32_t>(); break;
        case 5: v.i = r.get<int32_t>(); break;
        case 6: v.f = r.get<float>(); break;
        case 7: v.i = r.get<uint8_t>(); break;
        case 8: v.s = r.str(); break;
        case 10: v.i = (int64_t)r.get<uint64_t>(); break;
        case 11: v.i = r.get<int64_t>(); break;
        case 12: v.f = r.get<double>(); break;
        default: die("unsupported GGUF value type %u", t);
    }
}

struct GGUF {
    uint8_t* map = nullptr;
    size_t size = 0;
    std::map<std::string, KV> kv;
    std::map<std::string, Tensor> tensors;

    void load(const char* path, bool populate) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) die("cannot open %s", path);
        struct stat st;
        fstat(fd, &st);
        size = (size_t)st.st_size;
        map = (uint8_t*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE | (populate ? MAP_POPULATE : 0), fd, 0);
        if (map == MAP_FAILED) die("mmap failed");
        close(fd);
        Reader r{map, map + size};
        if (memcmp(map, "GGUF", 4) != 0) die("not a GGUF file");
        r.p += 4;
        uint32_t ver = r.get<uint32_t>();
        if (ver < 2 || ver > 3) die("unsupported GGUF version %u", ver);
        uint64_t nt = r.get<uint64_t>(), nkv = r.get<uint64_t>();
        for (uint64_t i = 0; i < nkv; i++) {
            std::string key = r.str();
            uint32_t t = r.get<uint32_t>();
            KV v;
            if (t == 9) {
                v.is_arr = true;
                v.elem_type = r.get<uint32_t>();
                uint64_t n = r.get<uint64_t>();
                if (v.elem_type == 9) die("nested arrays unsupported");
                for (uint64_t k = 0; k < n; k++) {
                    if (v.elem_type == 8) v.as.push_back(r.str());
                    else {
                        KV e;
                        read_scalar(r, v.elem_type, e);
                        if (v.elem_type == 6 || v.elem_type == 12) v.af.push_back(e.f);
                        else v.ai.push_back(e.i);
                    }
                }
            } else read_scalar(r, t, v);
            kv[key] = std::move(v);
        }
        std::vector<Tensor> ts(nt);
        for (auto& t : ts) {
            t.name = r.str();
            t.nd = (int)r.get<uint32_t>();
            if (t.nd < 1 || t.nd > 4) die("bad tensor dims");
            for (int d = 0; d < t.nd; d++) t.ne[d] = (int64_t)r.get<uint64_t>();
            t.type = r.get<uint32_t>();
            t.off = r.get<uint64_t>();
        }
        uint64_t align = 32;
        auto it = kv.find("general.alignment");
        if (it != kv.end()) align = (uint64_t)it->second.i;
        uint64_t pos = (uint64_t)(r.p - map);
        uint64_t data_start = (pos + align - 1) / align * align;
        for (auto& t : ts) {
            if (t.type >= 40 || !g_types[t.type].dq)
                die("tensor '%s' uses unsupported ggml type %u", t.name.c_str(), t.type);
            const TypeInfo& ti = g_types[t.type];
            if (t.ne[0] % ti.blck) die("tensor '%s': row length %lld not multiple of block %d", t.name.c_str(), (long long)t.ne[0], ti.blck);
            t.row_bytes = (size_t)(t.ne[0] / ti.blck) * ti.bytes;
            size_t nbytes = t.row_bytes * (size_t)(t.ne[1] * t.ne[2] * t.ne[3]);
            if (data_start + t.off + nbytes > size) die("tensor '%s' out of file bounds", t.name.c_str());
            t.file_off = data_start + t.off;
            t.data = map + t.file_off;
            tensors[t.name] = t;
        }
    }
    const KV* find(const std::string& k) const {
        auto it = kv.find(k);
        return it == kv.end() ? nullptr : &it->second;
    }
    int64_t geti(const std::string& k, int64_t def, bool required = false) const {
        auto p = find(k);
        if (!p) { if (required) die("missing metadata key %s", k.c_str()); return def; }
        return p->i;
    }
    double getf(const std::string& k, double def) const {
        auto p = find(k);
        if (!p) return def;
        return (p->elem_type == 6 || p->elem_type == 12 || p->f != 0) ? p->f : (double)p->i;
    }
    std::string gets(const std::string& k, const std::string& def = "") const {
        auto p = find(k);
        return p ? p->s : def;
    }
    const Tensor* tensor(const std::string& n, bool required = true) const {
        auto it = tensors.find(n);
        if (it == tensors.end()) { if (required) die("missing tensor %s", n.c_str()); return nullptr; }
        return &it->second;
    }
};

// ============================================================================
// Math kernels
// ============================================================================
static inline float dotf(const float* a, const float* b, int n) {
    float s = 0;
#pragma omp simd reduction(+ : s)
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

using eahmmf::WeightView;
static WeightView view_of(const Tensor& t) {
    WeightView v;
    v.data = t.data;
    v.bytes = t.row_bytes * (size_t)(t.ne[1] * t.ne[2] * t.ne[3]);
    v.type = t.type;
    v.ne0 = t.ne[0];
    v.ne1 = t.ne[1] * t.ne[2] * t.ne[3];
    v.row_bytes = t.row_bytes;
    return v;
}

// out[t*ldo + r] = dot(W.row(r), x[t*ldx ..]) for r in [0,W.ne1), t in [0,nb).
// The kernel only sees a WeightView: it does not know whether the bytes live in mmap, RAM or VRAM.
static void matmul(const WeightView& W, const float* x, int nb, int ldx, float* out, int ldo) {
    const TypeInfo& ti = g_types[W.type];
    const int blck = ti.blck;
    const int64_t nblk = W.ne0 / blck;
    const int64_t nrows = W.ne1;
    const int64_t row0 = 0;
#pragma omp parallel
    {
        float tmp[256];
#pragma omp for schedule(dynamic, 16)
        for (int64_t r = 0; r < nrows; r++) {
            const uint8_t* wr = W.data + (size_t)(row0 + r) * W.row_bytes;
            float acc[MAXB];
            for (int t = 0; t < nb; t++) acc[t] = 0.f;
            for (int64_t b = 0; b < nblk; b++) {
                ti.dq(wr + (size_t)b * ti.bytes, tmp);
                const float* xb = x + b * blck;
                for (int t = 0; t < nb; t++) acc[t] += dotf(tmp, xb + (size_t)t * ldx, blck);
            }
            for (int t = 0; t < nb; t++) out[(size_t)t * ldo + r] = acc[t];
        }
    }
}

#ifdef EAHMMF_CUDA_COMPUTE
extern "C" bool eahmmf_cuda_matmul(const uint8_t*, int, int, int, size_t,
                                   const float*, int, int, float*, int);
extern "C" bool eahmmf_cuda_matmul_pair(const uint8_t*, int, int, int, size_t, const uint8_t*, int, int, int, size_t, const float*, int, int, float*, int, float*, int);
static int g_shadow = 0;
static int g_cuda_compute = 0;
static long g_sh_calls = 0, g_sh_bad = 0, g_sh_skip = 0;
static double g_sh_max_rel = 0;
struct ShadowReport {
    ~ShadowReport() {
        if (g_shadow)
            fprintf(stderr, "cuda-shadow: calls=%ld bad=%ld skipped=%ld max_rel_err=%.3g\n",
                    g_sh_calls, g_sh_bad, g_sh_skip, g_sh_max_rel);
    }
} g_shadow_report;

static void matmul_shadow(const WeightView& W, const float* x, int nb, int ldx, float* out, int ldo) {
    matmul(W, x, nb, ldx, out, ldo);  // CPU result stays authoritative
    if (W.type != 8 && W.type != 12 && W.type != 13 && W.type != 14) { g_sh_skip++; return; }
    uint8_t* dw = nullptr;
    if (cudaMalloc(&dw, W.bytes) != cudaSuccess) { g_sh_skip++; return; }
    std::vector<float> g((size_t)nb * ldo, 0.f);
    bool ok = cudaMemcpy(dw, W.data, W.bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              eahmmf_cuda_matmul(dw, (int)W.type, (int)W.ne0, (int)W.ne1, W.row_bytes,
                                 x, nb, ldx, g.data(), ldo);
    cudaFree(dw);
    g_sh_calls++;
    if (!ok) { g_sh_bad++; return; }
    double maxerr = 0, maxout = 1e-3;
    for (int t = 0; t < nb; t++)
        for (int r = 0; r < (int)W.ne1; r++) {
            double c = out[(size_t)t * ldo + r], d = g[(size_t)t * ldo + r];
            maxerr = std::max(maxerr, std::fabs(c - d));
            maxout = std::max(maxout, std::fabs(c));
        }
    double rel = maxerr / maxout;
    g_sh_max_rel = std::max(g_sh_max_rel, rel);
    if (!(rel < 1e-4)) g_sh_bad++;
}
#endif

#ifdef EAHMMF_CUDA_COMPUTE
static long g_gpu_calls = 0, g_gpu_fallbacks = 0, g_pair_calls = 0;
struct ComputeReport {
    ~ComputeReport() {
        if (g_cuda_compute)
            fprintf(stderr, "cuda-compute: gpu_calls=%ld cpu_fallbacks=%ld pair_calls=%ld\n", g_gpu_calls, g_gpu_fallbacks, g_pair_calls);
    }
} g_compute_report;

static bool gpu_matmul(const WeightView& W, const float* x, int nb, int ldx, float* out, int ldo) {
    if (!W.dev || !(W.type == 8 || W.type == 12 || W.type == 13 || W.type == 14)) return false;
    return eahmmf_cuda_matmul(W.dev, (int)W.type, (int)W.ne0, (int)W.ne1, W.row_bytes, x, nb, ldx, out, ldo);
}

static bool gpu_matmul_pair(
    const WeightView& G, const WeightView& U,
    const float* x, int nb, int ldx,
    float* gout, int ldg, float* uout, int ldu)
{
    auto supported = [](const WeightView& W) {
        return W.dev &&
            (W.type == 8 || W.type == 12 || W.type == 13 || W.type == 14);
    };
    if (!supported(G) || !supported(U)) return false;
    if (G.ne0 != U.ne0 || G.ne1 != U.ne1) return false;
    return eahmmf_cuda_matmul_pair(
        G.dev, (int)G.type, (int)G.ne0, (int)G.ne1, G.row_bytes,
        U.dev, (int)U.type, (int)U.ne0, (int)U.ne1, U.row_bytes,
        x, nb, ldx, gout, ldg, uout, ldu);
}

// --cuda-compute + --cuda-shadow: the GPU result is in use; check it against the CPU kernel.
static void shadow_compare_cpu(const WeightView& W, const float* x, int nb, int ldx, const float* gpu_out, int ldo) {
    std::vector<float> c((size_t)nb * ldo, 0.f);
    matmul(W, x, nb, ldx, c.data(), ldo);
    double maxerr = 0, maxout = 1e-3;
    for (int t = 0; t < nb; t++)
        for (int r = 0; r < (int)W.ne1; r++) {
            double a = c[(size_t)t * ldo + r], b = gpu_out[(size_t)t * ldo + r];
            maxerr = std::max(maxerr, std::fabs(a - b));
            maxout = std::max(maxout, std::fabs(a));
        }
    double rel = maxerr / maxout;
    g_sh_calls++;
    g_sh_max_rel = std::max(g_sh_max_rel, rel);
    if (!(rel < 1e-4)) g_sh_bad++;
}
#endif

// Expert FFN matmuls go through here; with --cuda-shadow the GPU kernel is cross-checked against the CPU result.
static void expert_matmul(const WeightView& W, const float* x, int nb, int ldx, float* out, int ldo) {
#ifdef EAHMMF_CUDA_COMPUTE
    if (g_cuda_compute) {
        if (gpu_matmul(W, x, nb, ldx, out, ldo)) {
            g_gpu_calls++;
            if (g_shadow) shadow_compare_cpu(W, x, nb, ldx, out, ldo);
            return;
        }
        g_gpu_fallbacks++;  // not VRAM-resident (or no kernel for this type): CPU path below
    }
    if (g_shadow) { matmul_shadow(W, x, nb, ldx, out, ldo); return; }
#endif
    matmul(W, x, nb, ldx, out, ldo);
}

static void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float sc = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * sc * w[i];
}

static std::vector<float> tensor_f32(const Tensor& t) {
    const TypeInfo& ti = g_types[t.type];
    std::vector<float> out((size_t)t.nelem());
    int64_t rows = t.ne[1] * t.ne[2] * t.ne[3], nblk = t.ne[0] / ti.blck;
    for (int64_t r = 0; r < rows; r++)
        for (int64_t b = 0; b < nblk; b++)
            ti.dq(t.data + r * t.row_bytes + b * ti.bytes, &out[(size_t)(r * t.ne[0] + b * ti.blck)]);
    return out;
}

// ============================================================================
// Tokenizer (GPT-2 byte-level BPE + Qwen2 pre-tokenizer)
// ============================================================================
static bool in_ranges(const uint32_t (*r)[2], size_t n, uint32_t c) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (c < r[mid][0]) hi = mid;
        else if (c > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}
static bool is_letter(uint32_t c) {
    if (c < 128) return (c | 32) >= 'a' && (c | 32) <= 'z';
    return in_ranges(UNI_L, UNI_L_N, c);
}
static bool is_number(uint32_t c) {
    if (c < 128) return c >= '0' && c <= '9';
    return in_ranges(UNI_N, UNI_N_N, c);
}
static bool is_space(uint32_t c) {
    return (c >= 9 && c <= 13) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
static uint32_t utf8_next(const std::string& s, size_t i, int& len) {
    uint8_t c = (uint8_t)s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > s.size()) { len = 1; return 0xFFFD; }
    uint32_t cp = n == 1 ? c : c & (0xFF >> (n + 1));
    for (int k = 1; k < n; k++) {
        uint8_t cc = (uint8_t)s[i + k];
        if ((cc & 0xC0) != 0x80) { len = 1; return 0xFFFD; }
        cp = (cp << 6) | (cc & 0x3F);
    }
    len = n;
    return cp;
}
static std::string utf8_encode(uint32_t c) {
    std::string o;
    if (c < 0x80) o += (char)c;
    else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    return o;
}

// Equivalent of the Qwen2 regex:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
static void pretokenize(const std::string& s, std::vector<std::pair<size_t, size_t>>& out) {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
    for (size_t i = 0; i < s.size();) {
        int len;
        cp.push_back(utf8_next(s, i, len));
        off.push_back(i);
        i += len;
    }
    off.push_back(s.size());
    const size_t n = cp.size();
    auto L = [&](size_t k) { return k < n && is_letter(cp[k]); };
    auto N = [&](size_t k) { return k < n && is_number(cp[k]); };
    auto WS = [&](size_t k) { return k < n && is_space(cp[k]); };
    auto NL = [&](size_t k) { return k < n && (cp[k] == '\r' || cp[k] == '\n'); };
    auto SYM = [&](size_t k) { return k < n && !is_space(cp[k]) && !is_letter(cp[k]) && !is_number(cp[k]); };
    auto lower = [](uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
    size_t i = 0;
    while (i < n) {
        size_t j = 0;
        if (cp[i] == '\'' && i + 1 < n) {  // alt 1
            uint32_t c1 = lower(cp[i + 1]);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') j = i + 2;
            else if (i + 2 < n) {
                uint32_t c2 = lower(cp[i + 2]);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) j = i + 3;
            }
        }
        if (!j) {  // alt 2
            size_t k = 0;
            if (L(i)) k = i;
            else if (!NL(i) && !N(i) && L(i + 1)) k = i + 1;
            if (k || L(i)) {
                while (L(k)) k++;
                j = k;
            }
        }
        if (!j && N(i)) j = i + 1;  // alt 3
        if (!j) {                   // alt 4
            size_t k = i;
            bool ok = false;
            if (cp[i] == ' ' && SYM(i + 1)) { k = i + 1; ok = true; }
            else if (SYM(i)) ok = true;
            if (ok) {
                while (SYM(k)) k++;
                while (NL(k)) k++;
                j = k;
            }
        }
        if (!j && WS(i)) {  // alts 5,6,7
            size_t e = i;
            while (WS(e)) e++;
            size_t ln = 0;
            for (size_t k = e; k > i; k--) if (NL(k - 1)) { ln = k; break; }
            if (ln) j = ln;
            else if (e == n) j = e;
            else if (e - i >= 2) j = e - 1;
            else j = e;
        }
        if (!j) j = i + 1;
        out.push_back({off[i], off[j]});
        i = j;
    }
}

struct Tokenizer {
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> tok2id;
    std::unordered_map<uint64_t, std::pair<int, int>> merges;  // (a<<32|b) -> (rank, merged id)
    std::vector<std::pair<std::string, int>> specials;
    std::vector<std::string> piece;  // decoded bytes per id
    std::string byte2str[256];
    int eos = -1, im_start = -1, im_end = -1, endoftext = -1, nl = -1;

    void load(const GGUF& g) {
        const KV* toks = g.find("tokenizer.ggml.tokens");
        if (!toks || !toks->is_arr) die("no tokenizer.ggml.tokens in GGUF");
        std::string model = g.gets("tokenizer.ggml.model");
        if (model != "gpt2") fprintf(stderr, "warning: tokenizer model '%s' (expected gpt2)\n", model.c_str());
        std::string pre = g.gets("tokenizer.ggml.pre");
        if (pre != "qwen2" && pre != "qwen35" && !pre.empty()) fprintf(stderr, "warning: tokenizer pre-type '%s' (using qwen2 splitter)\n", pre.c_str());
        vocab = toks->as;
        const KV* types = g.find("tokenizer.ggml.token_type");
        std::unordered_map<uint32_t, int> cp2byte;
        {  // GPT-2 bytes_to_unicode
            std::vector<int> bs;
            for (int b = 33; b <= 126; b++) bs.push_back(b);
            for (int b = 161; b <= 172; b++) bs.push_back(b);
            for (int b = 174; b <= 255; b++) bs.push_back(b);
            bool in[256] = {false};
            for (int b : bs) in[b] = true;
            int n = 0;
            for (int b = 0; b < 256; b++) {
                uint32_t c = in[b] ? (uint32_t)b : (uint32_t)(256 + n++);
                byte2str[b] = utf8_encode(c);
                cp2byte[c] = b;
            }
        }
        piece.resize(vocab.size());
        for (size_t i = 0; i < vocab.size(); i++) {
            tok2id.emplace(vocab[i], (int)i);
            int ty = (types && i < types->ai.size()) ? (int)types->ai[i] : 1;
            if (ty == 3 || ty == 4) specials.push_back({vocab[i], (int)i});
            std::string out;
            bool ok = true;
            for (size_t k = 0; k < vocab[i].size();) {
                int len;
                uint32_t c = utf8_next(vocab[i], k, len);
                auto it = cp2byte.find(c);
                if (it == cp2byte.end()) { ok = false; break; }
                out += (char)it->second;
                k += len;
            }
            piece[i] = ok ? out : vocab[i];
        }
        std::sort(specials.begin(), specials.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
        const KV* mg = g.find("tokenizer.ggml.merges");
        if (mg) {
            for (size_t r = 0; r < mg->as.size(); r++) {
                const std::string& m = mg->as[r];
                size_t sp = m.find(' ', 1);
                if (sp == std::string::npos) continue;
                auto a = tok2id.find(m.substr(0, sp)), b = tok2id.find(m.substr(sp + 1));
                auto c = tok2id.find(m.substr(0, sp) + m.substr(sp + 1));
                if (a == tok2id.end() || b == tok2id.end() || c == tok2id.end()) continue;
                merges.emplace(((uint64_t)a->second << 32) | (uint32_t)b->second, std::make_pair((int)r, c->second));
            }
        }
        auto lookup = [&](const char* s) { auto it = tok2id.find(s); return it == tok2id.end() ? -1 : it->second; };
        im_start = lookup("<|im_start|>");
        im_end = lookup("<|im_end|>");
        endoftext = lookup("<|endoftext|>");
        nl = lookup(byte2str[10].c_str());
        eos = (int)g.geti("tokenizer.ggml.eos_token_id", -1);
        for (int b = 0; b < 256; b++)
            if (!tok2id.count(byte2str[b])) die("vocab lacks byte token for 0x%02x", b);
    }

    void bpe(const std::string& s, size_t a, size_t b, std::vector<int>& out) const {
        std::vector<int> sy;
        for (size_t i = a; i < b; i++) sy.push_back(tok2id.at(byte2str[(uint8_t)s[i]]));
        while (sy.size() > 1) {
            int best = INT32_MAX;
            size_t bi = 0;
            int bid = 0;
            for (size_t i = 0; i + 1 < sy.size(); i++) {
                auto it = merges.find(((uint64_t)sy[i] << 32) | (uint32_t)sy[i + 1]);
                if (it != merges.end() && it->second.first < best) { best = it->second.first; bi = i; bid = it->second.second; }
            }
            if (best == INT32_MAX) break;
            sy[bi] = bid;
            sy.erase(sy.begin() + bi + 1);
        }
        out.insert(out.end(), sy.begin(), sy.end());
    }
    void encode_plain(const std::string& s, std::vector<int>& out) const {
        std::vector<std::pair<size_t, size_t>> parts;
        pretokenize(s, parts);
        for (auto& p : parts) bpe(s, p.first, p.second, out);
    }
    std::vector<int> encode(const std::string& s, bool parse_special) const {
        std::vector<int> out;
        if (!parse_special) { encode_plain(s, out); return out; }
        size_t pos = 0;
        while (pos < s.size()) {
            size_t best = std::string::npos;
            int bid = -1;
            size_t blen = 0;
            for (auto& sp : specials) {
                size_t f = s.find(sp.first, pos);
                if (f != std::string::npos && (f < best || (f == best && sp.first.size() > blen))) { best = f; bid = sp.second; blen = sp.first.size(); }
            }
            if (best == std::string::npos) { encode_plain(s.substr(pos), out); break; }
            if (best > pos) encode_plain(s.substr(pos, best - pos), out);
            out.push_back(bid);
            pos = best + blen;
        }
        return out;
    }
};

// ============================================================================
// Engine
// ============================================================================
struct Layer {
    std::vector<float> attn_norm, q_norm, k_norm, ffn_norm;
    WeightView wq, wk, wv, wo, router;
    const Tensor *gate, *up, *down;  // stacked expert tensors (source of ExpertDescriptors)
};

struct Engine {
    GGUF g;
    int n_layer, D, n_head, n_kv, hd, QD, KD, n_expert, n_used, FF, n_vocab, n_ctx;
    float eps, rope_base;
    const Tensor *tok_embd, *output;
    WeightView out_v;
    std::vector<float> out_norm, inv_freq;
    std::vector<Layer> layers;
    float *kc = nullptr, *vc = nullptr;
    // activations
    std::vector<float> x, xn, q, k, v, att, tmp, rl, gbuf, ubuf, ybuf, xs, rcos, rsin;
    // EAHMMF expert residency layer (dev must outlive store)
    std::unique_ptr<eahmmf::DeviceBackend> dev;
    std::unique_ptr<eahmmf::ExpertStore> store;
    // routing telemetry (separate from physical residency telemetry)
    bool trace_routing = false;
    std::vector<int> trace_ids;
    std::vector<float> trace_w;

    ~Engine() { free(kc); free(vc); }

    // Reconfiguration requires external synchronization with inference.
    // Outstanding expert pins are rejected before either owner is replaced.
    bool init_experts(const eahmmf::Config& cfg, std::unique_ptr<eahmmf::DeviceBackend> d = nullptr) {
        if (store && store->stats().pinned_now != 0) {
            fprintf(stderr,
                    "eahmmf: cannot reinitialize expert store with %llu outstanding pin(s)\n",
                    static_cast<unsigned long long>(store->stats().pinned_now));
            return false;
        }

        store.reset();
        dev = d ? std::move(d) : std::unique_ptr<eahmmf::DeviceBackend>(new eahmmf::SimDevice());
        std::vector<std::array<eahmmf::TensorRef, 3>> refs(n_layer);
        for (int l = 0; l < n_layer; l++) {
            const Tensor* ts[3] = {layers[l].gate, layers[l].up, layers[l].down};
            for (int c = 0; c < 3; c++) {
                eahmmf::TensorRef& r = refs[l][c];
                r.file_offset = ts[c]->file_off;
                r.type = ts[c]->type;
                r.ne0 = ts[c]->ne[0];
                r.ne1 = ts[c]->ne[1];
                r.ne2 = ts[c]->ne[2];
                r.row_bytes = ts[c]->row_bytes;
            }
        }

        static bool printed_expert_types = false;
        if (!printed_expert_types) {
            for (int l = 0; l < n_layer; ++l) {
                const Tensor* ts[3] = {
                    layers[l].gate, layers[l].up, layers[l].down
                };

                for (int c = 0; c < 3; ++c) {
                    const TypeInfo& ti = g_types[ts[c]->type];

                    fprintf(stderr,
                            "[expert-format] layer=%d component=%s "
                            "type=%s id=%u ne0=%lld ne1=%lld row_bytes=%zu\n",
                            l,
                            c == 0 ? "gate" : c == 1 ? "up" : "down",
                            ti.name,
                            ts[c]->type,
                            (long long)ts[c]->ne[0],
                            (long long)ts[c]->ne[1],
                            ts[c]->row_bytes);
                }
            }
            printed_expert_types = true;
        }
        
        store.reset(new eahmmf::ExpertStore(g.map, g.size, n_layer, n_expert, refs, cfg, dev.get()));
        return true;
    }

    void load(const char* path, int ctx, bool populate) {
        g.load(path, populate);
        std::string arch = g.gets("general.architecture");
        if (arch != "qwen3moe") die("unsupported architecture '%s' (need qwen3moe)", arch.c_str());
        const std::string a = "qwen3moe.";
        n_layer = (int)g.geti(a + "block_count", 0, true);
        D = (int)g.geti(a + "embedding_length", 0, true);
        n_head = (int)g.geti(a + "attention.head_count", 0, true);
        n_kv = (int)g.geti(a + "attention.head_count_kv", 0, true);
        hd = (int)g.geti(a + "attention.key_length", D / n_head);
        n_expert = (int)g.geti(a + "expert_count", 0, true);
        n_used = (int)g.geti(a + "expert_used_count", 0, true);
        FF = (int)g.geti(a + "expert_feed_forward_length", 0, true);
        eps = (float)g.getf(a + "attention.layer_norm_rms_epsilon", 1e-6);
        rope_base = (float)g.getf(a + "rope.freq_base", 1e6);
        int n_ctx_train = (int)g.geti(a + "context_length", 4096);
        if (g.find(a + "rope.scaling.type") && g.gets(a + "rope.scaling.type") != "none")
            fprintf(stderr, "warning: rope scaling '%s' present in GGUF but not implemented\n", g.gets(a + "rope.scaling.type").c_str());
        QD = n_head * hd;
        KD = n_kv * hd;
        n_ctx = std::min(ctx, n_ctx_train);
        if (hd % 2 || n_head % n_kv) die("bad head config");
        if (D % 32 || FF % 32 || QD % 32) die("dimensions must be multiples of 32");

        tok_embd = g.tensor("token_embd.weight");
        output = g.tensor("output.weight", false);
        if (!output) output = tok_embd;  // tied embeddings
        out_v = view_of(*output);
        n_vocab = (int)output->ne[1];
        out_norm = tensor_f32(*g.tensor("output_norm.weight"));
        layers.resize(n_layer);
        for (int l = 0; l < n_layer; l++) {
            char b[96];
            auto T = [&](const char* s) { snprintf(b, sizeof b, "blk.%d.%s", l, s); return g.tensor(b); };
            Layer& L = layers[l];
            L.attn_norm = tensor_f32(*T("attn_norm.weight"));
            L.q_norm = tensor_f32(*T("attn_q_norm.weight"));
            L.k_norm = tensor_f32(*T("attn_k_norm.weight"));
            L.ffn_norm = tensor_f32(*T("ffn_norm.weight"));
            L.wq = view_of(*T("attn_q.weight"));
            L.wk = view_of(*T("attn_k.weight"));
            L.wv = view_of(*T("attn_v.weight"));
            L.wo = view_of(*T("attn_output.weight"));
            L.router = view_of(*T("ffn_gate_inp.weight"));
            L.gate = T("ffn_gate_exps.weight");
            L.up = T("ffn_up_exps.weight");
            L.down = T("ffn_down_exps.weight");
            if (L.gate->ne[2] != n_expert || L.gate->ne[1] != FF || L.down->ne[1] != D) die("unexpected expert tensor shape in layer %d", l);
        }
        inv_freq.resize(hd / 2);
        for (int i = 0; i < hd / 2; i++) inv_freq[i] = (float)pow((double)rope_base, -2.0 * i / hd);
        size_t kvn = (size_t)n_layer * n_ctx * KD;
        kc = (float*)calloc(kvn, sizeof(float));
        vc = (float*)calloc(kvn, sizeof(float));
        if (!kc || !vc) die("cannot allocate KV cache (%.2f GiB)", 2.0 * kvn * 4 / 1073741824.0);
        x.resize(MAXB * D); xn.resize(MAXB * D); q.resize(MAXB * QD); k.resize(MAXB * KD); v.resize(MAXB * KD);
        att.resize(MAXB * QD); tmp.resize(MAXB * D); rl.resize(MAXB * n_expert);
        gbuf.resize(MAXB * FF); ubuf.resize(MAXB * FF); ybuf.resize(MAXB * D); xs.resize(MAXB * D);
        rcos.resize(MAXB * hd / 2); rsin.resize(MAXB * hd / 2);
        if (!init_experts(eahmmf::Config()))
            die("cannot initialize default expert store");  // experts served from mmap (SSD tier)
        fprintf(stderr, "model: %d layers, d=%d, heads=%d/%d (hd=%d), experts=%d (top-%d, ff=%d), vocab=%d, ctx=%d (train %d), KV=%.2f GiB\n",
                n_layer, D, n_head, n_kv, hd, n_expert, n_used, FF, n_vocab, n_ctx, n_ctx_train, 2.0 * kvn * 4 / 1073741824.0);
    }

    void embed_row(int id, float* out) const {
        if (id < 0 || id >= tok_embd->ne[1]) die("token id %d out of range", id);
        const TypeInfo& ti = g_types[tok_embd->type];
        const uint8_t* wr = tok_embd->data + (size_t)id * tok_embd->row_bytes;
        for (int64_t b = 0; b < tok_embd->ne[0] / ti.blck; b++) ti.dq(wr + b * ti.bytes, out + b * ti.blck);
    }

    // Evaluate nb<=MAXB tokens at positions p0..p0+nb-1. If logits != null, writes logits of the last token.
    void forward(const int* toks, int nb, int p0, float* logits) {
        if (nb < 1 || nb > MAXB) die("bad batch size");
        if (p0 + nb > n_ctx) die("context overflow (%d > %d)", p0 + nb, n_ctx);
        const int half = hd / 2, grp = n_head / n_kv;
        const float scale = 1.0f / sqrtf((float)hd);
        for (int t = 0; t < nb; t++) embed_row(toks[t], &x[(size_t)t * D]);
        for (int t = 0; t < nb; t++)
            for (int i = 0; i < half; i++) {
                double ang = (double)(p0 + t) * inv_freq[i];
                rcos[t * half + i] = (float)cos(ang);
                rsin[t * half + i] = (float)sin(ang);
            }
        for (int l = 0; l < n_layer; l++) {
            Layer& L = layers[l];
            // ---- attention ----
            for (int t = 0; t < nb; t++) rmsnorm(&x[(size_t)t * D], L.attn_norm.data(), &xn[(size_t)t * D], D, eps);
            matmul(L.wq, xn.data(), nb, D, q.data(), QD);
            matmul(L.wk, xn.data(), nb, D, k.data(), KD);
            matmul(L.wv, xn.data(), nb, D, v.data(), KD);
            for (int t = 0; t < nb; t++) {
                for (int h = 0; h < n_head; h++) {
                    float* p = &q[(size_t)t * QD + h * hd];
                    rmsnorm(p, L.q_norm.data(), p, hd, eps);
                    for (int i = 0; i < half; i++) {
                        float a = p[i], b = p[i + half], c = rcos[t * half + i], s = rsin[t * half + i];
                        p[i] = a * c - b * s;
                        p[i + half] = a * s + b * c;
                    }
                }
                for (int h = 0; h < n_kv; h++) {
                    float* p = &k[(size_t)t * KD + h * hd];
                    rmsnorm(p, L.k_norm.data(), p, hd, eps);
                    for (int i = 0; i < half; i++) {
                        float a = p[i], b = p[i + half], c = rcos[t * half + i], s = rsin[t * half + i];
                        p[i] = a * c - b * s;
                        p[i + half] = a * s + b * c;
                    }
                }
                size_t o = ((size_t)l * n_ctx + p0 + t) * KD;
                memcpy(kc + o, &k[(size_t)t * KD], KD * sizeof(float));
                memcpy(vc + o, &v[(size_t)t * KD], KD * sizeof(float));
            }
            const float* lk = kc + (size_t)l * n_ctx * KD;
            const float* lv = vc + (size_t)l * n_ctx * KD;
#pragma omp parallel
            {
                std::vector<float> sc(p0 + nb);
#pragma omp for collapse(2) schedule(dynamic)
                for (int t = 0; t < nb; t++)
                    for (int h = 0; h < n_head; h++) {
                        const int np = p0 + t + 1, kvh = h / grp;
                        const float* qh = &q[(size_t)t * QD + h * hd];
                        float mx = -INFINITY;
                        for (int j = 0; j < np; j++) {
                            float s = dotf(qh, lk + (size_t)j * KD + kvh * hd, hd) * scale;
                            sc[j] = s;
                            mx = std::max(mx, s);
                        }
                        float sum = 0;
                        for (int j = 0; j < np; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                        float inv = 1.0f / sum;
                        float* o = &att[(size_t)t * QD + h * hd];
                        for (int d = 0; d < hd; d++) o[d] = 0;
                        for (int j = 0; j < np; j++) {
                            float w = sc[j] * inv;
                            const float* vj = lv + (size_t)j * KD + kvh * hd;
#pragma omp simd
                            for (int d = 0; d < hd; d++) o[d] += w * vj[d];
                        }
                    }
            }
            matmul(L.wo, att.data(), nb, QD, tmp.data(), D);
            for (int i = 0; i < nb * D; i++) x[i] += tmp[i];

            // ---- MoE FFN ----
            for (int t = 0; t < nb; t++) rmsnorm(&x[(size_t)t * D], L.ffn_norm.data(), &xn[(size_t)t * D], D, eps);
            matmul(L.router, xn.data(), nb, D, rl.data(), n_expert);
            std::vector<std::vector<std::pair<int, float>>> by_exp(n_expert);
            std::vector<int> idx(n_expert);
            for (int t = 0; t < nb; t++) {
                float* r = &rl[(size_t)t * n_expert];
                float mx = *std::max_element(r, r + n_expert), sum = 0;
                for (int e = 0; e < n_expert; e++) { r[e] = expf(r[e] - mx); sum += r[e]; }
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + n_used, idx.end(), [&](int a, int b) { return r[a] > r[b] || (r[a] == r[b] && a < b); });
                float tot = 0;
                for (int i = 0; i < n_used; i++) tot += r[idx[i]];
                for (int i = 0; i < n_used; i++) {
                    by_exp[idx[i]].push_back({t, r[idx[i]] / tot});
                    if (trace_routing) { trace_ids.push_back(idx[i]); trace_w.push_back(r[idx[i]] / tot); }
                }
            }
            memset(tmp.data(), 0, sizeof(float) * nb * D);
            for (int e = 0; e < n_expert; e++) {
                auto& lst = by_exp[e];
                if (lst.empty()) continue;
                int m = (int)lst.size();
                for (int i = 0; i < m; i++) memcpy(&xs[(size_t)i * D], &xn[(size_t)lst[i].first * D], D * sizeof(float));
                // Lease owns the pin through all weight-consuming operations.
                eahmmf::ExpertLease lease = store->acquire_lease(l, e);
                if (!lease) die("expert (layer %d, expert %d) is unavailable (see --expert-log)", l, e);
                const eahmmf::ExpertViews& ev = lease.views();
                bool paired = false;
#ifdef EAHMMF_CUDA_COMPUTE
                if (g_cuda_compute && getenv("EAHMMF_DISABLE_PAIR") == nullptr) {
                    paired = gpu_matmul_pair(
                        ev.gate, ev.up, xs.data(), m, D,
                        gbuf.data(), FF, ubuf.data(), FF);
                    if (paired) {
                        g_pair_calls++;
                        g_gpu_calls += 2;
                        if (g_shadow) {
                            shadow_compare_cpu(
                                ev.gate, xs.data(), m, D, gbuf.data(), FF);
                            shadow_compare_cpu(
                                ev.up, xs.data(), m, D, ubuf.data(), FF);
                        }
                    }
                }
#endif
                if (!paired) {
                    expert_matmul(ev.gate, xs.data(), m, D, gbuf.data(), FF);
                    expert_matmul(ev.up, xs.data(), m, D, ubuf.data(), FF);
                }
                for (int i = 0; i < m * FF; i++) {
                    float gv = gbuf[i];
                    gbuf[i] = gv / (1.0f + expf(-gv)) * ubuf[i];
                }
                expert_matmul(ev.down, gbuf.data(), m, FF, ybuf.data(), D);
                lease.reset();  // Preserve the previous unpin point.
                for (int i = 0; i < m; i++) {
                    float w = lst[i].second;
                    float* dst = &tmp[(size_t)lst[i].first * D];
                    const float* src = &ybuf[(size_t)i * D];
                    for (int d = 0; d < D; d++) dst[d] += w * src[d];
                }
            }
            for (int i = 0; i < nb * D; i++) x[i] += tmp[i];
        }
        if (logits) {
            rmsnorm(&x[(size_t)(nb - 1) * D], out_norm.data(), xn.data(), D, eps);
            matmul(out_v, xn.data(), 1, D, logits, n_vocab);
        }
    }

    // Evaluate an arbitrary number of tokens starting at position p0 (chunked).
    void eval(const std::vector<int>& toks, int p0, int n_batch, float* logits) {
        for (size_t i = 0; i < toks.size(); i += n_batch) {
            int nb = (int)std::min<size_t>(n_batch, toks.size() - i);
            bool last = i + nb == toks.size();
            forward(&toks[i], nb, p0 + (int)i, last ? logits : nullptr);
        }
    }
};

// ============================================================================
// Sampler
// ============================================================================
struct Sampler {
    float temp = 0.7f, top_p = 0.8f, min_p = 0.0f, rep = 1.05f;
    int top_k = 20, rep_n = 64;
    std::mt19937_64 rng;

    int sample(float* lg, int n, const std::vector<int>& hist) {
        if (rep != 1.0f) {
            size_t from = hist.size() > (size_t)rep_n ? hist.size() - rep_n : 0;
            for (size_t i = from; i < hist.size(); i++) {
                int t = hist[i];
                if (t >= 0 && t < n) lg[t] = lg[t] > 0 ? lg[t] / rep : lg[t] * rep;
            }
        }
        if (temp <= 0) return (int)(std::max_element(lg, lg + n) - lg);
        std::vector<std::pair<float, int>> c(n);
        for (int i = 0; i < n; i++) c[i] = {lg[i], i};
        size_t keep = (top_k > 0 && top_k < n) ? (size_t)top_k : (size_t)n;
        std::partial_sort(c.begin(), c.begin() + keep, c.end(), [](auto& a, auto& b) { return a.first > b.first; });
        c.resize(keep);
        float mx = c[0].first, sum = 0;
        std::vector<float> p(keep);
        for (size_t i = 0; i < keep; i++) { p[i] = expf((c[i].first - mx) / temp); sum += p[i]; }
        for (auto& v : p) v /= sum;
        size_t cut = keep;
        if (min_p > 0) {
            cut = 0;
            while (cut < keep && p[cut] >= min_p * p[0]) cut++;
        }
        if (top_p < 1.0f) {
            float cum = 0;
            size_t k = 0;
            while (k < cut) { cum += p[k++]; if (cum >= top_p) break; }
            cut = k;
        }
        float tot = 0;
        for (size_t i = 0; i < cut; i++) tot += p[i];
        float r = std::uniform_real_distribution<float>(0.f, tot)(rng), acc = 0;
        for (size_t i = 0; i < cut; i++) { acc += p[i]; if (r <= acc) return c[i].second; }
        return c[cut - 1].second;
    }
};

// ============================================================================
// CLI
// ============================================================================
[[maybe_unused]] static void usage() {
    fprintf(stderr,
            "usage: qwen3moe -m model.gguf [options]\n"
            "  -p TEXT          prompt (one-shot chat unless --raw); omit for interactive\n"
            "  -i               interactive chat\n"
            "  -s TEXT          system prompt\n"
            "  --raw            no chat template: prompt is tokenized as-is (special tokens parsed)\n"
            "  -n N             max new tokens per reply (default 1024)\n"
            "  -c N             context length (default 8192)\n"
            "  -t N             threads (default: all cores)\n"
            "  -b N             prefill batch (1..%d, default 32)\n"
            "  --temp F --top-k N --top-p F --min-p F --repeat-penalty F --seed N\n"
            "  --cuda-shadow    (CUDA=1 build) run each expert matmul on the GPU too and report max error vs CPU\n"
            "  --cuda-compute   (CUDA=1; needs --device cuda --expert-tier vram) run expert matmuls on the GPU from VRAM\n"
            "  --populate       pre-fault the mmap'd model\n"
            "  expert residency (EAHMMF):\n"
            "  --expert-tier ssd|ram|vram   tier experts are staged to before execution (default ssd = zero-copy mmap)\n"
            "  --ram-budget-mb N / --vram-budget-mb N   capacity of the managed RAM / VRAM expert pools (default unlimited)\n"
          "  --vram-reserve-mb N   keep N MiB of physical VRAM free for CUDA/runtime overhead (default 0)\n" 
            "  --expert-verify  checksum every migration against the GGUF   --expert-strict  fail instead of falling back\n"
            "  --expert-log FILE  residency event log (acquire/migrate/pin)  --expert-stats  print residency summary\n"
            "  --device sim|cuda  VRAM backend (sim = simulated device; cuda needs a build with CUDA=1)\n"
            "  --tokenize TEXT  debug: print token ids   --pretok TEXT  debug: print pre-tokenizer split\n"
            "  --logits-test 1,2,3  debug: print logits for token ids\n",
            MAXB);
    exit(1);
}

#ifndef QWEN3MOE_NO_MAIN
int main(int argc, char** argv) {
    init_types();
    std::string model, prompt, sys = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    std::string dbg_tok, dbg_pre, dbg_logits;
    bool interactive = false, raw = false, populate = false, have_prompt = false, have_dbg = false;
    int n_new = 1024, n_ctx = 8192, n_threads = 0, n_batch = 32;
    eahmmf::Config xcfg;
    bool xstats = false;
    std::string device = "sim";
    Sampler sp;
    uint64_t seed = (uint64_t)time(nullptr);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "-m") model = nxt();
        else if (a == "-p") { prompt = nxt(); have_prompt = true; }
        else if (a == "-i") interactive = true;
        else if (a == "-s") sys = nxt();
        else if (a == "--raw") raw = true;
        else if (a == "-n") n_new = atoi(nxt());
        else if (a == "-c") n_ctx = atoi(nxt());
        else if (a == "-t") n_threads = atoi(nxt());
        else if (a == "-b") n_batch = atoi(nxt());
        else if (a == "--temp") sp.temp = (float)atof(nxt());
        else if (a == "--top-k") sp.top_k = atoi(nxt());
        else if (a == "--top-p") sp.top_p = (float)atof(nxt());
        else if (a == "--min-p") sp.min_p = (float)atof(nxt());
        else if (a == "--repeat-penalty") sp.rep = (float)atof(nxt());
        else if (a == "--seed") seed = strtoull(nxt(), nullptr, 10);
        else if (a == "--cuda-compute") {
#ifdef EAHMMF_CUDA_COMPUTE
            g_cuda_compute = 1;
#else
            die("--cuda-compute needs a build with CUDA=1");
#endif
        }
        else if (a == "--cuda-shadow") {
#ifdef EAHMMF_CUDA_COMPUTE
            g_shadow = 1;
#else
            die("--cuda-shadow needs a build with CUDA=1");
#endif
        }
        else if (a == "--populate") populate = true;
        else if (a == "--expert-tier") {
            std::string v = nxt();
            if (v == "ssd") xcfg.exec_tier = eahmmf::Residency::SSD;
            else if (v == "ram") xcfg.exec_tier = eahmmf::Residency::RAM;
            else if (v == "vram") xcfg.exec_tier = eahmmf::Residency::VRAM;
            else usage();
        }
        else if (a == "--ram-budget-mb") xcfg.ram_budget = (size_t)(atof(nxt()) * 1048576);
        else if (a == "--vram-budget-mb") xcfg.vram_budget = (size_t)(atof(nxt()) * 1048576);
        else if (a == "--vram-reserve-mb") xcfg.vram_reserve = (size_t)(atof(nxt()) * 1048576);
        else if (a == "--expert-verify") xcfg.verify = true;
        else if (a == "--expert-strict") xcfg.strict = true;
        else if (a == "--expert-log") xcfg.log_path = nxt();
        else if (a == "--expert-stats") xstats = true;
        else if (a == "--device") device = nxt();
        else if (a == "--tokenize") { dbg_tok = nxt(); have_dbg = true; }
        else if (a == "--pretok") { dbg_pre = nxt(); have_dbg = true; }
        else if (a == "--logits-test") { dbg_logits = nxt(); have_dbg = true; }
        else usage();
    }
    if (model.empty()) usage();
    n_batch = std::max(1, std::min(n_batch, MAXB));
    if (n_threads > 0) omp_set_num_threads(n_threads);
    sp.rng.seed(seed);

    if (!dbg_pre.empty()) {  // needs no model weights beyond tables
        std::vector<std::pair<size_t, size_t>> parts;
        pretokenize(dbg_pre, parts);
        for (auto& p : parts) printf("%s\x1f", dbg_pre.substr(p.first, p.second - p.first).c_str());
        printf("\n");
        return 0;
    }

    Engine E;
    E.load(model.c_str(), n_ctx, populate);
    {
        std::unique_ptr<eahmmf::DeviceBackend> d;
        if (device == "cuda") {
#ifdef EAHMMF_CUDA
            d.reset(new eahmmf::CudaDevice());
#else
            die("this binary was built without CUDA (rebuild with: make CUDA=1)");
#endif
        } else if (device != "sim") usage();
        #ifdef EAHMMF_CUDA_COMPUTE
        if (g_cuda_compute && !(device == "cuda" && xcfg.exec_tier == eahmmf::Residency::VRAM))
            die("--cuda-compute needs --device cuda --expert-tier vram");
        xcfg.gpu_compute = g_cuda_compute != 0;
#endif
        if (!E.init_experts(xcfg, std::move(d)))
            die("cannot reinitialize expert store while leases are active");
        if (xcfg.exec_tier != eahmmf::Residency::SSD)
            fprintf(stderr, "experts: exec tier=%s device=%s ram_budget=%s vram_budget=%s\n", eahmmf::res_name(xcfg.exec_tier), E.dev->name(),
                    xcfg.ram_budget == SIZE_MAX ? "unlimited" : "set", xcfg.vram_budget == SIZE_MAX ? "unlimited" : "set");
    }
    Tokenizer tk;
    tk.load(E.g);
    fprintf(stderr, "tokenizer: %zu tokens, %zu merges, im_start=%d im_end=%d eos=%d, threads=%d\n", tk.vocab.size(), tk.merges.size(),
            tk.im_start, tk.im_end, tk.eos, omp_get_max_threads());
    if (have_dbg) {
        if (!dbg_tok.empty()) {
            for (int t : tk.encode(dbg_tok, true)) printf("%d ", t);
            printf("\n");
        }
        if (!dbg_logits.empty()) {
            std::vector<int> ids;
            for (char* s = strtok(&dbg_logits[0], ","); s; s = strtok(nullptr, ",")) ids.push_back(atoi(s));
            std::vector<float> lg(E.n_vocab);
            E.eval(ids, 0, n_batch, lg.data());
            for (int i = 0; i < std::min(E.n_vocab, 16); i++) printf("%.6f ", lg[i]);
            printf("\nargmax %d\n", (int)(std::max_element(lg.begin(), lg.end()) - lg.begin()));
        }
        return 0;
    }
    if (!have_prompt) interactive = true;

    std::vector<float> logits(E.n_vocab);
    std::vector<int> pending, hist;
    int n_past = 0;
    bool first_turn = true;
    auto stop_tok = [&](int t) { return t == tk.im_end || t == tk.endoftext || t == tk.eos; };

    auto generate = [&]() {
        auto t0 = std::chrono::steady_clock::now();
        int n_prompt = (int)pending.size(), n_gen = 0;
        double t_prefill = 0;
        for (int step = 0; step < n_new; step++) {
            if (n_past + (int)pending.size() > E.n_ctx) { fprintf(stderr, "\n[context full]\n"); pending.clear(); return; }
            E.eval(pending, n_past, n_batch, logits.data());
            n_past += (int)pending.size();
            pending.clear();
            if (step == 0) t_prefill = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            int tok = sp.sample(logits.data(), E.n_vocab, hist);
            hist.push_back(tok);
            if (stop_tok(tok)) {
                if (!raw) { pending = {tk.im_end, tk.nl}; }
                break;
            }
            const std::string& pc = tk.piece[tok];
            fwrite(pc.data(), 1, pc.size(), stdout);
            fflush(stdout);
            n_gen++;
            pending = {tok};
            if (step == n_new - 1 && !raw) pending = {tok, tk.im_end, tk.nl};
        }
        double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "\n[prompt %d tok in %.2fs (%.1f t/s) | gen %d tok in %.2fs (%.2f t/s)]\n", n_prompt, t_prefill,
                n_prompt / std::max(t_prefill, 1e-9), n_gen, tt - t_prefill, n_gen / std::max(tt - t_prefill, 1e-9));
    };
    auto add_turn = [&](const std::string& user) {
        auto app = [&](const std::vector<int>& v) { pending.insert(pending.end(), v.begin(), v.end()); };
        if (first_turn) {
            pending.push_back(tk.im_start);
            app(tk.encode("system\n" + sys, false));
            pending.push_back(tk.im_end);
            pending.push_back(tk.nl);
            first_turn = false;
        }
        pending.push_back(tk.im_start);
        app(tk.encode("user\n" + user, false));
        pending.push_back(tk.im_end);
        pending.push_back(tk.nl);
        pending.push_back(tk.im_start);
        app(tk.encode("assistant\n", false));
    };

    if (have_prompt) {
        if (raw) pending = tk.encode(prompt, true);
        else add_turn(prompt);
        generate();
        printf("\n");
        if (!interactive) { if (xstats) E.store->print_summary(stderr); return 0; }
    }
    char* line = nullptr;
    size_t cap = 0;
    while (interactive) {
        fprintf(stderr, "\n> ");
        ssize_t len = getline(&line, &cap, stdin);
        if (len <= 0) break;
        std::string u(line, len);
        while (!u.empty() && (u.back() == '\n' || u.back() == '\r')) u.pop_back();
        if (u == "/exit" || u == "/quit") break;
        if (u == "/reset") { n_past = 0; pending.clear(); hist.clear(); first_turn = true; fprintf(stderr, "[reset]\n"); continue; }
        if (u.empty()) continue;
        if (raw) pending = tk.encode(u, true);
        else add_turn(u);
        generate();
        printf("\n");
    }
    if (xstats) E.store->print_summary(stderr);
    return 0;
}
#endif  // QWEN3MOE_NO_MAIN
