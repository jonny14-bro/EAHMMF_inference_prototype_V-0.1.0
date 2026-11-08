#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_OK(call) ((call) == cudaSuccess)

__device__ static float half_to_float(uint16_t h) {
    unsigned sign = h >> 15;
    unsigned exp = (h >> 10) & 31;
    unsigned mant = h & 1023;
    float v;
    if (exp == 0)
        v = mant ? ldexpf((float)mant, -24) : 0.0f;
    else if (exp == 31)
        v = mant ? NAN : INFINITY;
    else
        v = ldexpf((float)(mant | 1024), (int)exp - 25);
    return sign ? -v : v;
}

__device__ static uint16_t read_u16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

__device__ static void scale_min_k4(
    int j, const uint8_t* q, uint8_t* d, uint8_t* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ static float decode_q4k(const uint8_t* b, int k) {
    float d = half_to_float(read_u16(b));
    float mn = half_to_float(read_u16(b + 2));
    const uint8_t* sc = b + 4;
    const uint8_t* q = b + 16;

    int group64 = k / 64;
    int within = k % 64;
    int group32 = k / 32;
    uint8_t s, m;
    scale_min_k4(group32, sc, &s, &m);

    int packed = q[group64 * 32 + (within % 32)];
    int quant = within < 32 ? (packed & 15) : (packed >> 4);
    return d * s * quant - mn * m;
}

__device__ static float decode_q5k(const uint8_t* b, int k) {
    float d = half_to_float(read_u16(b));
    float mn = half_to_float(read_u16(b + 2));
    const uint8_t* sc = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;

    int group64 = k / 64;
    int within = k % 64;
    int group32 = k / 32;
    uint8_t s, m;
    scale_min_k4(group32, sc, &s, &m);

    uint8_t packed = ql[group64 * 32 + (within % 32)];
    int quant = within < 32 ? (packed & 15) : (packed >> 4);
    int bit = 2 * group64 + (within >= 32 ? 1 : 0);
    if (qh[within % 32] & (1u << bit))
        quant += 16;

    return d * s * quant - mn * m;
}

__device__ static float decode_q6k(const uint8_t* b, int k) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = (const int8_t*)(b + 192);
    float d = half_to_float(read_u16(b + 208));

    int half = k / 128;
    int pos = k % 128;
    int quarter = pos / 32;
    int l = pos % 32;

    const uint8_t* low = ql + half * 64;
    const uint8_t* high = qh + half * 32;
    int packed = (quarter == 0 || quarter == 2)
        ? low[l] : low[l + 32];

    int quant = (quarter < 2)
        ? (packed & 15) : (packed >> 4);

    quant |= ((high[l] >> (2 * quarter)) & 3) << 4;
    quant -= 32;

    int scale_index = half * 8 + (l / 16) + 2 * quarter;
    return d * sc[scale_index] * quant;
}

__device__ static float decode_q8_0(const uint8_t* b, int k) {
    float d = half_to_float(read_u16(b));
    return d * (float)((const int8_t*)(b + 2))[k];
}

__device__ static float decode_weight(
    const uint8_t* row, int type, int k)
{
    if (type == 8)
        return decode_q8_0(row + (k / 32) * 34, k % 32);
    const int block = k / 256;
    const int within = k % 256;

    if (type == 12)
        return decode_q4k(row + block * 144, within);
    if (type == 13)
        return decode_q5k(row + block * 176, within);
    if (type == 14)
        return decode_q6k(row + block * 210, within);
    return 0.0f;
}

__global__ static void expert_matmul_kernel(
    const uint8_t* weights,
    const float* x,
    float* out,
    int type,
    int ne0,
    int ne1,
    size_t row_bytes,
    int nb,
    int ldx,
    int ldo)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ne1 * nb;
    if (index >= total) return;

    int row = index % ne1;
    int batch = index / ne1;
    const uint8_t* wr = weights + (size_t)row * row_bytes;
    const float* xr = x + (size_t)batch * ldx;

    float acc = 0.0f;
    for (int k = 0; k < ne0; ++k)
        acc += decode_weight(wr, type, k) * xr[k];

    out[(size_t)batch * ldo + row] = acc;
}

__device__ static __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// One warp per output row. Lane l handles values [8l, 8l+8) of every 256-value superblock, so the
// block header (d, dmin, scale/min) is decoded once per 8 weights instead of once per weight.
// Requires ne0 % 256 == 0. Up to TB tokens per launch (token tiles on grid.y).
template <int TB>
__global__ static void expert_matmul_warp_kernel(
    const uint8_t* weights, const float* x, float* out,
    int type, int ne0, int ne1, size_t row_bytes, int nb, int ldx, int ldo)
{
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= ne1) return;
    const int t0 = blockIdx.y * TB;
    const int ntb = min(TB, nb - t0);
    const uint8_t* wr = weights + (size_t)row * row_bytes;
    const int nsb = ne0 >> 8;
    const int k0 = lane * 8;

    float acc[TB];
#pragma unroll
    for (int t = 0; t < TB; t++) acc[t] = 0.f;

    for (int sb = 0; sb < nsb; sb++) {
        float w[8];
        if (type == 12 || type == 13) {
            const uint8_t* b = wr + (size_t)sb * (type == 12 ? 144 : 176);
            const float d = h2f(read_u16(b));
            const float mn = h2f(read_u16(b + 2));
            uint8_t s, m;
            scale_min_k4(lane >> 2, b + 4, &s, &m);
            const float ds = d * s, mm = mn * m;
            const int g64 = lane >> 3;
            const bool hi = ((lane >> 2) & 1) != 0;
            const int l0 = (lane & 3) * 8;
            const uint8_t* q = b + (type == 12 ? 16 : 48) + g64 * 32 + l0;
            const uint8_t* qh = b + 16 + l0;
            const int bit = 2 * g64 + (hi ? 1 : 0);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int packed = q[i];
                int quant = hi ? (packed >> 4) : (packed & 15);
                if (type == 13 && ((qh[i] >> bit) & 1)) quant += 16;
                w[i] = ds * quant - mm;
            }
        } else if (type == 14) {
            const uint8_t* b = wr + (size_t)sb * 210;
            const float d = h2f(read_u16(b + 208));
            const int half = lane >> 4;
            const int quarter = (lane & 15) >> 2;
            const int l0 = (lane & 3) * 8;
            const uint8_t* low = b + half * 64 + ((quarter == 0 || quarter == 2) ? 0 : 32);
            const uint8_t* high = b + 128 + half * 32;
            const int sidx = half * 8 + (l0 >> 4) + 2 * quarter;
            const float ds = d * (float)((const int8_t*)(b + 192))[sidx];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int l = l0 + i;
                const int packed = low[l];
                int quant = (quarter < 2) ? (packed & 15) : (packed >> 4);
                quant |= ((high[l] >> (2 * quarter)) & 3) << 4;
                quant -= 32;
                w[i] = ds * quant;
            }
        } else {  // type 8 (Q8_0): 8 blocks of 32 values per 256
            const uint8_t* b = wr + (size_t)(sb * 8 + (lane >> 2)) * 34;
            const float d = h2f(read_u16(b));
            const int8_t* q = (const int8_t*)(b + 2) + (lane & 3) * 8;
#pragma unroll
            for (int i = 0; i < 8; i++) w[i] = d * (float)q[i];
        }
#pragma unroll
        for (int t = 0; t < TB; t++) {
            if (t < ntb) {
                const float* xp = x + (size_t)(t0 + t) * ldx + (size_t)sb * 256 + k0;
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 8; i++) s += w[i] * xp[i];
                acc[t] += s;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TB; t++) {
        if (t < ntb) {
            float v = acc[t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) out[(size_t)(t0 + t) * ldo + row] = v;
        }
    }
}

// C interface called from qwen3moe.cpp.
// Supports GGML type IDs 12 (Q4_K), 13 (Q5_K), and 14 (Q6_K).
extern "C" bool eahmmf_cuda_matmul(
    const uint8_t* device_weights,
    int type,
    int ne0,
    int ne1,
    size_t row_bytes,
    const float* host_x,
    int nb,
    int ldx,
    float* host_out,
    int ldo)
{
    if (!device_weights || !host_x || !host_out ||
        ne0 <= 0 || ne1 <= 0 || nb <= 0 ||
        ldx < ne0 || ldo < ne1 ||
        (type != 8 && type != 12 && type != 13 && type != 14))
        return false;

    // Persistent device scratch (grown on demand, reused across calls; single-threaded caller).
    static float* sx = nullptr;
    static float* so = nullptr;
    static size_t cx = 0, co = 0;
    size_t xbytes = (size_t)nb * ldx * sizeof(float);
    size_t obytes = (size_t)nb * ldo * sizeof(float);
    if (xbytes > cx) {
        if (sx) cudaFree(sx);
        sx = nullptr; cx = 0;
        if (!CUDA_OK(cudaMalloc(&sx, xbytes))) return false;
        cx = xbytes;
    }
    if (obytes > co) {
        if (so) cudaFree(so);
        so = nullptr; co = 0;
        if (!CUDA_OK(cudaMalloc(&so, obytes))) return false;
        co = obytes;
    }
    float* dx = sx;
    float* dout = so;

    // Dedicated stream for activation copies, kernel execution, and results.
    // The output is ready for the CPU when this function returns.
    static cudaStream_t stream = nullptr;
    static bool stream_initialized = false;

    if (!stream_initialized) {
        if (!CUDA_OK(cudaStreamCreateWithFlags(
                &stream, cudaStreamNonBlocking))) {
            return false;
        }
        stream_initialized = true;
    }

    bool ok = CUDA_OK(cudaMemcpyAsync(
        dx, host_x, xbytes, cudaMemcpyHostToDevice, stream));

    if (ok) {
        static const bool use_old_kernel =
            getenv("EAHMMF_OLD_KERNEL") != nullptr;

        if (use_old_kernel || (ne0 & 255)) {
            int total = ne1 * nb;
            expert_matmul_kernel<<<
                (total + 127) / 128, 128, 0, stream>>>(
                device_weights, dx, dout, type, ne0, ne1,
                row_bytes, nb, ldx, ldo);
        } else {
            constexpr int TB = 8;
            dim3 grid((ne1 + 3) / 4, (nb + TB - 1) / TB);
            expert_matmul_warp_kernel<TB><<<grid, 128, 0, stream>>>(
                device_weights, dx, dout, type, ne0, ne1,
                row_bytes, nb, ldx, ldo);
        }

        ok = CUDA_OK(cudaGetLastError());
    }

    if (ok) {
        ok = CUDA_OK(cudaMemcpyAsync(
            host_out, dout, obytes, cudaMemcpyDeviceToHost, stream));
    }

    // Drain queued work even after an error before the caller releases
    // the expert lease or this function's scratch can be reused.
    cudaError_t sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) ok = false;

    return ok;
}


// Paired gate/up projection: one input transfer, two kernels, two output
// transfers, and one stream synchronization. Scratch is persistent.
// The caller is single-threaded, matching eahmmf_cuda_matmul().
extern "C" bool eahmmf_cuda_matmul_pair(
    const uint8_t* gw, int gt, int gne0, int gne1, size_t grb,
    const uint8_t* uw, int ut, int une0, int une1, size_t urb,
    const float* host_x, int nb, int ldx,
    float* host_g, int ldg, float* host_u, int ldu)
{
    auto supported = [](int t) {
        return t == 8 || t == 12 || t == 13 || t == 14;
    };
    if (!gw || !uw || !host_x || !host_g || !host_u ||
        !supported(gt) || !supported(ut) ||
        gne0 <= 0 || gne1 <= 0 || une0 != gne0 || une1 != gne1 ||
        nb <= 0 || ldx < gne0 || ldg < gne1 || ldu < une1)
        return false;

    const size_t xb = (size_t)nb * ldx * sizeof(float);
    const size_t gb = (size_t)nb * ldg * sizeof(float);
    const size_t ub = (size_t)nb * ldu * sizeof(float);

    static float* sx = nullptr;
    static float* sg = nullptr;
    static float* su = nullptr;
    static size_t cx = 0, cg = 0, cu = 0;

    auto grow = [](float** ptr, size_t* cap, size_t need) {
        if (need <= *cap) return true;
        if (*ptr) {
            if (!CUDA_OK(cudaFree(*ptr))) return false;
            *ptr = nullptr;
            *cap = 0;
        }
        if (!CUDA_OK(cudaMalloc((void**)ptr, need))) return false;
        *cap = need;
        return true;
    };

    if (!grow(&sx, &cx, xb) ||
        !grow(&sg, &cg, gb) ||
        !grow(&su, &cu, ub))
        return false;

    static cudaStream_t stream = nullptr;
    static bool stream_initialized = false;
    if (!stream_initialized) {
        if (!CUDA_OK(cudaStreamCreateWithFlags(
                &stream, cudaStreamNonBlocking)))
            return false;
        stream_initialized = true;
    }

    bool ok = CUDA_OK(cudaMemcpyAsync(
        sx, host_x, xb, cudaMemcpyHostToDevice, stream));

    static const bool use_old_kernel =
        getenv("EAHMMF_OLD_KERNEL") != nullptr;

    auto launch = [&](const uint8_t* weights, int type,
                      int ne0, int ne1, size_t row_bytes,
                      float* output, int ldo) {
        if (use_old_kernel || (ne0 & 255)) {
            int total = ne1 * nb;
            expert_matmul_kernel<<<
                (total + 127) / 128, 128, 0, stream>>>(
                weights, sx, output, type, ne0, ne1,
                row_bytes, nb, ldx, ldo);
        } else {
            constexpr int TB = 8;
            dim3 grid((ne1 + 3) / 4, (nb + TB - 1) / TB);
            expert_matmul_warp_kernel<TB><<<grid, 128, 0, stream>>>(
                weights, sx, output, type, ne0, ne1,
                row_bytes, nb, ldx, ldo);
        }
        return CUDA_OK(cudaGetLastError());
    };

    if (ok) ok = launch(gw, gt, gne0, gne1, grb, sg, ldg);
    if (ok) ok = launch(uw, ut, une0, une1, urb, su, ldu);

    if (ok) {
        ok = CUDA_OK(cudaMemcpyAsync(
            host_g, sg, gb, cudaMemcpyDeviceToHost, stream));
    }
    if (ok) {
        ok = CUDA_OK(cudaMemcpyAsync(
            host_u, su, ub, cudaMemcpyDeviceToHost, stream));
    }

    // Drain queued work even after an error, before scratch can be reused.
    cudaError_t sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) ok = false;

    return ok;
}
