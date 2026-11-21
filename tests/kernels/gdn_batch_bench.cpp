// tests/kernels/gdn_batch_bench.cpp - (Helix 2026-11-21, measurement only) the GDN recurrence's cost per layer for the
// 27B's shapes: the solo window's dec_gdn_rec (what a 1-session round launches, nsys `rec_kernel`) against the batch
// window's dec_gdn_gates + dec_gdn_rec_seg_split (nsys `gates_kernel` + `rec_split_seg_kernel` + `rec_norm_kernel`) and
// the cluster-per-head dec_gdn_rec_seg, at 1 x T and 2 x T rows.  Prints us per call.  Exit 77 without a GPU.
// Inputs built as in gdn_seg_split_parity.cpp; values do not matter for timing.
#include "basalt/kernels/gdn_window.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

using namespace basalt::kernels;

namespace {

std::mt19937 rng(7);
std::normal_distribution<float> nd(0.f, 1.f);
cudaStream_t S0;

template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
std::vector<float> randn(size_t n, float sd) {
    std::vector<float> v(n);
    for (auto& x : v) x = sd * nd(rng);
    return v;
}
std::vector<uint16_t> randbf(size_t n, float sd) {
    std::vector<uint16_t> v(n);
    for (auto& x : v) { const float f = sd * nd(rng); uint32_t u; std::memcpy(&u, &f, 4); x = (uint16_t) (u >> 16); }
    return v;
}
void* zeros(size_t bytes) { void* d = nullptr; cudaMalloc(&d, bytes); cudaMemset(d, 0, bytes); return d; }

double time_us(const std::function<void()>& f, int iters = 400) {
    for (int i = 0; i < 20; ++i) f();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, S0);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(e1, S0);
    cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    if (cudaGetLastError() != cudaSuccess) std::printf("  CUDA error\n");
    return 1000.0 * ms / iters;
}

void bench(int n_embd, int h_k, int h_v, int T) {
    const int SS = 128, C = 2 * h_k * SS + h_v * SS, VD = h_v * SS, E = n_embd, R = 2 * T;
    const size_t sf = (size_t) SS * h_v * SS + (size_t) C * 3;
    float* st[2] = {up(randn(sf, 0.05f)), up(randn(sf, 0.05f))};
    DecRecArgs a;
    a.qkv = up(randn((size_t) R * C, 1.f));
    a.conv_w = up(randn((size_t) C * 4, 0.5f));
    a.x = up(randn((size_t) R * E, 1.f));
    a.w_alpha = up(randbf((size_t) h_v * E, 0.02f));
    a.w_beta = up(randbf((size_t) h_v * E, 0.02f));
    a.dt = up(randn(h_v, 0.5f));
    a.ssm_a = up(randn(h_v, 0.5f));
    a.z = up(randn((size_t) R * VD, 1.f));
    a.gamma = up(randn(SS, 1.f));
    a.C = C; a.n_embd = E; a.h_k = h_k; a.h_v = h_v; a.eps = 1e-6f; a.gate_act = GdnGate::Sigmoid;
    a.h = (float*) zeros((size_t) 2 * kDecMaxT * C * 4);
    a.gate = (float*) zeros((size_t) 2 * kDecMaxT * h_v * 4);
    a.beta = (float*) zeros((size_t) 2 * kDecMaxT * h_v * 4);
    a.y = (float*) zeros((size_t) R * VD * 4);
    a.o = a.y;
    a.y16 = (uint16_t*) zeros((size_t) R * VD * 2);
    a.yq = zeros((size_t) R * (VD / 32) * 36);
    float* gates = (float*) zeros((size_t) 2 * R * h_v * 4);
    std::vector<SegRow> one{SegRow{st[0], 0, 0, T, T, 0, 0}};
    std::vector<SegRow> two{SegRow{st[0], 0, 0, T, T, 0, 0}, SegRow{st[1], (int32_t) T, 0, T, T, 1, 0}};
    SegRow* d1 = up(one);
    SegRow* d2 = up(two);
    DecRecArgs solo = a;
    solo.state = st[0];

    const double rec1 = time_us([&] { dec_gdn_rec(solo, T, S0); });
    const double rec1x2 = time_us([&] { dec_gdn_rec(solo, T, S0); dec_gdn_rec(solo, T, S0); });
    const double seg1 = time_us([&] { dec_gdn_rec_seg(a, d1, 0, 1, T, S0); });
    const double seg2 = time_us([&] { dec_gdn_rec_seg(a, d2, 0, 2, T, S0); });
    const double gates1 = time_us([&] { dec_gdn_gates(a, T, gates, S0); });
    const double gates2 = time_us([&] { dec_gdn_gates(a, R, gates, S0); });
    const double split1 = time_us([&] { dec_gdn_gates(a, T, gates, S0); dec_gdn_rec_seg_split(a, d1, 0, 1, T, T, S0, nullptr, 0, 0, gates); });
    const double split2 = time_us([&] { dec_gdn_gates(a, R, gates, S0); dec_gdn_rec_seg_split(a, d2, 0, 2, T, R, S0, nullptr, 0, 0, gates); });
    std::printf("n_embd %d h_v %d T=%2d | solo dec_gdn_rec 1x%d %6.1f us, two solo calls %6.1f | rec_seg 1x%d %6.1f, 2x%d %6.1f | "
                "gates %d rows %5.1f, %d rows %5.1f | gates+split 1x%d %6.1f, 2x%d %6.1f\n",
                E, h_v, T, T, rec1, rec1x2, T, seg1, T, seg2, T, gates1, R, gates2, T, split1, T, split2);
    for (void* p : {(void*) st[0], (void*) st[1], (void*) a.qkv, (void*) a.conv_w, (void*) a.x, (void*) a.w_alpha, (void*) a.w_beta,
                    (void*) a.dt, (void*) a.ssm_a, (void*) a.z, (void*) a.gamma, (void*) a.h, (void*) a.gate, (void*) a.beta,
                    (void*) a.y, (void*) a.y16, a.yq, (void*) gates, (void*) d1, (void*) d2})
        cudaFree(p);
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) return 77;
    cudaStreamCreate(&S0);
    for (int T : {4, 8}) bench(5120, 16, 48, T);   // the 27B's GDN layers (48 of 64)
    return 0;
}
