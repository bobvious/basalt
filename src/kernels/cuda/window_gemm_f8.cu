// src/kernels/cuda/window_gemm_f8.cu - the decode window products against FP8 E4M3 (window_gemm_sliced.cuh), BF16 and Q8
// matrices (window_gemm.cuh).
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.
#include "window_gemm_kernels.hpp"
#include "window_gemm_sliced.cuh"

namespace basalt::kernels::wgemm::detail {
namespace {
constexpr int RS = 64, CH = 128, ST = 4, MW = 1;
}

// The stages change only the overlap, never the sums: two where few wide blocks stream short slices.  At NT 2 a second
// stage's activations (the stage holds 8 * NT columns of the group) cost 16 KB of the SM's 100: with many tiles that
// drops the blocks an SM holds, and a second wave of blocks costs more than the overlap gains; DEEP is the few-tile call.
template <int KW, int NT, bool DEEP> Kernel make_sliced() {
    constexpr int KB = 1, SST = KW == 8 && (NT <= 1 || (NT == 2 && DEEP)) ? 2 : 1;
    using SC = SlicedCfg<KW, KB, NT, SST>;
    Kernel k;
    k.fn = wgemm_sliced_kernel<KW, KB, NT, SST>;
    k.threads = SC::kThreads;
    k.smem = SC::kSmem;
    k.rs = 16;
    k.ch = SC::kGroupK;
    return k;
}

const Kernel& f8(int kw, int nt, int tiles) {
    static const Kernel k[3][3] = {{make_sliced<2, 1, false>(), make_sliced<2, 2, false>(), make_sliced<2, 4, false>()},
                                   {make_sliced<4, 1, false>(), make_sliced<4, 2, false>(), make_sliced<4, 4, false>()},
                                   {make_sliced<8, 1, false>(), make_sliced<8, 2, false>(), make_sliced<8, 4, false>()}};
    static const Kernel deep8x2 = make_sliced<8, 2, true>();
    if (kw > 4 && nt == 2 && tiles < kDeepTiles) return deep8x2;
    return k[kw > 4 ? 2 : kw > 2 ? 1 : 0][nt > 2 ? 2 : nt > 1 ? 1 : 0];
}

const Kernel& bf16(int nt) {
    constexpr int B = (int) Format::Bf16;
    static const Kernel k1 = make_kernel<B, RS, 1, CH, ST, 1, MW>(), k2 = make_kernel<B, RS, 1, CH, ST, 2, MW>(),
                        k4 = make_kernel<B, RS, 1, CH, ST, 4, MW>();
    return nt > 2 ? k4 : nt > 1 ? k2 : k1;
}

const Kernel& q8(int nt) {
    constexpr int Q = (int) Format::Q8;
    static const Kernel k1 = make_kernel<Q, RS, 1, CH, ST, 1, MW>(), k2 = make_kernel<Q, RS, 1, CH, ST, 2, MW>(),
                        k4 = make_kernel<Q, RS, 1, CH, ST, 4, MW>();
    return nt > 2 ? k4 : nt > 1 ? k2 : k1;
}

}  // namespace basalt::kernels::wgemm::detail
