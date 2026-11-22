// src/models/inkling2/kernels/kv_q4.cuh - the device side of Inkling 2's 4-bit K/V cell (--kv q4; the layout and the format's rules
// are kernels.hpp's KvFmt::Q4 and dense/kv_format.hpp's, whose host reference include/basalt/models/inkling2/kv_q4.hpp is the
// oracle of kernels_parity.cpp / prompt_parity.cpp):
//   - store_row: the cache writers' quantizer (a warp holds one head row, 4 values a lane)
//   - codes32 / pairs16: the readers' unpack of one group (16 code bytes = 32 values) from a 16 B load
#pragma once

#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>

#include <cstdint>

namespace basalt::models::inkling2::kvq4 {

// The total order the group's "largest value" is taken in: larger magnitude, and of two equal magnitudes (opposite signs) the
// larger value, so any order of comparing the lanes' values lands on the same one (dense/kv_format.hpp q4_quantize_group).
__device__ __forceinline__ bool better(float a, float m) { return fabsf(a) > fabsf(m) || (fabsf(a) == fabsf(m) && a > m); }

// One head row of a cell: lane l of a whole warp holds values 4l .. 4l + 3 of the row's 128. Eight lanes are one 32-value group:
// the group's value of largest magnitude by xor shuffles (offsets 1, 2, 4 stay inside the eight), d = it / -8 rounded to fp16,
// code = clamp(floor(x / d + 8.5), 0, 15) against the stored d (the host reference's operations in its order, no fma), the
// lane's two bytes (value 2j low nibble, 2j + 1 high) at the head's codes, the group's scale written by its first lane.
// `cell` is the cell's first byte; every lane of the warp must call it.
__device__ __forceinline__ void store_row(uint8_t* cell, int head, int lane, const float (&x)[4]) {
    float m = x[0];
#pragma unroll
    for (int u = 1; u < 4; ++u)
        if (better(x[u], m)) m = x[u];
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) {
        const float t = __shfl_xor_sync(0xffffffffu, m, o);
        if (better(t, m)) m = t;
    }
    const __half h = __float2half_rn(m / -8.0f);
    const float d = __half2float(h);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int c[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) c[u] = max(0, min(15, (int) floorf(__fadd_rn(__fmul_rn(x[u], id), 8.5f))));
    const uint32_t pk = (uint32_t) (c[0] | (c[1] << 4)) | ((uint32_t) (c[2] | (c[3] << 4)) << 8);
    *reinterpret_cast<uint16_t*>(cell + kQ4CodeAt(head) + lane * 2) = (uint16_t) pk;
    if ((lane & 7) == 0) *reinterpret_cast<uint16_t*>(cell + kQ4ScaleAt(head, lane >> 3)) = __half_as_ushort(h);
}

// A group's 32 codes (a 16 B piece; word i holds values 8i .. 8i + 7, value 8i + j in nibble j) as the exact floats code - 8:
// (2^23 | n) as a float is 2^23 + n.
__device__ __forceinline__ void codes32(const uint4 u, float (&f)[32]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) f[8 * i + j] = __uint_as_float(((w[i] >> (4 * j)) & 0xFu) | 0x4B000000u) - 8388616.0f;
}

// A group's 32 codes times its scale as 16 fp16 pairs (pair k = values 2k, 2k + 1 = byte k): (1024 + n) - 1032 is exact in fp16,
// and the product with d is the correctly rounded fp16 of the exact (n - 8) d.
__device__ __forceinline__ void pairs16(const uint4 u, const __half2 d2, uint32_t (&out)[16]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
    const __half2 magic = __halves2half2(__ushort_as_half((unsigned short) 0x6408), __ushort_as_half((unsigned short) 0x6408));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t lo = w[i] & 0x0F0F0F0Fu, hi = (w[i] >> 4) & 0x0F0F0F0Fu;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            // bytes b of lo and of hi into the two halves (each byte twice: the mask keeps the low nibble of each half)
            const uint32_t t = (__byte_perm(lo, hi, (b) | (b << 4) | ((4 + b) << 8) | ((4 + b) << 12)) & 0x000F000Fu) | 0x64006400u;
            __half2 hv = *reinterpret_cast<const __half2*>(&t);
            hv = __hmul2(__hsub2(hv, magic), d2);
            out[4 * i + b] = *reinterpret_cast<const uint32_t*>(&hv);
        }
    }
}

// The scale k (0..3) of a head's four, from the 8 B load.
__device__ __forceinline__ __half scale_of(const uint2 s, int k) {
    const uint32_t w = k < 2 ? s.x : s.y;
    return __ushort_as_half((unsigned short) ((w >> (16 * (k & 1))) & 0xFFFFu));
}

}  // namespace basalt::models::inkling2::kvq4
