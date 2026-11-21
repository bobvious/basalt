// include/basalt/dense/kv_format.hpp - the dense stack's K/V cache formats: how many bytes a cell takes in each, and a host
// reference of the 4-bit format's group quantizer (the rule src/kernels/cuda/dense_ops.cu's attn_qk_prepare_kernel applies on the
// card and gqa_attn.cu's kernels undo).  Header only, no CUDA: the planner and a CPU test use it.
//
// 4-bit cache (kernels::KvInt8::q4; q4_0 style, symmetric, no rotation): per 32 values one fp16 scale d and 32 codes in 0 .. 15.
//   d     = (the value of largest magnitude, signed; the lower index on a tie of two equal magnitudes of opposite sign, the
//            larger value wins) / -8, rounded to fp16
//   code  = clamp(floor(x / d + 8.5), 0, 15) against the ROUNDED d (0 when d is 0: every code is 8)
//   value = (code - 8) d
// so the largest value lands on code 0 and the codes' range is [-8, 7] d.  A row of 256 values packs 128 bytes, byte j holding
// value 2j in its low nibble and 2j + 1 in its high one, and carries 8 scales.
#pragma once

#include "basalt/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdint>

namespace basalt::dense {

/// The attention layers' K/V cells: fp16, int8 codes with an fp16 scale per 64 values, or 4-bit codes with an fp16 scale per 32.
enum class KvFormat { Fp16, Int8, Q4 };

/// Bytes of one cell of one attention layer, K and V together.
constexpr uint64_t kv_cell_bytes(int64_t n_kv, int64_t head_dim, KvFormat f) {
    switch (f) {
        case KvFormat::Int8: return (uint64_t) n_kv * (uint64_t) (head_dim + head_dim / 64 * 2) * 2;
        case KvFormat::Q4: return (uint64_t) n_kv * (uint64_t) (head_dim / 2 + head_dim / 32 * 2) * 2;
        default: return (uint64_t) n_kv * (uint64_t) head_dim * 2 * 2;
    }
}

constexpr int kQ4Group = 32;   ///< values per 4-bit scale

/// One group of kQ4Group values: its fp16 scale (bits) and codes.
inline void q4_quantize_group(const float* x, uint16_t& d_bits, uint8_t* codes) {
    float m = x[0];
    for (int i = 1; i < kQ4Group; ++i)
        if (std::fabs(x[i]) > std::fabs(m) || (std::fabs(x[i]) == std::fabs(m) && x[i] > m)) m = x[i];
    d_bits = kernels::f16_from_f32(m / -8.0f);
    const float d = kernels::f32_from_f16(d_bits), id = d != 0.0f ? 1.0f / d : 0.0f;
    for (int i = 0; i < kQ4Group; ++i) {
        const int c = (int) std::floor(x[i] * id + 8.5f);
        codes[i] = (uint8_t) (c < 0 ? 0 : c > 15 ? 15 : c);
    }
}

/// A code's value.
inline float q4_value(uint8_t code, uint16_t d_bits) { return (float) ((int) code - 8) * kernels::f32_from_f16(d_bits); }

/// `n` (even) codes into n / 2 bytes: byte j = code 2j | code 2j + 1 << 4.
inline void q4_pack(const uint8_t* codes, uint8_t* bytes, int n) {
    for (int j = 0; j < n / 2; ++j) bytes[j] = (uint8_t) (codes[2 * j] | (codes[2 * j + 1] << 4));
}

/// Code `i` of a packed row.
inline uint8_t q4_code(const uint8_t* bytes, int i) { return (uint8_t) ((bytes[i / 2] >> (i % 2 ? 4 : 0)) & 15); }

}  // namespace basalt::dense
