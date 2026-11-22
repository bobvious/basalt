// include/basalt/models/inkling2/kv_q4.hpp - host reference of Inkling 2's 4-bit K/V cell (--kv q4): the cell's layout
// (kernels.hpp: KvFmt::Q4, kQ4CodeAt / kQ4ScaleAt, 576 B) written and read with the 27B stack's q4_0 group quantizer
// (dense/kv_format.hpp: the rounding, the tie rule and the nibble packing are THAT header's, not re-derived here).
// Header only, no CUDA: the CPU test and the GPU parity tests use it as the oracle the kernels are held to.
#pragma once

#include "basalt/dense/kv_format.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cstdint>
#include <cstring>

namespace basalt::models::inkling2 {

static_assert(dense::kQ4Group == kQ4Group, "the Inkling cell uses the dense stack's 32-value groups");

/// One head's row of kHeadDim values into a cell: its codes at kQ4CodeAt(head), its scales at kQ4ScaleAt(head, g).
inline void kv_q4_put_head(uint8_t* cell, int head, const float* x) {
    for (int g = 0; g < kQ4Groups; ++g) {
        uint16_t d = 0;
        uint8_t codes[kQ4Group];
        dense::q4_quantize_group(x + g * kQ4Group, d, codes);
        dense::q4_pack(codes, cell + kQ4CodeAt(head) + g * (kQ4Group / 2), kQ4Group);
        std::memcpy(cell + kQ4ScaleAt(head, g), &d, 2);
    }
}

/// A whole cell: x is [kKvHeads][kHeadDim].
inline void kv_q4_put_cell(uint8_t* cell, const float* x) {
    for (int h = 0; h < kKvHeads; ++h) kv_q4_put_head(cell, h, x + (size_t) h * kHeadDim);
}

/// One head's kHeadDim values out of a cell.
inline void kv_q4_get_head(const uint8_t* cell, int head, float* out) {
    for (int g = 0; g < kQ4Groups; ++g) {
        uint16_t d = 0;
        std::memcpy(&d, cell + kQ4ScaleAt(head, g), 2);
        const uint8_t* codes = cell + kQ4CodeAt(head) + g * (kQ4Group / 2);
        for (int i = 0; i < kQ4Group; ++i) out[g * kQ4Group + i] = dense::q4_value(dense::q4_code(codes, i), d);
    }
}

inline void kv_q4_get_cell(const uint8_t* cell, float* out) {
    for (int h = 0; h < kKvHeads; ++h) kv_q4_get_head(cell, h, out + (size_t) h * kHeadDim);
}

}  // namespace basalt::models::inkling2
