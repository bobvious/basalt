// tests/dense/kv_q4_test.cpp - the dense stack's 4-bit K/V cache (--kv q4): the bytes a cell takes in each format (what the start-up
// planner sizes the slots with) and the host reference of its group quantizer (include/basalt/dense/kv_format.hpp; the card's
// attn_qk_prepare_kernel applies the same rule).  Host only: the kernels themselves run in the GPU parity tests.
#include "basalt/dense/kv_format.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace basalt::dense;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

uint32_t lcg = 12345;
float rnd() {   // [-1, 1)
    lcg = lcg * 1664525u + 1013904223u;
    return (float) (lcg >> 8) / (float) (1u << 23) - 1.0f;
}

// one group: quantize, then every value's reconstruction error against half a step (|d| / 2), or a whole step for a value past
// +7.5 d (the code range is [-8, 7] d, so the end of the range opposite the largest value clamps, as q4_0's does), and the
// largest-magnitude value's reconstruction against its own fp16 rounding
void group(const char* what, std::vector<float> x, bool expect_zero = false) {
    uint16_t d_bits = 0;
    uint8_t codes[kQ4Group];
    q4_quantize_group(x.data(), d_bits, codes);
    const float d = basalt::kernels::f32_from_f16(d_bits);
    float worst_ratio = 0.0f, m = x[0];
    int clamped = 0;
    for (int i = 0; i < kQ4Group; ++i) {
        const bool past = d != 0.0f && x[i] / d > 7.5f;
        clamped += past;
        if (d != 0.0f) worst_ratio = std::fmax(worst_ratio, std::fabs(q4_value(codes[i], d_bits) - x[i]) / std::fabs(d) / (past ? 1.0f : 0.5f));
        if (std::fabs(x[i]) > std::fabs(m)) m = x[i];
    }
    if (expect_zero) {
        bool all = true;
        for (int i = 0; i < kQ4Group; ++i) all = all && codes[i] == 8 && q4_value(codes[i], d_bits) == 0.0f;
        check(all && d == 0.0f, std::string(what) + ": an all-zero group: scale 0, every code 8, every value 0");
        return;
    }
    check(worst_ratio <= 1.01f, std::string(what) + ": every error within its bound (worst " + std::to_string(worst_ratio) + " of it, " +
                                    std::to_string(clamped) + " clamped)");
    // the largest value is code 0 (-8 d), exact up to d's fp16 rounding (2^-11 relative)
    int at = 0;
    for (int i = 1; i < kQ4Group; ++i)
        if (std::fabs(x[i]) > std::fabs(x[at])) at = i;
    check(codes[at] == 0 && std::fabs(q4_value(codes[at], d_bits) - m) <= std::fabs(m) * 1e-3f,
          std::string(what) + ": the largest magnitude (" + std::to_string(m) + ") is code 0 and decodes to " +
              std::to_string(q4_value(codes[at], d_bits)));
}

}  // namespace

int main() {
    // cell sizes, Qwen3.8-27B: 4 KV heads of 256, one cell of one attention layer, K and V
    check(kv_cell_bytes(4, 256, KvFormat::Fp16) == 4096, "27B cell, fp16: 4096 B");
    check(kv_cell_bytes(4, 256, KvFormat::Int8) == 2112, "27B cell, int8: 2112 B (256 codes + 4 scales, K and V, 4 heads)");
    check(kv_cell_bytes(4, 256, KvFormat::Q4) == 1152, "27B cell, q4: 1152 B (128 code bytes + 8 scales of 2 B, K and V, 4 heads)");
    // 2 slots of 215,040 cells over 16 attention layers: what --max-context 430080 --parallel 2 holds
    const uint64_t slot = 215040ull * 16 * kv_cell_bytes(4, 256, KvFormat::Q4);
    check(slot == 3963617280ull && 2 * slot < (8ull << 30), "a q4 slot of 215,040 cells over 16 layers: 3,963,617,280 B, two under 8 GiB");
    check(2 * 215040ull * 16 * kv_cell_bytes(4, 256, KvFormat::Int8) > (13ull << 30), "the same two slots in int8 take over 13 GiB");

    // the quantizer
    std::vector<float> ramp(kQ4Group), noise(kQ4Group), zero(kQ4Group, 0.0f), up(kQ4Group, 0.01f), down(kQ4Group, 0.01f), tie(kQ4Group, 0.5f);
    for (int i = 0; i < kQ4Group; ++i) {
        ramp[i] = (float) i - 12.5f;
        noise[i] = rnd() * 3.0f;
    }
    up[7] = 5.0f;      // one large positive value: d negative
    down[7] = -5.0f;   // one large negative: d positive
    tie[3] = 3.0f;
    tie[20] = -3.0f;
    group("ramp", ramp);
    group("noise", noise);
    group("positive outlier", up);
    group("negative outlier", down);
    group("zeros", zero, true);
    group("tie of +3 and -3", tie);   // the negative partner clamps to code 15 (7 d): an error of one d

    // the tie goes to the larger value whichever lane or index holds it
    {
        std::vector<float> a = tie, b = tie;
        b[3] = -3.0f;
        b[20] = 3.0f;
        uint16_t da = 0, db = 0;
        uint8_t ca[kQ4Group], cb[kQ4Group];
        q4_quantize_group(a.data(), da, ca);
        q4_quantize_group(b.data(), db, cb);
        check(da == db && basalt::kernels::f32_from_f16(da) < 0.0f, "a +3 / -3 tie: the same (negative) scale wherever the +3 sits");
    }

    // packing: code i = i mod 16; byte j holds codes 2j (low) and 2j + 1 (high)
    {
        uint8_t codes[kQ4Group], bytes[kQ4Group / 2];
        for (int i = 0; i < kQ4Group; ++i) codes[i] = (uint8_t) (i % 16);
        q4_pack(codes, bytes, kQ4Group);
        bool ok = bytes[0] == 0x10 && bytes[1] == 0x32 && bytes[7] == 0xFE && bytes[8] == 0x10;
        for (int i = 0; i < kQ4Group; ++i) ok = ok && q4_code(bytes, i) == codes[i];
        check(ok, "packing: byte 0 = 0x10, byte 1 = 0x32, byte 7 = 0xFE; every code reads back");
    }
    std::printf("%s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
