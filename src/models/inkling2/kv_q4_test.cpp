// src/models/inkling2/kv_q4_test.cpp - the 4-bit K/V cell of Inkling 2 (--kv q4): the sizes, the byte offsets and the
// round trip through the 27B stack's q4_0 group quantizer (dense/kv_format.hpp). CPU only. The expectations are written out as
// literals (not read back from the layout's own constants), so a layout that moves, say the scales before the codes, fails here.
#include "basalt/models/inkling2/kv_q4.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace basalt::models::inkling2;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}
// a deterministic generator (no std::hash, no random_device)
struct Lcg {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    float next() {   // uniform in [-1, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (float) ((s >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
    }
};
}  // namespace

int main() {
    // ---- sizes: 8 heads x 128 values ----
    check(kv_cell_bytes(KvFmt::Fp16) == 2048, "an fp16 cell is 2,048 B");
    check(kv_cell_bytes(KvFmt::Fp8) == 1024, "an fp8 cell is 1,024 B");
    check(kv_cell_bytes(KvFmt::Q4) == 576, "a q4 cell is 576 B (512 B of codes + 64 B of scales)");
    check(kQ4Cell == 576 && kQ4ScalesAt == 512, "the q4 cell's constants: 576 B, the scales at byte 512");
    check(kv_cell_bytes(KvFmt::Q4) % 16 == 0, "a q4 cell is 16-byte aligned");
    check(ring_fmt(KvFmt::Q4) == KvFmt::Fp8 && ring_fmt(KvFmt::Fp8) == KvFmt::Fp8 && ring_fmt(KvFmt::Fp16) == KvFmt::Fp16,
          "q4 caches keep their rings in fp8; the others keep theirs as they are");
    // the whole cache: 8 full visits (K and V), one position
    check(kv_cache_bytes(8, 0, 1, 0, KvFmt::Fp8) == 16384, "fp8: 16 KiB a position over 8 full visits");
    check(kv_cache_bytes(8, 0, 1, 0, KvFmt::Fp16) == 32768, "fp16: 32 KiB a position");
    check(kv_cache_bytes(8, 0, 1, 0, KvFmt::Q4) == 9216, "q4: 9,216 B a position (2 x 8 x 576)");
    check(kv_cache_bytes(8, 0, 262144, 0, KvFmt::Fp8) * 2 == (8ull << 30), "fp8: two slots of 262,144 are 8 GiB");
    check(kv_cache_bytes(8, 0, 262144, 0, KvFmt::Q4) == 2415919104ull, "q4: one slot of 262,144 over 8 visits is 2,415,919,104 B");
    // a ring is fp8 under q4: 5 ring visits of 1,032 cells
    check(kv_cache_bytes(0, 5, 0, 1032, KvFmt::Q4) == 2ull * 5 * 1024 * 1032, "q4: the rings are fp8 cells");
    check(kv_cache_bytes(0, 5, 0, 1032, KvFmt::Fp16) == 2ull * 5 * 2048 * 1032, "fp16: the rings are fp16 cells");

    // ---- offsets, as literals ----
    check(kQ4CodeAt(0) == 0 && kQ4CodeAt(1) == 64 && kQ4CodeAt(7) == 448, "head h's codes at byte 64 h");
    check(kQ4ScaleAt(0, 0) == 512 && kQ4ScaleAt(0, 3) == 518 && kQ4ScaleAt(1, 0) == 520 && kQ4ScaleAt(7, 3) == 574,
          "scale (h, g) at byte 512 + 8 h + 2 g");
    check(kQ4ScaleAt(7, 3) + 2 == 576, "the last scale ends the cell");
    for (int h = 0; h < kKvHeads; ++h) {
        check(kQ4CodeAt(h) % 16 == 0, "a head's codes are 16-byte aligned (four groups, one 16 B load each)");
        check(kQ4ScaleAt(h, 0) % 8 == 0, "a head's four scales are one aligned 8 B load");
    }

    // ---- the round trip against the 27B reference, group by group, at literal offsets ----
    Lcg rng;
    std::vector<float> x((size_t) kKvHeads * kHeadDim);
    for (size_t i = 0; i < x.size(); ++i) x[i] = rng.next() * (1.0f + (float) (i / 32 % 5));   // spread of group magnitudes
    x[3 * 128 + 2 * 32 + 5] = -9.0f;   // a group whose largest value is negative
    std::vector<uint8_t> cell(576, 0xAA);
    kv_q4_put_cell(cell.data(), x.data());
    int wrong_code = 0, wrong_scale = 0;
    for (int h = 0; h < 8; ++h)
        for (int g = 0; g < 4; ++g) {
            uint16_t d = 0;
            uint8_t codes[32];
            basalt::dense::q4_quantize_group(&x[(size_t) h * 128 + (size_t) g * 32], d, codes);
            uint16_t got = (uint16_t) (cell[(size_t) (512 + h * 8 + g * 2)] | (cell[(size_t) (512 + h * 8 + g * 2 + 1)] << 8));
            if (got != d) ++wrong_scale;
            for (int j = 0; j < 16; ++j) {
                const uint8_t want = (uint8_t) (codes[2 * j] | (codes[2 * j + 1] << 4));   // byte j = value 2j | 2j+1 << 4
                if (cell[(size_t) (h * 64 + g * 16 + j)] != want) ++wrong_code;
            }
        }
    check(wrong_scale == 0, "every scale is where the layout puts it and equals the reference's");
    check(wrong_code == 0, "every code byte is where the layout puts it and equals the reference's packing");
    // decode: back to (code - 8) d, at most one d off (the range is [-8, +7] d: the end opposite the largest clamps)
    std::vector<float> y(x.size());
    kv_q4_get_cell(cell.data(), y.data());
    int out_of_bound = 0;
    double max_rel = 0;
    for (int h = 0; h < 8; ++h)
        for (int g = 0; g < 4; ++g) {
            uint16_t d = 0;
            std::memcpy(&d, &cell[(size_t) (512 + h * 8 + g * 2)], 2);
            const float df = basalt::kernels::f32_from_f16(d);
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[(size_t) h * 128 + (size_t) g * 32 + (size_t) i]));
            for (int i = 0; i < 32; ++i) {
                const size_t k = (size_t) h * 128 + (size_t) g * 32 + (size_t) i;
                if (std::fabs(y[k] - x[k]) > std::fabs(df) * 1.001f) ++out_of_bound;
                max_rel = std::fmax(max_rel, std::fabs(y[k] - x[k]) / amax);
            }
        }
    check(out_of_bound == 0, "every decoded value is within one scale of its source");
    check(max_rel < 0.14, "the worst error is under 1/8 of the group's largest value (plus the fp16 scale's rounding)");
    // the largest value of a group lands on code 0 exactly (the q4_0 rule): the group with -9.0 decodes it back to -9 within fp16
    check(std::fabs(y[3 * 128 + 2 * 32 + 5] + 9.0f) < 0.01f, "the largest-magnitude value decodes back");

    // ---- one head's put touches only its own 64 code bytes and 4 scales ----
    std::vector<uint8_t> a(576, 0x55), b(576, 0xAA);
    kv_q4_put_head(a.data(), 5, &x[(size_t) 5 * 128]);
    kv_q4_put_head(b.data(), 5, &x[(size_t) 5 * 128]);
    int touched = 0, covered = 0, disagree = 0;
    for (int i = 0; i < 576; ++i) {
        const bool mine = (i >= 5 * 64 && i < 6 * 64) || (i >= 512 + 5 * 8 && i < 512 + 6 * 8);
        if (!mine && (a[(size_t) i] != 0x55 || b[(size_t) i] != 0xAA)) ++touched;   // outside the head's bytes both fills survive
        if (mine) {
            ++covered;
            if (a[(size_t) i] != b[(size_t) i]) ++disagree;   // inside, the put overwrote both fills with the same bytes
        }
    }
    check(touched == 0, "a head's put writes its own bytes and nothing else");
    check(covered == 72 && disagree == 0, "a head's put covers its 64 code bytes and 8 scale bytes");
    std::vector<float> one(128);
    kv_q4_get_head(a.data(), 5, one.data());
    int same = 0;
    for (int i = 0; i < 128; ++i) same += one[(size_t) i] == y[(size_t) 5 * 128 + (size_t) i];
    check(same == 128, "a head read alone equals the same head read from the whole cell");

    // an all-zero group: scale 0, every code 8, values 0
    std::vector<float> z(128, 0.0f);
    std::vector<uint8_t> zc(576, 0xFF);
    kv_q4_put_head(zc.data(), 0, z.data());
    check(zc[0] == 0x88 && zc[63] == 0x88 && zc[512] == 0 && (zc[513] == 0 || zc[513] == 0x80), "an all-zero head: codes 8 (0x88 bytes), scales +-0");

    if (g_fail) {
        std::printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("inkling2 kv q4 cell: all checks passed\n");
    return 0;
}
