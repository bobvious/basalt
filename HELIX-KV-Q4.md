# HELIX-KV-Q4: 4-bit KV cache (`--kv q4`) for the Qwen3.8-27B dense stack

Branch `helix/kv-q4-dense` (local only, never pushed; base jesdga95/basalt v0.5.5 @ ff0b978d). Built 2026-11-21.

**STATUS: code complete for the scope below, COMPILED, CPU tests green (79/79), device kernels NEVER RUN.** No GPU was touched
(another job had the 5090). Everything the card executes (the append kernel, the decode kernel, the prompt kernel) is
compile-checked only. Read "Untested" before trusting it.

## Format (what was done)

q4_0 style, symmetric, **unrotated** (no Hadamard on V or K; Flash-Next rotates, this version does not). Per 32 values:
one fp16 scale `d` and 32 codes in 0..15. For K and V both.

- `d = (largest-magnitude value of the group, signed) / -8`, rounded to fp16 (the q4_0 rule; the lower-magnitude-tie rule below);
  `code = clamp(floor(x / d + 8.5), 0, 15)` computed against the ROUNDED `d`; `value = (code - 8) * d`.
  The largest value lands on code 0 exactly; range is [-8, +7] d, so the opposite end clamps (error up to one `d`, same as ggml q4_0).
  A tie of equal magnitudes and opposite signs goes to the larger (positive) value, so every lane of the warp agrees.
- Layout (same cell / KV-head interleave as int8): codes `[cell][n_kv][128]` bytes, **byte j = value 2j (low nibble) | value 2j+1 (high nibble)**
  (adjacent pairs, NOT ggml's j / j+16 pairing; chosen so 8 consecutive dims are 4 contiguous bytes), scales `[cell][n_kv][8]` fp16.
- Bytes per cell per attention layer (K and V), 27B (n_kv 4, head 256): fp16 4096, int8 2112, **q4 1152**.
  One slot of 215,040 cells x 16 attention layers = 3,963,617,280 B (3.69 GiB); two slots 7.38 GiB (int8: 13.5 GiB).
  Shared single source of the sizes: `include/basalt/dense/kv_format.hpp` (`kv_cell_bytes`), which also holds the host
  reference quantizer the CPU test checks.
- Plumbing: `kernels::KvInt8` (include/basalt/kernels/gqa_attn.hpp) gained `bool q4`. Under q4 `k`/`v` point at the packed bytes
  and `ks`/`vs` at the 32-value scales. Every kernel that took `KvInt8` branches on the flag, so `GqaSeg`, `KvFar::q8`, and the
  batched `gqa_decode_attn_segs` carry it with no signature change.
- `dense::Options`: `kv_int8` replaced by `kv_quant` (any quantized cache) + `kv_q4`, with `kv_format()`; the sizing functions take a
  `KvFormat` enum class (so any caller still passing the old bool fails to compile; that is how every site below was found).

## Where each piece lives

| Piece | File |
| --- | --- |
| append: quantize K and V into the cache (decode windows, batch windows, prompt chunks incl. the BF16 chunk form) | `src/kernels/cuda/dense_ops.cu` `attn_qk_prepare_kernel`, the `if (q8.q4)` block (~line 450). One warp = one 32-group; xor-shuffle max with the tie rule; even lanes pack with the neighbour lane |
| decode attention (`gqa_decode_attn`, and `gqa_decode_attn_segs` for 2 slots / drafted windows) | `src/kernels/cuda/gqa_attn.cu` `decode_mma_kernel<HD, I8, Q4>`: `q4_half8` decodes 8 dims to fp16 in the K/V shared tile (both the direct `load_kv` and the prefetched `fetch`/`commit` paths); `launch_segs` picks the `<HD,true,true>` instantiation |
| prompt attention (`gqa_prompt_attn`, `gqa_prompt_attn_bf16`) | `gqa_attn.cu` new `prompt_q4_kernel<IO>` (copy of `prompt_f16_kernel`: 96-row blocks, fp16 mma, fp32 online softmax), each 32-cell K/V tile decoded from the codes into the shared fp16 tiles; IO = fp32 or bf16 queries/output. `launch_prompt_q4` |
| memory planning | `include/basalt/dense/kv_format.hpp`, `src/dense/model.cpp` `cell_bytes` -> `session_layer_bytes` / `session_bytes` / `shared_bytes` / `split_inputs`; `src/models/qwen35/adapter.cpp` start-up `need_b` |
| allocation | `src/dense/session.cpp` `init_state` near-cell block (codes n/2, scales n/32) |
| flag / validation / help | `adapter.cpp` `run_options` (`kv == "q4"`), `finish_flags` (accepts fp16/int8/q4, refuses `--kv-resident`), `src/program/cli/help.cpp` |
| conversation parking / disk tier | `session.cpp` `cells_bytes`, `save_cells`, `restore_cells`: ConversationKv **format 3** (codes `rows`, scales `rows/8`). The store treats the buffers as opaque bytes (rates from sizes), so no store change |
| info words | `session.cpp` `info_words`: `kv=q4` |

## Every `kv_int8` / `q8` site and what was done

Searched: `src/dense`, `src/models/qwen35`, `include/basalt/dense`, `include/basalt/kernels`, `src/kernels/cuda`, and `src/core`
(no use there). The qwen4exp / deepseek41 / hy4flash / inkling2 stacks have their own KV code and are not the dense stack (not built here: `BASALT_MODELS=qwen35`).

| Site | Result |
| --- | --- |
| `dense/model.hpp` `Options::kv_int8` | replaced by `kv_quant` + `kv_q4` + `kv_format()`; enum `KvFormat` in `kv_format.hpp` |
| `dense/model.cpp` `cell_bytes`, `shared_bytes`, `session_host_bytes`, `session_layer_bytes`, `session_bytes`, `split_inputs` | implemented (take `KvFormat`) |
| `dense/model.cpp` `make_model` stage buffer (`o.kv_int8` -> `kv_quant`) | not needed for q4 (streaming refused); int8 behaviour unchanged |
| `dense/model.cpp` `make_model` | NEW validation: q4 refuses head width != 256, a head layout the 96-row prompt block cannot take, `--kv-resident`, a layer split. Clear error `--kv q4 does not support ...` |
| `dense/session.cpp` `init_state` sizing + near-cell alloc | implemented |
| `dense/session.cpp` `far_view` (host-streamed cells) | not needed: refused (never reached under q4) |
| `dense/session.cpp` `cells_bytes`/`save_cells`/`restore_cells` | implemented (format 3) |
| `dense/session.cpp` `info_words` | implemented |
| MTP drafter's own cache (`mtp_kc`/`mtp_vc`, `session_bytes`' drafter term) | stays fp16 (`KvFormat::Fp16`) as allowed |
| block drafter's ring (`dk`/`dv`, `gqa_window_attn`) | unchanged fp16 (own cache, default `KvInt8{}`) |
| `dense/forward.cpp` `win_attn`, `chunk_attn`, `batch_attn` (`q8` pass-through; `if (q8.k)` BF16 prompt path) | not needed: they thread the struct, and the BF16 prompt branch dispatches q4 inside `gqa_prompt_attn_bf16` |
| `dense/forward.cpp` `stage_rows` (far-cell copy) | not needed: streaming refused |
| `dense/stage.cpp` (second card: `cell_regions`, `take_cells`, calibration, `make_prompt_split`, `attach_split`) | **REFUSED** with `--kv q4 does not support a prompt split / layer split over two cards` (also refused earlier in the adapter when `--gpu` names a second card); the int8 code there only had `kv_int8` -> `kv_quant` |
| `adapter.cpp` `run_options`, `finish_flags`, load, `need_b`, `session_host_bytes` | implemented / refused as above |
| `kernels::attn_qk_prepare` / `_bf16` | implemented |
| `kernels::gqa_decode_attn` / `_segs` | implemented |
| `kernels::gqa_prompt_attn` / `_bf16` | implemented (new kernel; no key splitting, no double buffering: slower than the int8 kernel) |
| `kernels::gqa_window_attn` | not needed (drafter's fp16 ring) |
| `--kv-resident` K/V streaming | **REFUSED** (flag check and `make_model`) |
| second-card prompt split / layer split | **REFUSED** |
| idle-sleep | no such code in the dense stack |
| `tests/` for the int8 dense path | none exist for the dense stack on CPU; GPU `gqa_attn_parity` / `gqa_far_parity` have an int8 arm, NOT extended (see Untested) |

## Build and test status (exact)

Scripts: `D:\llm\.buildtmp\basalt-kvq4.cmd` (CRLF; sets vcvars64, TMP/TEMP/TMPDIR=D:\llm\.buildtmp, CUDA_PATH and CUDACXX to CUDA v13.4):

```
cmd /c D:\llm\.buildtmp\basalt-kvq4.cmd configure   # cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=qwen35 -DBUILD_TESTING=ON
cmd /c D:\llm\.buildtmp\basalt-kvq4.cmd build       # cmake --build build -j 8   (clean build 1m44s)
cmd /c D:\llm\.buildtmp\basalt-kvq4.cmd ctest       # ctest --test-dir build -L cpu --output-on-failure
```

- Build: all 435 targets link; binaries in `D:\llm\src\basalt\bin\` (basalt.exe, basalt-bench.exe, basalt-probe.exe, basalt-vision.exe). Only pre-existing MSVC warnings (C4324, C4996).
- `ctest -L cpu`: 78/78 passed after the main change (before the new test was added), then **79/79 passed** with the new `dense_kv_q4_test` (final run, after the last rebuild).
- New test `tests/dense/kv_q4_test.cpp` (registered in CMakeLists.txt next to `qwen35_split_plan_test`): cell bytes for the 27B in all three
  formats, the 2 x 215,040 budget, and the host reference quantizer (error bounds, extreme -> code 0, all-zero group, tie rule,
  packing order). **RED against four named mutants of `kv_format.hpp`, then GREEN restored**: M1 q4 cell bytes = int8's formula
  (fails the 1152 B and slot-size checks), M2 swapped nibble order (fails packing), M3 tie-break removed (fails the tie check),
  M4 truncating round `+8.0` instead of `+8.5` (fails 4 error-bound checks). This tests the format's SPEC and the sizing the planner
  uses; it does NOT execute the CUDA kernels (they mirror the reference by construction and by review only).
- `basalt.exe --help` shows the `--kv q4` line (checked; no GPU use). Nothing else was run.

## Untested (everything that runs on the card)

1. `attn_qk_prepare_kernel` q4 branch, `decode_mma_kernel<..,true,true>`, `prompt_q4_kernel`: compiled for sm_120, never executed.
2. No GPU parity test written for q4 (the existing `gqa_attn_parity` int8 arm would be the model; a q4 arm = quantize fp16 K/V
   with the host reference, dequantize, compare `gqa_decode_attn`/`gqa_prompt_attn` on the fp16 dequantized cache to the q4 cache: should agree to fp16 rounding).
3. Conversation parking format 3 (save/restore) and the disk tier with q4.
4. The start-up planner with q4 on the real pack, and 2 x 215,040 actually fitting in 32 GB with `--vram-reserve-mib 2048` (arithmetic says
   the KV is 7.38 GiB for both slots; weights, MTP/dflash drafter, chunk and window scratch are on top).
5. Quality: no PPL/KLD or greedy comparison against int8/fp16. Unrotated q4_0 is the cheapest 4-bit option; K outliers
   (Qwen K has a few large channels) may hurt, which is exactly what Flash-Next's rotation addresses. If quality is bad, the next step is a Hadamard on V (and/or K) as `kv_q4.hpp` does.
6. Speed: the prompt kernel decodes tiles synchronously with no double buffering and no key split, expect slower prefill than int8 at long context.
   Decode reads 55% of int8's bytes but dequantizes more per byte.
7. Wrong-bits risk inherited from the repo: `f16_bits.hpp` says `__float2half` once produced wrong bits and was replaced in quantize_act; the int8 append kernel
   uses `__float2half_rn` + `__half_as_ushort` as the q4 one does, so any such defect would hit both. If q4 output is garbage and int8 is fine, look there first.

## How to start it (when the 5090 is free; nothing here has been started)

Config `D:\llm\src\basalt\kv-q4-27b.json` (committed; README format `"version": 2`, engine flags allowed in the model entry; "kv" passes
straight through as `--kv q4`; I did not run the server, so the config loader itself is unexercised):

```json
{
  "version": 2,
  "models": [{
    "model": "D:/llm/models/basalt/qwen38-27b/Qwen3.8-27B-QUASAR-NVFP4.basalt",
    "gpu": "0",
    "parallel": 2,
    "max-context": 430080,
    "kv": "q4",
    "vram-reserve-mib": 2048
  }]
}
```

`--max-context` is shared by the slots (serve/config.py divides it: each slot holds 215,040). Start with the repo's server, e.g.
`bin\basalt.exe serve --config kv-q4-27b.json` from `D:\llm\src\basalt` (README quick start uses `bin/basalt serve --config ...`; the Python
server needs `pip install -r requirements.txt` on Windows, docs/platforms.md). Equivalent engine flags: `--model ... --gpu 0 --kv q4 --max-context 215040 --parallel 2 --vram-reserve-mib 2048`.
A second card in `--gpu` or `--kv-resident` makes the start FAIL with `--kv q4 does not support ...` (by design).

## What to run on the GPU first (in this order)

1. Plan only / load with a short context and `--kv q4 --parallel 1`; read the load log: it should say `kv=q4` and the planned KV bytes should be ~55% of `--kv int8`.
2. One short prompt, greedy, `--kv fp16` vs `--kv int8` vs `--kv q4`: the output should be coherent and close. Then `tools/ppl_gate.py`-style PPL/KLD on the same chunks for int8 vs q4.
3. Long prompt (>= 32K) to exercise the prompt kernel's tile loop and cells past 32, then decode, and `--spec` (drafted windows hit `gqa_decode_attn_segs`).
4. Two concurrent requests (the batched segs kernel) and a conversation park/resume (format 3).
5. Only then the full 2 x 215,040 start; watch free VRAM (the 2 GiB reserve is the WDDM guard).

## Known risks

- Untested kernels (above). A bad index would be silent corruption or an illegal-address fault; the likeliest spots are the byte
  offsets `row * (HD/2) + d8/2` and scale index `row * (HD/32) + d8/32`, which are identical in the writer and both readers by construction.
- `prompt_q4_kernel` needs `96 % (n_head / n_kv) == 0` (24 heads / 4 KV heads = 6, ok) and 84,480 B of dynamic shared memory (opt-in done per device/instantiation).
- Quality unmeasured; unrotated q4 on K may be noticeably worse than int8.
- The disk conversation tier / parking uses a new format code (3); an old image with another format is refused, a q4 image is refused by an int8 server (`other caches`).

---

# Round 2 (2026-11-21): the fast q4 prompt kernel

Coordinator's result for round 1 (D:/llm/benchmarks/runs/basalt-kvq4-20261121/RESULTS.md): q4 works on the GPU, decode = int8, gate KLD 0.02909 / top-1 92.54%,
but prefill at depth was far behind int8 (the fp16-tile prompt kernel above: synchronous tile loads, no key split, no double buffering).
This round replaces that kernel. **The `prompt_q4_kernel` / `launch_prompt_q4` described in the round-1 tables no longer exists.**

## What was done

`prompt_i8_kernel` (gqa_attn.cu) is now templated `<W, Split, IO, Q4>`; the int8 instantiations are the old code (the int8 gate below is bit-identical to
the published row). `Q4=true` is the int8 kernel (int8 tensor-core QK^T, u8 x s8 P V, key splitting through the same plan and merge kernel, 8- or 4-warp blocks)
fed from the 4-bit cache:

- **Tile feed.** A 64-cell key tile arrives packed by cp.async (K and V codes 8 KiB each, one buffer) with its fp16 scales (64 x 8 per tensor, double buffered).
  At the top of the iteration the block unpacks the packed codes into the int8 tile of the int8 layout (16-byte pieces swizzled by the cell, so the int8
  loops are unchanged): offset-binary SWAR (`q4_sx`: `((x | 0x80808080) - 0x08080808) ^ 0x80808080` = signed c - 8, no borrow between bytes), nibble
  interleave by `__byte_perm`. Only after the unpack barrier is the next tile's copy issued into the same packed buffer, so it runs under the compute.
- **K scales per 32 values** (the int8 kernel's are per 64): the QK^T k32 step's integer sum is scaled by its own `q_scale * k_scale` (two accumulators per 64-dim
  group, the two scales read as one 32-bit load).
- **V scales per 32 values, sign.** A 4-bit group's scale is negative when its largest value is positive (q4_0's `d = max / -8`), but P V rounds
  `p * scale` to u8, which needs a non-negative scale. This was the bug in the first version of this kernel (KLD 0.53, top-1 74%): the fix is that the unpack negates
  the codes of a negative-scale group (c -> 8 - c, branch-free: `((u ^ N) + (N & 0x01010101)) ^ 0x80808080`) and stores |d| in a `[group][cell]` buffer that P V reads
  as two 32-bit loads. P V's per-group probability rounding now runs over 8 groups of 32 dims instead of 4 of 64.
- 4-bit only: the two divisions per P V group use `__fdividef` (the inverse is then rounded down to fp16, so it moves a code only for a product within
  2 ulp of a half step).
- Shared memory: 87,040 B (8 warps) of the 101,376 B (99 KiB) a block may opt into on this card; the first attempt (102,400 B) was refused with
  `invalid argument`, which is why the packed buffer is single and only the scales are double buffered. The opt-in error now prints the size and the CUDA error.
- Removed: `prompt_q4_kernel`, `launch_prompt_q4`, `io_st2`, and the "96 % (n_head/n_kv) == 0" head-layout requirement in `make_model` and `gqa_prompt_attn*` for q4.

## Measurement

Same binary (the build at this commit), `basalt-bench`, RTX 5090 only (`CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID`, `--gpu 0`, TEMP/TMP=D:/llm/work/basalt/tmp, nothing else on GPU 0 at the start),
order int8, q4, q4, int8. Script: `D:/llm/work/basalt/kvq4perf/kvq4-bench.cmd <fmt> <out>`:

```
bin/basalt-bench.exe --model D:/llm/models/basalt/qwen38-27b/Qwen3.8-27B-QUASAR-NVFP4.basalt -p 16384,65536,131072 --logs <dir> -- --gpu 0 --kv <fmt> --max-context 140000 --vram-reserve-mib 2048
```

Prefill t/s (engine), the two runs of each format and their mean (raw outputs: `D:/llm/work/basalt/kvq4perf/final-*.txt`):

| | 16K | 64K | 128K |
|---|--:|--:|--:|
| int8 run 1 / run 4 | 15,620 / 15,327 | 11,442 / 11,232 | 8,217 / 8,096 |
| **int8 mean** | **15,474** | **11,337** | **8,157** |
| q4 run 2 / run 3 | 14,691 / 14,587 | 9,869 / 9,808 | 6,737 / 6,705 |
| **q4 mean (after)** | **14,639 (94.6% of int8)** | **9,839 (86.8%)** | **6,721 (82.4%)** |
| q4 before (same-session baseline, one run, round-1 kernel) | 12,111 (77.1% of that day's int8 15,701) | 5,951 (51.9% of 11,464) | 3,460 (42.1% of 8,224) |
| q4 after / before | x1.21 | x1.65 | x1.94 |

**The goal (within ~10% of int8 at 64K and 128K) is NOT met**: 64K is 13% behind, 128K 18% behind (16K is within 6%). int8's own run-to-run spread here is about 1.5-2%.

## Quality gate (numerics), final binary

`D:/llm/work/basalt/kvq4perf/kvq4-gate.cmd <fmt> <out>`:
`bin/basalt-bench.exe --model <same> --quality 100 --kld-base D:/llm/models/basalt/kld-bases/qwen38-27b-bf16-base.kld -- --gpu 0 --kv <fmt> --vram-reserve-mib 2048`

| run | PPL | KLD mean | median | p99 | same top-1 |
|---|--:|--:|--:|--:|--:|
| q4, round 1 (fp16-tile kernel; RESULTS.md) | 6.8533 | 0.02909 | 0.01351 | 0.2715 | 92.54% |
| q4, **final kernel** | **6.8331** | **0.02871** | 0.01352 | 0.2784 | **92.47%** |
| q4, an intermediate build (same kernel before `__fdividef` and the abs-scale buffer) | 6.8423 | 0.02907 | 0.01365 | 0.2726 | 92.19% |
| int8, final binary | 6.8260 | 0.02585 | 0.01185 | 0.2522 | 92.89% (= the published row: the int8 path is unchanged) |
| q4, first version of the new kernel (negative V scales unhandled) | 10.8816 | 0.53269 | 0.138 | 6.49 | 74.25% (broken, fixed above) |

Required band KLD 0.0291 +/- 0.0005 and top-1 92.5 +/- 0.2: the final kernel is inside both (0.02871 is 0.0004 under the nominal; 92.47%). **The math did change in principle**
(the old q4 kernel took fp16 queries and fp16 P V; this one rounds queries to int8 per 64 values and probabilities to u8, as the int8 kernel does), and one
intermediate build landed 0.11 pt of top-1 below the band, so treat the two q4 kernels as the same quality only to within this gate's noise (100 chunks, one run each;
about +/-0.0004 KLD and +/-0.3 pt top-1 between builds).
CPU tests: `ctest -L cpu` 79/79 passed.

## Where the remaining time is (measured, not guessed)

Experiment (temporary, reverted): the same kernel with the unpack loops compiled out (garbage tile, same barriers and copies) gave 15,270 / 10,702 / 7,499 t/s at 16K/64K/128K,
against 14,838 / 9,940 / 6,775 with them: **the unpack costs about 8% of total prefill at 64K and 11% at 128K**, and everything else q4 does differently (8 P V
rounding groups instead of 4, a second scale multiply per QK^T step) costs about 5% at 64K and 8% at 128K. With the unpack free q4 would be 94.6% / 92% of int8 at 64K / 128K.

The unpack is not overlapped with compute: the block unpacks, hits a barrier, then computes, with all 8 warps in the same phase, so the tensor cores idle during it. Hiding
it needs a second int8 K/V tile (or at least V) so the next tile can be unpacked during this tile's compute. The queries (32 KiB) + one int8 K/V tile (32 KiB) + the packed
copy (16 KiB) + the scales (4 KiB) + a second tile (32 KiB) = 116 KiB > the 99 KiB opt-in limit; without the packed staging buffer
(unpacking from registers loaded straight from global memory) and with single-buffered scales it is exactly 101,376 B, and the kernel already uses 255 registers with small spills,
so it lacks the register budget. Cheaper options not done: (1) change the nibble pairing to ggml's (byte j = value j | value j+16), which removes the interleave permutes
from the unpack (about 30% fewer ops; touches the append, decode and test code); (2) apply the V sign as a nibble NOT plus a different subtract constant per group (2 ops fewer per word);
(3) a 32-cell compute tile with double-buffered 16 KiB int8 tiles. Estimated together: +3-4% at 64K/128K, still short of 90% at 128K.

Other items still slow or untested: the key-split plan (`plan_prompt_i8`) is the int8 kernel's cost model, not retuned for q4; no GPU parity test (gqa_attn_parity) was added; 4+ streams untested.
