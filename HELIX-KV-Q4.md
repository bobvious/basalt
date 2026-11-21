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
