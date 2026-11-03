# Muse MoE W4A16: custom NVFP4 MoE kernel vs SGLang Marlin

My entry for the Muse Spark Open Kernel Challenge, problem `muse_moe_w4a16`: the routed-experts
layer (top-8 of 256 experts, 32 local, NVFP4 weights, BF16 activations) on one H200, timed
against SGLang's Marlin MoE path.

**How to read this repo:** `main` starts with the challenge starter and `submission.py` = the
SGLang baseline. Each speedup landed as its own pull request, so every idea is a reviewable diff.
Lines marked `// SPEEDUP (vN):` point at the code that made that version faster.

| PR | Version | Idea | Judge score (× SGLang) |
| --- | --- | --- | --- |
| 1 | v7 | First custom kernel (swap-AB mma.sync, TMA weights, fused SiLU×up) | 1.0118 |
| 2 | v8 | In-CTA split-K | 1.2211 |
| 3 | v9 | Balanced GEMM2 | 1.2583 |
| 4 | v15 | Per-shape token-tile search in `prepare()` | 1.4018 |
| 5 | v16 | Hoisted activation gather | 1.5532 |
| 6 | v21 | Memoized FP8→BF16 scales | 1.6253 |
| 7 | v22 | Programmatic dependent launch (PDL) | 1.6294 |
| 8 | v24 | Early weight prefetch across kernels | 1.6565 |
| 9 | v26 | Per-block completion counters | 1.6776 |
| 10 | v27 | Decode moved from ALU pipe to FMA pipe | 1.8042 |
| 11 | v28 | Two scales per 32-bit word | 1.8458 |
| 12 | v31 | Router folded into the GEMMs (decode-1) | 1.8541 |
| 13 | v35 | Final board kernel | **1.8696** |

Measured hardware ceiling (TMA byte-stream on the judge's H200): ~3.06×.

- Challenge files: `TASK.md`, `reference.py`, `checks.py`, `nvfp4.py`, `baseline.py`
- SGLang reference kernel: [`sglang_baseline/`](sglang_baseline/) (Apache-2.0, unmodified)
