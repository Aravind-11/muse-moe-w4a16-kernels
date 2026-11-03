# SGLang baseline kernel (read-only reference)

Unmodified copies of the SGLang files behind the challenge baseline (Marlin MoE, NVFP4 path), from
[sgl-project/sglang](https://github.com/sgl-project/sglang) at commit
[`a55ab7d`](https://github.com/sgl-project/sglang/tree/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f).
Licensed under Apache-2.0 (see `LICENSE` here). Paths mirror the upstream tree.

| File | Role |
| --- | --- |
| `kernels/ops/moe/moe_wna16_marlin.py` | Python JIT wrapper called by `baseline.py` |
| `kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` | Host side: picks thread/tile config, launches the kernel |
| `kernels/jit/csrc/gemm/marlin_moe/marlin_template.h` | Device kernel: the MoE GEMM itself |
| `kernels/jit/csrc/gemm/marlin/dequant.h` | FP4 → BF16 dequant helpers |
| `srt/layers/quantization/marlin_utils_fp4.py` | Repacks NVFP4 weights/scales into Marlin layout |

These files are not edited in this repo. The custom kernel in `submission.py` is a specialised
rewrite of the same job (routed NVFP4 MoE, both GEMMs + SiLU×up + weighted sum), not a patch to them.
