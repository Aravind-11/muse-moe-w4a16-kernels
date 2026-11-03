# v7 — first custom kernel to beat SGLang

**Judged score:** 1.0118× (check `ddc11e6cef6449a5`)  
**Previous best:** 0.9910× (Marlin fallback, scored `b7364df108d748cc`)  
**Built on:** the S2 fused prototype (v4_fused/v5/v6), which always fell back to Marlin  
**File:** `submission_v7.py` (search it for `SPEEDUP (v7)`)

## What this kernel does

Both expert GEMMs (gate/up, then down) run on one custom **swap-AB** kernel: weights sit on the
MMA's M side (`mma.m16n8k16` BF16) and tokens on N, because decode batches have far fewer tokens
than weight rows. GEMM1 fuses `silu(gate) * up` into its epilogue (one warp owns 16 gate rows and
the matching 16 up rows) and writes GEMM2's input directly. `prepare()` repacks the 32 local experts
(same byte count, still exact NVFP4), compiles one `.cu` per (GEMM, token tile), and a **gate** times
each scored shape against Marlin and keeps the custom path only where it is correct and ≥3% faster.

## Why it is faster

The prototypes spent ~277 instructions per 64-K tile and lost to Marlin. v7 cut that to ~190:
- weights repacked K-tile-major, so one CTA's stage is a single TMA bulk copy issued by one thread;
- E2M1 → bf16x2 in **9 integer ops per 8 codes** (a rotate/mask bit layout found by search), exact;
- E4M3 block scales recoded at repack time, so decoding one is a `byte_perm` + one IMAD;
- activation fragments via `ldmatrix.x4` from padded, bank-conflict-free shared memory.
It is the first version where the custom path won on some shapes, so the board moved above 1.0×.

## The code that produced the speedup

### Exact 9-op NVFP4 decode

From `submission_v7.py`, original line 282:

```cpp
// Packed word -> four bf16x2 A registers (E2M1 value * 2^-126). Layout in _word_sources().
__device__ __forceinline__ void deq(uint32_t w, uint32_t r[4]) {
  const uint32_t M = 0x81C081C0u;
  r[0] = w & M;
  r[1] = __funnelshift_l(w, w, 4) & M;
  r[2] = __funnelshift_l(w, w, 10) & M;
  r[3] = (__funnelshift_l(w, w, 6) & 0x80C080C0u) | (__funnelshift_l(w, w, 14) & 0x01000100u);
}
```

### Scale decode: one byte_perm + one IMAD

From `submission_v7.py`, original line 291:

```cpp
// Coded scale byte s of u -> bf16x2 (both halves) = scale * 2^60. Then A = w * 2^-66.
__device__ __forceinline__ uint32_t scale_dup(uint32_t u, int s) {
  uint32_t x = __byte_perm(u, 0u, 0x4040u | ((unsigned)s << 8) | (unsigned)s);
  return x * 16u + 0x58805880u;
}
```

### Swap-AB inner loop

From `submission_v7.py`, original line 420:

```cpp
      for (int h = 0; h < 2; ++h) {
        const int s = sp + h;
#pragma unroll
        for (int m = 0; m < MATS; ++m) {
          uint32_t r[4];
          deq(wv[m][s], r);
          const uint32_t a0 = hmul2(r[0], sc[m][0][s]);
          const uint32_t a1 = hmul2(r[1], sc[m][1][s]);
          const uint32_t a2 = hmul2(r[2], sc[m][0][s]);
          const uint32_t a3 = hmul2(r[3], sc[m][1][s]);
#pragma unroll
          for (int g = 0; g < G; ++g) {
            if (g * 8 < nact) mma16816(acc[m][g], a0, a1, a2, a3, bfr[g][2 * h], bfr[g][2 * h + 1]);
          }
        }
      }
    }
```
