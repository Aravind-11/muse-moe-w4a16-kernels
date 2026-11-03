# v9 — GEMM2 does GEMM1's per-tile work; late B loads for big tiles

**Judged score:** 1.2583× scored (`7a5b622a61874984`; check 1.2691×)  
**Previous best:** 1.2211× (v8)  
**Built on:** v8  
**File:** `submission_v9.py` (search it for `SPEEDUP (v9)`)

## What this kernel does

Two changes to the shared kernel:
1. **GEMM2 units carry two matrices** like GEMM1: rows `r` and `r + 2048` of W2. Both GEMMs now do the
   same amount of work per K tile, so one tuned pipeline fits both and GEMM2 launches half as many,
   fuller CTAs.
2. For token tiles of 32+ (`G >= 4`), each MMA atom's activation fragments are loaded just before use
   (`ldmatrix.x2`), instead of holding every group's fragments live, which cut register pressure.

## Why it is faster

GEMM2 used to process one matrix per unit, so each CTA streamed half the weight bytes per tile
through the same fixed per-tile overhead (barriers, address math, TMA issue). Doubling the work per
tile amortizes that overhead. The late B-fragment loads lowered registers at big tiles, so more CTAs
fit per SM there.

## The code that produced the speedup

### Unit definition per GEMM

From `submission_v9.py`, original line 301:

```cpp
// KIND == 1: GEMM1. A 16-row unit is 16 gate rows + the matching 16 up rows; epilogue silu(gate)*up;
//            activation row = token; 256 units.
// KIND == 2: GEMM2. A unit is 16 rows of W2 + the same 16 rows offset by 2048 (same per-tile work as
//            GEMM1); epilogue plain bf16; activation row = slot (cache2); 128 units.
```

### GEMM2 epilogue writes both matrices

From `submission_v9.py`, original line 528:

```cpp
          o[row0] = __float2bfloat16(acc[0][g][c] * gf);
          o[row0 + 8] = __float2bfloat16(acc[0][g][2 + c] * gf);
          o[row0 + 2048] = __float2bfloat16(acc[1][g][c] * gf);
          o[row0 + 2056] = __float2bfloat16(acc[1][g][2 + c] * gf);
```

### Late B-fragment loads for big tiles

From `submission_v9.py`, original line 430:

```cpp
    if constexpr (G >= 4) {
      // Large token tiles: load each atom's B fragments just before use (2 live regs per group).
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        uint32_t ag[MATS][4];
#pragma unroll
        for (int m = 0; m < MATS; ++m) {
          uint32_t r[4];
          deq(wv[m][s], r);
          ag[m][0] = hmul2(r[0], sc[m][0][s]);
          ag[m][1] = hmul2(r[1], sc[m][1][s]);
          ag[m][2] = hmul2(r[2], sc[m][0][s]);
          ag[m][3] = hmul2(r[3], sc[m][1][s]);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
          if (g * 8 < nact) {
            uint32_t b[2];
            ldsm2(b, ab + (g * 8 + (lane & 7)) * ACT_LD + s * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
            for (int m = 0; m < MATS; ++m) mma16816(acc[m][g], ag[m][0], ag[m][1], ag[m][2], ag[m][3], b[0], b[1]);
          }
        }
      }
```
