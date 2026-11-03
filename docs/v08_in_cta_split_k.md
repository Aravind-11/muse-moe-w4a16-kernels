# v8 — in-CTA split-K (more warps per weight unit)

**Judged score:** 1.2211× (check `3ad16a1427c34bd4`)  
**Previous best:** 1.0118× (v7)  
**Built on:** v7  
**File:** `submission_v8.py` (search it for `SPEEDUP (v8)`)

## What this kernel does

Same kernel as v7, with a new template parameter **KS**: a CTA now runs `NW × KS` warps, where
`KS` warps work on the *same* 16-row weight unit but on different K tiles of each pipeline stage.
At the end, the KS partial accumulators are summed through shared memory and one warp writes the
epilogue.

## Why it is faster

At small batch a v7 CTA had few warps and each walked all 64 K tiles alone, so there was not
enough work in flight to hide the dequant → MMA latency chain. Splitting K inside the CTA multiplies
the number of independent instruction streams per weight unit without changing the bytes read, and
the reduction at the end costs one shared-memory round trip per unit. +21% overall.

## The code that produced the speedup

### Warp roles: unit × K-slice

From `submission_v8.py`, original line 329:

```cpp
  const int unit = warp % NW;   // which 16-row unit
  const int ks = warp / NW;     // which K tile of each stage
```

### Cross-warp K reduction

From `submission_v8.py`, original line 450:

```cpp
  if constexpr (KS > 1) {
    // Sum the KS partial accumulators of each 16-row unit through shared memory (lane-fastest).
    constexpr int NACC = MATS * G * 4;
    __syncthreads();
    float* red = reinterpret_cast<float*>(stages);
    if (ks > 0) {
#pragma unroll
      for (int m = 0; m < MATS; ++m)
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            red[(((ks - 1) * NW + unit) * NACC + (m * G + g) * 4 + i) * 32 + lane] = acc[m][g][i];
    }
    __syncthreads();
    if (ks > 0) return;
#pragma unroll
    for (int k2 = 1; k2 < KS; ++k2)
#pragma unroll
      for (int m = 0; m < MATS; ++m)
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            acc[m][g][i] += red[(((k2 - 1) * NW + unit) * NACC + (m * G + g) * 4 + i) * 32 + lane];
  }
```
