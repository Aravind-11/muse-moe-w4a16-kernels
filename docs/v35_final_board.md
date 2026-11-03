# v35 — final board version (v32 correctness guard + wgmma variants)

**Judged score:** 1.8696× scored (`366c4a635e5f4b01`), rank 508 of 1,194  
**Previous best:** 1.8541× (v31)  
**Built on:** v32 (v31 + a guard that the router fold runs only with the custom norm)  
**File:** `submission_v35.py` (search it for `SPEEDUP (v35)`)

## What this kernel does

v35 is the best scored file. It contains:
1. **v32's correctness guard**: the in-kernel routing (v31) is used only together with the custom norm,
   because SGLang's norm treats every id ≥ 0 as a local slot.
2. **Hopper wgmma variants** for big token tiles, built for `sm_90a` and guarded so a plain `sm_90`
   build still compiles. The gate never selected them: they tied mma.sync at best.

## Why it is faster

Honest note: the +0.8% over v31 is **within judge noise** (CIs overlap). No new code path was chosen
by the gate, so the gain is not attributable to the wgmma code. v35 is included because it is the
final scored submission and the safest one to build on.

## The code that produced the speedup

### v32 guard

From `submission_v35.py`, original line 1377:

```python
        if (pdl_box[0] & 16) and norm and "ctr" in state_ref[0]:  # v32: custom norm only
```

### wgmma inner loop (not selected)

From `submission_v35.py`, original line 733:

```cpp
    if constexpr (WGM != 0) {
      // Warp w of a warpgroup holds the 16 rows of its unit (mma.m16n8k16 A layout); one wgmma.m64nTk16 per
      // matrix per K atom reads this K slice's whole token tile from shared memory once for all 4 warps.
      uint32_t ag[4][MATS][4];
#pragma unroll
      for (int s = 0; s < 4; ++s)
#pragma unroll
        for (int m = 0; m < MATS; ++m) {
          uint32_t r[4];
          deq(wv[m][s], r);
          ag[s][m][0] = hmul2_sc<SF>(r[0], sc[m][0][s], s);
          ag[s][m][1] = hmul2_sc<SF>(r[1], sc[m][1][s], s);
          ag[s][m][2] = hmul2_sc<SF>(r[2], sc[m][0][s], s);
          ag[s][m][3] = hmul2_sc<SF>(r[3], sc[m][1][s], s);
        }
      const unsigned bb = smem_u32(ab);
#pragma unroll
      for (int m = 0; m < MATS; ++m)
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
          for (int i = 0; i < 4; ++i) keep_f(acc[m][g][i]);
      wg_fence();
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const uint64_t desc = wg_desc(bb + s * 256, 128, 1024);  // LBO: K-adjacent core matrices, SBO: 8-token groups
#pragma unroll
        for (int m = 0; m < MATS; ++m) wgmma_rs<G>(&acc[m][0][0], ag[s][m], desc);
      }
      wg_commit();
```
