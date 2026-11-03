# v26 — per-token-block completion counters (+ v25's NW=1 GEMM2)

**Judged score:** 1.6776× (`cf3d1f39d2f644ff`)  
**Previous best:** 1.6565× (v24)  
**Built on:** v25 (v24 + NW=1 GEMM2 variants)  
**File:** `submission_v26.py` (search it for `SPEEDUP (v26)`)

## What this kernel does

1. **v25 (included)**: GEMM2 had half of GEMM1's row units but used the same variant table, so at equal
   NW it launched half as many CTAs. NW=1 variants give GEMM2 GEMM1's CTA count.
2. **v26**: GEMM2 token block *b* only needs the rows that GEMM1's CTAs for block *b* wrote. The router
   zeroes one counter per block; each GEMM1 CTA does a release-add after its epilogue stores; each
   GEMM2 CTA spins (acquire-load) on its own block's counter only, instead of waiting for all of GEMM1.
   After 2^16 polls it falls back to the grid wait, so a counting bug costs time, never a wrong result.

## Why it is faster

Waiting for the whole of GEMM1 serialized the two GEMMs at the grid level. Per-block counters let
GEMM2 start on finished blocks while GEMM1 is still running others (the CUTLASS arrive/wait pattern).
verify-128 dropped 9 µs in the gate; NW=1 was chosen for decode-1, decode-16 and verify-8. +1.3%.

## The code that produced the speedup

### GEMM2: wait on one block's counter

From `submission_v26.py`, original line 448:

```cpp
    if ((mode & 8) && (mode & 3) != 1) {
      // Wait only for the GEMM1 CTAs that wrote this token block (256 units); fall back to the grid wait.
      if (tid == 0) {
        const int* done = n_padded + 1 + blk;
        int spins = 0;
        while (ld_acquire_gpu(done) < 256 && ++spins < (1 << 16)) __nanosleep(32);
        if (spins >= (1 << 16)) pdl_wait();
      }
      __syncthreads();
    } else {
      pdl_wait();
    }
#pragma unroll
```

### GEMM1: publish after the epilogue

From `submission_v26.py`, original line 642:

```cpp
  if constexpr (KIND == 1) {
    if (mode & 8) {
      // Only the NW ks == 0 warps reach here (KS > 1 retires the rest after the reduction).
      asm volatile("bar.sync 1, %0;\n" ::"r"(NW * 32) : "memory");
      if (tid == 0) red_release_gpu(const_cast<int*>(n_padded) + 1 + blk, NW);
    }
```

### Release / acquire helpers

From `submission_v26.py`, original line 248:

```cpp
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_gpu(int* p, int v) {
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
```
