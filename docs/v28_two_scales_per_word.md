# v28 — two scales per register for 1-byte scales

**Judged score:** 1.8458× scored (`196ad392b4524623`)  
**Previous best:** 1.8042× (v27)  
**Built on:** v27  
**File:** `submission_v28.py` (search it for `SPEEDUP (v28)`)

## What this kernel does

1. Since v27's multiply broadcasts one half of a scale word, a word may carry two *different* scales.
   Each 32-bit row word of scale codes now becomes two words: atoms 0|2 (`x02`) and 1|3 (`x13`):
   1 LOP + 1 PRMT + 2 IMAD instead of 4 PRMT + 4 IMAD.
2. 1-byte-scale twins of the 3-stage configs, so verify-128/256 could try 1-byte scales too.

## Why it is faster

Saves 8 ALU + 8 IMAD ops per tile per warp on the pipe v27 showed is the bottleneck. The patcher
checks every scale against v27's decode for every code byte before writing the file. decode-32:
148.6 → 135.3 µs. +2.3% overall.

## The code that produced the speedup

### Two-scale decode

From `submission_v28.py`, original line 531:

```cpp
        // two scales per word: atoms 0|2 and 1|3 (hmul2_sc<0> picks the half); == scale_dup per half
        const uint32_t r0_02 = (u.x & 0x00FF00FFu) * k16 + 0x58805880u;
        const uint32_t r0_13 = __byte_perm(u.x, 0u, 0x4341u) * k16 + 0x58805880u;
        const uint32_t r1_02 = (u.y & 0x00FF00FFu) * k16 + 0x58805880u;
        const uint32_t r1_13 = __byte_perm(u.y, 0u, 0x4341u) * k16 + 0x58805880u;
        sc[m][0][0] = r0_02;
        sc[m][0][1] = r0_13;
        sc[m][0][2] = r0_02;
        sc[m][0][3] = r0_13;
        sc[m][1][0] = r1_02;
        sc[m][1][1] = r1_13;
        sc[m][1][2] = r1_02;
        sc[m][1][3] = r1_13;
```

### Half selection in the multiply

From `submission_v28.py`, original line 336:

```cpp
template <int SF>
__device__ __forceinline__ uint32_t hmul2_sc(uint32_t a, uint32_t b, int s) {
  if constexpr (SF == 1) return (s & 1) ? hmul2_hi(a, b) : hmul2_lo(a, b);
  else if constexpr (SF == 0) return (s & 2) ? hmul2_hi(a, b) : hmul2_lo(a, b);  // x02 / x13 pairs
  else return hmul2(a, b);
}
```
