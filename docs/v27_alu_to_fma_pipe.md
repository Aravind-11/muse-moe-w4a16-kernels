# v27 — decode work moved from the ALU pipe to the FMA pipe

**Judged score:** 1.8042× scored (`01b2da2380c44ff8`)  
**Previous best:** 1.6776× (v26)  
**Built on:** v26  
**File:** `submission_v27.py` (search it for `SPEEDUP (v27)`)

## What this kernel does

Four exact, instruction-level changes (results bit-identical):
1. Dequant: `rotl(w,4) & M` → `(w*16) & M` and `rotl(w,6)` → `w*64`. Every bit a rotate would wrap is
   cleared by the mask, so shift == rotate (checked on 200k random words + edge cases).
2. 1-byte scale decode: the `* 16` multiplier is made opaque to the compiler (`16 + (mode >> 30)`), so
   it compiles to IMAD instead of LEA.
3. bf16 scales: the 8 `__byte_perm` duplications per matrix per tile are gone; `__hmul2` with
   `__low2bfloat162`/`__high2bfloat162` broadcasts the needed half through HMUL2's operand selector.
4. L2 hints: weight TMA copies evict-first (read once), GEMM1's outputs evict-last (GEMM2 rereads them).

## Why it is faster

The 4-bit decode saturates the integer **ALU** pipe. `SHF`/`LEA` run there; `IMAD` runs on the
**FMA** pipe, which had room. Moving 2 of the 9 dequant ops and the scale multiply across, and deleting
the PRMTs, unblocked the binding pipe. Bonus: 1-byte scales became cheap enough that the gate switched
decode-16/32 and verify-8 back to them (10% fewer bytes): decode-16 GEMM1 102.9 → 77.2 µs.
**+7.5% overall, the biggest single step.**

## The code that produced the speedup

### Dequant shifts as multiplies

From `submission_v27.py`, original line 309:

```cpp
__device__ __forceinline__ void deq(uint32_t w, uint32_t r[4]) {
  const uint32_t M = 0x81C081C0u;
  r[0] = w & M;
  r[1] = (w * 16u) & M;  // == rotl(w,4) & M (wrapped bits are masked); IMAD, not SHF
  r[2] = __funnelshift_l(w, w, 10) & M;
  r[3] = ((w * 64u) & 0x80C080C0u) | (__funnelshift_l(w, w, 14) & 0x01000100u);  // w*64 == rotl(w,6) under the mask
}
```

### Opaque ×16 and in-multiply broadcast

From `submission_v27.py`, original line 318:

```cpp
__device__ __forceinline__ uint32_t scale_dup(uint32_t u, int s, uint32_t k16) {
  uint32_t x = __byte_perm(u, 0u, 0x4040u | ((unsigned)s << 8) | (unsigned)s);
  return x * k16 + 0x58805880u;  // k16 == 16 at run time, opaque to ptxas: IMAD (FMA pipe), not LEA (ALU)
}

// bf16 scale word b holds two scales; multiply a by its low or high half broadcast to both lanes.
// __hmul2 with __low2bfloat162/__high2bfloat162 folds into HMUL2's .H0_H0/.H1_H1 operand selector.
__device__ __forceinline__ uint32_t hmul2_lo(uint32_t a, uint32_t b) {
  const __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a),
                                   __low2bfloat162(*reinterpret_cast<const __nv_bfloat162*>(&b)));
  return *reinterpret_cast<const uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t hmul2_hi(uint32_t a, uint32_t b) {
  const __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a),
                                   __high2bfloat162(*reinterpret_cast<const __nv_bfloat162*>(&b)));
  return *reinterpret_cast<const uint32_t*>(&r);
}
// scale multiply for format SF: SF 1 keeps raw words (scale s = half s & 1); others are pre-duplicated
template <int SF>
__device__ __forceinline__ uint32_t hmul2_sc(uint32_t a, uint32_t b, int s) {
  if constexpr (SF == 1) return (s & 1) ? hmul2_hi(a, b) : hmul2_lo(a, b);
  else return hmul2(a, b);
}

```

### bf16 scales without byte_perm

From `submission_v27.py`, original line 538:

```cpp
        // raw words; hmul2_sc<1> broadcasts half (s & 1) inside the multiply (no PRMT)
        sc[m][0][0] = v.x;
        sc[m][0][1] = v.x;
        sc[m][0][2] = v.y;
        sc[m][0][3] = v.y;
        sc[m][1][0] = v.z;
        sc[m][1][1] = v.z;
        sc[m][1][2] = v.w;
        sc[m][1][3] = v.w;
```
