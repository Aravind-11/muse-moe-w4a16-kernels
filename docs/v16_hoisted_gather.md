# v16 — activation-gather addresses hoisted out of the K loop

**Judged score:** 1.5532× (`2a8a1508d44f46d4`; second run 1.5585×)  
**Previous best:** 1.4018× (v15)  
**Built on:** v15  
**File:** `submission_v16.py` (search it for `SPEEDUP (v16)`)

## What this kernel does

Each thread's activation copies (source pointer, shared-memory destination, valid flag) are now
computed **once before the K loop**. Inside the loop, a tile only adds its K offset and stage offset.

## Why it is faster

v15's 16-token kernels recomputed every copy address on every K tile: a token-offset shared load
plus ~10 IMAD/SHF/LEA, inside a strided loop the compiler could not prove ran once. That was pure
per-tile overhead on an instruction-bound kernel. The static loop shrank from 490 to 258 instructions
(~14% fewer executed per tile). Same data, same results. +11% overall.

## The code that produced the speedup

### Gather plan computed once

From `submission_v16.py`, original line 370:

```cpp
  // Activation gather plan, fixed for the whole K loop: chunk i = tid + j*NTH is token tok's 16-byte
  // piece c of K sub-tile k2. Per tile only the K offset (and the stage) changes.
  constexpr int NCH = KS * T * 8;
  constexpr int CPT = (NCH + NTH - 1) / NTH;
  const __nv_bfloat16* asrc[CPT];
  unsigned adst[CPT];
  bool aval[CPT];
  const unsigned sbase_u32 = smem_u32(stages);
#pragma unroll
  for (int j = 0; j < CPT; ++j) {
    const int i = tid + j * NTH;
    const int k2 = i / (T * 8);
    const int jj = i - k2 * (T * 8);
    const int tok = jj >> 3;
    const int c = jj & 7;
    const int off = (i < NCH) ? tokoff[tok] : -1;
    aval[j] = off >= 0;
    asrc[j] = X + (aval[j] ? off : 0) + k2 * 64 + c * 8;
    adst[j] = sbase_u32 + (unsigned)(k2 * SUB + SUB_W + tok * (ACT_LD * 2) + c * 16);
```

### Per-tile copy issue

From `submission_v16.py`, original line 391:

```cpp
  auto issue = [&](int kto, int st) {
    char* sbase = stages + st * STAGE;
    if (tid == 0) {
      mbar_expect_tx(&bars[st], (unsigned)(KS * SUB_W));
#pragma unroll
      for (int k2 = 0; k2 < KS; ++k2) {
        tma_1d(sbase + k2 * SUB, wsrc + (size_t)(kto * KS + k2) * NTU * PIECE, (unsigned)SUB_W, &bars[st]);
      }
    }
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      if (aval[j]) cp16a(adst[j] + (unsigned)(st * STAGE), asrc[j] + kto * (KS * 64));
    }
  };
```
