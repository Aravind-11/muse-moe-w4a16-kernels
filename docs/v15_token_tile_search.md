# v15 — per-shape token-tile search

**Judged score:** 1.4018× (`8eb6b3747d7d4ad9`)  
**Previous best:** 1.2583× (v9 scored)  
**Built on:** v9g (v9 with gate fixes); the CUDA kernel is unchanged  
**File:** `submission_v15.py` (search it for `SPEEDUP (v15)`)

## What this kernel does

No kernel change. The gate now also times the full custom path with token tiles **smaller** than the
shape's default (8, 16, 32 tokens per CTA), every GEMM1 × GEMM2 variant pair, each checked for
correctness, and keeps the fastest.

## Why it is faster

Measured on the judge: verify-128/256 defaulted to 48/64-token tiles, which needed 88–102 registers
and fit 2 CTAs per SM. A 16-token tile needs 55 registers and fits 4 per SM, and most local experts
receive ≤16 tokens anyway, so the bigger tiles mostly multiplied padding. Forcing 16-token tiles
cut full time 21% / 25% on verify-128 / verify-256. +11% overall.

## The code that produced the speedup

### Tile search in the gate

From `submission_v15.py`, original line 1239:

```python
        for tb in TILE_CANDIDATES:
            if tb >= bsm:
                continue
            if _elapsed() > PREPARE_SOFT_LIMIT_S:
                print(f"TILE_SKIP shape={name} bsm={tb} prepare_elapsed_s={_elapsed():.1f}", flush=True)
                break
            for tv1 in range(len(G1_VARIANTS[tb])):
                for tv2 in range(len(G2_VARIANTS[tb])):
                    ok, us = _try(name, f"TILE bsm={tb} g1={tv1} g2={tv2} norm=True",
                                  lambda d, s, a=tv1, b=tv2, t=tb: new_fn(d, s, v1=a, v2=b, norm=True, full=True, bsm=t),
                                  data, state, ref, mode_box)
                    if ok and us < best[3]:
                        best = (tv1, tv2, True, us)
                        full, tile = True, tb
        if tile is not None:
```
