# v22 — programmatic dependent launch (PDL) + lighter loop bookkeeping

**Judged score:** 1.6294× scored (`a2cff897065f4452`)  
**Previous best:** 1.6253× (v21)  
**Built on:** v21  
**File:** `submission_v22.py` (search it for `SPEEDUP (v22)`)

## What this kernel does

1. **PDL**: route → GEMM1 → GEMM2 → norm are launched with the programmatic-stream-serialization
   attribute. A kernel's CTAs get scheduled while its predecessor is still finishing and wait at
   `griddepcontrol.wait`; producers call `griddepcontrol.launch_dependents` once their main work is done.
2. The K loop keeps stage index, ring parity and issue stage as running counters (no `%`/`/` per tile).

## Why it is faster

Launch latency of each kernel now overlaps the previous kernel's tail: 1–4 µs per shape. Small,
but the scored run moved up and the gate kept it on every shape where it measured faster.

## The code that produced the speedup

### PDL primitives

From `submission_v22.py`, original line 246:

```cpp
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" :::); }
```

### Wait at kernel entry

From `submission_v22.py`, original line 339:

```cpp
  pdl_wait();  // inputs come from the previous kernel (PDL launch); no-op on a plain launch
```

### Trigger after the main loop

From `submission_v22.py`, original line 547:

```cpp
  pdl_trigger();  // main work done: let the next kernel's CTAs start launching
```
