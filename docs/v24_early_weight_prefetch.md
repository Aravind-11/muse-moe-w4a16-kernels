# v24 — GEMM2 prefetches its weights before waiting on GEMM1

**Judged score:** 1.6565× scored (`37b2b48ef0f24d22`)  
**Previous best:** 1.6294× (v22)  
**Built on:** v22  
**File:** `submission_v24.py` (search it for `SPEEDUP (v24)`)

## What this kernel does

New mode bit (`early`): GEMM1 triggers its dependents right after its own wait, so GEMM2's CTAs can
become resident during GEMM1's last wave. GEMM2 reads the router's outputs, issues the TMA copies of
its first S−1 **weight** stages, and only then waits for GEMM1, before gathering activations. A
bit-exact self-test in `prepare()` must pass before the gate may use it; the gate keeps it per shape
only when faster. Also added: one `SUMMARY` line per shape printed at the end of `prepare()`, so the
gate's decisions survive the judge's 16 KB log tail.

## Why it is faster

In v22 every GEMM waited as its first instruction, so PDL only hid launch latency. GEMM2's weights
do not depend on GEMM1 at all, so streaming them while GEMM1 finishes fills SMs that GEMM1's tail wave
leaves idle. Kept on 5 of 6 shapes, 1–2.5 µs each. +1.7% overall.

## The code that produced the speedup

### Early trigger / delayed wait

From `submission_v24.py`, original line 340:

```cpp
  const bool early = (mode & 4) != 0;
  if (KIND == 1 || !early) pdl_wait();  // inputs come from the previous kernel (PDL launch); no-op on a plain launch
  if (KIND == 1 && early) pdl_trigger();
```

### Weight stages issued before the wait

From `submission_v24.py`, original line 435:

```cpp
  if (KIND == 2 && early) {
    // Weights need only the router's output; stream them while GEMM1 finishes, then wait for its activations.
#pragma unroll
    for (int s = 0; s < S - 1; ++s) issue_w(s, s);
    pdl_wait();
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
      issue_a(s, s);
      cp_commit();
    }
  } else {
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
      issue(s, s);
      cp_commit();
    }
```
