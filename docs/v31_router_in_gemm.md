# v31 — routing folded into the GEMM CTAs (decode-1)

**Judged score:** 1.8541× scored (`cfd709c24f694754`)  
**Previous best:** 1.8458× (v28)  
**Built on:** v28  
**File:** `submission_v31.py` (search it for `SPEEDUP (v31)`)

## What this kernel does

Mode bit 16 (`inroute`): each GEMM CTA computes its own block's routing from `topk_ids`: per-expert
counts (warp `__match_any_sync` + shared atomics), block-padded offsets, its block's expert, then its
block's token slots as the j0..j0+T−1-th occurrences of that expert (ballots + a block scan). It is
deterministic, so every CTA of a block in both GEMMs agrees. The GEMM1→GEMM2 counters move to a
persistent buffer that the block's last GEMM2 CTA re-arms. Gate-selected, bit-exact self-test first.

## Why it is faster

The judge forgives one graph launch per call, not one per kernel. Folding the router removes a
launch and the 2.2 µs router kernel from GEMM1's critical path. It only pays when the scan is tiny,
so the gate kept it on **decode-1 only** (40.2 → 37.8 µs, first shape above 2.0×); on verify-128 every
CTA re-scanning all ids cost more (285 vs 224 µs).

## The code that produced the speedup

### Per-CTA routing

From `submission_v31.py`, original line 380:

```cpp
  // mode bit 16 (inroute): sorted_ids is topk_ids (local_start 0); this CTA derives its block's expert,
  // the padded total and (below) its slots itself, deterministically, instead of reading the router's output.
  const bool inroute = (mode & 16) != 0;
  __shared__ int r_cnt[32];
  __shared__ int r_info[3];  // expert, first occurrence index j0, padded total
  int expert, j0 = 0, npad_v;
  if (inroute) {
    const int t0 = (int)threadIdx.x;
    if (t0 < 32) r_cnt[t0] = 0;
    __syncthreads();
    for (int base = 0; base < numel; base += NTH) {  // uniform trip count: all lanes reach the match
      const int i = base + t0;
      const int e = i < numel ? sorted_ids[i] : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if ((unsigned)e < 32u && (t0 & 31) == __ffs(peers) - 1) atomicAdd(&r_cnt[e], __popc(peers));
    }
    __syncthreads();
    if (t0 == 0) {
      int off = 0, eh = -1, jj = 0;
      const int pos = blk * T;
      for (int k = 0; k < 32; ++k) {
        const int nx = off + ((r_cnt[k] + T - 1) / T) * T;
        if (pos >= off && pos < nx) {
          eh = k;
          jj = pos - off;
        }
        off = nx;
      }
      r_info[0] = eh;
      r_info[1] = jj;
      r_info[2] = off;
    }
    __syncthreads();
    expert = r_info[0];
    j0 = r_info[1];
    npad_v = r_info[2];
  } else {
    npad_v = *n_padded;
    expert = blk * T < npad_v ? block_experts[blk] : -1;
  }
  if (blk * T >= npad_v) return;
  if (expert < 0) return;

```

### Block slot selection

From `submission_v31.py`, original line 446:

```cpp
    // slots = the j0 .. j0+T-1-th occurrences of `expert` in topk_ids, in index order
    __shared__ int r_w[32];
    if (tid < T) slots[tid] = -1;
    __syncthreads();
    int run = 0;
    for (int base = 0; base < numel && run < j0 + T; base += NTH) {  // run is CTA-uniform
      const int i = base + tid;
      const bool hit = i < numel && sorted_ids[i] == expert;
      const unsigned bal = __ballot_sync(0xffffffffu, hit);
      if (lane == 0) r_w[warp] = __popc(bal);
      __syncthreads();
      int woff = 0, tot = 0;
#pragma unroll 1
      for (int w = 0; w < NTH / 32; ++w) {
        const int c = r_w[w];
        woff += w < warp ? c : 0;
        tot += c;
      }
      const int rank = run + woff + __popc(bal & ((1u << lane) - 1u));
      if (hit && rank >= j0 && rank < j0 + T) slots[rank - j0] = i;
      run += tot;
      __syncthreads();
    }
    if (tid < T) {
      const int slot = slots[tid];
```
