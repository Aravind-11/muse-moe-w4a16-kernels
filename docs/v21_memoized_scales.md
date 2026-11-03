# v21 — scales pre-converted (memoized) in prepare()

**Judged score:** 1.6253× (`27dd3086169e4155`)  
**Previous best:** 1.5532× (v16)  
**Built on:** v16  
**File:** `submission_v21.py` (search it for `SPEEDUP (v21)`)

## What this kernel does

`prepare()` stores the block scales a second and third time, already converted to the bf16 bits the
kernel used to compute every tile: `SF=1` (bf16 per scale) and `SF=2` (pre-duplicated bf16x2). The
kernel reads them instead of decoding. The variant's stage field carries the format
(`100*SF + stages`), and the gate measures the bytes-vs-instructions trade per shape.

## Why it is faster

Every K tile each warp spent 32 instructions (PRMT + LEA, ~13% of the loop) converting the same
constant scale bytes, on every call. Moving that to setup removes up to 16 (SF=1) or 32 (SF=2)
instructions per tile for 11% / 33% more scale bytes. The kernel was instruction-bound, so this won,
most clearly on decode-32 (+7%) and decode-16 (+2–3%). Bits are identical to the old decode.

## The code that produced the speedup

### Setup-time conversion

From `submission_v21.py`, original line 215:

```python
        # Memoized scale conversion: the exact bf16 bits the kernel's scale_dup produces, (code << 4) + 0x5880,
        # stored as bf16 (fmt 1) or as the duplicated bf16x2 word (fmt 2), little-endian.
        h = (sc.to(torch.int32) << 4) + 0x5880
        h = h.to(torch.int16) if fmt == 1 else (h | (h << 16)).to(torch.int32)
        sc = h.contiguous().view(torch.uint8).reshape(kt_n, nt_n, mats, -1)
```

### Kernel: read instead of decode

From `submission_v21.py`, original line 452:

```cpp
      if constexpr (SF == 0) {
        const uint2 u = *reinterpret_cast<const uint2*>(pw + m * MATB + 512 + grp * 8);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          sc[m][0][s] = scale_dup(u.x, s);
          sc[m][1][s] = scale_dup(u.y, s);
        }
      } else if constexpr (SF == 1) {
        // bf16 per scale: rows grp / grp+8, atoms 0..3, two per word -> duplicate the half.
        const uint4 v = *reinterpret_cast<const uint4*>(pw + m * MATB + 512 + grp * 16);
        sc[m][0][0] = __byte_perm(v.x, 0u, 0x1010u);
        sc[m][0][1] = __byte_perm(v.x, 0u, 0x3232u);
        sc[m][0][2] = __byte_perm(v.y, 0u, 0x1010u);
        sc[m][0][3] = __byte_perm(v.y, 0u, 0x3232u);
        sc[m][1][0] = __byte_perm(v.z, 0u, 0x1010u);
        sc[m][1][1] = __byte_perm(v.z, 0u, 0x3232u);
        sc[m][1][2] = __byte_perm(v.w, 0u, 0x1010u);
        sc[m][1][3] = __byte_perm(v.w, 0u, 0x3232u);
```
