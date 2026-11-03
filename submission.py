# okc: problem=muse_moe_w4a16
# ==================================================================================================
# v7 — first custom kernel to beat SGLang
# Judged score: 1.0118× (check `ddc11e6cef6449a5`)   (previous best: 0.9910× (Marlin fallback, scored `b7364df108d748cc`))
# Built on: the S2 fused prototype (v4_fused/v5/v6), which always fell back to Marlin
#
# What this version does and why it is faster: see README.md in this folder.
# Every line that produced this version's speedup is marked with a comment starting
# 'SPEEDUP (v7)' (search for it). Those comments are the only change from the judged file;
# no code was altered.
# ==================================================================================================
"""NVFP4 routed experts on H200.

S3: both expert GEMMs run on one swap-AB mma.m16n8k16 BF16 kernel (weights on M, tokens on N).
GEMM1 fuses silu(gate)*up into its epilogue (a warp owns 16 gate rows and the matching 16 up rows)
and writes the GEMM2 input directly; GEMM2 writes y for the Triton post-norm.

Per 64-K tile and 16-row unit, the kernel spends ~190 instructions instead of ~277 in S2:
  * weights repacked kt-major so one CTA's stage is a single TMA bulk copy (one thread, mbarrier),
  * E2M1 -> bf16x2 in 9 integer ops per 8 codes (rotate/mask layout found by search),
  * E4M3 block scales recoded at repack so decode is one byte_perm + one IMAD,
  * activation fragments via ldmatrix.x4, padded rows, conflict-free,
  * fp32 accumulate, gs * 2^66 folded into the epilogue.
prepare() repacks the 32 local experts (same byte count), builds one .cu per (GEMM, block size) in
parallel, and per scored shape keeps the fastest of {Marlin, custom GEMM1 + Marlin GEMM2, both custom}
by CUDA-graph replay with a cold L2. A configuration must match Marlin under the checks.py rule and
beat it by 3%; anything else stays on Marlin.
"""
from typing import Callable, TypedDict

import types

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8

# checks.py: relative L2 <= 1.5%, and every row <= 5x that, with a floor at
# 10% of the mean row norm. Inlined so prepare() does not import the reference.
REL_L2_TOL = 1.5e-2
ROW_FACTOR = 5.0
ROW_NORM_FLOOR = 0.1

# Judge shapes, in leaderboard order. Names are the load plus the token count.
SCORED_SHAPES = (
    ("decode-bs1", 1),
    ("decode-bs16", 16),
    ("decode-bs32", 32),
    ("verify-bs8", 8),
    ("verify-bs128", 128),
    ("verify-bs256", 256),
)
GATE_WARMUP = 3
GATE_ITERS = 7
GRAPH_ITERS = 21
GEMM1_EXPERT_BYTES = 8192 * (2048 + 256)   # packed weights + scales of one local expert's W13


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


# ---PACK-BEGIN---
# Bit layout of one packed 32-bit word (8 E2M1 codes, code bits = [sign e1 e0 m]).
# The kernel makes four bf16x2 A registers from it with 9 integer ops:
#   r0 = w & M, r1 = rotl(w,4) & M, r2 = rotl(w,10) & M,
#   r3 = (rotl(w,6) & 0x80C080C0) | (rotl(w,14) & 0x01000100),   M = 0x81C081C0.
# A code c placed at bf16 bits c<<6 (sign at bit 15) is the E2M1 value times 2^-126
# (denormal for c == 1); the scale multiply restores the exponent.
_P = (6, 7, 8, 15, 22, 23, 24, 31)          # bf16x2 positions: lo (m,e0,e1,sign), hi (m,e0,e1,sign)
_ROT = (0, 4, 10)                           # regs 0..2: rotl(w, _ROT[j]) & M
_SRC_REG3 = (0, 1, 26, 9, 16, 17, 10, 25)   # reg 3 sources, in _P order


def _word_sources():
    """Return, for each word bit q, the (slot, code_bit) stored there. slot = 2*reg + (0 lo, 1 hi)."""
    src = {}
    for j in range(3):
        for i in range(8):
            q = (_P[i] - _ROT[j]) % 32
            assert q not in src
            src[q] = (2 * j + (i >> 2), i & 3)
    for i in range(8):
        assert _SRC_REG3[i] not in src
        src[_SRC_REG3[i]] = (6 + (i >> 2), i & 3)
    assert sorted(src) == list(range(32))
    return [src[q] for q in range(32)]


def _pack_indices(nt_n, kt_n, mats, half_rows):
    """numpy gather indices. Layout: [kt][nt][mat][lane][atom][4 bytes] + per (kt,nt,mat) 64 scale bytes.

    kt-major, so one CTA's NW consecutive 16-row units for one K tile are one contiguous TMA copy."""
    import numpy as np

    shape = (kt_n, nt_n, mats, 32, 4)
    kt = np.arange(kt_n).reshape(-1, 1, 1, 1, 1)
    nt = np.arange(nt_n).reshape(1, -1, 1, 1, 1)
    mat = np.arange(mats).reshape(1, 1, -1, 1, 1)
    lane = np.arange(32).reshape(1, 1, 1, -1, 1)
    atom = np.arange(4).reshape(1, 1, 1, 1, -1)
    grp = lane >> 2
    tig = lane & 3
    ix = {"n": [], "kb": [], "sh": []}
    for i in range(8):
        dk = i & 1
        rsel = (i >> 1) & 1
        kh = i >> 2
        n = mat * half_rows + nt * 16 + grp + 8 * rsel
        k = kt * 64 + atom * 16 + tig * 2 + dk + 8 * kh
        n = np.broadcast_to(n, shape).astype(np.int64)
        k = np.broadcast_to(k, shape).astype(np.int64)
        ix["n"].append(n)
        ix["kb"].append(k >> 1)
        ix["sh"].append(((k & 1) * 4).astype(np.uint8))
    shape_s = (kt_n, nt_n, mats, 8, 2, 4)
    kt = np.arange(kt_n).reshape(-1, 1, 1, 1, 1, 1)
    nt = np.arange(nt_n).reshape(1, -1, 1, 1, 1, 1)
    mat = np.arange(mats).reshape(1, 1, -1, 1, 1, 1)
    grp = np.arange(8).reshape(1, 1, 1, -1, 1, 1)
    half = np.arange(2).reshape(1, 1, 1, 1, -1, 1)
    atom = np.arange(4).reshape(1, 1, 1, 1, 1, -1)
    ix["sn"] = np.broadcast_to(mat * half_rows + nt * 16 + grp + 8 * half, shape_s).astype(np.int64)
    ix["sk"] = np.broadcast_to(kt * 4 + atom, shape_s).astype(np.int64)
    return ix


def _scale_code_lut():
    """e4m3 byte -> 8-bit code (e'<<3 | mant), e' = true exponent + 10, exact for every finite e4m3.

    The kernel turns a code into bf16 bits with one multiply-add: (code << 4) + (177 << 7) in both
    halves, i.e. scale * 2^60. Zero maps to 0; its blocks get their codes zeroed at repack."""
    import numpy as np

    lut = np.zeros(256, dtype=np.uint8)
    for b in range(256):
        v = b & 0x7F
        e, m = v >> 3, v & 7
        if v == 0:
            continue
        if e == 0:
            p = m.bit_length() - 1
            et = p - 9
            mant = int((m / (1 << p) - 1.0) * 8)
        else:
            et = e - 7
            mant = m
        lut[b] = ((et + 10) << 3) | mant
    return lut


def _pack_expert(w_e, sfc_e, ix, stack, cat):
    """w_e uint8 [rows, K/2] (low nibble = even k), sfc_e coded scales [rows, K/16] -> flat pieces."""
    g = [(w_e[ix["n"][i], ix["kb"][i]] >> ix["sh"][i]) & 15 for i in range(8)]
    # A zero block scale makes the block exactly zero in the reference; zero its codes too.
    g = [g[i] * (sfc_e[ix["n"][i], ix["kb"][i] >> 3] != 0) for i in range(8)]
    byts = [None] * 4
    for q, (i, cb) in enumerate(_word_sources()):
        bit = ((g[i] >> cb) & 1) << (q % 8)
        b = q // 8
        byts[b] = bit if byts[b] is None else (byts[b] | bit)
    words = stack(byts)
    kt_n, nt_n, mats = words.shape[0], words.shape[1], words.shape[2]
    words = words.reshape(kt_n, nt_n, mats, 512)
    sc = sfc_e[ix["sn"], ix["sk"]].reshape(kt_n, nt_n, mats, 64)
    return cat([words, sc]).reshape(-1)
# ---PACK-END---

# Device code only (no PyTorch headers: nvcc on the judge image cannot parse them).
_CUDA_COMMON = r'''
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int KT = 64;        // K = 4096 in tiles of 64
constexpr int ACT_LD = 72;    // bf16 per smem activation row (64 + 8 pad: conflict-free ldmatrix)
constexpr int KDIM = 4096;
constexpr int NOUT = 4096;    // output rows of both GEMMs (gate/up pairs for GEMM1, H for GEMM2)
constexpr int NT16 = 256;     // 16-row units in NOUT

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(smem)), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_1d(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t r[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ void mma16816(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// SPEEDUP (v7): exact NVFP4 decode in 9 integer ops per 8 weights. The repack (_word_sources) places
// each E2M1 code's bits so a rotate + mask lands them in bf16 exponent/mantissa fields (value * 2^-126).
// Packed word -> four bf16x2 A registers (E2M1 value * 2^-126). Layout in _word_sources().
__device__ __forceinline__ void deq(uint32_t w, uint32_t r[4]) {
  const uint32_t M = 0x81C081C0u;
  r[0] = w & M;
  r[1] = __funnelshift_l(w, w, 4) & M;
  r[2] = __funnelshift_l(w, w, 10) & M;
  r[3] = (__funnelshift_l(w, w, 6) & 0x80C080C0u) | (__funnelshift_l(w, w, 14) & 0x01000100u);
}

// SPEEDUP (v7): scale decode is one byte_perm (duplicate the byte) + one IMAD (shift into bf16 + rebias).
// Coded scale byte s of u -> bf16x2 (both halves) = scale * 2^60. Then A = w * 2^-66.
__device__ __forceinline__ uint32_t scale_dup(uint32_t u, int s) {
  uint32_t x = __byte_perm(u, 0u, 0x4040u | ((unsigned)s << 8) | (unsigned)s);
  return x * 16u + 0x58805880u;
}

// MATS == 2: GEMM1, gate+up rows per 16-row unit, epilogue silu(gate)*up, activation row = token.
// MATS == 1: GEMM2, activation row = slot (cache2), epilogue plain bf16.
template <int MATS, int NW, int G, int S>
__global__ void __launch_bounds__(NW * 32)
moe_gemm_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W,
                const float* __restrict__ GS, const int* __restrict__ sorted_ids,
                const int* __restrict__ block_experts, const int* __restrict__ n_padded,
                __nv_bfloat16* __restrict__ OUT, int numel, int snumel, int mode) {
  constexpr int T = G * 8;
  constexpr int NTH = NW * 32;
  constexpr int PIECE = MATS * 576;
  constexpr int W_STAGE = NW * PIECE;
  constexpr int A_STAGE = T * ACT_LD * 2;
  constexpr int STAGE = W_STAGE + A_STAGE;

  const int blk = (int)blockIdx.y;
  if (blk * T >= *n_padded) return;
  const int expert = block_experts[blk];
  if (expert < 0) return;

  extern __shared__ __align__(128) unsigned char dyn[];
  char* stages = reinterpret_cast<char*>(dyn);
  int* tokoff = reinterpret_cast<int*>(dyn + S * STAGE);
  int* slots = tokoff + T;
  __shared__ int nact_s;
  __shared__ __align__(8) uint64_t bars[S];

  const int tid = (int)threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int grp = lane >> 2;
  const int tig = lane & 3;

  if (tid == 0) {
    nact_s = 0;
#pragma unroll
    for (int i = 0; i < S; ++i) mbar_init(&bars[i], 1u);
    mbar_fence_init();
  }
  __syncthreads();
  if (tid < T) {
    const int ii = blk * T + tid;
    const int slot = ii < snumel ? sorted_ids[ii] : numel;
    const bool valid = slot < numel;
    slots[tid] = valid ? slot : -1;
    tokoff[tid] = valid ? (MATS == 2 ? (slot >> 3) : slot) * KDIM : -1;
    if (valid) atomicMax(&nact_s, tid + 1);
  }
  __syncthreads();
  const int nact = nact_s;
  if (nact <= 0) return;

  const int nchunk = (int)blockIdx.x;
  const uint8_t* wsrc = W + ((size_t)expert * KT * NT16 + (size_t)nchunk * NW) * PIECE;

  auto issue = [&](int kt, int st) {
    char* sbase = stages + st * STAGE;
    if (tid == 0) {
      mbar_expect_tx(&bars[st], (unsigned)W_STAGE);
      tma_1d(sbase, wsrc + (size_t)kt * NT16 * PIECE, (unsigned)W_STAGE, &bars[st]);
    }
    char* adst = sbase + W_STAGE;
    for (int i = tid; i < T * 8; i += NTH) {
      const int tok = i >> 3;
      const int c = i & 7;
      const int off = tokoff[tok];
      if (off >= 0) cp16(adst + tok * (ACT_LD * 2) + c * 16, X + off + kt * 64 + c * 8);
    }
  };

  float acc[MATS][G][4];
#pragma unroll
  for (int m = 0; m < MATS; ++m)
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[m][g][i] = 0.0f;

#pragma unroll
  for (int s = 0; s < S - 1; ++s) {
    issue(s, s);
    cp_commit();
  }

  for (int kt = 0; kt < KT; ++kt) {
    const int st = kt % S;
    cp_wait<S - 2>();
    mbar_wait(&bars[st], (unsigned)((kt / S) & 1));
    __syncthreads();
    {
      const int nk = kt + S - 1;
      if (nk < KT) issue(nk, nk % S);
      cp_commit();
    }
    if (mode == 1) continue;

    const char* sbase = stages + st * STAGE;
    const char* pw = sbase + warp * PIECE;
    uint32_t wv[MATS][4];
    uint32_t sc[MATS][2][4];
#pragma unroll
    for (int m = 0; m < MATS; ++m) {
      const uint4 w4 = *reinterpret_cast<const uint4*>(pw + m * 576 + lane * 16);
      wv[m][0] = w4.x;
      wv[m][1] = w4.y;
      wv[m][2] = w4.z;
      wv[m][3] = w4.w;
      const uint2 u = *reinterpret_cast<const uint2*>(pw + m * 576 + 512 + grp * 8);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        sc[m][0][s] = scale_dup(u.x, s);
        sc[m][1][s] = scale_dup(u.y, s);
      }
    }
    const __nv_bfloat16* ab = reinterpret_cast<const __nv_bfloat16*>(sbase + W_STAGE);
#pragma unroll
    for (int sp = 0; sp < 4; sp += 2) {
      uint32_t bfr[G][4];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        if (g * 8 < nact) ldsm4(bfr[g], ab + (g * 8 + (lane & 7)) * ACT_LD + sp * 16 + (lane >> 3) * 8);
      }
#pragma unroll
      // SPEEDUP (v7): swap-AB inner loop: dequantize the warp's 16 weight rows once, multiply by the block scale,
      // then reuse those A fragments for every 8-token group of the tile (tokens are the small N dimension).
      for (int h = 0; h < 2; ++h) {
        const int s = sp + h;
#pragma unroll
        for (int m = 0; m < MATS; ++m) {
          uint32_t r[4];
          deq(wv[m][s], r);
          const uint32_t a0 = hmul2(r[0], sc[m][0][s]);
          const uint32_t a1 = hmul2(r[1], sc[m][1][s]);
          const uint32_t a2 = hmul2(r[2], sc[m][0][s]);
          const uint32_t a3 = hmul2(r[3], sc[m][1][s]);
#pragma unroll
          for (int g = 0; g < G; ++g) {
            if (g * 8 < nact) mma16816(acc[m][g], a0, a1, a2, a3, bfr[g][2 * h], bfr[g][2 * h + 1]);
          }
        }
      }
    }
  }

  if (mode == 1) return;
  const float gf = GS[expert] * 73786976294838206464.0f;  // gs * 2^66
  const int row0 = nchunk * (NW * 16) + warp * 16 + grp;
#pragma unroll
  for (int g = 0; g < G; ++g) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int idx = g * 8 + tig * 2 + c;
      const int slot = slots[idx];
      if (slot >= 0) {
        float v0, v1;
        if constexpr (MATS == 2) {
          const float x0 = acc[0][g][c] * gf;
          const float y0 = acc[MATS - 1][g][c] * gf;
          const float x1 = acc[0][g][2 + c] * gf;
          const float y1 = acc[MATS - 1][g][2 + c] * gf;
          v0 = x0 * __frcp_rn(1.0f + __expf(-x0)) * y0;
          v1 = x1 * __frcp_rn(1.0f + __expf(-x1)) * y1;
        } else {
          v0 = acc[0][g][c] * gf;
          v1 = acc[0][g][2 + c] * gf;
        }
        __nv_bfloat16* o = OUT + (size_t)slot * NOUT;
        o[row0] = __float2bfloat16(v0);
        o[row0 + 8] = __float2bfloat16(v1);
      }
    }
  }
}

template <int MATS, int NW, int G, int S>
struct Cfg {
  static constexpr size_t smem =
      (size_t)S * (NW * MATS * 576 + G * 8 * ACT_LD * 2) + (size_t)G * 8 * 8 + 128;
};

template <int MATS, int NW, int G, int S>
int run(const void* x, const void* w, const void* gs, const void* sorted_ids, const void* bexp,
        const void* npad, void* out, int numel, int snumel, int nblocks, int mode, cudaStream_t stream) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(moe_gemm_kernel<MATS, NW, G, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)Cfg<MATS, NW, G, S>::smem);
    init = true;
  }
  dim3 grid(NT16 / NW, nblocks);
  dim3 block(NW * 32);
  moe_gemm_kernel<MATS, NW, G, S><<<grid, block, Cfg<MATS, NW, G, S>::smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const uint8_t*>(w),
      reinterpret_cast<const float*>(gs), reinterpret_cast<const int*>(sorted_ids),
      reinterpret_cast<const int*>(bexp), reinterpret_cast<const int*>(npad),
      reinterpret_cast<__nv_bfloat16*>(out), numel, snumel, mode);
  return (int)cudaGetLastError();
}

template <int MATS, int NW, int G, int S>
void attrs_one(int* o) {
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, moe_gemm_kernel<MATS, NW, G, S>);
  cudaFuncSetAttribute(moe_gemm_kernel<MATS, NW, G, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)Cfg<MATS, NW, G, S>::smem);
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, moe_gemm_kernel<MATS, NW, G, S>, NW * 32,
                                                Cfg<MATS, NW, G, S>::smem);
  o[0] = a.numRegs;
  o[1] = (int)a.localSizeBytes;
  o[2] = NW;
  o[3] = (int)Cfg<MATS, NW, G, S>::smem;
  o[4] = blocks;
  o[5] = S;
}

}  // namespace
'''

# (warps per CTA, token groups of 8 = bsm/8, cp.async/TMA stages) per variant. Source of truth for
# both the generated CUDA dispatchers and the Python gate.
G1_VARIANTS = {8: [(4, 1, 6), (2, 1, 8)], 16: [(4, 2, 6), (2, 2, 8)], 32: [(8, 4, 5), (4, 4, 6)],
               48: [(8, 6, 4)], 64: [(8, 8, 4), (4, 8, 5)]}
G2_VARIANTS = {8: [(8, 1, 8), (4, 1, 8)], 16: [(8, 2, 8), (4, 2, 8)], 32: [(8, 4, 6)],
               48: [(8, 6, 5)], 64: [(8, 8, 5)]}
_ARGS_DECL = ("const void* x, const void* w, const void* gs, const void* sorted_ids, const void* bexp, "
              "const void* npad, void* out, int numel, int snumel, int nblocks, int mode, cudaStream_t stream")
_ARGS_PASS = "x, w, gs, sorted_ids, bexp, npad, out, numel, snumel, nblocks, mode, stream"


def _cuda_unit(kind: int, bsm: int) -> str:
    """One .cu translation unit: common code + dispatchers for (kind, bsm). Units compile in parallel."""
    mats = 2 if kind == 1 else 1
    table = (G1_VARIANTS if kind == 1 else G2_VARIANTS)[bsm]
    launch = "\n".join(f"  if (variant == {v}) return run<{mats}, {nw}, {g}, {s}>({_ARGS_PASS});"
                       for v, (nw, g, s) in enumerate(table))
    attrs = "\n".join(f"  if (variant == {v}) {{ attrs_one<{mats}, {nw}, {g}, {s}>(o); return; }}"
                      for v, (nw, g, s) in enumerate(table))
    return (_CUDA_COMMON
            + f'\nextern "C" int muse_launch_g{kind}_{bsm}(int variant, {_ARGS_DECL}) {{\n{launch}\n  return -1;\n}}\n'
            + f'extern "C" void muse_attrs_g{kind}_{bsm}(int variant, int* o) {{\n{attrs}\n  o[0] = -1;\n}}\n')


def _cpp_unit() -> str:
    bsms = sorted(G1_VARIANTS)
    decls = []
    for kind in (1, 2):
        for b in bsms:
            decls.append(f'extern "C" int muse_launch_g{kind}_{b}(int variant, const void*, const void*, const void*, '
                         f'const void*, const void*, const void*, void*, int, int, int, int, void*);')
            decls.append(f'extern "C" void muse_attrs_g{kind}_{b}(int variant, int* o);')
    lcases = "\n".join(f"    case {kind * 1000 + b}: rc = muse_launch_g{kind}_{b}(v, x.data_ptr(), w.data_ptr(), "
                       f"gs.data_ptr(), sid.data_ptr(), bexp.data_ptr(), npad.data_ptr(), out.data_ptr(), numel, "
                       f"snumel, nblocks, md, stream); break;" for kind in (1, 2) for b in bsms)
    acases = "\n".join(f"    case {kind * 1000 + b}: muse_attrs_g{kind}_{b}(v, o); break;" for kind in (1, 2) for b in bsms)
    return r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
''' + "\n".join(decls) + r'''

void launch(int64_t kind, torch::Tensor x, torch::Tensor w, torch::Tensor gs, torch::Tensor sid,
            torch::Tensor bexp, torch::Tensor npad, torch::Tensor out, int64_t bsm, int64_t variant, int64_t mode) {
  TORCH_CHECK(kind == 1 || kind == 2, "kind");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && x.size(1) == 4096, "x");
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kUInt8, "w");
  TORCH_CHECK(gs.is_cuda() && gs.is_contiguous() && gs.scalar_type() == torch::kFloat32 && gs.numel() == 32, "gs");
  TORCH_CHECK(sid.is_cuda() && sid.is_contiguous() && sid.scalar_type() == torch::kInt32, "sorted_ids");
  TORCH_CHECK(bexp.is_cuda() && bexp.is_contiguous() && bexp.scalar_type() == torch::kInt32, "block_experts");
  TORCH_CHECK(npad.is_cuda() && npad.scalar_type() == torch::kInt32 && npad.numel() >= 1, "n_padded");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kBFloat16 && out.size(1) == 4096, "out");
  const int64_t pieces = (int64_t)32 * 64 * 256;
  TORCH_CHECK(w.numel() == pieces * (kind == 1 ? 1152 : 576), "w bytes");
  const int64_t numel64 = kind == 1 ? x.size(0) * 8 : x.size(0);
  TORCH_CHECK(out.size(0) == numel64, "out rows");
  const int numel = static_cast<int>(numel64);
  const int snumel = static_cast<int>(sid.numel());
  const int nblocks = static_cast<int>(bexp.numel());
  const int v = static_cast<int>(variant);
  const int md = static_cast<int>(mode);
  void* stream = at::cuda::getCurrentCUDAStream().stream();
  int rc = -1;
  switch (static_cast<int>(kind * 1000 + bsm)) {
''' + lcases + r'''
    default: rc = -2;
  }
  TORCH_CHECK(rc == 0, "muse gemm launch failed rc=", rc, " kind=", kind, " bsm=", bsm, " variant=", variant);
}

std::vector<int64_t> attrs(int64_t kind, int64_t bsm, int64_t variant) {
  int o[8] = {-1, 0, 0, 0, 0, 0, 0, 0};
  const int v = static_cast<int>(variant);
  switch (static_cast<int>(kind * 1000 + bsm)) {
''' + acases + r'''
    default: break;
  }
  return std::vector<int64_t>(o, o + 6);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch);
  m.def("attrs", &attrs);
}
'''


def _repack(w, sf, mats, half_rows):
    """Repack 32 local experts into the kernel's kt-major fragment layout. Same byte count as the input."""
    rows = w.shape[1]
    if w.shape != (32, rows, 2048) or sf.shape != (32, rows, 256) or rows != mats * 4096:
        raise RuntimeError(f"unexpected weight shape {tuple(w.shape)} scales {tuple(sf.shape)}")
    dev = w.device
    ix_np = _pack_indices(256, 64, mats, half_rows)
    ix = {
        "n": [torch.from_numpy(a).to(dev) for a in ix_np["n"]],
        "kb": [torch.from_numpy(a).to(dev) for a in ix_np["kb"]],
        "sh": [torch.from_numpy(a).to(dev) for a in ix_np["sh"]],
        "sn": torch.from_numpy(ix_np["sn"]).to(dev),
        "sk": torch.from_numpy(ix_np["sk"]).to(dev),
    }
    del ix_np
    lut = torch.from_numpy(_scale_code_lut()).to(dev)
    sf_u8 = sf.contiguous().view(torch.uint8)
    out = torch.empty((32, 64 * 256 * mats * 576), dtype=torch.uint8, device=dev)
    for e in range(32):
        out[e] = _pack_expert(
            w[e].contiguous(), lut[sf_u8[e].long()], ix,
            lambda lst: torch.stack(lst, dim=-1),
            lambda lst: torch.cat(lst, dim=-1),
        )
    del ix
    torch.cuda.empty_cache()
    return out


def _build_extension():
    """Write one .cu per (kind, bsm) and build them in parallel with ninja."""
    import os
    import tempfile

    from torch.utils.cpp_extension import load

    root = tempfile.mkdtemp(prefix="muse_v7_")
    sources = []
    cpp = os.path.join(root, "muse_ext.cpp")
    with open(cpp, "w") as f:
        f.write(_cpp_unit())
    sources.append(cpp)
    for kind in (1, 2):
        for bsm in sorted(G1_VARIANTS):
            path = os.path.join(root, f"muse_g{kind}_{bsm}.cu")
            with open(path, "w") as f:
                f.write(_cuda_unit(kind, bsm))
            sources.append(path)
    os.environ.setdefault("MAX_JOBS", str(max(4, min(16, os.cpu_count() or 4))))
    return load(
        name="muse_v7_ext",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=root,
        with_cuda=True,
        verbose=False,
    )


def _compile_new_kernels(weights):
    """Return (forward, mode_box). prepare() catches failures."""
    w13 = weights["w13"][:32]
    w2 = weights["w2"][:32]
    gs13 = weights["w13_gs"][:32].contiguous().float()
    gs2 = weights["w2_gs"][:32].contiguous().float()
    p13 = _repack(w13, weights["w13_sf"][:32], 2, 4096)
    p2 = _repack(w2, weights["w2_sf"][:32], 1, 0)
    print(f"REPACK w13={p13.nbytes} w2={p2.nbytes} elapsed_s={_elapsed():.1f}", flush=True)
    mod = _build_extension()
    print(f"BUILD_OK elapsed_s={_elapsed():.1f}", flush=True)
    for kind, table in ((1, G1_VARIANTS), (2, G2_VARIANTS)):
        for bsm, vs in table.items():
            for v in range(len(vs)):
                regs, lmem, nw, smem_d, occ, stages = mod.attrs(kind, bsm, v)
                print(f"KERNEL_ATTR gemm{kind} bsm={bsm} variant={v} warps={nw} stages={stages} regs={regs} "
                      f"local_bytes={lmem} smem_dyn={smem_d} ctas_per_sm={occ}", flush=True)
    mode_box = [0]

    def _forward(data, state, v1=-1, v2=-1):
        cfg = data["config"]
        if (
            cfg["local_start"] != 0
            or cfg["local_count"] != 32
            or cfg["d_hidden"] != 4096
            or cfg["d_expert"] != 4096
            or cfg["top_k"] != TOP_K
            or cfg["n_experts"] != N_EXPERTS
        ):
            return _marlin_forward(data, state)
        if v1 == -1:
            v1, v2 = state["cfg"].get(cfg["bs"], (None, None))
        g1 = g2 = None
        if v1 is not None:
            def g1(x, sorted_ids, block_experts, n_padded, out, bsm):
                mod.launch(1, x, p13, gs13, sorted_ids, block_experts, n_padded, out, bsm,
                           min(v1, len(G1_VARIANTS[bsm]) - 1), mode_box[0])
        if v2 is not None:
            def g2(x, sorted_ids, block_experts, n_padded, out, bsm):
                mod.launch(2, x, p2, gs2, sorted_ids, block_experts, n_padded, out, bsm,
                           min(v2, len(G2_VARIANTS[bsm]) - 1), mode_box[0])
        return _marlin_forward(data, state, gemm1=g1, gemm2=g2)

    return _forward, mode_box


def _metrics(output, ref) -> tuple[bool, float, float]:
    """Return (ok, relative L2, worst-row relative L2) using the checks.py rule."""
    if not isinstance(output, torch.Tensor):
        return False, float("inf"), float("inf")
    if output.shape != ref.shape or output.dtype != ref.dtype:
        return False, float("inf"), float("inf")
    out = output.float()
    if not torch.isfinite(out).all():
        return False, float("inf"), float("inf")
    got = ref.float()
    diff = out - got
    err = float(diff.norm() / got.norm().clamp(min=1e-12))
    rows_diff = diff.reshape(-1, got.shape[-1]).norm(dim=-1)
    rows_ref = got.reshape(-1, got.shape[-1]).norm(dim=-1)
    floor = (ROW_NORM_FLOOR * rows_ref.mean()).clamp(min=1e-12)
    row_err = float((rows_diff / torch.maximum(rows_ref, floor)).max())
    ok = err <= REL_L2_TOL and row_err <= ROW_FACTOR * REL_L2_TOL
    return ok, err, row_err


def _random_routed(bs: int, device, hidden: int, d_expert: int, seed: int) -> Inputs:
    """One routed batch like reference.make_inputs(dist="skewed", rank="busiest") on `device`.

    Not the judge's inputs. Not used on the timed path.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    hidden_states = torch.randn((bs, hidden), generator=gen, dtype=torch.float32, device="cpu")
    # A few outlier activation channels, as in the skewed distribution.
    hot = torch.randperm(hidden, generator=gen)[:8]
    hidden_states[:, hot] *= 20.0
    hidden_states = hidden_states.to(device=device, dtype=torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS), generator=gen)
    scores = scores + torch.rand((N_EXPERTS,), generator=gen).pow(6) * 4
    vals, idx = scores.topk(TOP_K, dim=-1)
    pairs = torch.bincount((idx // 32).flatten(), minlength=N_EXPERTS // 32)
    idx = (idx - 32 * pairs.argmax()) % N_EXPERTS   # busiest GPU's experts become 0-31
    weights = vals / vals.sum(dim=-1, keepdim=True)
    return {
        "hidden_states": hidden_states,
        "topk_ids": idx.to(device=device, dtype=torch.int32),
        "topk_weights": weights.to(device=device, dtype=torch.float32),
        "config": {
            "bs": bs,
            "d_hidden": hidden,
            "d_expert": d_expert,
            "n_experts": N_EXPERTS,
            "local_start": 0,
            "local_count": 32,
            "top_k": TOP_K,
            "sharding": "ep8",
        },
    }


def _median_us(fn: Callable, data, state, warmup: int = GATE_WARMUP, iters: int = GATE_ITERS) -> float:
    """Eager CUDA-event median, in microseconds. Fallback only; the judge replays a CUDA graph."""
    for _ in range(warmup):
        fn(data, state)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn(data, state)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


_FLUSH = {}


def _graph_us(fn: Callable, data, state, iters: int = GRAPH_ITERS) -> float:
    """CUDA-graph replay median in microseconds with a cold L2, like the judge. Falls back to eager."""
    try:
        dev = data["hidden_states"].device
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn(data, state)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(data, state)
        if "buf" not in _FLUSH:
            _FLUSH["buf"] = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
        flush = _FLUSH["buf"]
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            flush.zero_()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1e3)
        samples.sort()
        del graph
        return samples[len(samples) // 2]
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"GRAPH_TIMING_FALLBACK {type(exc).__name__}: {exc}", flush=True)
        return _median_us(fn, data, state)


_T0 = [0.0]
PREPARE_SOFT_LIMIT_S = 230.0   # TASK.md: prepare() <= 300 s; keep shapes on Marlin past this


def _elapsed() -> float:
    import time

    return time.monotonic() - _T0[0]


def _bsm_for(bs: int) -> int:
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if bs * TOP_K / 32 / block_size_m < 0.9:
            return block_size_m
    return BLOCK_SIZE_M_CANDIDATES[-1]


def _breakdown(data, state, name):
    """Graph time of the Marlin pipeline cut after each stage; differences are per-stage costs."""
    try:
        t = [_graph_us(lambda d, s, u=u: _marlin_forward(d, s, upto=u), data, state) for u in (1, 2, 3, 4, 5)]
        print(f"BREAKDOWN shape={name} ids_align={t[0]:.2f} gemm1={t[1]-t[0]:.2f} silu={t[2]-t[1]:.2f} "
              f"gemm2={t[3]-t[2]:.2f} norm={t[4]-t[3]:.2f} total={t[4]:.2f}", flush=True)
    except Exception as exc:
        print(f"BREAKDOWN_FAIL shape={name} {type(exc).__name__}: {exc}", flush=True)


def _try(name, label, fn, data, state, ref, mode_box):
    """Correctness against Marlin, CUDA-graph time, and loads-only time for one configuration."""
    try:
        ok, err, row = _metrics(fn(data, state), ref)
        us = _graph_us(fn, data, state) if ok else float("inf")
        mode_box[0] = 1
        try:
            loads = _graph_us(fn, data, state)
        finally:
            mode_box[0] = 0
    except Exception as exc:
        mode_box[0] = 0
        print(f"CONFIG_FAIL shape={name} {label} {type(exc).__name__}: {exc}", flush=True)
        return False, float("inf")
    print(f"CONFIG shape={name} {label} ok={ok} err={err:.3e} row={row:.3e} us={us:.2f} loads_only_us={loads:.2f}",
          flush=True)
    return ok, us


def _gate_shapes(state, device) -> None:
    """Pick per shape: Marlin, custom GEMM1 + Marlin GEMM2, or both custom. Failures stay on Marlin."""
    new_fn = state["new_fn"]
    mode_box = state["mode_box"]
    bank = state["banks"][(0, 32)]
    hidden = state["norm_w"].shape[0]
    d_expert = bank["size_n"]
    for name, bs in SCORED_SHAPES:
        state["enabled"][bs] = False
        if _elapsed() > PREPARE_SOFT_LIMIT_S:
            print(f"GATE_SKIP shape={name} prepare_elapsed_s={_elapsed():.1f}", flush=True)
            continue
        try:
            data = _random_routed(bs, device, hidden, d_expert, seed=5252 + bs)
            local = bank["expert_map"][data["topk_ids"].long()]
            live = int(torch.unique(local[local >= 0]).numel())
            ref = _marlin_forward(data, state)
            marlin_us = _graph_us(_marlin_forward, data, state)
            _breakdown(data, state, name)
        except Exception as exc:
            print(f"GATE_FAIL shape={name} marlin {type(exc).__name__}: {exc}", flush=True)
            continue
        if new_fn is None:
            print(f"GATE shape={name} bs={bs} marlin_us={marlin_us:.2f} choice=marlin kernel=none", flush=True)
            continue
        bsm = _bsm_for(bs)
        best = (None, None, marlin_us)
        best_v1, best_v1_us = None, float("inf")
        for v1 in range(len(G1_VARIANTS[bsm])):
            ok, us = _try(name, f"g1={v1} g2=marlin", lambda d, s, v1=v1: new_fn(d, s, v1=v1, v2=None),
                          data, state, ref, mode_box)
            if ok and us < best_v1_us:
                best_v1, best_v1_us = v1, us
        if best_v1 is not None and best_v1_us < best[2]:
            best = (best_v1, None, best_v1_us)
        for v2 in range(len(G2_VARIANTS[bsm])):
            ok, us = _try(name, f"g1={best_v1} g2={v2}", lambda d, s, v2=v2: new_fn(d, s, v1=best_v1, v2=v2),
                          data, state, ref, mode_box)
            if ok and us < best[2]:
                best = (best_v1, v2, us)
        # SPEEDUP (v7): the gate keeps the custom kernel only where it beat Marlin (correctness-checked first).
        use_new = (best[0] is not None or best[1] is not None) and best[2] < marlin_us * 0.97
        state["enabled"][bs] = use_new
        state["cfg"][bs] = (best[0], best[1])
        print(f"GATE shape={name} bs={bs} live_experts={live} bsm={bsm} g1={best[0]} g2={best[1]} "
              f"new_us={best[2]:.2f} marlin_us={marlin_us:.2f} speedup={marlin_us / best[2]:.3f} "
              f"choice={'new' if use_new else 'marlin'}", flush=True)


def prepare(weights):
    import time

    _T0[0] = time.monotonic()
    device = weights["w13"].device
    # Our repack reads the raw NVFP4 tensors, so it runs before Marlin's (which may reuse storage).
    new_fn, mode_box, compile_err = None, [0], None
    try:
        new_fn, mode_box = _compile_new_kernels(weights)
    except Exception as exc:
        msg = str(exc)
        compile_err = f"{type(exc).__name__}: {msg[-4000:]}"
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    state = {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "enabled": {},
        "new_fn": new_fn,
        "mode_box": mode_box,
        "cfg": {},
    }
    if new_fn is None:
        print(f"COMPILE_FAIL {compile_err}", flush=True)
    else:
        print(f"COMPILE_OK s3-swapab-tma prepare_elapsed_s={_elapsed():.1f}", flush=True)
    try:
        _gate_shapes(state, device)
    except Exception as exc:
        state["enabled"] = {}
        state["new_fn"] = None
        print(f"GATE_ABORT {type(exc).__name__}: {exc}", flush=True)
    torch.cuda.synchronize()
    print(f"PREPARE_DONE elapsed_s={_elapsed():.1f} enabled={state['enabled']} cfg={state['cfg']}", flush=True)
    return state


def _marlin_forward(data, state, gemm1=None, gemm2=None, upto=5):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break

    # Ids are local (or -1). Histogram over the local expert count, not all 256.
    sorted_ids, block_experts, n_padded = moe_align_block_size(
        local_ids, block_size_m, bank["count"])
    if upto <= 1:
        return sorted_ids

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    if gemm1 is None:
        cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
        cache1 = moe_wna16_marlin_gemm(
            x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
            size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
        if upto <= 2:
            return cache1
        silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    else:
        # Fused GEMM1 + silu(gate) * up, straight into cache2.
        gemm1(x, sorted_ids, block_experts, n_padded, cache2, block_size_m)
    if upto <= 3:
        return cache2
    if gemm2 is None:
        y = moe_wna16_marlin_gemm(
            cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
            size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    else:
        gemm2(cache2, sorted_ids, block_experts, n_padded, cache3, block_size_m)
        y = cache3.view(num_tokens, TOP_K, size_k)
    if upto <= 4:
        return y
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def kernel_fn(data, state):
    # Shape dispatch is constant for a captured graph. Shapes that failed the
    # gate, and every batch size prepare() did not time, take Marlin.
    fn = state["new_fn"]
    if fn is not None and state["enabled"].get(data["config"]["bs"], False):
        return fn(data, state)
    return _marlin_forward(data, state)
