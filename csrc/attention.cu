// FlashAttention-style fused attention (forward, fp32).
//
// Computes softmax(scale * Q K^T + causal_mask) V without ever materialising the
// [Tq, Tk] score matrix in global memory. K and V are streamed through shared
// memory in tiles of BC keys; each query row keeps a running max m, running
// denominator l and an unnormalised output accumulator in registers, rescaled
// whenever a new tile raises the max (the "online softmax" trick).
//
// Work decomposition:
//   grid  = (ceil(Tq / BR), B * H)
//   block = NWARPS warps; each warp owns ROWS_PER_WARP query rows (BR total)
//   For each K/V tile of BC = 32 keys:
//     lane j computes the score for key j      (one dot product of length D)
//     warp_max / warp_sum give the tile max and tile denominator
//     lane d accumulates output dims d, d+32   (P broadcast via __shfl_sync)
//
// Shared-memory layout notes:
//   sK is padded to D+1 columns: lane j reads sK[j][d], so without the pad all
//   32 lanes would hit the same bank (stride 64 floats) -> 32-way conflict.
//   sQ[r][d] is read with the same address by every lane (broadcast, free).
//   sV[j][lane + 32*i] is read with consecutive addresses (conflict-free).
#include "common.cuh"
#include "kernels.h"

namespace gpt2cuda {

template <int D, int BC, int NWARPS, int ROWS_PER_WARP>
__global__ void __launch_bounds__(NWARPS * kWarpSize)
flash_attention_kernel(const AttnParams p) {
  static_assert(BC == kWarpSize, "one key per lane");
  static_assert(D % kWarpSize == 0, "head_dim must be a multiple of 32");
  constexpr int BR = NWARPS * ROWS_PER_WARP;
  constexpr int DPL = D / kWarpSize;  // output dims per lane
  constexpr int NT = NWARPS * kWarpSize;

  __shared__ float sQ[BR][D];
  __shared__ float sK[BC][D + 1];
  __shared__ float sV[BC][D];

  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int b = blockIdx.y / p.H;
  const int h = blockIdx.y % p.H;
  const int q0 = blockIdx.x * BR;
  const int offset = p.Tk - p.Tq;  // > 0 when decoding with a KV cache

  const float* qb = p.q + b * p.q_sb + h * p.q_sh;
  const float* kb = p.k + b * p.k_sb + h * p.k_sh;
  const float* vb = p.v + b * p.v_sb + h * p.v_sh;

  // Load the query tile once, pre-multiplied by the softmax scale.
  for (int i = tid; i < BR * D; i += NT) {
    const int r = i / D, d = i % D;
    const int qi = q0 + r;
    sQ[r][d] = qi < p.Tq ? qb[static_cast<int64_t>(qi) * p.q_st + d] * p.scale : 0.f;
  }

  float m[ROWS_PER_WARP], l[ROWS_PER_WARP], acc[ROWS_PER_WARP][DPL];
#pragma unroll
  for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
    m[rr] = -INFINITY;
    l[rr] = 0.f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[rr][i] = 0.f;
  }

  // With a causal mask, keys past the last query row of this tile are never needed.
  const int kv_end = p.causal ? min(p.Tk, min(q0 + BR, p.Tq) + offset) : p.Tk;

  for (int k0 = 0; k0 < kv_end; k0 += BC) {
    __syncthreads();  // previous tile fully consumed (and sQ visible on the first pass)
    for (int i = tid; i < BC * D; i += NT) {
      const int r = i / D, d = i % D;
      const int kj = k0 + r;
      const bool ok = kj < p.Tk;
      sK[r][d] = ok ? kb[static_cast<int64_t>(kj) * p.k_st + d] : 0.f;
      sV[r][d] = ok ? vb[static_cast<int64_t>(kj) * p.v_st + d] : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
      const int r = warp * ROWS_PER_WARP + rr;
      const int qi = q0 + r;
      if (qi >= p.Tq) continue;                          // warp-uniform
      const int last_key = p.causal ? qi + offset : p.Tk - 1;
      if (k0 > last_key) continue;                       // whole tile masked for this row

      const int kj = k0 + lane;
      float sc = 0.f;
#pragma unroll 16
      for (int d = 0; d < D; ++d) sc += sQ[r][d] * sK[lane][d];
      const bool valid = kj < p.Tk && kj <= last_key;
      if (!valid) sc = -INFINITY;

      const float m_new = fmaxf(m[rr], warp_max(sc));
      const float pj = valid ? __expf(sc - m_new) : 0.f;
      const float alpha = __expf(m[rr] - m_new);         // 0 on the first tile (m = -inf)
      l[rr] = l[rr] * alpha + warp_sum(pj);
#pragma unroll
      for (int i = 0; i < DPL; ++i) acc[rr][i] *= alpha;
#pragma unroll 8
      for (int j = 0; j < BC; ++j) {
        const float pjj = __shfl_sync(kFullMask, pj, j);
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc[rr][i] += pjj * sV[j][lane + i * kWarpSize];
      }
      m[rr] = m_new;
    }
  }

  // Write directly in [B, Tq, H, D] order so the caller can skip the transpose.
#pragma unroll
  for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
    const int qi = q0 + warp * ROWS_PER_WARP + rr;
    if (qi >= p.Tq) continue;
    const float inv = 1.f / l[rr];
    float* orow = p.o + ((static_cast<int64_t>(b) * p.Tq + qi) * p.H + h) * D;
#pragma unroll
    for (int i = 0; i < DPL; ++i) orow[lane + i * kWarpSize] = acc[rr][i] * inv;
  }
}

template <int D>
static void launch_flash_attention_d(const AttnParams& p, cudaStream_t stream) {
  constexpr int kBC = 32, kWarps = 4, kRowsPerWarp = 4;
  constexpr int kBR = kWarps * kRowsPerWarp;
  const dim3 grid((p.Tq + kBR - 1) / kBR, p.B * p.H);
  const dim3 block(kWarps * kWarpSize);
  flash_attention_kernel<D, kBC, kWarps, kRowsPerWarp><<<grid, block, 0, stream>>>(p);
}

void launch_flash_attention(const AttnParams& p, int head_dim, cudaStream_t stream) {
  switch (head_dim) {
    case 64: launch_flash_attention_d<64>(p, stream); break;
    case 128: launch_flash_attention_d<128>(p, stream); break;
    default: break;  // rejected in ops.cpp
  }
}

}  // namespace gpt2cuda
