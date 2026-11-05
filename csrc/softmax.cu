// Scaled, causally-masked softmax over attention scores.
//
// The PyTorch reference does scale -> masked_fill -> softmax: three kernels and
// three round trips through DRAM. Both versions here fuse all of it.
//
// v1: one 256-thread block per row, shared-memory reductions, three passes over
//     the row (max, sum, write). Wasteful for short rows (decode: tk ~ 100).
// v2: one warp per row with an "online softmax": a single pass computes the max
//     and the rescaled sum together, merged across lanes with warp shuffles.
//     Masked positions are never read (they are only written as 0).
#include "common.cuh"
#include "kernels.h"

namespace gpt2cuda {

__device__ __forceinline__ int causal_limit(int row, int tq, int tk, bool causal) {
  if (!causal) return tk;
  const int qi = row % tq;
  return min(tk, qi + (tk - tq) + 1);  // number of visible keys
}

template <int BLOCK>
__global__ void softmax_v1_kernel(const float* __restrict__ x, float* __restrict__ y,
                                  int tq, int tk, float scale, bool causal) {
  __shared__ float smem[BLOCK];
  const int row = blockIdx.x;
  const int limit = causal_limit(row, tq, tk, causal);
  const float* xr = x + static_cast<int64_t>(row) * tk;
  float* yr = y + static_cast<int64_t>(row) * tk;

  float m = -INFINITY;
  for (int j = threadIdx.x; j < limit; j += BLOCK) m = fmaxf(m, xr[j] * scale);
  m = block_max_smem<BLOCK>(m, smem);

  float s = 0.f;
  for (int j = threadIdx.x; j < limit; j += BLOCK) s += expf(xr[j] * scale - m);
  const float inv = 1.f / block_sum_smem<BLOCK>(s, smem);

  for (int j = threadIdx.x; j < tk; j += BLOCK)
    yr[j] = j < limit ? expf(xr[j] * scale - m) * inv : 0.f;
}

__global__ void softmax_v2_kernel(const float* __restrict__ x, float* __restrict__ y,
                                  int rows, int tq, int tk, float scale, bool causal) {
  const int warps_per_block = blockDim.x / kWarpSize;
  const int row = blockIdx.x * warps_per_block + threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  if (row >= rows) return;

  const int limit = causal_limit(row, tq, tk, causal);
  const float* xr = x + static_cast<int64_t>(row) * tk;
  float* yr = y + static_cast<int64_t>(row) * tk;

  // Per-lane running max m and sum s of exp(v - m).
  float m = -INFINITY, s = 0.f;
  for (int j = lane; j < limit; j += kWarpSize) {
    const float v = xr[j] * scale;
    if (v > m) {
      s = s * __expf(m - v) + 1.f;
      m = v;
    } else {
      s += __expf(v - m);
    }
  }

  // Merge (m, s) pairs across the warp. Guard -inf so empty lanes don't produce NaN.
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const float m2 = __shfl_xor_sync(kFullMask, m, offset);
    const float s2 = __shfl_xor_sync(kFullMask, s, offset);
    const float mn = fmaxf(m, m2);
    s = (m == -INFINITY ? 0.f : s * __expf(m - mn)) + (m2 == -INFINITY ? 0.f : s2 * __expf(m2 - mn));
    m = mn;
  }

  const float inv = 1.f / s;
  for (int j = lane; j < tk; j += kWarpSize)
    yr[j] = j < limit ? __expf(xr[j] * scale - m) * inv : 0.f;
}

void launch_softmax_v1(const float* x, float* y, int rows, int tq, int tk,
                       float scale, bool causal, cudaStream_t stream) {
  constexpr int kBlock = 256;
  softmax_v1_kernel<kBlock><<<rows, kBlock, 0, stream>>>(x, y, tq, tk, scale, causal);
}

void launch_softmax_v2(const float* x, float* y, int rows, int tq, int tk,
                       float scale, bool causal, cudaStream_t stream) {
  constexpr int kWarpsPerBlock = 8;
  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  softmax_v2_kernel<<<grid, block, 0, stream>>>(x, y, rows, tq, tk, scale, causal);
}

}  // namespace gpt2cuda
