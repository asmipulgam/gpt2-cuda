// Shared device helpers for all kernels.
#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace gpt2cuda {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// Butterfly reductions: after the loop every lane holds the full result.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(kFullMask, v, offset);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    v = fmaxf(v, __shfl_xor_sync(kFullMask, v, offset));
  return v;
}

// Classic shared-memory tree reduction (used by the v1 kernels so the
// progression to warp shuffles is visible). BLOCK must be a power of two.
// Every thread gets the result. Ends with a barrier so smem can be reused.
template <int BLOCK>
__device__ __forceinline__ float block_sum_smem(float v, float* smem) {
  smem[threadIdx.x] = v;
  __syncthreads();
#pragma unroll
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  const float result = smem[0];
  __syncthreads();
  return result;
}

template <int BLOCK>
__device__ __forceinline__ float block_max_smem(float v, float* smem) {
  smem[threadIdx.x] = v;
  __syncthreads();
#pragma unroll
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  const float result = smem[0];
  __syncthreads();
  return result;
}

// GPT-2's "gelu_new": tanh approximation.
__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kCoeff = 0.044715f;
  return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + kCoeff * x * x * x)));
}

}  // namespace gpt2cuda
