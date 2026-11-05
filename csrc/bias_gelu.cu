// Fused bias add + GELU (tanh approximation, GPT-2's "gelu_new").
//
// PyTorch eager runs this as two kernels (add, then gelu), so the
// [rows, 3072] intermediate goes to DRAM and back. Fusing it reads x once and
// writes y once.
//
// v1: one thread per element, scalar loads.
// v2: grid-stride loop over float4 chunks (4 elements per load/store).
#include "common.cuh"
#include "kernels.h"

#include <algorithm>

namespace gpt2cuda {

__global__ void bias_gelu_v1_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                    float* __restrict__ y, int64_t n, int cols) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = gelu_tanh(x[i] + bias[i % cols]);
}

__global__ void bias_gelu_v2_kernel(const float4* __restrict__ x, const float4* __restrict__ bias,
                                    float4* __restrict__ y, int64_t n4, int cols4) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 v = x[i];
    const float4 bb = bias[i % cols4];
    float4 o;
    o.x = gelu_tanh(v.x + bb.x);
    o.y = gelu_tanh(v.y + bb.y);
    o.z = gelu_tanh(v.z + bb.z);
    o.w = gelu_tanh(v.w + bb.w);
    y[i] = o;
  }
}

void launch_bias_gelu_v1(const float* x, const float* bias, float* y,
                         int64_t n, int cols, cudaStream_t stream) {
  constexpr int kBlock = 256;
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  bias_gelu_v1_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(x, bias, y, n, cols);
}

void launch_bias_gelu_v2(const float* x, const float* bias, float* y,
                         int64_t n, int cols, cudaStream_t stream) {
  constexpr int kBlock = 256;
  const int64_t n4 = n / 4;
  // Enough blocks to fill the GPU several times over; the grid-stride loop covers the rest.
  const int64_t blocks = std::min<int64_t>((n4 + kBlock - 1) / kBlock, 65535);
  bias_gelu_v2_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
      reinterpret_cast<const float4*>(x), reinterpret_cast<const float4*>(bias),
      reinterpret_cast<float4*>(y), n4, cols / 4);
}

}  // namespace gpt2cuda
