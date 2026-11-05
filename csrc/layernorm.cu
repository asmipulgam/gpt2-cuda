// LayerNorm kernels.
//
// v1: one 256-thread block per row, two shared-memory tree reductions
//     (mean, then variance), and x is read from global memory three times.
// v2: one warp per row. The row is loaded once with float4 loads and kept in
//     registers, reductions use warp shuffles (no shared memory, no
//     __syncthreads), and the output is written with float4 stores.
#include "common.cuh"
#include "kernels.h"

namespace gpt2cuda {

template <int BLOCK>
__global__ void layernorm_v1_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    const float* __restrict__ b, float* __restrict__ y,
                                    int cols, float eps) {
  __shared__ float smem[BLOCK];
  const float* xr = x + static_cast<int64_t>(blockIdx.x) * cols;
  float* yr = y + static_cast<int64_t>(blockIdx.x) * cols;

  float sum = 0.f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) sum += xr[i];
  const float mean = block_sum_smem<BLOCK>(sum, smem) / cols;

  float sq = 0.f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    const float d = xr[i] - mean;
    sq += d * d;
  }
  const float rstd = rsqrtf(block_sum_smem<BLOCK>(sq, smem) / cols + eps);

  for (int i = threadIdx.x; i < cols; i += BLOCK) yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
}

// MAX_VEC = maximum number of float4s each lane holds (cols <= MAX_VEC * 128).
template <int MAX_VEC>
__global__ void layernorm_v2_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    const float* __restrict__ b, float* __restrict__ y,
                                    int rows, int cols, float eps) {
  const int warps_per_block = blockDim.x / kWarpSize;
  const int row = blockIdx.x * warps_per_block + threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  if (row >= rows) return;  // whole warp exits together

  const int n4 = cols / 4;
  const float4* x4 = reinterpret_cast<const float4*>(x + static_cast<int64_t>(row) * cols);
  float4 buf[MAX_VEC];

  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < MAX_VEC; ++i) {
    const int idx = lane + i * kWarpSize;
    if (idx < n4) {
      buf[i] = x4[idx];
      sum += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
    }
  }
  const float mean = warp_sum(sum) / cols;

  // Second pass over registers, not memory: numerically stable two-pass variance for free.
  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < MAX_VEC; ++i) {
    const int idx = lane + i * kWarpSize;
    if (idx < n4) {
      const float dx = buf[i].x - mean, dy = buf[i].y - mean;
      const float dz = buf[i].z - mean, dw = buf[i].w - mean;
      sq += dx * dx + dy * dy + dz * dz + dw * dw;
    }
  }
  const float rstd = rsqrtf(warp_sum(sq) / cols + eps);

  const float4* w4 = reinterpret_cast<const float4*>(w);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  float4* y4 = reinterpret_cast<float4*>(y + static_cast<int64_t>(row) * cols);
#pragma unroll
  for (int i = 0; i < MAX_VEC; ++i) {
    const int idx = lane + i * kWarpSize;
    if (idx < n4) {
      const float4 wv = w4[idx];
      const float4 bv = b4[idx];
      float4 o;
      o.x = (buf[i].x - mean) * rstd * wv.x + bv.x;
      o.y = (buf[i].y - mean) * rstd * wv.y + bv.y;
      o.z = (buf[i].z - mean) * rstd * wv.z + bv.z;
      o.w = (buf[i].w - mean) * rstd * wv.w + bv.w;
      y4[idx] = o;
    }
  }
}

void launch_layernorm_v1(const float* x, const float* w, const float* b, float* y,
                         int rows, int cols, float eps, cudaStream_t stream) {
  constexpr int kBlock = 256;
  layernorm_v1_kernel<kBlock><<<rows, kBlock, 0, stream>>>(x, w, b, y, cols, eps);
}

void launch_layernorm_v2(const float* x, const float* w, const float* b, float* y,
                         int rows, int cols, float eps, cudaStream_t stream) {
  constexpr int kWarpsPerBlock = 4;
  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (cols <= 512) {
    layernorm_v2_kernel<4><<<grid, block, 0, stream>>>(x, w, b, y, rows, cols, eps);
  } else if (cols <= 1024) {
    layernorm_v2_kernel<8><<<grid, block, 0, stream>>>(x, w, b, y, rows, cols, eps);
  } else {
    layernorm_v2_kernel<16><<<grid, block, 0, stream>>>(x, w, b, y, rows, cols, eps);
  }
}

}  // namespace gpt2cuda
