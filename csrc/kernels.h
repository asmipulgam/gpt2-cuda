// Host-callable kernel launchers. Implemented in the .cu files, called from ops.cpp.
// All launchers take raw float32 device pointers and a stream; shape/dtype
// validation happens in ops.cpp before these are called.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gpt2cuda {

// LayerNorm over the last dimension: y = (x - mean) / sqrt(var + eps) * w + b
void launch_layernorm_v1(const float* x, const float* w, const float* b, float* y,
                         int rows, int cols, float eps, cudaStream_t stream);
void launch_layernorm_v2(const float* x, const float* w, const float* b, float* y,
                         int rows, int cols, float eps, cudaStream_t stream);
// Largest row width launch_layernorm_v2 supports (row is cached in registers).
constexpr int kLayerNormV2MaxCols = 2048;

// y = gelu_tanh(x + bias), bias broadcast over rows of width `cols`.
void launch_bias_gelu_v1(const float* x, const float* bias, float* y,
                         int64_t n, int cols, cudaStream_t stream);
void launch_bias_gelu_v2(const float* x, const float* bias, float* y,
                         int64_t n, int cols, cudaStream_t stream);

// Row softmax of (x * scale) over the last dim of a [rows = batch*tq, tk] matrix.
// With causal=true, query i (of tq) may see keys j <= i + (tk - tq); the rest become 0.
void launch_softmax_v1(const float* x, float* y, int rows, int tq, int tk,
                       float scale, bool causal, cudaStream_t stream);
void launch_softmax_v2(const float* x, float* y, int rows, int tq, int tk,
                       float scale, bool causal, cudaStream_t stream);

// Fused attention. q/k/v are [B, H, T, D] with arbitrary strides on the first
// three dims (last dim must be contiguous). Output is written as a contiguous
// [B, Tq, H * D] tensor, i.e. already in the layout the output projection wants.
struct AttnParams {
  const float* q;
  const float* k;
  const float* v;
  float* o;
  int B, H, Tq, Tk;
  int64_t q_sb, q_sh, q_st;
  int64_t k_sb, k_sh, k_st;
  int64_t v_sb, v_sh, v_st;
  float scale;
  bool causal;
};
void launch_flash_attention(const AttnParams& p, int head_dim, cudaStream_t stream);

}  // namespace gpt2cuda
