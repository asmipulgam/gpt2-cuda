// Host side: validates tensors, allocates outputs, launches kernels on the
// current PyTorch stream, and exposes everything to Python as gpt2_cuda._C.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#include "kernels.h"

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_FLOAT(x)

bool aligned16(const torch::Tensor& t) {
  return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0;
}

torch::Tensor layer_norm(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                         double eps, int64_t version) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  const int64_t cols = x.size(-1);
  TORCH_CHECK(weight.numel() == cols && bias.numel() == cols,
              "weight and bias must have x.size(-1) = ", cols, " elements");
  TORCH_CHECK(cols > 0, "last dimension must be non-empty");
  const int64_t rows = x.numel() / cols;

  auto y = torch::empty_like(x);
  if (rows == 0) return y;
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (version == 1) {
    gpt2cuda::launch_layernorm_v1(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                  y.data_ptr<float>(), rows, cols, static_cast<float>(eps), stream);
  } else if (version == 2) {
    TORCH_CHECK(cols % 4 == 0, "layer_norm v2 needs the last dim to be a multiple of 4, got ", cols);
    TORCH_CHECK(cols <= gpt2cuda::kLayerNormV2MaxCols, "layer_norm v2 supports at most ",
                gpt2cuda::kLayerNormV2MaxCols, " columns, got ", cols);
    TORCH_CHECK(aligned16(x) && aligned16(weight) && aligned16(bias),
                "layer_norm v2 needs 16-byte aligned tensors");
    gpt2cuda::launch_layernorm_v2(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                  y.data_ptr<float>(), rows, cols, static_cast<float>(eps), stream);
  } else {
    TORCH_CHECK(false, "unknown layer_norm version ", version);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor bias, int64_t version) {
  CHECK_INPUT(x);
  CHECK_INPUT(bias);
  const int64_t cols = x.size(-1);
  TORCH_CHECK(bias.numel() == cols, "bias must have x.size(-1) = ", cols, " elements");
  const int64_t n = x.numel();

  auto y = torch::empty_like(x);
  if (n == 0) return y;
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (version == 1) {
    gpt2cuda::launch_bias_gelu_v1(x.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(),
                                  n, cols, stream);
  } else if (version == 2) {
    TORCH_CHECK(cols % 4 == 0, "bias_gelu v2 needs the last dim to be a multiple of 4, got ", cols);
    TORCH_CHECK(aligned16(x) && aligned16(bias), "bias_gelu v2 needs 16-byte aligned tensors");
    gpt2cuda::launch_bias_gelu_v2(x.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(),
                                  n, cols, stream);
  } else {
    TORCH_CHECK(false, "unknown bias_gelu version ", version);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor softmax(torch::Tensor scores, double scale, bool causal, int64_t version) {
  CHECK_INPUT(scores);
  TORCH_CHECK(scores.dim() >= 2, "scores must be [..., Tq, Tk]");
  const int64_t tq = scores.size(-2);
  const int64_t tk = scores.size(-1);
  TORCH_CHECK(tk > 0, "Tk must be positive");
  TORCH_CHECK(!causal || tk >= tq, "causal softmax needs Tk >= Tq (got Tq=", tq, ", Tk=", tk, ")");
  const int64_t rows = scores.numel() / tk;

  auto y = torch::empty_like(scores);
  if (rows == 0) return y;
  const c10::cuda::CUDAGuard guard(scores.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (version == 1) {
    gpt2cuda::launch_softmax_v1(scores.data_ptr<float>(), y.data_ptr<float>(), rows, tq, tk,
                                static_cast<float>(scale), causal, stream);
  } else if (version == 2) {
    gpt2cuda::launch_softmax_v2(scores.data_ptr<float>(), y.data_ptr<float>(), rows, tq, tk,
                                static_cast<float>(scale), causal, stream);
  } else {
    TORCH_CHECK(false, "unknown softmax version ", version);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              double scale, bool causal) {
  for (const auto* t : {&q, &k, &v}) {
    CHECK_CUDA(*t);
    CHECK_FLOAT(*t);
    TORCH_CHECK(t->dim() == 4, "q, k, v must be [B, H, T, D]");
    TORCH_CHECK(t->stride(3) == 1, "q, k, v must be contiguous in the last (head_dim) dimension");
  }
  const int64_t B = q.size(0), H = q.size(1), Tq = q.size(2), D = q.size(3);
  const int64_t Tk = k.size(2);
  TORCH_CHECK(k.size(0) == B && k.size(1) == H && k.size(3) == D, "k shape must be [B, H, Tk, D]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must have the same shape as k");
  TORCH_CHECK(D == 64 || D == 128, "flash_attention supports head_dim 64 or 128, got ", D);
  TORCH_CHECK(Tk > 0, "Tk must be positive");
  TORCH_CHECK(!causal || Tk >= Tq, "causal attention needs Tk >= Tq (got Tq=", Tq, ", Tk=", Tk, ")");
  TORCH_CHECK(B * H <= 65535, "B * H must fit in gridDim.y");

  auto o = torch::empty({B, Tq, H * D}, q.options());
  if (B * H * Tq == 0) return o;
  const c10::cuda::CUDAGuard guard(q.device());

  gpt2cuda::AttnParams p;
  p.q = q.data_ptr<float>();
  p.k = k.data_ptr<float>();
  p.v = v.data_ptr<float>();
  p.o = o.data_ptr<float>();
  p.B = B; p.H = H; p.Tq = Tq; p.Tk = Tk;
  p.q_sb = q.stride(0); p.q_sh = q.stride(1); p.q_st = q.stride(2);
  p.k_sb = k.stride(0); p.k_sh = k.stride(1); p.k_st = k.stride(2);
  p.v_sb = v.stride(0); p.v_sh = v.stride(1); p.v_st = v.stride(2);
  p.scale = static_cast<float>(scale);
  p.causal = causal;
  gpt2cuda::launch_flash_attention(p, D, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Custom CUDA kernels for GPT-2 inference";
  m.def("layer_norm", &layer_norm, "LayerNorm over the last dim",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5, py::arg("version") = 2);
  m.def("bias_gelu", &bias_gelu, "gelu_tanh(x + bias)",
        py::arg("x"), py::arg("bias"), py::arg("version") = 2);
  m.def("softmax", &softmax, "Scaled (optionally causal) softmax over the last dim",
        py::arg("scores"), py::arg("scale") = 1.0, py::arg("causal") = true, py::arg("version") = 2);
  m.def("flash_attention", &flash_attention, "Fused attention; returns [B, Tq, H*D]",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true);
}
