// SYCL port of the v2 causal softmax kernel (csrc/softmax.cu), for Intel GPUs.
//
// Build (Intel oneAPI DPC++):
//   icpx -fsycl -O3 sycl/softmax_sycl.cpp -o sycl/softmax_sycl
// Run:
//   ./sycl/softmax_sycl [batch heads seq]        (default 8 12 512)
//   ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./sycl/softmax_sycl
//
// CUDA -> SYCL mapping used here:
//   warp (32 lanes)            -> sub-group (16 work-items, the native width on Intel Xe)
//   __shfl_xor_sync reduction  -> sycl::reduce_over_group(sub_group, v, op)
//   blockIdx / threadIdx       -> nd_item::get_group / get_local_id
//   cudaEvent timing           -> queue with enable_profiling + event::get_profiling_info
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int kSubGroup = 16;
constexpr int kSubGroupsPerGroup = 8;

// One sub-group per row. Two passes over visible keys (max, then sum of exp),
// then the write pass; masked keys are never read.
sycl::event softmax_causal(sycl::queue& q, const float* x, float* y, int rows, int tq, int tk, float scale) {
  const int groups = (rows + kSubGroupsPerGroup - 1) / kSubGroupsPerGroup;
  const sycl::nd_range<1> range(groups * kSubGroupsPerGroup * kSubGroup, kSubGroupsPerGroup * kSubGroup);
  return q.parallel_for(range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
    const auto sg = it.get_sub_group();
    const int row = it.get_group(0) * kSubGroupsPerGroup + sg.get_group_linear_id();
    if (row >= rows) return;  // uniform across the sub-group
    const int lane = sg.get_local_linear_id();
    const int qi = row % tq;
    const int limit = std::min(tk, qi + (tk - tq) + 1);
    const float* xr = x + static_cast<size_t>(row) * tk;
    float* yr = y + static_cast<size_t>(row) * tk;

    float m = -INFINITY;
    for (int j = lane; j < limit; j += kSubGroup) m = sycl::fmax(m, xr[j] * scale);
    m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());

    float s = 0.f;
    for (int j = lane; j < limit; j += kSubGroup) s += sycl::exp(xr[j] * scale - m);
    s = sycl::reduce_over_group(sg, s, sycl::plus<float>());

    const float inv = 1.f / s;
    for (int j = lane; j < tk; j += kSubGroup) yr[j] = j < limit ? sycl::exp(xr[j] * scale - m) * inv : 0.f;
  });
}

void softmax_reference(const std::vector<float>& x, std::vector<float>& y, int rows, int tq, int tk, float scale) {
  for (int r = 0; r < rows; ++r) {
    const int limit = std::min(tk, r % tq + (tk - tq) + 1);
    const float* xr = &x[static_cast<size_t>(r) * tk];
    float* yr = &y[static_cast<size_t>(r) * tk];
    float m = -INFINITY;
    for (int j = 0; j < limit; ++j) m = std::max(m, xr[j] * scale);
    double s = 0;
    for (int j = 0; j < limit; ++j) s += std::exp(xr[j] * scale - m);
    for (int j = 0; j < tk; ++j) yr[j] = j < limit ? static_cast<float>(std::exp(xr[j] * scale - m) / s) : 0.f;
  }
}

int main(int argc, char** argv) {
  const int B = argc > 1 ? std::atoi(argv[1]) : 8;
  const int H = argc > 2 ? std::atoi(argv[2]) : 12;
  const int T = argc > 3 ? std::atoi(argv[3]) : 512;
  const int rows = B * H * T;
  const size_t n = static_cast<size_t>(rows) * T;
  const float scale = 0.125f;

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};
  std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  std::printf("shape: [%d, %d, %d, %d]\n", B, H, T, T);

  std::vector<float> hx(n), hy(n), href(n);
  std::mt19937 rng(0);
  std::normal_distribution<float> dist(0.f, 5.f);
  for (auto& v : hx) v = dist(rng);

  float* dx = sycl::malloc_device<float>(n, q);
  float* dy = sycl::malloc_device<float>(n, q);
  q.memcpy(dx, hx.data(), n * sizeof(float)).wait();

  softmax_causal(q, dx, dy, rows, T, T, scale).wait();  // warmup
  q.memcpy(hy.data(), dy, n * sizeof(float)).wait();

  softmax_reference(hx, href, rows, T, T, scale);
  double max_err = 0;
  for (size_t i = 0; i < n; ++i) max_err = std::max(max_err, static_cast<double>(std::fabs(hy[i] - href[i])));
  std::printf("max abs error vs CPU reference: %.3e  %s\n", max_err, max_err < 1e-5 ? "OK" : "MISMATCH");

  constexpr int kIters = 50;
  double total_ns = 0;
  for (int i = 0; i < kIters; ++i) {
    auto e = softmax_causal(q, dx, dy, rows, T, T, scale);
    e.wait();
    total_ns += e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                e.get_profiling_info<sycl::info::event_profiling::command_start>();
  }
  const double ms = total_ns / kIters / 1e6;
  const double gbps = 2.0 * n * sizeof(float) / (ms * 1e6);
  std::printf("time: %.4f ms   effective bandwidth: %.1f GB/s\n", ms, gbps);

  sycl::free(dx, q);
  sycl::free(dy, q);
  return max_err < 1e-5 ? 0 : 1;
}
