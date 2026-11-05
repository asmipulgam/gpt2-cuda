# SYCL port: causal softmax

A port of the v2 softmax kernel (`csrc/softmax.cu`) to SYCL / oneAPI so it runs on
Intel GPUs. It checks the result against a CPU reference and reports time and
effective bandwidth, the same metrics `benchmarks/bench_kernels.py` reports for CUDA.

## Run on Intel hardware

The easiest free option is the [Intel Tiber AI Cloud](https://ai.cloud.intel.com/)
(previously Intel Developer Cloud), which has oneAPI and Intel Data Center GPUs.

```bash
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -O3 sycl/softmax_sycl.cpp -o sycl/softmax_sycl
sycl-ls                                   # list devices
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./sycl/softmax_sycl 8 12 512
```

It also runs on a CPU device (`ONEAPI_DEVICE_SELECTOR=opencl:cpu`), which is handy
for checking correctness on a laptop with oneAPI installed.

## What changed from CUDA

| CUDA | SYCL |
|---|---|
| warp of 32 lanes | sub-group of 16 (`[[sycl::reqd_sub_group_size(16)]]`, native on Intel Xe) |
| `__shfl_xor_sync` butterfly | `sycl::reduce_over_group(sub_group, v, op)` |
| online (max, sum) merge across lanes | two plain reductions (max, then sum). `reduce_over_group` only takes built-in operators, and a custom pair merge would need hand-written shuffles |
| `cudaMalloc` / `cudaMemcpy` | USM: `sycl::malloc_device`, `queue::memcpy` |
| CUDA events | `enable_profiling` queue + `event::get_profiling_info` |

## Status

Not benchmarked on Intel hardware, and not yet run on any SYCL device. The port is
written to mirror the CUDA kernel, and the program checks itself against a CPU
reference (`max abs error ... OK`) when you run it with the commands above. No
timings are reported here.
