# gpt2-cuda: accelerating GPT-2 inference with custom CUDA kernels

GPT-2 small (124M) running in PyTorch, with its slowest non-GEMM operations replaced by
hand-written CUDA kernels. The kernels are bound to Python as a PyTorch C++ extension
and benchmarked end to end against PyTorch eager and `torch.compile`.

| Kernel | Versions | Key techniques |
|---|---|---|
| LayerNorm | v1 → v2 | block reduction in shared memory → warp-per-row, register-cached row, `float4`, warp shuffles |
| Bias + GELU | v1 → v2 | fused add+activation (one DRAM round trip saved) → `float4` grid-stride |
| Causal softmax | v1 → v2 | fused scale+mask+softmax → warp-per-row online softmax, masked keys never read |
| Fused attention | flash | FlashAttention-style tiling: QKᵀ, softmax and ·V in one kernel, no T×T matrix in memory |
| KV cache | (stretch) | preallocated cache, consumed as strided views by the attention kernel |
| SYCL softmax | (stretch) | port of softmax v2 to SYCL sub-groups for Intel GPUs |

## Results

**GPU:** Tesla T4 (sm_75, ~320 GB/s peak DRAM bandwidth)

End-to-end greedy generation, 64 new tokens:

| Variant | B=1, T=32 tok/s | B=8, T=256 tok/s | vs HF eager (B=1,T=32 / B=8,T=256) |
|---|---|---|---|
| HF eager | 122.2 | 726.5 | 1.00× / 1.00× |
| HF + `torch.compile` | 195.1 | 741.0 | 1.60× / 1.02× |
| ours, pure-PyTorch ops (`torch`) | 164.0 | 949.1 | 1.34× / 1.31× |
| ours, `cuda_flash` | 318.0 | 1208.2 | 2.60× / 1.66× |

Part of the gain over HF eager comes from the compact model itself (the `torch` row), not
the kernels. Against that fairer baseline, `cuda_flash` is 1.94× at B=1,T=32 and 1.27× at
B=8,T=256. Full sweep (T up to 768) is in [benchmarks/results/e2e.md](benchmarks/results/e2e.md).

Per-kernel results (time, effective bandwidth, % of peak) are in
[benchmarks/results/kernels.md](benchmarks/results/kernels.md), with charts in
`benchmarks/results/*.png`. The optimization write-up is in
[docs/optimization_notes.md](docs/optimization_notes.md).

## Repo layout

```
csrc/            CUDA kernels (.cu), launchers (kernels.h), C++ checks + pybind (ops.cpp)
gpt2_cuda/       Python package
  ops.py         wrappers around the compiled extension (gpt2_cuda._C)
  reference.py   plain-PyTorch reference ops (ground truth for tests/benchmarks)
  model.py       compact GPT-2 that loads HF weights; pluggable backends; KV cache
  patch.py       monkey-patches a Hugging Face GPT2LMHeadModel (LayerNorm + MLP)
tests/           pytest correctness tests vs PyTorch and vs Hugging Face
benchmarks/      baseline profiling, per-kernel and end-to-end benchmarks, Nsight scripts, plots
sycl/            SYCL/oneAPI port of the softmax kernel
docs/            optimization notes
notebooks/       Colab quickstart
```

## Setup

You need an NVIDIA GPU with the CUDA toolkit (`nvcc`). A free Colab or Kaggle T4 is
enough. The quickest path is [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb).

```bash
pip install -r requirements.txt
TORCH_CUDA_ARCH_LIST="7.5" pip install -e . --no-build-isolation   # 7.5 = T4, 8.0 = A100, 8.6 = A10/3090, 8.9 = L4/4090
python -m pytest -q tests
```

`--no-build-isolation` matters: the build imports your installed `torch`, so the
extension is compiled against the same version you run with. After editing a `.cu`
file, rerun the `pip install -e .` line to rebuild.

## Usage

```python
import torch
from transformers import GPT2TokenizerFast
from gpt2_cuda import GPT2, ops

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2.from_pretrained("gpt2", backend="cuda_flash")   # or torch / cuda_v1 / cuda_softmax
ids = tok("The future of GPU computing is", return_tensors="pt").input_ids.cuda()
print(tok.decode(model.generate(ids, max_new_tokens=40)[0]))

# Individual kernels
x = torch.randn(4096, 768, device="cuda")
w, b = torch.ones(768, device="cuda"), torch.zeros(768, device="cuda")
y = ops.layer_norm(x, w, b, eps=1e-5, version=2)
```

Patching an existing Hugging Face model in place (LayerNorm + fused bias/GELU MLP):

```python
from transformers import GPT2LMHeadModel
from gpt2_cuda import patch_hf_model
hf = patch_hf_model(GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval())
```

### Why a separate model for fused attention?

`GPT2Attention`'s forward signature, mask format and cache object have changed across
`transformers` releases, while LayerNorm and the MLP have kept a plain
`forward(hidden_states)`. `patch.py` swaps in those two. `gpt2_cuda/model.py` is a
compact GPT-2 that loads the same Hugging Face weights and owns its forward pass,
so the fused attention kernel and the KV cache don't depend on `transformers`
internals. All its backends run the same graph, so differences between them come
only from the kernels. `tests/test_model.py` checks that its logits match HF's and
that greedy generation produces identical text for every backend.

## Running the benchmarks

```bash
# Week 1: baseline, sample text, throughput sweep, torch.profiler table + Chrome trace
python benchmarks/baseline.py

# Nsight Systems timeline of the baseline
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi \
    -o benchmarks/results/baseline python benchmarks/baseline.py --nsys

# Every kernel version vs PyTorch across sizes -> results/kernels.{csv,md}
python benchmarks/bench_kernels.py            # --peak-bw 320 if your GPU isn't auto-detected

# End to end: HF eager / torch.compile / patched HF / our backends -> results/e2e.{csv,md}
python benchmarks/bench_e2e.py

# Nsight Compute: DRAM bandwidth % of peak, occupancy, bank conflicts
bash benchmarks/ncu.sh          # summary CSV
bash benchmarks/ncu.sh full     # full report for the GUI

# Charts
python benchmarks/plot.py
```

On Colab, `ncu` may need `apt-get install nsight-compute` or may lack permission to
read GPU performance counters (ERR_NVGPUCTRPERM); Kaggle blocks them too. It needs a
machine where you have root, such as a rented cloud GPU VM or your own NVIDIA GPU.

## Kernel design notes

**LayerNorm v2.** One warp per row. Each lane loads its slice of the row as `float4`
and keeps it in registers (6 × `float4` per lane for C=768), so the row is read from
memory once. Mean and variance come from two warp-shuffle reductions over registers.
That is a two-pass variance, numerically stable even when |mean| ≫ std, at no extra
memory cost.

**Softmax v2.** One warp per row with an online softmax: each lane keeps a running
`(max, sum)`, and lanes merge pairs with `s = s₁·e^(m₁−m) + s₂·e^(m₂−m)`. The causal
mask is applied by index (`j ≤ i + Tk − Tq`), so masked scores are never loaded. The
same formula handles the KV-cache case, where `Tk > Tq`.

**Fused attention.** Grid `(⌈Tq/16⌉, B·H)`, 4 warps per block, 4 query rows per warp.
K and V tiles of 32 keys are staged in shared memory. Lane *j* computes the score for
key *j*, warp reductions give the tile max and sum, and the softmax weights are
broadcast with `__shfl_sync` while lane *d* accumulates output dims *d* and *d*+32.
The K tile is padded to 65 columns to avoid 32-way bank conflicts. Causal blocks
stop at their last visible key. Q, K and V are read through arbitrary strides, so the
model passes views of the fused QKV projection and slices of the KV cache without
copying them. The output is written directly as `[B, T, H·D]`, which removes the
transpose before the output projection.

**KV cache.** `KVCache` preallocates `[layers, B, H, max_len, D]`. Each step writes the
new K/V into position `pos` and attention reads `cache[:, :, :pos+T]` as a view, with
no `torch.cat` per step.

## Testing

```bash
python -m pytest -q tests                  # everything
python -m pytest -q tests/test_attention.py
```

The kernel tests compare against PyTorch with `torch.testing.assert_close` across
odd sizes, decode shapes (Tq=1), cache offsets (Tq<Tk), strided inputs and extreme
values. The model tests compare logits and greedy text against Hugging Face. They
skip automatically when no GPU is present, or when the extension isn't built or
GPT-2 can't be downloaded.

## Stretch goals

- [x] KV cache (`gpt2_cuda/model.py`)
- [x] SYCL port of softmax for Intel GPUs, written but not yet run or benchmarked ([sycl/](sycl/README.md))
- [ ] FP16 / Tensor Cores with WMMA (a T4 supports FP16 WMMA)
- [ ] Split-K "flash-decoding" for better occupancy at Tq = 1
