# Optimization notes

One section per kernel: what was measured, what changed, and how much it helped.
Numbers come from `benchmarks/bench_kernels.py` (time, effective GB/s) and the
end-to-end sweep in `benchmarks/results/e2e.md`. Raw tables:
[kernels.md](../benchmarks/results/kernels.md), [e2e.md](../benchmarks/results/e2e.md),
[baseline.md](../benchmarks/results/baseline.md).

GPU: Tesla T4 (sm_75, FP32 path, ~320 GB/s peak DRAM, 4 MB L2) · all kernels FP32

**How to read the bandwidth numbers.** "GB/s" is *effective* bandwidth: bytes the
op must logically move divided by time. Small shapes fit in the T4's 4 MB L2, so
they can report more than 100% of DRAM peak (e.g. bias+GELU at 128×3072 shows
110%). Those rows measure cache and launch overhead, not DRAM. Trust the large
shapes for bandwidth claims.

**What was not measured.** Nsight Compute failed on the Kaggle machine
(`ERR_NVGPUCTRPERM`, no access to GPU performance counters), so
`ncu_metrics.csv` is empty. There are no measured occupancy, DRAM-% or bank-conflict
counters anywhere in this document. Where a section explains *why* something
is fast or slow, it is a hypothesis from the timings, marked as such.

---

## 0. Baseline: where the time goes

Source: `benchmarks/baseline.py` → `baseline_profile.txt` (torch profiler, HF GPT-2 eager, T4).

- **GEMMs dominate GPU time:** `addmm` + `mm` kernels are about 42% + 25% of self CUDA
  time. These stay on cuBLAS.
- **HF eager already uses an efficient attention kernel.** The profile shows
  `fmha_cutlassF_f32_aligned_64x64_rf_sm75` (PyTorch SDPA, memory-efficient backend),
  13.8% of CUDA time. So the HF baseline is not a naive attention implementation. This
  matters for section 4.
- **`aten::cat` is 7.4% of CUDA time** (1560 calls). That is the HF KV cache growing by
  concatenation every step, which is the motivation for the preallocated KV cache.
- **LayerNorm is only 2.2%** and the elementwise ops (mul/add/pow/tanh, i.e. unfused GELU)
  about 2–3% each. The non-GEMM kernels we replace are a small slice of GPU time.
- **Decode is CPU-bound.** Total self CPU time was 1.19 s against 0.34 s of CUDA time, so
  the GPU is idle most of the time during single-token steps. At B=1 the dominant cost
  is launch overhead, not kernel speed. That is why `torch.compile` and *fewer launches*
  help so much at B=1.

Baseline (profile run, HF eager): 97–104 tok/s at B=1, 665–777 tok/s at B=8
([baseline.md](../benchmarks/results/baseline.md)). The later e2e sweep measured HF eager at
122 / 726 tok/s for the same shapes. The two runs used different harnesses and warmup,
so **all speedups in this document use the e2e sweep as the denominator**.

---

## 1. LayerNorm

| Version | Idea | 4096×768 ms | GB/s | % peak | vs PyTorch |
|---|---|---|---|---|---|
| PyTorch | `F.layer_norm` | 0.171 | 147.5 | 46% | 1.00× |
| v1 | one block per row, shared-memory tree reductions | 0.274 | 91.9 | 29% | 0.62× |
| v2 | one warp per row, row in registers, `float4`, warp shuffles | 0.116 | 216.5 | 68% | 1.47× |

v2 speedup over PyTorch across sizes: 1.74× (128 rows), 1.64× (1024), 1.47× (4096), 1.13× (16384).
v1 is *slower* than PyTorch at every size except 128 rows (1.24×).

- **Why v1 lost (hypothesis, not profiled):** a 256-thread block for a 768-wide row leaves
  each thread with 3 elements, and every step of the tree reduction needs a
  `__syncthreads()`. For short rows the synchronization costs more than the arithmetic.
- **v2 change:** each lane holds 6 `float4`s (24 floats) in registers, so x is read from
  global memory once. Reductions use 5 `__shfl_xor_sync` steps with no shared memory
  or barriers, and 128-bit loads cut instruction count 4×.
- **Result:** v2 is 1.1–1.7× faster than PyTorch and reaches 72% of peak at the largest
  size. The advantage shrinks as the problem grows because PyTorch's kernel also
  approaches the bandwidth limit (64% at 16384 rows).

## 2. Bias + GELU

| Version | Idea | 4096×3072 ms | GB/s | % peak | vs PyTorch |
|---|---|---|---|---|---|
| PyTorch | `x + b`, then `gelu` (2 kernels) | 0.854 | 117.8 | 37% | 1.00× |
| v1 | fused, one thread per element | 0.404 | 249.4 | 78% | 2.12× |
| v2 | fused, `float4` grid-stride loop | 0.418 | 240.9 | 75% | 2.04× |

- **Fusion is the win.** It halves DRAM traffic (the add's intermediate is never written
  and read back), giving about 2.1× over PyTorch, which matches the traffic ratio.
- **float4 did not help; v2 is 3–4% slower than v1 at large shapes**
  (2.02–2.04× vs 2.09–2.12× for rows ≥ 1024). It was only ahead at 128 rows
  (2.45× vs 2.29×), where L2 effects dominate.
- **Why (hypothesis):** v1 is already at 75–78% of peak, and the practical ceiling of a
  streaming kernel on a T4 is not much above that. There is no bandwidth left for
  vectorization to recover. Scalar loads from adjacent threads are already coalesced
  into full transactions, so `float4` only reduces instruction count, which isn't the
  bottleneck on a memory-bound op. The grid-stride loop may also add loop overhead and
  a worse tail (fewer, longer-lived blocks).
- **Takeaway:** keep v1 as the default for large inputs. v2 stays in the repo as the
  documented negative result. I did not test this on other GPUs, and an A100 with much
  higher bandwidth might behave differently.

## 3. Causal softmax

| Version | Idea | 4×12×512×512 ms | GB/s | % peak | vs PyTorch |
|---|---|---|---|---|---|
| PyTorch | scale → mask → softmax (3 kernels) | 1.744 | 57.7 | 18% | 1.00× |
| v1 | fused; block per row | 0.637 | 158.1 | 49% | 2.74× |
| v2 | fused; warp per row, online softmax, masked keys never read | 0.384 | 262.1 | 82% | 4.54× |

v2 speedup over PyTorch: 7.25× (1×12×128×128), 4.58× (8×12×128×128), 4.54× (4×12×512×512),
4.27× (8×12×1024×1024). v1 is *slower* than PyTorch at 8×12×128×128 (0.93×).

- **Where the gain comes from:** fusing three kernels into one, plus an online
  (single-pass) softmax and skipping masked keys, which cuts the data actually read by
  about half for causal masks.
- **v1 vs v2:** equal at the largest shape (3.45 vs 3.37 ms) but v2 is far better for
  short rows, where a 256-thread block per row leaves most threads idle. That also
  explains v1's 0.93× on the small batched shape.
- **Caveats:** the 7.25× on the smallest shape is 8 µs in absolute terms, i.e. launch
  overhead and cache effects rather than a bandwidth result. "Effective GB/s" counts the
  full matrix but v2 skips masked keys, so v2's real DRAM traffic is lower than the
  figure suggests.
- **End to end:** swapping in v2 softmax (`cuda_softmax` vs `cuda_v1`) changed throughput
  by less than ±4% in either direction (e.g. 1199.8 vs 1161.9 tok/s at B=8, T=256).
  Softmax is not a large share of the model's time.

## 4. Fused attention (FlashAttention-style)

**Prefill-shaped attention (Tq = Tk):**

| Impl | 4×12×512×64 ms | speedup vs unfused | 8×12×1024×64 ms | speedup vs unfused |
|---|---|---|---|---|
| PyTorch unfused (matmul, mask, softmax, matmul) | 2.701 | 1.00× | 21.370 | 1.00× |
| PyTorch SDPA | 0.743 | **3.64×** | 4.896 | **4.37×** |
| matmul + v2 softmax | 1.525 | 1.77× | 12.340 | 1.73× |
| flash (ours) | 2.068 | 1.31× | 15.318 | 1.40× |

**Decode-shaped attention (Tq = 1):**

| Impl | 8×12×1×512 ms | 8×12×1×1024 ms | 1×12×1×512 ms |
|---|---|---|---|
| PyTorch unfused | 0.186 | 0.247 | 0.190 |
| PyTorch SDPA | 0.359 | 0.697 | 0.085 |
| flash (ours) | **0.101** | **0.200** | **0.049** |
| ours vs unfused | 1.84× | 1.23× | 3.86× |

### Where SDPA beats this kernel

For prefill, **PyTorch SDPA is 3–4× faster than our flash kernel** and even our simple
"matmul + v2 softmax" path beats it (1.7–1.8× over unfused vs 1.3–1.4×). At 8×12×128×64
the flash kernel is no faster than unfused (0.99×). The fused kernel only wins where
it matters least:

- **Why it loses (hypothesis, not profiled):** the kernel is scalar FP32 on CUDA cores.
  Each block handles 16 query rows with one lane per key and one lane per output
  dimension, and the QKᵀ dot product reads Q from shared memory. It is arithmetic- and
  shared-memory-bound at about 2–8% of DRAM peak, so avoiding the T×T matrix in memory
  doesn't help once compute is the limit. SDPA's efficient backend uses larger tiles
  and register blocking. It is not using tensor cores either (FP32 on sm_75), but its tiling is far better.
- **The profile baseline already contained SDPA.** HF eager calls
  `fmha_cutlassF_f32_aligned_64x64_rf_sm75`. Replacing it with our kernel in prefill is not
  an improvement; the end-to-end gain from `cuda_flash` comes from elsewhere (below).
- **Where it does win, decode:** with Tq = 1 SDPA picks a poor path (it is *slower* than
  unfused at batch 8: 0.52× and 0.35×) and our kernel reaches 249–252 GB/s (78–79% of
  peak) at batch 8, reading the K/V cache close to the bandwidth limit. This is the one
  attention regime where the kernel is clearly the best option on this GPU. At
  batch 8, sequence 1024 the margin over unfused narrows to 1.23×, because unfused is
  already at 64% of peak.

### Long-context regression

End-to-end, `cuda_flash` is not uniformly better than the unfused-attention variants:

| B=1 | cuda_softmax tok/s | cuda_flash tok/s | decode ms/token (softmax → flash) |
|---|---|---|---|
| T=32 | 256.0 | **318.0** | 3.89 → 3.13 |
| T=256 | 254.5 | **311.3** | 3.77 → 3.03 |
| T=768 | **248.1** | 230.1 | 3.28 → 3.57 |

At prompt length 768 the flash kernel's decode step gets *slower* (3.57 vs 3.28 ms/token)
while it gets faster at 32 and 256, so at B=1, T=768 it loses 7% against the simpler variant.
At B=8 the two variants are within 1–5% of each other at every length.

- **Likely cause (hypothesis, unconfirmed):** decode uses one block per (batch, head), with
  1 of 16 query rows active, so at B=1 there are only 12 blocks on a 40-SM T4. The key loop
  inside each block grows linearly with context, and nothing parallelises across the key
  dimension. That is a serial latency problem rather than a bandwidth one, and it gets
  worse with context length, which fits the data. The 1×12×1×512 microbenchmark
  (49 µs, 64 GB/s, 20% of peak) is also far from the bandwidth limit.
- **To confirm:** Nsight Compute on the decode kernel at T=768 (achieved occupancy, warp
  stall reasons). I could not get counter access on Kaggle.
- **Fix:** a split-K "flash-decoding" variant that partitions the keys across blocks and
  merges `(max, sum, acc)` partials in a second small kernel.

---

## 5. End to end

Greedy generation, 64 new tokens, T4. Speedup is vs HF eager. "torch" is our compact
model using plain PyTorch ops, which isolates how much of the gain is the model code
rather than the kernels.

| Variant | B=1 T=32 | B=1 T=768 | B=8 T=256 | B=8 T=768 |
|---|---|---|---|---|
| HF eager (tok/s) | 122.2 | 115.8 | 726.5 | 362.4 |
| HF + `torch.compile` | 195.1 (1.60×) | 153.2 (1.32×) | 741.0 (1.02×) | 351.9 (0.97×) |
| HF patched (LN + MLP) | 153.5 (1.26×) | 142.8 (1.23×) | 708.0 (0.97×) | 334.1 (0.92×) |
| ours, `torch` ops | 164.0 (1.34×) | 160.8 (1.39×) | 949.1 (1.31×) | 549.3 (1.52×) |
| ours, `cuda_v1` | 255.6 (2.09×) | 246.2 (2.13×) | 1161.9 (1.60×) | 614.2 (1.70×) |
| ours, `cuda_softmax` | 256.0 (2.10×) | 248.1 (2.14×) | 1199.8 (1.65×) | 634.5 (1.75×) |
| ours, `cuda_flash` | 318.0 (2.60×) | 230.1 (1.99×) | 1208.2 (1.66×) | 631.1 (1.74×) |

What this shows:

1. **A large part of the gain is not from the kernels.** The compact model with plain
   PyTorch ops is already 1.3–1.5× faster than HF eager (less Python and fewer launches,
   no growing-`cat` KV cache). The fair comparison for the kernels is against the `torch`
   row: `cuda_flash` is 1.94× at B=1, T=32, but only 1.27× at B=8, T=256 and 1.15× at B=8, T=768.
2. **Most of the kernel gain arrives with `cuda_v1`** (fused bias+GELU and a first
   LayerNorm/softmax). `cuda_softmax` and `cuda_flash` add little at B=8: the three
   variants are within 5% of each other at every B=8 length.
3. **Patching HF with just LayerNorm and the MLP is not enough.** `hf_patched` is faster
   than HF eager at B=1 (1.23–1.26×) but *slower* at B=8 for T ≥ 256 (0.92–0.97×),
   so those kernels alone can't overcome the surrounding HF overhead when attention
   and GEMMs dominate.
4. **`torch.compile` helps at B=1 and not at large batch.** 1.60× at B=1, T=32, but
   0.97× at B=8, T=768, consistent with the baseline profile showing small-batch decode is launch bound.
5. **Prefill at B=8, T=768 is the slowest phase (~410–520 ms)** and is nearly unchanged
   across variants, since it is dominated by cuBLAS GEMMs we don't touch.

Correctness: `python -m pytest -q tests` on the T4 gave **100 passed, 0 skipped, 0 failed**
(34.9 s). The tests check the kernels against PyTorch references and the model's logits and
greedy output against Hugging Face, so the throughput numbers above are for correct kernels.

## 6. What I'd do next

- Get Nsight Compute working (a machine with counter access, or a cloud GPU with
  `--cap-add=SYS_ADMIN`) to turn the hypotheses above into measurements: DRAM %, occupancy
  and warp stalls for the flash decode kernel; bank conflicts for the padded K tile.
- Split-K flash-decoding to fix the long-context B=1 regression.
- FP16 with WMMA/tensor cores. The T4 has 65 TFLOPS FP16 versus 8 TFLOPS FP32, and every
  kernel here is FP32. This is the real route to closing the prefill gap with SDPA.
- Preallocated KV cache in the HF-patched path (`aten::cat` was 7.4% of baseline GPU time).
- Re-run on an A100/L4 to see whether the float4 result (section 2) changes with higher bandwidth.
- Record CUDA and PyTorch versions in the benchmark outputs (they are missing here).
