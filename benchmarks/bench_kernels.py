"""Per-kernel benchmark: every version of every kernel vs PyTorch, across sizes.

    python benchmarks/bench_kernels.py [--peak-bw 320]

Writes benchmarks/results/kernels.csv and kernels.md. Bandwidth is "effective"
bandwidth: the minimum bytes the op must move (read inputs once, write outputs
once) divided by time, so it is directly comparable to the GPU's peak.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so gpt2_cuda imports without pip install

import torch
import torch.nn.functional as F

from gpt2_cuda import ops, reference
from utils import RESULTS_DIR, cuda_time_ms, peak_bandwidth, write_csv, write_markdown

C, H, D = 768, 12, 64
ROW_SHAPES = [(1, 128), (8, 128), (8, 512), (16, 1024)]  # (batch, seq) -> rows = batch * seq
ATTN_SHAPES = [(1, 128), (8, 128), (4, 512), (8, 1024)]


def record(rows, kernel, impl, shape, ms, nbytes, peak, baseline_ms):
    gbps = nbytes / ms / 1e6
    rows.append(dict(kernel=kernel, impl=impl, shape=shape, ms=ms, gbps=gbps,
                     pct_peak=(100 * gbps / peak) if peak else float("nan"),
                     speedup=baseline_ms / ms))


def bench_layernorm(rows, peak):
    for B, T in ROW_SHAPES:
        x = torch.randn(B * T, C, device="cuda")
        w, b = torch.randn(C, device="cuda"), torch.randn(C, device="cuda")
        nbytes = (2 * x.numel() + 2 * C) * 4
        impls = {
            "pytorch": lambda: F.layer_norm(x, (C,), w, b, 1e-5),
            "v1 block+smem": lambda: ops.layer_norm(x, w, b, 1e-5, 1),
            "v2 warp+float4": lambda: ops.layer_norm(x, w, b, 1e-5, 2),
        }
        base = None
        for name, fn in impls.items():
            ms = cuda_time_ms(fn)
            base = base or ms
            record(rows, "layernorm", name, f"{B * T}x{C}", ms, nbytes, peak, base)


def bench_bias_gelu(rows, peak):
    N = 4 * C
    for B, T in ROW_SHAPES:
        x = torch.randn(B * T, N, device="cuda")
        b = torch.randn(N, device="cuda")
        nbytes = (2 * x.numel() + N) * 4
        impls = {
            "pytorch (add + gelu)": lambda: F.gelu(x + b, approximate="tanh"),
            "v1 scalar": lambda: ops.bias_gelu(x, b, 1),
            "v2 float4": lambda: ops.bias_gelu(x, b, 2),
        }
        base = None
        for name, fn in impls.items():
            ms = cuda_time_ms(fn)
            base = base or ms
            record(rows, "bias_gelu", name, f"{B * T}x{N}", ms, nbytes, peak, base)


def bench_softmax(rows, peak):
    scale = 1 / math.sqrt(D)
    for B, T in ATTN_SHAPES:
        s = torch.randn(B, H, T, T, device="cuda")
        mask = reference.causal_mask(T, T, "cuda")
        nbytes = 2 * s.numel() * 4
        impls = {
            "pytorch (scale + mask + softmax)": lambda: reference.causal_softmax(s, scale, True, mask),
            "v1 block+smem": lambda: ops.causal_softmax(s, scale, True, 1),
            "v2 warp online": lambda: ops.causal_softmax(s, scale, True, 2),
        }
        base = None
        for name, fn in impls.items():
            ms = cuda_time_ms(fn)
            base = base or ms
            record(rows, "softmax", name, f"{B}x{H}x{T}x{T}", ms, nbytes, peak, base)


def bench_attention(rows, peak):
    scale = 1 / math.sqrt(D)
    for B, T in ATTN_SHAPES:
        q, k, v = (torch.randn(B, H, T, D, device="cuda") for _ in range(3))
        nbytes = 4 * q.numel() * 4  # q, k, v in, o out: the fused kernel's minimum traffic

        def unfused_ours():
            p = ops.causal_softmax(q @ k.transpose(-1, -2), scale, True, 2)
            return (p @ v).transpose(1, 2).reshape(B, T, C)

        impls = {
            "pytorch unfused": lambda: reference.attention(q, k, v, scale),
            "pytorch sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            "matmul + v2 softmax": unfused_ours,
            "flash (fused)": lambda: ops.flash_attention(q, k, v, scale),
        }
        base = None
        for name, fn in impls.items():
            ms = cuda_time_ms(fn, iters=20)
            base = base or ms
            record(rows, "attention", name, f"{B}x{H}x{T}x{D}", ms, nbytes, peak, base)

    # Decode: one query against a cache of length T.
    for B, T in [(1, 512), (8, 512), (8, 1024)]:
        q = torch.randn(B, H, 1, D, device="cuda")
        k, v = (torch.randn(B, H, T, D, device="cuda") for _ in range(2))
        nbytes = (2 * q.numel() + 2 * k.numel()) * 4
        impls = {
            "pytorch unfused": lambda: reference.attention(q, k, v, scale),
            "pytorch sdpa": lambda: F.scaled_dot_product_attention(q, k, v),
            "flash (fused)": lambda: ops.flash_attention(q, k, v, scale),
        }
        base = None
        for name, fn in impls.items():
            ms = cuda_time_ms(fn)
            base = base or ms
            record(rows, "attention_decode", name, f"{B}x{H}x1x{T}", ms, nbytes, peak, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-bw", type=float, help="peak DRAM bandwidth in GB/s (auto-detected for common GPUs)")
    ap.add_argument("--only", nargs="+", choices=["layernorm", "bias_gelu", "softmax", "attention"])
    args = ap.parse_args()
    if not ops.is_available():
        raise SystemExit("Build the extension first: pip install -e . --no-build-isolation")

    torch.backends.cuda.matmul.allow_tf32 = False
    peak = peak_bandwidth(args.peak_bw)
    print(f"peak bandwidth: {peak} GB/s" if peak else "peak bandwidth unknown; pass --peak-bw")

    rows = []
    benches = {"layernorm": bench_layernorm, "bias_gelu": bench_bias_gelu,
               "softmax": bench_softmax, "attention": bench_attention}
    for name, fn in benches.items():
        if args.only is None or name in args.only:
            fn(rows, peak)
            for r in rows:
                if r["kernel"].startswith(name):
                    print(f"{r['kernel']:>16} {r['shape']:>18} {r['impl']:>34}  {r['ms']:8.4f} ms  "
                          f"{r['gbps']:7.1f} GB/s  {r['pct_peak']:5.1f}%  x{r['speedup']:.2f}")

    write_csv(RESULTS_DIR / "kernels.csv", rows)
    write_markdown(RESULTS_DIR / "kernels.md", "Kernel benchmarks", rows,
                   ["kernel", "shape", "impl", "ms", "gbps", "pct_peak", "speedup"])


if __name__ == "__main__":
    main()
