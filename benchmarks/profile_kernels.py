"""Launch each custom kernel a few times at GPT-2 shapes so Nsight Compute can
profile it. Run through benchmarks/ncu.sh rather than directly."""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so gpt2_cuda imports without pip install

import torch

from gpt2_cuda import ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    B, T, C, H, D = args.batch, args.seq, 768, 12, 64
    x = torch.randn(B * T, C, device="cuda")
    w, b = torch.randn(C, device="cuda"), torch.randn(C, device="cuda")
    u = torch.randn(B * T, 4 * C, device="cuda")
    fb = torch.randn(4 * C, device="cuda")
    s = torch.randn(B, H, T, T, device="cuda")
    q, k, v = (torch.randn(B, H, T, D, device="cuda") for _ in range(3))
    kd, vd = (torch.randn(B, H, T, D, device="cuda") for _ in range(2))
    qd = torch.randn(B, H, 1, D, device="cuda")
    scale = 1 / math.sqrt(D)

    for _ in range(args.iters):
        for version in (1, 2):
            ops.layer_norm(x, w, b, 1e-5, version)
            ops.bias_gelu(u, fb, version)
            ops.causal_softmax(s, scale, True, version)
        ops.flash_attention(q, k, v, scale)     # prefill
        ops.flash_attention(qd, kd, vd, scale)  # decode
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
