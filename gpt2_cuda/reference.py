"""Plain-PyTorch reference implementations: the ground truth for tests and the
"PyTorch eager" side of every benchmark."""
import math

import torch
import torch.nn.functional as F


def layer_norm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.size(-1),), weight, bias, eps)


def bias_gelu(x, bias):
    return F.gelu(x + bias, approximate="tanh")


def causal_mask(tq, tk, device):
    """True where query i may attend to key j (j <= i + tk - tq)."""
    return torch.ones(tq, tk, dtype=torch.bool, device=device).tril(diagonal=tk - tq)


def causal_softmax(scores, scale=1.0, causal=True, mask=None):
    s = scores * scale
    if causal:
        if mask is None:
            mask = causal_mask(s.size(-2), s.size(-1), s.device)
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1)


def attention(q, k, v, scale=None, causal=True):
    """Unfused attention: materialises the full [Tq, Tk] matrix. Returns [B, Tq, H * D]."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    B, H, Tq, D = q.shape
    p = causal_softmax(q @ k.transpose(-1, -2), scale, causal)
    return (p @ v).transpose(1, 2).reshape(B, Tq, H * D)
