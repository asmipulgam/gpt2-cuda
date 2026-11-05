"""Python wrappers around the compiled extension (gpt2_cuda._C)."""
import math

import torch

try:
    from . import _C
    _IMPORT_ERROR = None
except ImportError as e:  # not built yet, or no CUDA toolkit on this machine
    _C = None
    _IMPORT_ERROR = e


def is_available() -> bool:
    return _C is not None and torch.cuda.is_available()


def _ext():
    if _C is None:
        raise RuntimeError(
            "gpt2_cuda._C is not built. On a machine with a CUDA GPU run:\n"
            "    pip install -e . --no-build-isolation"
        ) from _IMPORT_ERROR
    return _C


def layer_norm(x, weight, bias, eps=1e-5, version=2):
    """LayerNorm over the last dimension (v1: block/shared-mem, v2: warp/float4)."""
    return _ext().layer_norm(x.contiguous(), weight.contiguous(), bias.contiguous(), eps, version)


def bias_gelu(x, bias, version=2):
    """gelu_tanh(x + bias) in one pass (v1: scalar, v2: float4)."""
    return _ext().bias_gelu(x.contiguous(), bias.contiguous(), version)


def causal_softmax(scores, scale=1.0, causal=True, version=2):
    """softmax(scores * scale) over the last dim, with the causal mask fused in.

    scores is [..., Tq, Tk]. With Tk > Tq (KV cache) query i sees keys <= i + Tk - Tq.
    """
    return _ext().softmax(scores.contiguous(), scale, causal, version)


def flash_attention(q, k, v, scale=None, causal=True):
    """Fused attention. q: [B, H, Tq, D], k/v: [B, H, Tk, D] -> [B, Tq, H * D].

    Inputs may be strided views (e.g. a slice of a preallocated KV cache) as long
    as the last dimension is contiguous.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    return _ext().flash_attention(q, k, v, scale, causal)
