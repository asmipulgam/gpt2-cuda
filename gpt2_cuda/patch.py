"""Monkey-patch a Hugging Face GPT2LMHeadModel to use the custom kernels.

Only LayerNorm and the MLP are patched here: both have had a stable
`forward(hidden_states)` interface across transformers releases. Fused
attention (and the KV cache that feeds it) lives in gpt2_cuda.model.GPT2,
which owns its forward pass so it doesn't depend on transformers internals.
"""
import torch.nn as nn

from . import ops


class CudaLayerNorm(nn.Module):
    """Drop-in for nn.LayerNorm that shares the original parameters."""

    def __init__(self, ln: nn.LayerNorm, version=2):
        super().__init__()
        self.weight = ln.weight
        self.bias = ln.bias
        self.eps = ln.eps
        self.version = version

    def forward(self, x):
        return ops.layer_norm(x, self.weight, self.bias, self.eps, self.version)


class FusedGPT2MLP(nn.Module):
    """c_fc matmul -> fused bias+GELU kernel -> c_proj. Replaces GPT2MLP (eval only)."""

    def __init__(self, mlp, version=2):
        super().__init__()
        self.c_fc = mlp.c_fc        # HF Conv1D: weight is [in, out]
        self.c_proj = mlp.c_proj
        self.version = version

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        u = ops.bias_gelu(x @ self.c_fc.weight, self.c_fc.bias, self.version)
        out = self.c_proj(u)
        return out.view(*shape[:-1], out.size(-1))


def patch_hf_model(model, layer_norm=True, mlp=True, version=2):
    """Swap kernels into a transformers GPT2LMHeadModel in place. Returns the model."""
    if model.training:
        raise ValueError("patch_hf_model is for inference; call model.eval() first")
    transformer = model.transformer
    if layer_norm:
        for block in transformer.h:
            block.ln_1 = CudaLayerNorm(block.ln_1, version)
            block.ln_2 = CudaLayerNorm(block.ln_2, version)
        transformer.ln_f = CudaLayerNorm(transformer.ln_f, version)
    if mlp:
        for block in transformer.h:
            block.mlp = FusedGPT2MLP(block.mlp, version)
    return model
