"""A compact GPT-2 that loads Hugging Face weights and runs on pluggable ops.

Why a separate model instead of only patching transformers' GPT2Attention?
The attention module's forward signature and cache handling have changed
several times across transformers releases, while LayerNorm and the MLP have
stayed stable (see patch.py for those). Owning the forward pass makes the
fused-attention integration and the KV cache independent of the transformers
version, and lets every backend run the exact same graph, so benchmark
differences come only from the kernels.

Backends (see BACKENDS):
    torch         reference PyTorch ops (F.layer_norm, add+gelu, unfused attention)
    cuda_v1       first-version kernels: v1 LayerNorm, v1 bias+GELU, v1 softmax
    cuda_softmax  optimised kernels with matmul + fused softmax attention
    cuda_flash    optimised kernels with the fused FlashAttention-style kernel
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops, reference


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5


class TorchOps:
    name = "torch"

    def layer_norm(self, x, ln):
        return reference.layer_norm(x, ln.weight, ln.bias, ln.eps)

    def bias_gelu(self, x, bias):
        return reference.bias_gelu(x, bias)

    def attention(self, q, k, v, scale):
        return reference.attention(q, k, v, scale, causal=True)


class CudaOps:
    def __init__(self, name, attention="flash", ln_version=2, gelu_version=2, softmax_version=2):
        self.name = name
        self.attention_impl = attention
        self.ln_version = ln_version
        self.gelu_version = gelu_version
        self.softmax_version = softmax_version

    def layer_norm(self, x, ln):
        return ops.layer_norm(x, ln.weight, ln.bias, ln.eps, self.ln_version)

    def bias_gelu(self, x, bias):
        return ops.bias_gelu(x, bias, self.gelu_version)

    def attention(self, q, k, v, scale):
        if self.attention_impl == "flash":
            return ops.flash_attention(q, k, v, scale, causal=True)
        B, H, Tq, D = q.shape
        p = ops.causal_softmax(q @ k.transpose(-1, -2), scale, True, self.softmax_version)
        return (p @ v).transpose(1, 2).reshape(B, Tq, H * D)


BACKENDS = {
    "torch": lambda: TorchOps(),
    "cuda_v1": lambda: CudaOps("cuda_v1", attention="softmax", ln_version=1, gelu_version=1, softmax_version=1),
    "cuda_softmax": lambda: CudaOps("cuda_softmax", attention="softmax"),
    "cuda_flash": lambda: CudaOps("cuda_flash", attention="flash"),
}


class KVCache:
    """Preallocated [n_layer, B, H, max_len, D] key/value buffers.

    Attention reads k[layer, :, :, :length], a strided view the flash kernel
    consumes directly, so nothing is concatenated or copied per step.
    """

    def __init__(self, config, batch_size, max_len, device, dtype=torch.float32):
        head_dim = config.n_embd // config.n_head
        shape = (config.n_layer, batch_size, config.n_head, max_len, head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.length = 0


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        C = cfg.n_embd
        self.n_head = cfg.n_head
        self.ln_1 = nn.LayerNorm(C, eps=cfg.layer_norm_epsilon)
        # Stored as [in, out] like HF's Conv1D, so y = x @ W + b.
        self.c_attn_w = nn.Parameter(torch.empty(C, 3 * C))
        self.c_attn_b = nn.Parameter(torch.empty(3 * C))
        self.attn_proj_w = nn.Parameter(torch.empty(C, C))
        self.attn_proj_b = nn.Parameter(torch.empty(C))
        self.ln_2 = nn.LayerNorm(C, eps=cfg.layer_norm_epsilon)
        self.c_fc_w = nn.Parameter(torch.empty(C, 4 * C))
        self.c_fc_b = nn.Parameter(torch.empty(4 * C))
        self.mlp_proj_w = nn.Parameter(torch.empty(4 * C, C))
        self.mlp_proj_b = nn.Parameter(torch.empty(C))

    def forward(self, x, backend, layer_idx, cache=None):
        B, T, C = x.shape
        H = self.n_head
        D = C // H

        h = backend.layer_norm(x, self.ln_1)
        qkv = torch.addmm(self.c_attn_b, h.view(-1, C), self.c_attn_w).view(B, T, 3, H, D)
        # [B, H, T, D] views into qkv -- no copies.
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        if cache is not None:
            pos = cache.length
            cache.k[layer_idx, :, :, pos:pos + T] = k
            cache.v[layer_idx, :, :, pos:pos + T] = v
            k = cache.k[layer_idx, :, :, :pos + T]
            v = cache.v[layer_idx, :, :, :pos + T]
        a = backend.attention(q, k, v, 1.0 / math.sqrt(D))  # [B, T, C]
        x = x + torch.addmm(self.attn_proj_b, a.reshape(-1, C), self.attn_proj_w).view(B, T, C)

        h = backend.layer_norm(x, self.ln_2)
        u = backend.bias_gelu(h.view(-1, C) @ self.c_fc_w, self.c_fc_b)
        x = x + torch.addmm(self.mlp_proj_b, u, self.mlp_proj_w).view(B, T, C)
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config = GPT2Config(), backend="torch"):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.set_backend(backend)

    def set_backend(self, backend):
        if isinstance(backend, str):
            if backend not in BACKENDS:
                raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
            backend = BACKENDS[backend]()
        self.backend = backend
        return self

    @classmethod
    @torch.no_grad()
    def from_pretrained(cls, name="gpt2", backend="torch", device="cuda"):
        from transformers import GPT2LMHeadModel

        hf = GPT2LMHeadModel.from_pretrained(name)
        c = hf.config
        config = GPT2Config(vocab_size=c.vocab_size, n_positions=c.n_positions, n_embd=c.n_embd,
                            n_layer=c.n_layer, n_head=c.n_head, layer_norm_epsilon=c.layer_norm_epsilon)
        model = cls(config, backend)
        sd = hf.state_dict()
        model.wte.weight.copy_(sd["transformer.wte.weight"])
        model.wpe.weight.copy_(sd["transformer.wpe.weight"])
        model.ln_f.weight.copy_(sd["transformer.ln_f.weight"])
        model.ln_f.bias.copy_(sd["transformer.ln_f.bias"])
        names = {
            "ln_1.weight": "ln_1.weight", "ln_1.bias": "ln_1.bias",
            "c_attn_w": "attn.c_attn.weight", "c_attn_b": "attn.c_attn.bias",
            "attn_proj_w": "attn.c_proj.weight", "attn_proj_b": "attn.c_proj.bias",
            "ln_2.weight": "ln_2.weight", "ln_2.bias": "ln_2.bias",
            "c_fc_w": "mlp.c_fc.weight", "c_fc_b": "mlp.c_fc.bias",
            "mlp_proj_w": "mlp.c_proj.weight", "mlp_proj_b": "mlp.c_proj.bias",
        }
        for i, block in enumerate(model.h):
            params = dict(block.named_parameters())
            for ours, theirs in names.items():
                params[ours].copy_(sd[f"transformer.h.{i}.{theirs}"])
        del hf
        return model.to(device).eval().requires_grad_(False)  # inference only: kernels have no backward

    def forward(self, idx, cache=None, last_only=False):
        """idx: [B, T] token ids -> logits [B, T, vocab] (or [B, 1, vocab] if last_only)."""
        B, T = idx.shape
        pos = cache.length if cache is not None else 0
        if pos + T > self.config.n_positions:
            raise ValueError(f"sequence length {pos + T} exceeds n_positions={self.config.n_positions}")
        if cache is not None and pos + T > cache.max_len:
            raise ValueError(f"KV cache full ({cache.max_len} positions)")

        positions = torch.arange(pos, pos + T, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        for i, block in enumerate(self.h):
            x = block(x, self.backend, i, cache)
        if cache is not None:
            cache.length += T
        if last_only:
            x = x[:, -1:]
        x = self.backend.layer_norm(x.contiguous(), self.ln_f)
        return F.linear(x, self.wte.weight)  # tied embeddings

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=True):
        """Greedy decoding. Returns [B, T + max_new_tokens] token ids."""
        B, T = idx.shape
        cache = KVCache(self.config, B, T + max_new_tokens, idx.device) if use_cache else None
        logits = self.forward(idx, cache, last_only=True)
        for step in range(max_new_tokens):
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if step == max_new_tokens - 1:
                break
            logits = self.forward(nxt if use_cache else idx, cache, last_only=True)
        return idx
