import pytest
import torch

from gpt2_cuda import ops, reference

pytestmark = pytest.mark.skipif(not ops.is_available(), reason="CUDA extension not built / no GPU")

TOL = dict(atol=1e-4, rtol=1e-4)


def _qkv(B, H, Tq, Tk, D=64):
    q = torch.randn(B, H, Tq, D, device="cuda")
    k = torch.randn(B, H, Tk, D, device="cuda")
    v = torch.randn(B, H, Tk, D, device="cuda")
    return q, k, v


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("B,H,Tq,Tk", [
    (1, 1, 1, 1),
    (1, 12, 16, 16),
    (2, 12, 64, 64),
    (1, 12, 100, 100),   # not a multiple of the 16x32 tile
    (2, 4, 257, 257),
    (1, 12, 1, 77),      # decode: one query against the cache
    (2, 12, 5, 130),     # several new tokens against a longer cache
    (1, 2, 1024, 1024),
])
def test_matches_reference(causal, B, H, Tq, Tk):
    q, k, v = _qkv(B, H, Tq, Tk)
    out = ops.flash_attention(q, k, v, causal=causal)
    ref = reference.attention(q, k, v, causal=causal)
    assert out.shape == (B, Tq, H * 64)
    torch.testing.assert_close(out, ref, **TOL)


def test_head_dim_128():
    q, k, v = _qkv(2, 4, 70, 70, D=128)
    torch.testing.assert_close(ops.flash_attention(q, k, v), reference.attention(q, k, v), **TOL)


def test_strided_views_from_qkv_and_cache():
    """The model passes q as a view of the fused qkv projection and k/v as slices of
    a preallocated cache; the kernel must honour those strides."""
    B, T, H, D, max_len, past = 2, 9, 12, 64, 64, 20
    qkv = torch.randn(B, T, 3, H, D, device="cuda")
    q = qkv[:, :, 0].transpose(1, 2)
    assert not q.is_contiguous()

    k_cache = torch.randn(B, H, max_len, D, device="cuda")
    v_cache = torch.randn(B, H, max_len, D, device="cuda")
    k = k_cache[:, :, :past + T]
    v = v_cache[:, :, :past + T]

    out = ops.flash_attention(q, k, v)
    ref = reference.attention(q.contiguous(), k.contiguous(), v.contiguous())
    torch.testing.assert_close(out, ref, **TOL)


def test_matches_pytorch_sdpa():
    q, k, v = _qkv(2, 12, 128, 128)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = ref.transpose(1, 2).reshape(2, 128, 12 * 64)
    torch.testing.assert_close(ops.flash_attention(q, k, v), ref, **TOL)


def test_rejects_bad_head_dim():
    q, k, v = _qkv(1, 1, 4, 4, D=48)
    with pytest.raises(RuntimeError, match="head_dim"):
        ops.flash_attention(q, k, v)
