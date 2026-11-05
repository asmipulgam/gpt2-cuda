import pytest
import torch

from gpt2_cuda import ops, reference

pytestmark = pytest.mark.skipif(not ops.is_available(), reason="CUDA extension not built / no GPU")

SHAPES = [
    (1, 1, 1, 1),
    (2, 12, 64, 64),
    (1, 12, 1, 100),     # decode step with a KV cache
    (2, 3, 17, 40),      # chunk of queries against a longer cache
    (1, 2, 300, 300),    # rows longer than a warp/block
    (1, 1, 5, 1500),
]


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_pytorch(version, causal, shape):
    scores = torch.randn(shape, device="cuda") * 5
    out = ops.causal_softmax(scores, 0.125, causal, version)
    ref = reference.causal_softmax(scores, 0.125, causal)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("version", [1, 2])
def test_rows_sum_to_one_and_mask_is_zero(version):
    tq, tk = 50, 80
    scores = torch.randn(2, 4, tq, tk, device="cuda") * 10
    out = ops.causal_softmax(scores, 1.0, True, version)
    torch.testing.assert_close(out.sum(-1), torch.ones(2, 4, tq, device="cuda"))
    masked = ~reference.causal_mask(tq, tk, "cuda")
    assert (out[..., masked] == 0).all()


def test_large_logits_do_not_overflow():
    scores = torch.full((1, 1, 4, 4), 1e4, device="cuda")
    scores[..., 0] = 2e4
    for version in (1, 2):
        out = ops.causal_softmax(scores, 1.0, False, version)
        assert torch.isfinite(out).all()


def test_causal_requires_tk_ge_tq():
    with pytest.raises(RuntimeError, match="Tk >= Tq"):
        ops.causal_softmax(torch.randn(1, 1, 8, 4, device="cuda"), 1.0, True)
