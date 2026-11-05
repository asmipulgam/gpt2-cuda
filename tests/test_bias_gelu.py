import pytest
import torch
import torch.nn.functional as F

from gpt2_cuda import ops

pytestmark = pytest.mark.skipif(not ops.is_available(), reason="CUDA extension not built / no GPU")


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("shape", [(1, 3072), (77, 3072), (2, 9, 3072), (3, 4), (1000, 768)])
def test_matches_pytorch(version, shape):
    x = torch.randn(shape, device="cuda") * 4
    b = torch.randn(shape[-1], device="cuda")
    out = ops.bias_gelu(x, b, version)
    ref = F.gelu(x + b, approximate="tanh")
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cols", [1, 10, 3071])
def test_v1_odd_widths(cols):
    x = torch.randn(13, cols, device="cuda")
    b = torch.randn(cols, device="cuda")
    torch.testing.assert_close(ops.bias_gelu(x, b, 1), F.gelu(x + b, approximate="tanh"), atol=1e-5, rtol=1e-5)


def test_extreme_values():
    x = torch.tensor([[-50.0, -5.0, 0.0, 5.0, 50.0, 1e-8, -1e-8, 3.0]], device="cuda")
    b = torch.zeros(8, device="cuda")
    for version in (1, 2):
        out = ops.bias_gelu(x, b, version)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, F.gelu(x, approximate="tanh"), atol=1e-5, rtol=1e-5)
