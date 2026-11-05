import pytest
import torch
import torch.nn.functional as F

from gpt2_cuda import ops

pytestmark = pytest.mark.skipif(not ops.is_available(), reason="CUDA extension not built / no GPU")


def _inputs(shape, device="cuda"):
    cols = shape[-1]
    x = torch.randn(shape, device=device) * 3 + 1
    w = torch.randn(cols, device=device)
    b = torch.randn(cols, device=device)
    return x, w, b


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("shape", [(1, 768), (7, 768), (128, 768), (2, 5, 768), (33, 1024), (5, 2048), (3, 4)])
def test_matches_pytorch(version, shape):
    x, w, b = _inputs(shape)
    out = ops.layer_norm(x, w, b, 1e-5, version)
    ref = F.layer_norm(x, (shape[-1],), w, b, 1e-5)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cols", [1, 100, 777, 5000])
def test_v1_odd_widths(cols):
    x, w, b = _inputs((9, cols))
    torch.testing.assert_close(ops.layer_norm(x, w, b, 1e-5, 1),
                               F.layer_norm(x, (cols,), w, b, 1e-5), atol=1e-5, rtol=1e-5)


def test_large_mean_is_stable():
    # Two-pass variance shouldn't lose precision when |mean| >> std.
    x, w, b = _inputs((16, 768))
    x = x + 1000.0
    for version in (1, 2):
        torch.testing.assert_close(ops.layer_norm(x, w, b, 1e-5, version),
                                   F.layer_norm(x, (768,), w, b, 1e-5), atol=1e-3, rtol=1e-3)


def test_v2_rejects_unsupported_width():
    x, w, b = _inputs((4, 770))
    with pytest.raises(RuntimeError, match="multiple of 4"):
        ops.layer_norm(x, w, b, 1e-5, 2)


def test_rejects_cpu_tensor():
    x, w, b = _inputs((4, 768), device="cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        ops.layer_norm(x, w, b)
