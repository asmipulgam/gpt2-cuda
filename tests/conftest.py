import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    # Keep the PyTorch reference in true fp32 on Ampere+ so comparisons are fair.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield


@pytest.fixture(scope="session")
def hf_gpt2():
    """(model, tokenizer) for Hugging Face GPT-2 small, or skip if it can't be loaded."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval()
    except OSError as e:  # offline / no HF access
        pytest.skip(f"could not download gpt2: {e}")
    return model, tok
