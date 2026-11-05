"""End-to-end checks: our GPT-2 vs Hugging Face, for every backend."""
import pytest
import torch

from gpt2_cuda import BACKENDS, GPT2, ops, patch_hf_model

PROMPTS = [
    "The quick brown fox",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
]
NEW_TOKENS = 24


@pytest.fixture(scope="module")
def ours(hf_gpt2):
    return GPT2.from_pretrained("gpt2", backend="torch")


def _hf_greedy(model, tok, prompt, n):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                         min_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return ids, out


def _backends():
    names = ["torch"]
    if ops.is_available():
        names += [b for b in BACKENDS if b != "torch"]
    return names


@pytest.mark.parametrize("backend", _backends())
@torch.no_grad()
def test_logits_match_hf(hf_gpt2, ours, backend):
    hf, tok = hf_gpt2
    ids = tok(PROMPTS[1], return_tensors="pt").input_ids.cuda()
    ref = hf(ids).logits
    out = ours.set_backend(backend)(ids)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=1e-4)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("prompt", PROMPTS)
@torch.no_grad()
def test_greedy_text_matches_hf(hf_gpt2, ours, backend, prompt):
    hf, tok = hf_gpt2
    ids, ref = _hf_greedy(hf, tok, prompt, NEW_TOKENS)
    out = ours.set_backend(backend).generate(ids, NEW_TOKENS)
    assert tok.decode(out[0]) == tok.decode(ref[0])


@pytest.mark.parametrize("backend", _backends())
@torch.no_grad()
def test_kv_cache_matches_full_recompute(hf_gpt2, ours, backend):
    _, tok = hf_gpt2
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.cuda().repeat(3, 1)
    model = ours.set_backend(backend)
    assert torch.equal(model.generate(ids, 16, use_cache=True), model.generate(ids, 16, use_cache=False))


@pytest.mark.skipif(not ops.is_available(), reason="CUDA extension not built / no GPU")
@torch.no_grad()
def test_patched_hf_model_matches(hf_gpt2):
    from transformers import GPT2LMHeadModel

    hf, tok = hf_gpt2
    patched = patch_hf_model(GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval())
    ids = tok(PROMPTS[1], return_tensors="pt").input_ids.cuda()
    torch.testing.assert_close(patched(ids).logits, hf(ids).logits, atol=2e-3, rtol=1e-4)
    for prompt in PROMPTS:
        _, ref = _hf_greedy(hf, tok, prompt, NEW_TOKENS)
        _, out = _hf_greedy(patched, tok, prompt, NEW_TOKENS)
        assert tok.decode(out[0]) == tok.decode(ref[0])
