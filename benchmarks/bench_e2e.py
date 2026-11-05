"""End-to-end GPT-2 generation benchmark.

    python benchmarks/bench_e2e.py
    python benchmarks/bench_e2e.py --variants hf_eager cuda_flash --batch-sizes 1 8

Variants:
    hf_eager       transformers GPT2LMHeadModel, eager mode
    hf_compile     same, with torch.compile on the forward
    hf_patched     transformers model with our LayerNorm + bias/GELU kernels patched in
    torch          our GPT-2 with plain PyTorch ops (isolates model-code differences)
    cuda_v1        our GPT-2, first-version kernels
    cuda_softmax   our GPT-2, optimised kernels + fused softmax
    cuda_flash     our GPT-2, optimised kernels + fused attention

All variants do greedy decoding with a KV cache and generate exactly --new-tokens.
"""
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so gpt2_cuda imports without pip install

import torch
from transformers import GPT2LMHeadModel

from gpt2_cuda import GPT2, KVCache, ops, patch_hf_model
from utils import RESULTS_DIR, gpu_name, write_csv, write_markdown

HF_VARIANTS = ["hf_eager", "hf_compile", "hf_patched"]
OUR_VARIANTS = ["torch", "cuda_v1", "cuda_softmax", "cuda_flash"]


def load_hf(variant):
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval()
    if variant == "hf_compile":
        model.forward = torch.compile(model.forward, dynamic=True)
    elif variant == "hf_patched":
        patch_hf_model(model)
    return model


def make_runner(variant, model):
    """Returns (prefill_fn(ids), generate_fn(ids, n))."""
    if variant in HF_VARIANTS:
        pad = model.config.eos_token_id

        def prefill(ids):
            return model(ids, use_cache=True)

        def generate(ids, n):
            return model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                                  min_new_tokens=n, do_sample=False, pad_token_id=pad)
        return prefill, generate

    model.set_backend(variant)

    def prefill(ids):
        cache = KVCache(model.config, ids.size(0), ids.size(1), ids.device)
        return model(ids, cache, last_only=True)

    return prefill, model.generate


def time_s(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.inference_mode()
def bench_variant(variant, model, batch_sizes, prompt_lens, new_tokens, repeats):
    prefill, generate = make_runner(variant, model)
    rows = []
    for B in batch_sizes:
        for T in prompt_lens:
            ids = torch.randint(0, 50257, (B, T), device="cuda", generator=torch.Generator("cuda").manual_seed(0))
            generate(ids, 8)  # warmup (and compilation for hf_compile)
            prefill(ids)
            prefill_ms = min(time_s(lambda: prefill(ids)) for _ in range(repeats)) * 1e3
            gen_s = min(time_s(lambda: generate(ids, new_tokens)) for _ in range(repeats))
            decode_ms = max(gen_s * 1e3 - prefill_ms, 0.0) / max(new_tokens - 1, 1)
            rows.append(dict(variant=variant, batch=B, prompt_len=T, new_tokens=new_tokens,
                             prefill_ms=prefill_ms, decode_ms_per_token=decode_ms,
                             generate_s=gen_s, tokens_per_s=B * new_tokens / gen_s))
            r = rows[-1]
            print(f"{variant:>13} B={B:<3} T={T:<5} prefill {r['prefill_ms']:8.2f} ms   "
                  f"decode {r['decode_ms_per_token']:6.2f} ms/tok   {r['tokens_per_s']:9.1f} tok/s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=HF_VARIANTS + OUR_VARIANTS,
                    choices=HF_VARIANTS + OUR_VARIANTS)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[32, 256, 768])
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    needs_ext = [v for v in args.variants if v.startswith("cuda") or v == "hf_patched"]
    if needs_ext and not ops.is_available():
        raise SystemExit(f"{needs_ext} need the extension: pip install -e . --no-build-isolation")
    if max(args.prompt_lens) + args.new_tokens > 1024:
        raise SystemExit("prompt_len + new_tokens must be <= 1024 (GPT-2 context)")

    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU: {gpu_name()}")
    rows = []
    ours = None
    for variant in args.variants:
        if variant in HF_VARIANTS:
            model = load_hf(variant)
        else:
            ours = ours or GPT2.from_pretrained("gpt2")
            model = ours
        try:
            rows += bench_variant(variant, model, args.batch_sizes, args.prompt_lens, args.new_tokens, args.repeats)
        except Exception as e:  # e.g. torch.compile failing on an unusual setup
            print(f"{variant} failed: {type(e).__name__}: {e}")
        if variant in HF_VARIANTS:
            del model
            torch.cuda.empty_cache()

    # Speedup vs HF eager for the same (batch, prompt_len).
    base = {(r["batch"], r["prompt_len"]): r["tokens_per_s"] for r in rows if r["variant"] == "hf_eager"}
    for r in rows:
        b = base.get((r["batch"], r["prompt_len"]))
        r["speedup_vs_hf_eager"] = r["tokens_per_s"] / b if b else float("nan")

    write_csv(RESULTS_DIR / "e2e.csv", rows)
    write_markdown(RESULTS_DIR / "e2e.md", "End-to-end GPT-2 generation", rows,
                   ["variant", "batch", "prompt_len", "prefill_ms", "decode_ms_per_token",
                    "tokens_per_s", "speedup_vs_hf_eager"])


if __name__ == "__main__":
    main()
