"""Week 1 baseline: Hugging Face GPT-2 in PyTorch eager mode.

    python benchmarks/baseline.py                 # sample text, throughput sweep, torch.profiler
    python benchmarks/baseline.py --skip-sweep    # profiler only

For an Nsight Systems timeline (NVTX ranges mark prefill/decode):
    nsys profile -t cuda,nvtx,osrt -o benchmarks/results/baseline \\
        --capture-range=cudaProfilerApi python benchmarks/baseline.py --nsys
"""
import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils import RESULTS_DIR, gpu_name, write_csv, write_markdown


def hf_generate(model, ids, new_tokens, pad_id):
    return model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=new_tokens,
                          min_new_tokens=new_tokens, do_sample=False, pad_token_id=pad_id)


@torch.no_grad()
def throughput_sweep(model, pad_id, batch_sizes, prompt_lens, new_tokens):
    rows = []
    for B in batch_sizes:
        for T in prompt_lens:
            ids = torch.randint(0, model.config.vocab_size, (B, T), device="cuda")
            hf_generate(model, ids, 4, pad_id)  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(ids)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1e3

            t0 = time.perf_counter()
            hf_generate(model, ids, new_tokens, pad_id)
            torch.cuda.synchronize()
            total = time.perf_counter() - t0
            rows.append(dict(batch=B, prompt_len=T, new_tokens=new_tokens, prefill_ms=prefill_ms,
                             generate_s=total, tokens_per_s=B * new_tokens / total,
                             ms_per_step=total * 1e3 / new_tokens))
            print(rows[-1])
    return rows


@torch.no_grad()
def run_profiler(model, pad_id, batch, prompt_len, new_tokens):
    ids = torch.randint(0, model.config.vocab_size, (batch, prompt_len), device="cuda")
    hf_generate(model, ids, 4, pad_id)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("prefill"):
            model(ids)
        with record_function("generate"):
            hf_generate(model, ids, new_tokens, pad_id)
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    print(table)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baseline_profile.txt").write_text(f"GPU: {gpu_name()}\n\n{table}\n")
    prof.export_chrome_trace(str(RESULTS_DIR / "baseline_trace.json"))
    print(f"wrote {RESULTS_DIR / 'baseline_profile.txt'} and baseline_trace.json (open in chrome://tracing or Perfetto)")


@torch.no_grad()
def run_nsys(model, pad_id, batch, prompt_len, new_tokens):
    ids = torch.randint(0, model.config.vocab_size, (batch, prompt_len), device="cuda")
    hf_generate(model, ids, 4, pad_id)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push("prefill")
    model(ids)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("generate")
    hf_generate(model, ids, new_tokens, pad_id)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--profile-batch", type=int, default=4)
    ap.add_argument("--profile-prompt-len", type=int, default=128)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--nsys", action="store_true", help="only run the region captured by nsys")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval()
    pad_id = tok.eos_token_id

    if args.nsys:
        run_nsys(model, pad_id, args.profile_batch, args.profile_prompt_len, args.new_tokens)
        return

    print(f"GPU: {gpu_name()}")
    ids = tok("Alan Turing theorized that computers would one day", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        print("Sample:", tok.decode(hf_generate(model, ids, 40, pad_id)[0]), "\n")

    if not args.skip_sweep:
        rows = throughput_sweep(model, pad_id, args.batch_sizes, args.prompt_lens, args.new_tokens)
        write_csv(RESULTS_DIR / "baseline.csv", rows)
        write_markdown(RESULTS_DIR / "baseline.md", "Baseline: HF GPT-2 eager", rows)

    run_profiler(model, pad_id, args.profile_batch, args.profile_prompt_len, args.new_tokens)


if __name__ == "__main__":
    main()
