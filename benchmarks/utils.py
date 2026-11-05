"""Timing and reporting helpers shared by the benchmark scripts."""
import csv
from pathlib import Path
import statistics

import torch

RESULTS_DIR = Path(__file__).parent / "results"

# Theoretical DRAM bandwidth (GB/s) for GPUs you're likely to get on Colab/Kaggle/cloud.
PEAK_BANDWIDTH_GBPS = {
    "T4": 320, "P100": 732, "V100": 900, "A10G": 600, "L4": 300,
    "A100-SXM4-80GB": 2039, "A100": 1555, "H100": 3350,
    "RTX 3090": 936, "RTX 4090": 1008,
}


def gpu_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def peak_bandwidth(override=None):
    if override:
        return override
    name = gpu_name()
    for key, bw in PEAK_BANDWIDTH_GBPS.items():  # most specific keys first
        if key in name:
            return bw
    return None


def cuda_time_ms(fn, warmup=10, iters=50, repeats=5):
    """Median over `repeats` of the mean time per call (ms), measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def markdown_table(rows, columns=None):
    columns = columns or list(rows[0].keys())

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def write_markdown(path, title, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\nGPU: {gpu_name()}\n\n{markdown_table(rows, columns)}\n")
    print(f"wrote {path}")
