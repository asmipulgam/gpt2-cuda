"""Build the CUDA extension.

    pip install -e . --no-build-isolation

Set TORCH_CUDA_ARCH_LIST to target a specific GPU (e.g. "7.5" for a T4,
"8.0" for an A100); by default PyTorch builds for the GPU it can see.
"""
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
CSRC = ROOT / "csrc"

setup(
    name="gpt2_cuda",
    version="0.1.0",
    description="Custom CUDA kernels (LayerNorm, bias+GELU, softmax, fused attention) for GPT-2 inference",
    packages=find_packages(include=["gpt2_cuda", "gpt2_cuda.*"]),
    ext_modules=[
        CUDAExtension(
            name="gpt2_cuda._C",
            sources=[str(p.relative_to(ROOT)) for p in sorted(CSRC.glob("*.cpp")) + sorted(CSRC.glob("*.cu"))],
            include_dirs=[str(CSRC)],
            extra_compile_args={
                "cxx": ["-O3"],
                # -lineinfo lets Nsight Compute map metrics back to source lines.
                "nvcc": ["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
