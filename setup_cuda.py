"""Build the exq_dispatch CUDA extension."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Force CUDA 12.5 nvcc
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.5")
os.environ.setdefault("PATH", f"/usr/local/cuda-12.5/bin:{os.environ.get('PATH','')}")

setup(
    name="exq_dispatch_cuda",
    ext_modules=[
        CUDAExtension(
            name="exq_dispatch_cuda",
            sources=["exq/kernels/cuda_src/exq_dispatch.cu"],
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_86",           # A10G = Ampere sm_86
                    "--use_fast_math",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
