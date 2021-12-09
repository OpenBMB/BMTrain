from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os

def get_avx_flags():
    if os.environ.get("BMP_AVX256", "").lower() in ["1", "true", "on"]:
        return ["-mavx", "-mfma", "-mf16c"]
    elif os.environ.get("BMP_AVX512", "").lower() in ["1", "true", "on"]:
        return ["-mavx512f"]
    else:
        return ["-march=native"]

def main():
    print(find_packages())

    avx_flag = get_avx_flags()
    setup(
        name='bmpretrain',
        version='0.0.9',
        packages=find_packages(),
        install_requires=[
            "torch>=1.10",
            "numpy",
            "typing-extensions>=4.0.0"
        ],
        ext_modules=[
            CUDAExtension('bmpretrain.nccl._C', [
                'csrc/nccl.cpp',
            ], include_dirs=["csrc/nccl/build/include"], extra_compile_args={}),
            CUDAExtension('bmpretrain.optim._cuda', [
                'csrc/adam_cuda.cpp',
                'csrc/cuda/adam.cu',
                'csrc/cuda/has_inf_nan.cu'
            ], extra_compile_args={}),
            CppExtension("bmpretrain.optim._cpu", [
                "csrc/adam_cpu.cpp",
            ], extra_compile_args=[
                '-fopenmp', 
                *avx_flag
            ], extra_link_args=['-lgomp'])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })

if __name__ == '__main__':
    main()