from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
print(find_packages())
setup(
    name='bmpretrain',
    version='0.0.8',
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "cpm_kernels>=1.0.9",
        "numpy",
        "typing-extensions>=4.0.0"
    ],
    ext_modules=[
        CUDAExtension('bmpretrain.nccl._C', [
            'csrc/nccl.cpp',
        ], include_dirs=["csrc/nccl/build/include"], extra_compile_args={}),
        CUDAExtension('bmpretrain.optim._cuda', [
            'csrc/adam_cuda.cpp',
            'csrc/cuda/adam.cu'
        ], extra_compile_args={}),
        CppExtension("bmpretrain.optim._cpu", [
            "csrc/adam_cpu.cpp",
        ], extra_compile_args=[
            '-fopenmp', 
            # '-march=native'
            '-mavx512f'
        ], extra_link_args=['-lgomp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
