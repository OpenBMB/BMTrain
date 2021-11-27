from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths
print(find_packages())
setup(
    name='bmpretrain',
    version='0.0.5',
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "cpm_kernels>=1.0.8",
        "numpy"
    ],
    ext_modules=[
        CUDAExtension('bmpretrain.nccl._C', [
            'csrc/nccl.cpp',
        ], include_paths=["csrc/nccl/build/include"], extra_compile_args={}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
