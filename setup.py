from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name='BMPretrain',
    version='test',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('BMPretrain._c', [
            "csrc/bind.cpp",
            'csrc/igemm/igemm_kernel.cu',
            'csrc/igemm/igemm.cpp',
            'csrc/position_bias/position_bias.cpp',
            'csrc/position_bias/position_bias_kernel.cu',
            'csrc/norm/norm_kernel.cu',
            'csrc/norm/norm.cpp',
            'csrc/softmax/softmax.cpp',
            'csrc/softmax/softmax_kernel.cu',
        ], extra_compile_args={},
        include_dirs = [
            "csrc/common"
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })