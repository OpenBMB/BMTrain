from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from glob import glob

files =  glob("csrc/**/*.cu") + \
            glob("csrc/**/*.cpp") + \
            glob("csrc/*.cpp")

print(files)
setup(
    name='BMPretrain',
    version='test',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('BMPretrain._c',
           files
        , extra_compile_args={},
        include_dirs = [
            "csrc/includes"
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })