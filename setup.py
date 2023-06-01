from setuptools import setup, find_packages
import torch
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import os
import os
import setuptools
import subprocess
from pkg_resources import packaging
from typing import Dict, List, Optional, Union, Tuple
def CppExtension(name, sources, *args, **kwargs):
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths()
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths()
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    kwargs['libraries'] = libraries

    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])


    include_dirs += include_paths(cuda=True)
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    dlink_libraries = kwargs.get('dlink_libraries', [])
    dlink = kwargs.get('dlink', False) or dlink_libraries
    if dlink:
        extra_compile_args = kwargs.get('extra_compile_args', {})

        extra_compile_args_dlink = extra_compile_args.get('nvcc_dlink', [])
        extra_compile_args_dlink += ['-dlink']
        extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
        extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]


        extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink

        kwargs['extra_compile_args'] = extra_compile_args

    return setuptools.Extension(name, sources, *args, **kwargs)

def _find_cuda_home() -> Optional[str]:
    r'''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'],
                                               stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            #Guess #3
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    return cuda_home

CUDA_HOME = _find_cuda_home()
_BMT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _join_cuda_home(*paths) -> str:
    if CUDA_HOME is None:
        raise EnvironmentError('CUDA_HOME environment variable is not set. '
                               'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)

def CppExtension(name, sources, *args, **kwargs):
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('cudart')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])

    include_dirs += include_paths(cuda=True)
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    dlink_libraries = kwargs.get('dlink_libraries', [])
    dlink = kwargs.get('dlink', False) or dlink_libraries
    if dlink:
        extra_compile_args = kwargs.get('extra_compile_args', {})

        extra_compile_args_dlink = extra_compile_args.get('nvcc_dlink', [])
        extra_compile_args_dlink += ['-dlink']
        extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
        extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]

        extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink

        kwargs['extra_compile_args'] = extra_compile_args

    return setuptools.Extension(name, sources, *args, **kwargs)


def include_paths(cuda: bool = False) -> List[str]:
    paths = [
        os.path.join(_BMT_PATH, 'csrc', 'include'),
    ]
    if cuda:
        cuda_home_include = _join_cuda_home('include')
        # if we have the Debian/Ubuntu packages for cuda, we get /usr as cuda home.
        # but gcc doesn't like having /usr/include passed explicitly
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)
    return paths


def library_paths(cuda: bool = False) -> List[str]:
    paths = []

    if cuda:
        lib_dir = 'lib64'
        if (not os.path.exists(_join_cuda_home(lib_dir)) and
                os.path.exists(_join_cuda_home('lib'))):
            lib_dir = 'lib'

        paths.append(_join_cuda_home(lib_dir))
    return paths

def get_avx_flags():
    if os.environ.get("BMT_AVX256", "").lower() in ["1", "true", "on"]:
        return ["-mavx", "-mfma", "-mf16c"]
    elif os.environ.get("BMT_AVX512", "").lower() in ["1", "true", "on"]:
        return ["-mavx512f"]
    else:
        return ["-march=native"]

def get_device_cc():
    try:
        CC_SET = set()
        for i in range(torch.cuda.device_count()):
            CC_SET.add(torch.cuda.get_device_capability(i))
        
        if len(CC_SET) == 0:
            return None
        
        ret = ""
        for it in CC_SET:
            if len(ret) > 0:
                ret = ret + " "
            ret = ret + ("%d.%d" % it)
        return ret
    except RuntimeError:
        return None

avx_flag = get_avx_flags()
device_cc = get_device_cc()
if device_cc is None:
    if not torch.cuda.is_available():
        os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "6.0 6.1 7.0 7.5 8.0+PTX")
    else:
        if torch.version.cuda.startswith("10"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "6.0 6.1 7.0 7.5+PTX")
        else:
            if not torch.version.cuda.startswith("11.0"):
                os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "6.0 6.1 7.0 7.5 8.0 8.6+PTX")
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "6.0 6.1 7.0 7.5 8.0+PTX")
else:
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", device_cc)

ext_modules = []

if os.environ.get("GITHUB_ACTIONS", "false") == "false":
    ext_modules = [
        CUDAExtension('bmtrain.nccl._C', [
            'csrc/nccl.cpp',
        ], include_dirs=["csrc/nccl/build/include"], extra_compile_args={}, extra_link_args=["../libnccl_static.a"]),
        CUDAExtension('bmtrain.optim._cuda', [
            'csrc/adam_cuda.cpp',
            'csrc/cuda/adam.cu',
            'csrc/cuda/has_inf_nan.cu'
        ], extra_compile_args={}),
        CppExtension("bmtrain.optim._cpu", [
            "csrc/adam_cpu.cpp",
        ], extra_compile_args=[
            '-fopenmp', 
            *avx_flag
        ], extra_link_args=['-lgomp']),
        CUDAExtension('bmtrain.loss._cuda', [
            'csrc/cross_entropy_loss.cpp',
            'csrc/cuda/cross_entropy.cu',
        ], extra_compile_args={}),
    ]
else:
    ext_modules = []

setup(
    name='bmtrain',
    version='0.2.2',
    author="Guoyang Zeng",
    author_email="qbjooo@qq.com",
    description="A toolkit for training big models",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': build_ext
    })

