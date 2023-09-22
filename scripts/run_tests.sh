export CUDA_HOME="/usr/local/cuda"
export NCCL_LIB_DIR=$CUDA_HOME/lib64
export NCCL_INCLUDE_DIR=$CUDA_HOME/include
export PATH=/usr/local/cuda/bin:/tmp/build_tools/cmake/bin:$PATH
export CMAKE_BUILD_PARALLEL_LEVEL=16
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$($HOME'/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate py3.10
#apt install rustc cargo -y
#pip3 install setuptools_rust -i https://mirror.in.zhihu.com/simple
pip install pytest pytest-custom_exit_code numpy pytrie -i https://mirror.in.zhihu.com/simple
pip install torch==1.12.0+cu113 -i https://mirror.sjtu.edu.cn/pytorch-wheels/
DEBUG=1 pip install .
# pytest has possibly mem leaks between tests, so we run each file independently.
find tests/ -name 'test_*.py'|xargs -n 1 pytest --suppress-no-test-exit-code --import-mode=importlib --verbose
