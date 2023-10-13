#!/bin/bash
TOOLS_URI="http://inner-hadoop06.dev.rack.zhihu.com:50070"
ZPYPI="https://mirror.in.zhihu.com/simple/"
TOOLS_PATH="/data/klara/tools"
USER="ici_search"
PY_VERSIONS=`seq 7 10`
CUDA_HOME="/usr/local/cuda"
CUDA_VERSION="11.6.2"
NCCL_VERSION="2.12.12"


function init_env() {
    # git
    git submodule update --init

    # cmake & cuda & conda & py3.{7-10}
    rm -rf /tmp/build_tools*
    pip3 install hdfs -i $ZPYPI

    python3 -c "import hdfs; hdfs.InsecureClient('$TOOLS_URI', '$USER').download('$TOOLS_PATH/nccl_$NCCL_VERSION.tgz', '/tmp', n_threads=0)"
    python3 -c "import hdfs; hdfs.InsecureClient('$TOOLS_URI', '$USER').download('$TOOLS_PATH/build_tools_cuda$CUDA_VERSION.tgz', '/tmp', n_threads=0)"
    tar -xzvf /tmp/build_tools_cuda$CUDA_VERSION.tgz -C /tmp &>/dev/null && ln -sf /tmp/build_tools_cuda$CUDA_VERSION /tmp/build_tools
    bash /tmp/build_tools/miniconda -b
    bash /tmp/build_tools/cuda.run --silent --toolkit

    tar -xzvf /tmp/nccl_$NCCL_VERSION.tgz -C /tmp &>/dev/null
    mv /tmp/nccl_$NCCL_VERSION/lib/* $CUDA_HOME/lib64/ && export NCCL_LIB_DIR=$CUDA_HOME/lib64
    mv /tmp/nccl_$NCCL_VERSION/include/* $CUDA_HOME/include/  && export NCCL_INCLUDE_DIR=$CUDA_HOME/include
    export PATH=/usr/local/cuda/bin:/tmp/build_tools/cmake/bin:$PATH
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
    for ver in $PY_VERSIONS
    do
        conda create -y -n py3.$ver python=3.$ver &>/dev/null && echo "install py3.$ver success." || echo "install py3.$ver failed."
    done
    cmake --version && nvcc --version && conda --version || (echo "init build env failed"; exit 1)
    conda env list

    # pypi
    cat ./.pypirc | sed "s/__PYPI_USER__/$PUBLISHER_USERNAME/g" | sed "s/__PYPI_PASSWORD__/$PUBLISHER_PASSWORD/g" &> $HOME/.pypirc
}

function check_version() {
    tags=`git tag -l | xargs`
    python -c "import version; print(version.is_newer_version('${tags}'))"
}

function main() {
    env
    # init build env
    init_env
}

main $@