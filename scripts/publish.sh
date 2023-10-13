#!/bin/bash
TOOLS_URI="http://inner-hadoop06.dev.rack.zhihu.com:50070"
ZPYPI="https://mirror.in.zhihu.com/simple/"
QH_PYPI="https://pypi.tuna.tsinghua.edu.cn/simple"
TOOLS_PATH="/data/klara/tools"
USER="ici_search"
PY_VERSIONS=10
CUDA_HOME="/usr/local/cuda"


function init_env() {
    # git
    # git submodule update --init
    env

    # cmake & cuda & conda & py3.{7-10}
    rm -rf /tmp/build_tools*
    pip install hdfs -i $ZPYPI

    python -c "import hdfs; hdfs.InsecureClient('$TOOLS_URI', '$USER').download('$TOOLS_PATH/nccl_2.12.10.tgz', '/tmp', n_threads=0)"
    python -c "import hdfs; hdfs.InsecureClient('$TOOLS_URI', '$USER').download('$TOOLS_PATH/build_tools.tgz', '/tmp', n_threads=0)"
    tar -xzvf /tmp/build_tools.tgz -C /tmp &>/dev/null
    bash /tmp/build_tools/miniconda -b
    bash /tmp/build_tools/cuda112.run --silent --toolkit

    tar -xzvf /tmp/nccl_2.12.10.tgz -C /tmp &>/dev/null
    mv /tmp/nccl_2.12.10/lib/* $CUDA_HOME/lib64/ && export NCCL_LIB_DIR=$CUDA_HOME/lib64
    mv /tmp/nccl_2.12.10/include/* $CUDA_HOME/include/  && export NCCL_INCLUDE_DIR=$CUDA_HOME/include
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

function build() {
    UPLOAD=""
    if [ "X$1" == "Xtrue" ]; then
        UPLOAD="upload"
    fi
    export CMAKE_BUILD_PARALLEL_LEVEL=16
    for ver in $PY_VERSIONS
    do
        # TODO do parallel
        conda activate py3.$ver && pip install -r doc_requirements.txt -i $QH_PYPI && pip install -r other_requirements.txt -i $QH_PYPI && \
	python setup.py sdist $UPLOAD || (echo "publish py3.$ver failed."; exit 2)
    done
    package_name=`python -c "import version; print(version.__package__)"`
    if [ "X$new_tag" != "X" ]; then
        scripts/weixin.sh "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=7ac0328d-0ed5-4fa9-83de-09b20a064447" "$package_name release $new_tag" "@all"
    fi
}

function main() {
    release="false"
    if [ "$BUILD_TYPE" == "master" ]; then
        tags=`git tag -l | xargs`
        res=`python -c "import version; print(version.is_newer_version('${tags}'))"`
        if [ "$res" == "True" ]; then
            #PY_VERSIONS=`seq 7 10`
            new_tag=`python -c "import version; print(version.__version__)"`
            git tag $new_tag && git push origin --tags && echo "start publish tag=$new_tag" || (echo "publish tag=$new_tag failed."; exit 3)
            release="true"
        fi
    fi
    # init build env
    init_env
    # build and publish if needed
    build $release && echo "publish bmtrain done!"
}

main $@
