#!/bin/bash
PY_VERSIONS=10
CUDA_HOME="/usr/local/cuda"
export NCCL_LIB_DIR=$CUDA_HOME/lib64
export NCCL_INCLUDE_DIR=$CUDA_HOME/include
PATH=/usr/local/cuda/bin:/tmp/build_tools/cmake/bin:$PATH
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

function build() {
    UPLOAD=""
    if [ "X$1" == "Xtrue" ]; then
        UPLOAD="upload"
    fi
    export CMAKE_BUILD_PARALLEL_LEVEL=16
    export BMT_AVX512=1
    for ver in $PY_VERSIONS
    do
        # TODO do parallel
        conda activate py3.$ver && pip install pybind11 && python setup.py bdist_wheel $UPLOAD || (echo "publish py3.$ver failed."; exit 2)
    done
    package_name=`python -c "import version; print(version.__package__)"`
    if [ "X$new_tag" != "X" ]; then
        scripts/weixin.sh "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=7ac0328d-0ed5-4fa9-83de-09b20a064447" "$package_name release $new_tag" "@all"
    fi
}

function main() {
    env
    release="false"
    if [ "$BUILD_TYPE" == "master" ]; then
        tags=`git tag -l | xargs`
        res=`python -c "import version; print(version.is_newer_version('${tags}'))"`
        if [ "$res" == "True" ]; then
            PY_VERSIONS=`seq 7 10`
            new_tag=`python -c "import version; print(version.__version__)"`
            git tag $new_tag && git push origin --tags && echo "start publish tag=$new_tag" || (echo "publish tag=$new_tag failed."; exit 3)
            release="true"
        fi
        build $release && echo "publish libcpm done!"
    fi

}

main $@