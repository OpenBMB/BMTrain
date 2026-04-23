FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
WORKDIR /build

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    cmake \
    ninja-build \
    git \
    iputils-ping opensm libopensm-dev libibverbs1 libibverbs-dev

RUN pip3 install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install --break-system-packages torch==2.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --break-system-packages numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV BMT_AVX512=1

ADD other_requirements.txt other_requirements.txt
RUN pip3 install --break-system-packages -r other_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

ADD . .
RUN pip3 install --break-system-packages .

WORKDIR /root
ADD example example
ADD tests tests