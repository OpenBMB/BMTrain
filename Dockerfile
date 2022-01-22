FROM nvidia/cuda:10.2-devel
WORKDIR /build
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel
RUN pip3 install torch==1.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt install iputils-ping opensm libopensm-dev libibverbs1 libibverbs-dev -y --no-install-recommends
ENV TORCH_CUDA_ARCH_LIST=6.1;7.0;7.5
ENV BMP_AVX512=1
ADD other_requirements.txt other_requirements.txt
RUN pip3 install -r other_requirements.txt
ADD . .
RUN python3 setup.py install

# training V2

RUN groupadd ma-group -g 1000 && \
    useradd -d /home/ma-user -m -u 1000 -g 1000 -s /bin/bash ma-user && \
    chmod 770 /home/ma-user && \
    chmod 770 /root && \
    # or silver bullet of files permission
    # chmod -R 777 /root && \
    usermod -a -G root ma-user

RUN apt-get install -y sudo && \
    awk 'BEGIN{print "Defaults !env_reset\nroot    ALL=(ALL:ALL) ALL\n\nma-user  ALL=(ALL) NOPASSWD:ALL\n%admin ALL=(ALL) ALL\n%sudo   ALL=(ALL:ALL) ALL"}' > /etc/sudoers

USER ma-user
WORKDIR /home/ma-user

ADD example example