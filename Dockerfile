FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONUNBUFFERED=1

USER root
WORKDIR /root

RUN apt-get -y -qq update && \
    apt-get -y -qq --no-install-recommends \
    install \
    build-essential \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    nbtscan \
    git \
    iputils-ping \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libgl1 \
    unp \
    htop \
    swig \
    vim \
    ninja-build \
    rsync && \
    rm -rf /var/lib/apt/lists/*

# Install btop
RUN git clone https://github.com/aristocratos/btop.git && \
    cd btop && \
    make GPU_SUPPORT=true && \
    make install

# Setup unison
RUN wget https://github.com/bcpierce00/unison/releases/download/v2.53.0/unison-v2.53.0+ocaml-4.10.2+x86_64.linux.tar.gz && \
    unp unison-v2.53.0+ocaml-4.10.2+x86_64.linux.tar.gz && \
    cp bin/* /usr/local/bin && \
    rm -rf unison-v2.53.0+ocaml-4.10.2+x86_64.linux.tar.gz bin

# Install Python
ENV PYTHON=3.10.7
RUN wget -q https://www.python.org/ftp/python/${PYTHON}/Python-${PYTHON}.tgz && \
    tar -xf Python-${PYTHON}.tgz && \
    cd Python-${PYTHON} && \
    ./configure --enable-optimizations && \
    make -j install && \
    cd .. && \
    rm -rf Python-${PYTHON} Python-${PYTHON}.tgz && \
    python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel

# COPY requirements.txt .
# COPY setup .
# RUN ./setup && \
#     rm -rf /root/.cache/pip
