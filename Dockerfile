FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

USER root
WORKDIR /root


RUN apt-get -qq -y update && \
    apt-get -qq -y install \
    build-essential \
    wget \
    zlib1g-dev \
    libbz2-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    curl \
    rsync \
    git

ENV PY_VERSION 3.9.7
RUN wget -q https://www.python.org/ftp/python/${PY_VERSION}/Python-${PY_VERSION}.tgz && \
    tar -xf Python-${PY_VERSION}.tgz && \
    cd Python-${PY_VERSION} && \
    ./configure --enable-optimizations && \
    make -j install

RUN python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel

COPY requirements.txt .
RUN pip install -r requirements.txt
