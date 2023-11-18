# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

LABEL maintainer="yifeiwu1997@outlook.com"

# setup environment
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip install numpy tokenizers prompt_toolkit rwkv --upgrade

# obtain code
RUN mkdir -p /workspace/kqwyf && \
    git clone --depth=1 https://github.com/kqwyf/RWKV-LM /workspace/kqwyf/RWKV-LM

