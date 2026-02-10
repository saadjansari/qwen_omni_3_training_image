FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git wget curl ffmpeg libsndfile1 j\
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch 2.6.0 + CUDA 12.6
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Install ms-swift and training deps
RUN pip install ms-swift deepspeed==0.14.5 accelerate transformers wandb awscli "qwen-omni-utils[decord]" soundfile

# Install flash-attn (prebuilt wheel)
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# Install SageMaker training toolkit
RUN pip install sagemaker-training

# SageMaker expects the training script here
ENV SAGEMAKER_PROGRAM=train.sh
ENV PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code
