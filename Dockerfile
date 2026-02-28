# Text2ObjectSDF - Docker image with CUDA support for training and inference
# Uses PyTorch CUDA base so that tinycudann (HashGrid) and GPU training work.

ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=12.1
# Use -devel image so we have nvcc to build tinycudann from source (not on PyPI)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system deps: build-essential for compiling, libspatialindex for rtree
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libspatialindex-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Project files (configs, src, scripts, sampling, requirements)
COPY configs/           ./configs/
COPY src/               ./src/
COPY scripts/            ./scripts/
COPY sampling/           ./sampling/
COPY requirements.txt   requirements.txt
COPY requirements-docker.txt requirements-docker.txt

# tinycudann is not on PyPI; build from source. No GPU at build time, so we must set
# target architectures explicitly (see https://developer.nvidia.com/cuda-gpus).
# 61=Pascal, 75=Turing, 80=A100, 86=RTX30/Ampere, 89=RTX40/Ada.
# Override example:
#   docker build --build-arg TCNN_CUDA_ARCHITECTURES="61;75;80;86;89" .
ARG TCNN_CUDA_ARCHITECTURES=61;75;80;86;89
ENV TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}
RUN pip install --no-cache-dir "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Ensure project root is on PYTHONPATH for "from src.xxx" imports
ENV PYTHONPATH=/app

# Default: bash so you can run any command (train, inference, compute_sdf)
# Example: docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints <image> python scripts/train.py
CMD ["/bin/bash"]
