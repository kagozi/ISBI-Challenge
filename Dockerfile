FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# PyTorch CUDA 12.1
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision

# Only what your training code needs
RUN pip install \
    numpy scipy pandas tqdm \
    scikit-learn \
    matplotlib seaborn \
    pillow \
    opencv-python-headless \
    timm>=0.9.0 \
    albumentations>=1.3.0 \
    einops safetensors \
    huggingface_hub

COPY . /workspace
ENV PYTHONPATH=/workspace