FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN python -m pip install --upgrade pip

# PyTorch CUDA 12.1 wheels
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Your deps (add HF + timm + albu)
RUN pip install \
    numpy scipy pandas tqdm pyyaml rich click \
    opencv-python-headless pillow matplotlib seaborn \
    scikit-learn einops sentencepiece regex \
    omegaconf hydra-core tensorboard \
    timm>=0.9.0 albumentations>=1.3.0 safetensors \
    huggingface_hub

# Copy code (adjust to your repo layout)
COPY . /workspace
ENV PYTHONPATH=/workspace