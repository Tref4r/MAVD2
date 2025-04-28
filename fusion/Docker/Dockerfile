FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace/fusion

RUN apt-get update && apt-get install -y wget curl git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==1.26.3 tqdm==4.66.5 scikit-learn==1.5.1 einops==0.8.0 opencv-python torchvggish ffmpeg-python soundfile matplotlib

CMD tail -f /dev/null