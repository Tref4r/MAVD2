FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace/mmaction

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/workspace:$PYTHONPATH"

RUN apt-get update && apt-get install -y wget curl git ninja-build libglib2.0-0 \
    libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3.8-distutils python3-pip \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --set python /usr/bin/python3.8 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --set python3 /usr/bin/python3.8

RUN pip install --upgrade pip==23.2.1 setuptools wheel
RUN pip install cython xtcocotools
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install openmim
RUN mim install mmengine
RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
RUN mim install mmdet
RUN mim install mmpose
RUN pip install fastapi uvicorn python-multipart requests einops

CMD pip install -v -e . && python main.py
