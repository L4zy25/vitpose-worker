FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install Python and system deps
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv wget \
    libgl1-mesa-glx libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch + dependencies
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    mmengine mmdet mmpose runpod numpy opencv-python-headless

RUN pip install --no-cache-dir \
    mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# Download ViTPose-L weights
RUN mkdir -p /workspace/vitpose && \
    cd /workspace/vitpose && \
    wget -q https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth -O vitpose-l-coco.pth

# Copy handler
COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
