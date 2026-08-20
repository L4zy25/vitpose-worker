FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev python3.10-venv wget git gcc g++ \
    libgl1-mesa-glx libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir mmengine mmdet

RUN pip install --no-cache-dir \
    mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

RUN pip install --no-cache-dir --no-binary xtcocotools xtcocotools

RUN pip install --no-cache-dir mmpose runpod opencv-python-headless einops timm

# Force numpy back down and rebuild xtcocotools
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4" && \
    pip install --no-cache-dir --force-reinstall --no-binary xtcocotools xtcocotools

# Download ViTPose-L weights
RUN mkdir -p /workspace/vitpose && \
    cd /workspace/vitpose && \
    wget -q https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth -O vitpose-l-coco.pth

# Clone MotionBERT repo
RUN git clone https://github.com/Walter0807/MotionBERT.git /workspace/motionbert

# Download MotionBERT weights
RUN cd /workspace/motionbert && \
    wget --no-check-certificate https://github.com/Walter0807/MotionBERT/releases/download/v0.1/motionbert_lite.pth -O motionbert_lite.pth || \
    echo "MotionBERT weights download failed - will need manual upload"

ENV PYTHONPATH="${PYTHONPATH}:/workspace/motionbert"

COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
