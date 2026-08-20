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
    "torch==2.1.2+cu121" "torchvision==0.16.2+cu121" --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir mmengine mmdet mmpretrain
RUN pip install --no-cache-dir \
    mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
RUN pip install --no-cache-dir --no-deps mmpose
RUN pip install --no-cache-dir scipy matplotlib pillow opencv-python-headless
RUN pip install --no-cache-dir runpod einops timm
RUN pip install --no-cache-dir munkres json-tricks
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4" && \
    pip install --no-cache-dir --force-reinstall --no-binary xtcocotools "numpy==1.26.4" xtcocotools && \
    python -c "import numpy; print('numpy:', numpy.__version__); assert numpy.__version__.startswith('1.26')"
RUN mkdir -p /workspace/vitpose && \
    cd /workspace/vitpose && \
    wget -q https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth -O vitpose-l-coco.pth
RUN git clone https://github.com/Walter0807/MotionBERT.git /workspace/motionbert
RUN cd /workspace/motionbert && \
    wget --no-check-certificate https://github.com/Walter0807/MotionBERT/releases/download/v0.1/motionbert_lite.pth -O motionbert_lite.pth || \
    echo "weights download failed"
ENV PYTHONPATH="/workspace/motionbert"
COPY handler.py /workspace/handler.py
CMD ["python", "/workspace/handler.py"]
