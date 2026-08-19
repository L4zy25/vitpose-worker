FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

WORKDIR /workspace

# Install mmcv from pip with CUDA support
RUN pip install mmengine mmdet && \
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html && \
    pip install mmpose

# Download ViTPose-L config and weights
RUN mkdir -p /workspace/vitpose && \
    cd /workspace/vitpose && \
    wget -q https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth -O vitpose-l-coco.pth

# Download config file
RUN cd /workspace/vitpose && \
    wget -q https://raw.githubusercontent.com/ViTAE-Transformer/ViTPose/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py || \
    echo "Config will be created manually"

# Install runpod
RUN pip install runpod

# Copy handler
COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
