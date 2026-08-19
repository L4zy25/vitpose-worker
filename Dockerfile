FROM runpod/base:0.6.2-cuda12.1.0

WORKDIR /workspace

# Install Python dependencies
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install mmengine mmdet && \
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html && \
    pip install mmpose && \
    pip install runpod numpy opencv-python-headless

# Download ViTPose-L weights
RUN mkdir -p /workspace/vitpose && \
    cd /workspace/vitpose && \
    wget -q https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth -O vitpose-l-coco.pth

# Copy handler
COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
