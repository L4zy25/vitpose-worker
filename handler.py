import runpod
import base64
import numpy as np
import cv2
import torch
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import PoseDataSample

# Load model once at startup (stays in memory between requests)
model = None

def load_model():
    global model
    if model is None:
        config = '/workspace/vitpose/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py'
        checkpoint = '/workspace/vitpose/vitpose-l-coco.pth'
        model = init_model(config, checkpoint, device='cuda:0')
    return model

def handler(event):
    """
    Receives base64-encoded cropped person images.
    Returns 17 COCO keypoints per person.
    """
    input_data = event.get("input", {})
    frames = input_data.get("frames", [])
    
    model = load_model()
    results = []
    
    for frame_data in frames:
        # Decode base64 image
        img_bytes = base64.b64decode(frame_data["image"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Create bounding box for the full crop (entire image is the person)
        h, w = img.shape[:2]
        bbox = [0, 0, w, h]
        
        # Run ViTPose inference
        pose_results = inference_topdown(model, img, [bbox])
        
        keypoints = []
        if pose_results:
            kpts = pose_results[0].pred_instances.keypoints[0]  # 17x2
            scores = pose_results[0].pred_instances.keypoint_scores[0]  # 17
            for i in range(17):
                keypoints.append({
                    "x": float(kpts[i][0]),
                    "y": float(kpts[i][1]),
                    "score": float(scores[i])
                })
        
        results.append({
            "frame_id": frame_data.get("frame_id", 0),
            "keypoints": keypoints
        })
    
    return {"results": results}

# Pre-load model on startup
load_model()

runpod.serverless.start({"handler": handler})
