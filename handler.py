import runpod
import base64
import numpy as np
import cv2
import os
import json

model = None
device = None

def load_model():
    global model, device
    if model is None:
        import torch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load ViTPose-L directly with torch
        checkpoint = torch.load('/workspace/vitpose/vitpose-l-coco.pth', map_location=device, weights_only=False)
        
        # For now, use a simpler approach with mmpose
        # Try importing step by step to find the error
        try:
            import mmpose
            print(f"mmpose version: {mmpose.__version__}")
        except Exception as e:
            print(f"mmpose import error: {e}")
        
        try:
            import mmcv
            print(f"mmcv version: {mmcv.__version__}")
        except Exception as e:
            print(f"mmcv import error: {e}")
            
        try:
            import mmengine
            print(f"mmengine version: {mmengine.__version__}")
        except Exception as e:
            print(f"mmengine import error: {e}")
        
        try:
            import mmdet
            print(f"mmdet version: {mmdet.__version__}")
        except Exception as e:
            print(f"mmdet import error: {e}")
        
        print("All imports checked")
        model = True  # placeholder
    return model

def handler(event):
    input_data = event.get("input", {})
    
    # Debug mode - just return import status
    if input_data.get("debug"):
        load_model()
        return {"status": "ok", "message": "imports checked - see logs"}
    
    frames = input_data.get("frames", [])
    if not frames:
        return {"results": [], "status": "no frames provided"}
    
    return {"results": [], "status": "handler reached"}

print("Starting ViTPose worker...")
try:
    load_model()
except Exception as e:
    print(f"Model load error: {e}")

runpod.serverless.start({"handler": handler})
