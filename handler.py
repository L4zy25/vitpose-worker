import runpod
import base64
import numpy as np
import cv2
import os
import torch

model = None

def load_model():
    global model
    if model is None:
        try:
            from mmpretrain.models import backbones
        except:
            pass
        
        from mmpose.apis import init_model
        import mmpose
        import glob
        
        # Find mmpose's bundled ViTPose config
        mmpose_dir = os.path.dirname(mmpose.__file__)
        candidates = glob.glob(os.path.join(mmpose_dir, '**', '*ViTPose*large*256x192*.py'), recursive=True)
        print(f"Config candidates: {candidates}")
        
        if candidates:
            config_path = candidates[0]
        else:
            # Try .mim directory
            candidates = glob.glob(os.path.join(mmpose_dir, '.mim', '**', '*ViTPose*large*256x192*.py'), recursive=True)
            print(f"Mim candidates: {candidates}")
            if candidates:
                config_path = candidates[0]
            else:
                raise RuntimeError("No ViTPose config found in mmpose installation")
        
        print(f"Using config: {config_path}")
        checkpoint = '/workspace/vitpose/vitpose-l-coco.pth'
        model = init_model(config_path, checkpoint, device='cuda:0')
        print("ViTPose-L loaded on GPU")
    return model

def handler(event):
    from mmpose.apis import inference_topdown

    input_data = event.get("input", {})
    frames = input_data.get("frames", [])

    if not frames:
        return {"results": [], "status": "no frames"}

    pose_model = load_model()
    results = []
    image_cache = {}

    for frame_data in frames:
        img_b64 = frame_data["image"]
        if img_b64 not in image_cache:
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image_cache[img_b64] = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = image_cache[img_b64]

        bbox = frame_data.get("bbox")
        if bbox:
            bboxes = np.array([bbox], dtype=np.float32)
        else:
            h, w = img.shape[:2]
            bboxes = np.array([[0, 0, w, h]], dtype=np.float32)

        keypoints = []
        try:
            pose_results = inference_topdown(pose_model, img, bboxes)
            if pose_results:
                pr = pose_results[0]
                kpts = pr.pred_instances.keypoints
                scores = pr.pred_instances.keypoint_scores
                if kpts is not None and len(kpts) > 0:
                    s = scores[0] if len(scores.shape) > 1 else scores
                    for i in range(min(17, kpts.shape[1])):
                        keypoints.append({
                            "x": float(kpts[0][i][0]),
                            "y": float(kpts[0][i][1]),
                            "score": float(s[i])
                        })
        except Exception as e:
            keypoints = []
            print(f"Pose error: {e}")
            import traceback
            traceback.print_exc()

        results.append({
            "frame_id": frame_data.get("frame_id", 0),
            "keypoints": keypoints
        })

    image_cache.clear()
    return {"results": results}

print("Loading ViTPose-L...")
try:
    load_model()
    print("Model ready")
except Exception as e:
    print(f"Model load error (will retry): {e}")

runpod.serverless.start({"handler": handler})
