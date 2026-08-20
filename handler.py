import runpod
import base64
import numpy as np
import cv2
import os
import torch

vitpose_model = None
motionbert_model = None

def load_vitpose():
    global vitpose_model
    if vitpose_model is None:
        try:
            from mmpretrain.models import backbones
        except:
            pass
        from mmpose.apis import init_model
        import mmpose, glob
        mmpose_dir = os.path.dirname(mmpose.__file__)
        candidates = glob.glob(os.path.join(mmpose_dir, '**', '*ViTPose*large*256x192*.py'), recursive=True)
        if not candidates:
            candidates = glob.glob(os.path.join(mmpose_dir, '.mim', '**', '*ViTPose*large*256x192*.py'), recursive=True)
        config_path = candidates[0] if candidates else None
        if not config_path:
            raise RuntimeError("No ViTPose config found")
        checkpoint = '/workspace/vitpose/vitpose-l-coco.pth'
        vitpose_model = init_model(config_path, checkpoint, device='cuda:0')
        print("ViTPose-L loaded on GPU")
    return vitpose_model

def load_motionbert():
    global motionbert_model
    if motionbert_model is None:
        import sys
        sys.path.insert(0, '/workspace/motionbert')
        from lib.utils.tools import get_config
        from lib.utils.learning import load_backbone
        import torch.nn as nn

        args = get_config('/workspace/motionbert/configs/pose3d/MB_ft_h36m_global_lite.yaml')
        model = load_backbone(args)
        if torch.cuda.is_available():
            model = nn.DataParallel(model)
            model = model.cuda()

        ckpt_path = '/workspace/motionbert/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin'
        if not os.path.exists(ckpt_path):
            # Try the lite weights
            ckpt_path = '/workspace/motionbert/motionbert_lite.pth'

        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        key = 'model_pos' if 'model_pos' in checkpoint else 'model'
        model.load_state_dict(checkpoint[key], strict=True)
        model.eval()
        motionbert_model = model
        print(f"MotionBERT loaded from {ckpt_path}")
    return motionbert_model

def handler(event):
    input_data = event.get("input", {})

    # Route: ViTPose 2D pose estimation
    if "frames" in input_data:
        return handle_vitpose(input_data)

    # Route: MotionBERT 3D lifting
    if "keypoints_2d" in input_data:
        return handle_motionbert(input_data)

    if input_data.get("test"):
        return {"results": [], "status": "no frames"}

    return {"error": "Unknown input format"}

def handle_vitpose(input_data):
    from mmpose.apis import inference_topdown
    frames = input_data.get("frames", [])
    if not frames:
        return {"results": [], "status": "no frames"}

    pose_model = load_vitpose()
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
            print(f"Pose error: {e}")

        results.append({
            "frame_id": frame_data.get("frame_id", 0),
            "keypoints": keypoints
        })

    image_cache.clear()
    return {"results": results}

def handle_motionbert(input_data):
    """Lift 2D keypoint sequences to 3D using MotionBERT.
    Input: keypoints_2d = list of frames, each frame = list of 17 [x, y, score] joints
    Output: keypoints_3d = list of frames, each frame = list of 17 [x, y, z] joints
    """
    try:
        model = load_motionbert()
    except Exception as e:
        return {"error": f"MotionBERT load failed: {e}", "keypoints_3d": []}

    kps_2d = input_data["keypoints_2d"]  # [[{x,y,score}, ...], ...]
    clip_len = 243

    # Convert to numpy array [T, 17, 3] (x, y, confidence)
    T = len(kps_2d)
    if T == 0:
        return {"keypoints_3d": []}

    seq = np.zeros((T, 17, 3), dtype=np.float32)
    for t, frame_kps in enumerate(kps_2d):
        for j, kp in enumerate(frame_kps[:17]):
            if isinstance(kp, dict):
                seq[t, j] = [kp["x"], kp["y"], kp.get("score", 1.0)]
            else:
                seq[t, j] = kp[:3]

    # Normalize to [-1, 1] range
    # Find bbox of all keypoints
    valid = seq[:, :, 2] > 0.1
    if valid.any():
        xs = seq[:, :, 0][valid]
        ys = seq[:, :, 1][valid]
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        scale = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0
        if scale < 1:
            scale = 1
        seq[:, :, 0] = (seq[:, :, 0] - cx) / scale
        seq[:, :, 1] = (seq[:, :, 1] - cy) / scale

    # Pad or clip to clip_len
    if T < clip_len:
        pad = np.zeros((clip_len - T, 17, 3), dtype=np.float32)
        # Repeat last frame
        for i in range(clip_len - T):
            pad[i] = seq[-1]
        seq_padded = np.concatenate([seq, pad], axis=0)
    else:
        seq_padded = seq[:clip_len]

    # Run model
    batch = torch.from_numpy(seq_padded).unsqueeze(0).cuda()  # [1, 243, 17, 3]

    with torch.no_grad():
        pred_3d = model(batch)  # [1, 243, 17, 3]

    pred_3d = pred_3d.cpu().numpy()[0]  # [243, 17, 3]

    # Only return the original T frames (not padding)
    pred_3d = pred_3d[:T]

    # Convert to list format
    keypoints_3d = []
    for t in range(T):
        frame_joints = []
        for j in range(17):
            frame_joints.append({
                "x": float(pred_3d[t, j, 0]),
                "y": float(pred_3d[t, j, 1]),
                "z": float(pred_3d[t, j, 2])
            })
        keypoints_3d.append(frame_joints)

    return {"keypoints_3d": keypoints_3d}

print("Loading models...")
try:
    load_vitpose()
    print("ViTPose ready")
except Exception as e:
    print(f"ViTPose load error (will retry): {e}")

try:
    load_motionbert()
    print("MotionBERT ready")
except Exception as e:
    print(f"MotionBERT load error (will retry): {e}")

runpod.serverless.start({"handler": handler})
