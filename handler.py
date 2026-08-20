import runpod
import base64
import numpy as np
import cv2
import os
import torch

vitpose_model = None
motionbert_model = None
device = None

def load_models():
    global vitpose_model, motionbert_model, device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if vitpose_model is None:
        from mmpose.apis import init_model
        config_path = '/workspace/vitpose/config.py'
        if not os.path.exists(config_path):
            config_content = """
_base_ = ['mmpose::_base_/default_runtime.py']
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='large',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.55,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=''),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=1024,
        out_channels=17,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2)),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))
"""
            with open(config_path, 'w') as f:
                f.write(config_content)
        checkpoint = '/workspace/vitpose/vitpose-l-coco.pth'
        vitpose_model = init_model(config_path, checkpoint, device='cuda:0')
        print("ViTPose-L loaded")
    
    if motionbert_model is None:
        try:
            import sys
            sys.path.insert(0, '/workspace/motionbert')
            from lib.model.DSTformer import DSTformer
            mb_checkpoint = torch.load('/workspace/motionbert/motionbert_lite.pth', map_location=device, weights_only=False)
            mb_model = DSTformer(
                dim_in=2, dim_out=3, dim_feat=256,
                dim_rep=512, depth=5, num_heads=8,
                mlp_ratio=4, maxlen=243
            )
            if 'model' in mb_checkpoint:
                mb_model.load_state_dict(mb_checkpoint['model'], strict=False)
            elif 'model_pos' in mb_checkpoint:
                mb_model.load_state_dict(mb_checkpoint['model_pos'], strict=False)
            else:
                mb_model.load_state_dict(mb_checkpoint, strict=False)
            motionbert_model = mb_model.to(device).eval()
            print("MotionBERT loaded")
        except Exception as e:
            print(f"MotionBERT load failed: {e}")
            motionbert_model = None

def run_vitpose(img):
    from mmpose.apis import inference_topdown
    h, w = img.shape[:2]
    bbox = [[0, 0, w, h]]
    pose_results = inference_topdown(vitpose_model, img, bbox)
    keypoints = []
    if pose_results and len(pose_results) > 0:
        kpts = pose_results[0].pred_instances.keypoints[0]
        scores = pose_results[0].pred_instances.keypoint_scores[0]
        for i in range(17):
            keypoints.append({
                "x": float(kpts[i][0]),
                "y": float(kpts[i][1]),
                "score": float(scores[i])
            })
    return keypoints

def lift_to_3d(keypoints_sequence):
    if motionbert_model is None or len(keypoints_sequence) == 0:
        return []
    
    # COCO 17 to Human3.6M mapping
    coco_to_h36m = [0, 12, 14, 16, 11, 13, 15, 0, 0, 0, 5, 7, 9, 6, 8, 10, 0]
    
    seq_len = len(keypoints_sequence)
    target_len = 243
    
    input_2d = np.zeros((target_len, 17, 2), dtype=np.float32)
    for t in range(min(seq_len, target_len)):
        frame_kps = keypoints_sequence[t]
        for j in range(17):
            src_idx = coco_to_h36m[j]
            if src_idx < len(frame_kps):
                input_2d[t, j, 0] = frame_kps[src_idx]["x"]
                input_2d[t, j, 1] = frame_kps[src_idx]["y"]
    
    if seq_len < target_len:
        for t in range(seq_len, target_len):
            input_2d[t] = input_2d[min(seq_len - 1, t)]
    
    w_max = np.abs(input_2d[:, :, 0]).max()
    h_max = np.abs(input_2d[:, :, 1]).max()
    if w_max > 0:
        input_2d[:, :, 0] = input_2d[:, :, 0] / w_max * 2 - 1
    if h_max > 0:
        input_2d[:, :, 1] = input_2d[:, :, 1] / h_max * 2 - 1
    
    input_tensor = torch.FloatTensor(input_2d).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_3d = motionbert_model(input_tensor)
    
    output_np = output_3d[0, :seq_len].cpu().numpy()
    
    results_3d = []
    for t in range(min(seq_len, len(output_np))):
        frame_3d = []
        for j in range(17):
            frame_3d.append({
                "x": float(output_np[t, j, 0]),
                "y": float(output_np[t, j, 1]),
                "z": float(output_np[t, j, 2])
            })
        results_3d.append(frame_3d)
    
    return results_3d

def handler(event):
    input_data = event.get("input", {})
    frames = input_data.get("frames", [])
    use_3d = input_data.get("use_3d", False)
    
    if not frames:
        return {"results": [], "status": "no frames"}
    
    load_models()
    
    results_2d = []
    for frame_data in frames:
        img_bytes = base64.b64decode(frame_data["image"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        try:
            keypoints = run_vitpose(img)
        except Exception as e:
            keypoints = []
            print(f"ViTPose error: {e}")
        
        results_2d.append({
            "frame_id": frame_data.get("frame_id", 0),
            "keypoints": keypoints
        })
    
    if use_3d and len(results_2d) > 1:
        try:
            all_kps = [r["keypoints"] for r in results_2d]
            results_3d = lift_to_3d(all_kps)
            for i, r in enumerate(results_2d):
                if i < len(results_3d):
                    r["keypoints_3d"] = results_3d[i]
        except Exception as e:
            print(f"3D lifting error: {e}")
    
    return {"results": results_2d}

print("Loading models...")
try:
    load_models()
    print("All models ready")
except Exception as e:
    print(f"Model load error (will retry): {e}")

runpod.serverless.start({"handler": handler})
