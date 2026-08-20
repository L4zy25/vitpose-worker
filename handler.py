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
        from mmpose.apis import init_model
        try:
            from mmpretrain.models import backbones
        except:
            pass

        config_path = '/workspace/vitpose/config.py'
        # Always rewrite config to ensure it's correct
        config_content = """
_base_ = ['mmpose::_base_/default_runtime.py']

codec = dict(type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

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
        decoder=codec),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(type='CocoDataset', data_root='', ann_file='', data_prefix=dict(img='')),
)
test_dataloader = val_dataloader
"""
        with open(config_path, 'w') as f:
            f.write(config_content)

        # Clear cached pyc
        for ext in ['.pyc', 'c']:
            p = config_path + ext if ext == 'c' else config_path + ext
            if os.path.exists(p):
                os.remove(p)

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
            if pose_results and len(pose_results) > 0:
                pr = pose_results[0]
                if hasattr(pr, 'pred_instances'):
                    pi = pr.pred_instances
                    kpts = pi.keypoints
                    scores = pi.keypoint_scores
                    if len(kpts) > 0:
                        if len(scores.shape) > 1:
                            scores = scores[0]
                        for i in range(17):
                            keypoints.append({
                                "x": float(kpts[0][i][0]),
                                "y": float(kpts[0][i][1]),
                                "score": float(scores[i])
                            })
        except Exception as e:
            keypoints = []
            print(f"Pose error: {e}")

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
