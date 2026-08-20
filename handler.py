import runpod
import base64
import numpy as np
import cv2
import os

# Register mmpretrain models in mmpose registry
try:
    import mmpretrain
    print("mmpretrain imported OK")
except Exception as e:
    print(f"mmpretrain import warning: {e}")

model = None

def load_model():
    global model
    if model is None:
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
            bboxes = [bbox]
        else:
            h, w = img.shape[:2]
            bboxes = [[0, 0, w, h]]

        try:
            pose_results = inference_topdown(pose_model, img, bboxes)
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
