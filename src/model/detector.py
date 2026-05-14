"""
YOLOv8 Aerial Surveillance Detector
=====================================
Wraps Ultralytics YOLOv8 for aerial object detection.

Classes:
  0 person               4 artificial_net_3d
  1 tarpaulin_gray        5 artificial_grass_mat
  2 tarpaulin_green       6 artificial_hedge
  3 artificial_net_2d     7 structure
"""

import os
import sys
import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

CLASSES = [
    "person", "tarpaulin_gray", "tarpaulin_green",
    "artificial_net_2d", "artificial_net_3d",
    "artificial_grass_mat", "artificial_hedge", "structure",
]
NUM_CLASSES = len(CLASSES)

CLASS_COLORS = {
    "person":               (255,  80,  80),
    "tarpaulin_gray":       (160, 160, 160),
    "tarpaulin_green":      ( 60, 200,  60),
    "artificial_net_2d":    (255, 180,   0),
    "artificial_net_3d":    (255, 120,   0),
    "artificial_grass_mat": ( 40, 220, 100),
    "artificial_hedge":     ( 20, 160,  60),
    "structure":            ( 80,  80, 255),
}

THREAT_LEVEL = {
    "person":               "HIGH",
    "tarpaulin_gray":       "HIGH",
    "tarpaulin_green":      "HIGH",
    "artificial_net_2d":    "MEDIUM",
    "artificial_net_3d":    "MEDIUM",
    "artificial_grass_mat": "MEDIUM",
    "artificial_hedge":     "LOW",
    "structure":            "LOW",
}

CAMOUFLAGE_CLASSES = [
    "tarpaulin_gray", "tarpaulin_green",
    "artificial_net_2d", "artificial_net_3d",
    "artificial_grass_mat", "artificial_hedge",
]

WEIGHTS_PATH = os.path.join(ROOT, "weights", "aerial_yolov8.pt")


def load_model(weights_path=None, device=None):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Run: pip install ultralytics")

    if weights_path is None:
        weights_path = WEIGHTS_PATH
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(weights_path):
        print(f"  ✓ Loaded weights: {weights_path}")
        model = YOLO(weights_path)
    else:
        print(f"  [INFO] No trained weights found. Loading YOLOv8s pretrained on COCO.")
        print(f"  [INFO] Train first: python src/model/train.py")
        model = YOLO("yolov8s.pt")

    return model


def _map_coco_to_ours(coco_name):
    coco_name = coco_name.lower()
    if any(p in coco_name for p in ["person", "people", "pedestrian", "human"]):
        return "person"
    return "structure"


def predict(model, img_bgr, conf_thresh=0.25, iou_thresh=0.45, img_size=640):
    results = model.predict(
        source=img_bgr,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=img_size,
        verbose=False,
    )

    detections = []
    if not results:
        return detections

    result = results[0]
    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf   = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        if cls_id < len(CLASSES):
            cls_name = CLASSES[cls_id]
        else:
            try:
                cls_name = _map_coco_to_ours(model.names[cls_id])
            except Exception:
                cls_name = "structure"

        detections.append({
            "class_id":   cls_id,
            "class_name": cls_name,
            "confidence": round(conf, 4),
            "threat":     THREAT_LEVEL.get(cls_name, "LOW"),
            "bbox":       [x1, y1, x2, y2],
        })

    return detections


def draw_detections(img_bgr, detections):
    img = img_bgr.copy()
    threat_bgr = {
        "HIGH":   (50,  50, 255),
        "MEDIUM": (50, 165, 255),
        "LOW":    (50, 200,  50),
    }
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        name   = det["class_name"]
        conf   = det["confidence"]
        threat = det.get("threat", "LOW")
        color  = CLASS_COLORS.get(name, (200, 200, 200))
        bgr    = (color[2], color[1], color[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
        cv2.rectangle(img, (x1, y1), (x1+6, y2), threat_bgr.get(threat, bgr), -1)

        label = f"{name.replace('_',' ')} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1-th-8), (x1+tw+6, y1), bgr, -1)
        cv2.putText(img, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def compute_threat_assessment(detections):
    if not detections:
        return {
            "level": "CLEAR", "score": 0,
            "summary": "No suspicious objects detected.",
            "high_count": 0, "medium_count": 0,
            "low_count": 0, "camouflage_count": 0,
        }

    high   = sum(1 for d in detections if d.get("threat") == "HIGH")
    medium = sum(1 for d in detections if d.get("threat") == "MEDIUM")
    low    = sum(1 for d in detections if d.get("threat") == "LOW")
    camo   = sum(1 for d in detections if d["class_name"] in CAMOUFLAGE_CLASSES)
    score  = high * 3 + medium * 2 + low

    if score >= 10 or high >= 3:
        level   = "CRITICAL"
        summary = f"Critical threat. {high} high-risk objects. {camo} camouflage installations detected."
    elif score >= 6 or high >= 1:
        level   = "HIGH"
        summary = f"High threat level. Suspected concealment activity detected."
    elif score >= 3 or medium >= 2:
        level   = "MEDIUM"
        summary = f"Moderate threat. {camo} camouflage objects identified."
    else:
        level   = "LOW"
        summary = f"Low threat. {len(detections)} objects of interest identified."

    return {
        "level": level, "score": score, "summary": summary,
        "high_count": high, "medium_count": medium,
        "low_count": low, "camouflage_count": camo,
    }
