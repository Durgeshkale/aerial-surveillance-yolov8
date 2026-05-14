"""
YOLOv8 Evaluation Script
=========================
Computes mAP50, mAP50-95, Precision, Recall, F1
per class and overall.

Run:
  python src/model/evaluate.py
  python src/model/evaluate.py --weights weights/aerial_yolov8.pt
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def evaluate(weights_path, data_yaml, device="cpu", conf=0.25, iou=0.45):
    from ultralytics import YOLO
    from src.model.detector import CLASSES

    print(f"\n{'='*65}")
    print(f"  Aerial Surveillance — Evaluation")
    print(f"  Weights : {weights_path}")
    print(f"  Data    : {data_yaml}")
    print(f"  Device  : {device}")
    print(f"{'='*65}\n")

    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights not found: {weights_path}")
        print("Run training first: python src/model/train.py --stage 1")
        sys.exit(1)

    model = YOLO(weights_path)

    metrics = model.val(
        data    = data_yaml,
        imgsz   = 640,
        conf    = conf,
        iou     = iou,
        device  = device,
        verbose = True,
    )

    # Extract results
    try:
        mp   = float(metrics.results_dict.get("metrics/precision(B)", 0))
        mr   = float(metrics.results_dict.get("metrics/recall(B)", 0))
        map50     = float(metrics.results_dict.get("metrics/mAP50(B)", 0))
        map50_95  = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0))
        f1   = 2 * mp * mr / (mp + mr + 1e-6)
    except Exception:
        mp = mr = map50 = map50_95 = f1 = 0.0

    print(f"\n{'='*65}")
    print(f"  {'METRIC':<30} {'VALUE':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Mean Precision':<30} {mp:>10.4f}")
    print(f"  {'Mean Recall':<30} {mr:>10.4f}")
    print(f"  {'Mean F1':<30} {f1:>10.4f}")
    print(f"  {'mAP @ IoU=0.50':<30} {map50:>10.4f}")
    print(f"  {'mAP @ IoU=0.50:0.95':<30} {map50_95:>10.4f}")
    print(f"{'='*65}\n")

    # Save results
    out_dir  = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_results.json")

    summary = {
        "mean_precision": round(mp, 4),
        "mean_recall":    round(mr, 4),
        "mean_f1":        round(f1, 4),
        "mAP50":          round(map50, 4),
        "mAP50_95":       round(map50_95, 4),
        "weights":        weights_path,
        "classes":        CLASSES,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {out_path}")
    return summary


if __name__ == "__main__":
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/aerial_yolov8.pt", type=str)
    p.add_argument("--data",    default="dataset/dataset.yaml",      type=str)
    p.add_argument("--conf",    default=0.25,                        type=float)
    p.add_argument("--iou",     default=0.45,                        type=float)
    args = p.parse_args()

    weights = args.weights if os.path.isabs(args.weights) \
              else os.path.join(ROOT, args.weights)
    data = args.data if os.path.isabs(args.data) \
           else os.path.join(ROOT, args.data)
    device = "0" if torch.cuda.is_available() else "cpu"

    evaluate(weights, data, device, args.conf, args.iou)
