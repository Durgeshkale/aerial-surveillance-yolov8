"""
YOLOv8 Training Script — Aerial Surveillance
==============================================
3-stage training pipeline:
  Stage 1: Train on VisDrone (transfer learning from COCO)
  Stage 2: Fine-tune at lower LR
  Stage 3: Final fine-tune with mixed data (optional)

Run from project root:
  python src/model/train.py
  python src/model/train.py --stage 1 --epochs 100
  python src/model/train.py --stage 2 --epochs 50
"""

import os
import sys
import json
import argparse
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def get_device():
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def stage1_train(args):
    """Stage 1: Transfer learning from COCO pretrained YOLOv8."""
    from ultralytics import YOLO

    device   = get_device()
    yaml     = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    out_dir  = os.path.join(ROOT, "outputs", "stage1")
    weights  = os.path.join(ROOT, "weights", "aerial_yolov8.pt")

    print(f"\n{'='*60}")
    print(f"  STAGE 1 — Transfer Learning from COCO")
    print(f"  Model    : YOLOv8{args.model_size}")
    print(f"  Data     : {yaml}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")

    if not os.path.exists(yaml):
        print(f"[ERROR] dataset.yaml not found at {yaml}")
        print("Run: python src/data/visdrone_prep.py first")
        sys.exit(1)

    model = YOLO(f"yolov8{args.model_size}.pt")

    results = model.train(
        data        = yaml,
        epochs      = args.epochs,
        imgsz       = 640,
        batch       = args.batch,
        lr0         = 0.01,
        lrf         = 0.01,
        momentum    = 0.937,
        weight_decay= 0.0005,
        warmup_epochs= 3,
        device      = device,
        project     = out_dir,
        name        = "train",
        exist_ok    = True,
        patience    = 20,
        save        = True,
        plots       = True,
        verbose     = True,
    )

    # Copy best weights
    best = os.path.join(out_dir, "train", "weights", "best.pt")
    if os.path.exists(best):
        os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)
        shutil.copy2(best, weights)
        print(f"\n  ✓ Best weights saved → {weights}")
    else:
        print(f"  [WARN] best.pt not found at {best}")

    save_metrics(results, "stage1")
    return weights


def stage2_finetune(args):
    """Stage 2: Fine-tune at lower learning rate."""
    from ultralytics import YOLO

    device  = get_device()
    yaml    = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    out_dir = os.path.join(ROOT, "outputs", "stage2")
    weights_in  = os.path.join(ROOT, "weights", "aerial_yolov8.pt")
    weights_out = os.path.join(ROOT, "weights", "aerial_yolov8.pt")

    print(f"\n{'='*60}")
    print(f"  STAGE 2 — Fine-tuning")
    print(f"  Base weights : {weights_in}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  LR           : 0.001 → 0.0001")
    print(f"{'='*60}\n")

    if not os.path.exists(weights_in):
        print(f"[ERROR] Run Stage 1 first: python src/model/train.py --stage 1")
        sys.exit(1)

    model = YOLO(weights_in)

    results = model.train(
        data         = yaml,
        epochs       = args.epochs,
        imgsz        = 640,
        batch        = args.batch,
        lr0          = 0.001,
        lrf          = 0.1,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 1,
        device       = device,
        project      = out_dir,
        name         = "finetune",
        exist_ok     = True,
        patience     = 15,
        save         = True,
        plots        = True,
        freeze       = 10,       # freeze first 10 layers, train head only
        verbose      = True,
    )

    best = os.path.join(out_dir, "finetune", "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy2(best, weights_out)
        print(f"\n  ✓ Fine-tuned weights saved → {weights_out}")

    save_metrics(results, "stage2")
    return weights_out


def stage3_final(args):
    """Stage 3: Final fine-tune at very low LR."""
    from ultralytics import YOLO

    device  = get_device()
    yaml    = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    out_dir = os.path.join(ROOT, "outputs", "stage3")
    weights_in  = os.path.join(ROOT, "weights", "aerial_yolov8.pt")
    weights_out = os.path.join(ROOT, "weights", "aerial_yolov8_final.pt")

    print(f"\n{'='*60}")
    print(f"  STAGE 3 — Final Fine-tuning (lowest LR)")
    print(f"  Epochs : {args.epochs}")
    print(f"  LR     : 0.0001")
    print(f"{'='*60}\n")

    if not os.path.exists(weights_in):
        print(f"[ERROR] Run Stages 1 and 2 first.")
        sys.exit(1)

    model = YOLO(weights_in)

    results = model.train(
        data         = yaml,
        epochs       = args.epochs,
        imgsz        = 640,
        batch        = args.batch,
        lr0          = 0.0001,
        lrf          = 1.0,
        momentum     = 0.9,
        weight_decay = 0.0005,
        warmup_epochs= 0,
        device       = device,
        project      = out_dir,
        name         = "final",
        exist_ok     = True,
        patience     = 10,
        save         = True,
        plots        = True,
        verbose      = True,
    )

    best = os.path.join(out_dir, "final", "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy2(best, weights_out)
        # Also update main weights
        shutil.copy2(best, os.path.join(ROOT, "weights", "aerial_yolov8.pt"))
        print(f"\n  ✓ Final weights saved → {weights_out}")

    save_metrics(results, "stage3")
    return weights_out


def save_metrics(results, stage_name):
    """Save training metrics to JSON."""
    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    try:
        metrics = {
            "stage": stage_name,
            "box_loss":  float(results.results_dict.get("train/box_loss", 0)),
            "cls_loss":  float(results.results_dict.get("train/cls_loss", 0)),
            "mAP50":     float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50_95":  float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall":    float(results.results_dict.get("metrics/recall(B)", 0)),
        }
        path = os.path.join(out_dir, f"{stage_name}_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics saved → {path}")
        print(f"  mAP@50     : {metrics['mAP50']:.4f}")
        print(f"  Precision  : {metrics['precision']:.4f}")
        print(f"  Recall     : {metrics['recall']:.4f}")
    except Exception as e:
        print(f"  [WARN] Could not save metrics: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train YOLOv8 Aerial Detector")
    p.add_argument("--stage",      type=int,   default=1,                              help="Training stage: 1, 2, or 3")
    p.add_argument("--epochs",     type=int,   default=100,                            help="Number of epochs")
    p.add_argument("--batch",      type=int,   default=16,                             help="Batch size")
    p.add_argument("--data",       type=str,   default="dataset/dataset.yaml",         help="Path to dataset.yaml")
    p.add_argument("--model-size", type=str,   default="s",                            help="YOLOv8 size: n/s/m/l/x")
    args = p.parse_args()

    print(f"\n  Running Stage {args.stage}...")

    if args.stage == 1:
        stage1_train(args)
    elif args.stage == 2:
        stage2_finetune(args)
    elif args.stage == 3:
        stage3_final(args)
    else:
        print("[ERROR] --stage must be 1, 2, or 3")
        sys.exit(1)

    print("\n  ✓ Training complete!")
    print(f"  Weights: {os.path.join(ROOT, 'weights', 'aerial_yolov8.pt')}")
    print(f"  Next: python src/model/evaluate.py")
