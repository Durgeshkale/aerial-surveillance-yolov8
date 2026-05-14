"""
VisDrone → YOLO Format Converter
==================================
Converts VisDrone2019-DET dataset to YOLO format
and maps VisDrone classes to our 8 surveillance classes.

VisDrone classes:
  0: ignored        5: van
  1: pedestrian     6: tricycle
  2: people         7: awning-tricycle
  3: bicycle        8: bus
  4: car            9: motor

Our classes:
  0: person             4: artificial_net_3d
  1: tarpaulin_gray     5: artificial_grass_mat
  2: tarpaulin_green    6: artificial_hedge
  3: artificial_net_2d  7: structure

Mapping:
  pedestrian(1) → person(0)
  people(2)     → person(0)
  car(4)        → structure(7)
  van(5)        → structure(7)
  bus(8)        → structure(7)
  tricycle(6)   → structure(7)
  bicycle(3)    → ignore
  motor(9)      → ignore
  ignored(0)    → ignore
  awning(7)     → ignore

Run:
  python src/data/visdrone_prep.py --visdrone-dir path/to/VisDrone2019-DET-train
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# VisDrone class ID → our class ID (-1 = ignore)
VISDRONE_MAP = {
    0: -1,   # ignored
    1:  0,   # pedestrian → person
    2:  0,   # people → person
    3: -1,   # bicycle → ignore
    4:  7,   # car → structure
    5:  7,   # van → structure
    6:  7,   # tricycle → structure
    7: -1,   # awning-tricycle → ignore
    8:  7,   # bus → structure
    9: -1,   # motor → ignore
}


def convert_annotation(ann_path, img_w, img_h):
    """Convert one VisDrone annotation file to YOLO format lines."""
    yolo_lines = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue

            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score       = int(parts[4])
            vis_cls     = int(parts[5])

            if score == 0 or w == 0 or h == 0:
                continue

            our_cls = VISDRONE_MAP.get(vis_cls, -1)
            if our_cls == -1:
                continue

            # Convert to YOLO normalized format
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))

            yolo_lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return yolo_lines


def prepare_split(visdrone_dir, out_img_dir, out_lbl_dir, split_name="train"):
    """Convert one VisDrone split to YOLO format."""
    import cv2

    img_dir = os.path.join(visdrone_dir, "images")
    ann_dir = os.path.join(visdrone_dir, "annotations")

    if not os.path.exists(img_dir):
        print(f"  [ERROR] images folder not found at {img_dir}")
        return 0

    if not os.path.exists(ann_dir):
        print(f"  [ERROR] annotations folder not found at {ann_dir}")
        return 0

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    converted = 0
    skipped   = 0

    print(f"  Converting {len(img_files)} {split_name} images...")

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        ann_name = os.path.splitext(img_file)[0] + ".txt"
        ann_path = os.path.join(ann_dir, ann_name)

        if not os.path.exists(ann_path):
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]
        yolo_lines = convert_annotation(ann_path, img_w, img_h)

        if not yolo_lines:
            skipped += 1
            continue

        # Copy image
        out_img = os.path.join(out_img_dir, img_file)
        shutil.copy2(img_path, out_img)

        # Write label
        out_lbl = os.path.join(out_lbl_dir, ann_name)
        with open(out_lbl, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    print(f"  ✓ {split_name}: {converted} converted, {skipped} skipped")
    return converted


def write_yaml(out_dir, class_names):
    """Write dataset.yaml for YOLOv8 training."""
    yaml_content = f"""# Aerial Surveillance Dataset — VisDrone converted
path: {os.path.abspath(out_dir)}
train: train/images
val:   val/images
test:  test/images

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  ✓ dataset.yaml written → {yaml_path}")
    return yaml_path


def prepare_visdrone(
    train_dir=None,
    val_dir=None,
    test_dir=None,
    out_dir=None,
):
    from src.model.detector import CLASSES

    if out_dir is None:
        out_dir = os.path.join(ROOT, "dataset")

    print(f"\n{'='*55}")
    print(f"  VisDrone → YOLO Conversion")
    print(f"  Output: {out_dir}")
    print(f"{'='*55}\n")

    total = 0

    if train_dir and os.path.exists(train_dir):
        total += prepare_split(
            train_dir,
            os.path.join(out_dir, "train", "images"),
            os.path.join(out_dir, "train", "labels"),
            "train",
        )
    else:
        print("  [WARN] train_dir not provided or not found, skipping train split")

    if val_dir and os.path.exists(val_dir):
        total += prepare_split(
            val_dir,
            os.path.join(out_dir, "val", "images"),
            os.path.join(out_dir, "val", "labels"),
            "val",
        )
    else:
        print("  [WARN] val_dir not provided or not found, skipping val split")

    if test_dir and os.path.exists(test_dir):
        total += prepare_split(
            test_dir,
            os.path.join(out_dir, "test", "images"),
            os.path.join(out_dir, "test", "labels"),
            "test",
        )

    yaml_path = write_yaml(out_dir, CLASSES)

    print(f"\n  Total images converted: {total}")
    print(f"  Ready to train with: python src/model/train.py")
    print(f"{'='*55}\n")
    return yaml_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert VisDrone to YOLO format")
    p.add_argument("--train-dir", type=str, default=None, help="Path to VisDrone2019-DET-train")
    p.add_argument("--val-dir",   type=str, default=None, help="Path to VisDrone2019-DET-val")
    p.add_argument("--test-dir",  type=str, default=None, help="Path to VisDrone2019-DET-test-dev")
    p.add_argument("--out-dir",   type=str, default=None, help="Output dataset directory")
    args = p.parse_args()

    prepare_visdrone(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        out_dir=args.out_dir,
    )
