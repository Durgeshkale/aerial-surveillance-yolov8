# 🛰️ Aerial Surveillance Using Satellite Images — ML System

> **BTech CSE Final Year Project**  
> Object Detection in Aerial Imagery using Custom CNN (PyTorch)

---

## 📁 Project Structure

```
aerial-surveillance/
├── src/
│   ├── data/
│   │   └── generate_dataset.py     ← Synthetic aerial image generator
│   ├── model/
│   │   ├── detector.py             ← AerialDetector CNN + inference
│   │   ├── train.py                ← Training loop
│   │   └── evaluate.py             ← Precision / Recall / F1 / mAP
│   └── utils/
│       └── enhance.py              ← CLAHE / denoise / sharpen pipeline
├── dashboard/
│   ├── app.py                      ← Flask web server
│   └── templates/index.html        ← Web dashboard UI
├── notebooks/
│   └── Aerial_Surveillance_Pipeline.ipynb  ← Google Colab notebook
├── dataset/                        ← Auto-generated (after step 2)
├── weights/                        ← Model weights saved here
├── outputs/                        ← Training plots, eval results
└── requirements.txt
```

---

## 🚀 Quick Start (Local)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate synthetic dataset
```bash
python src/data/generate_dataset.py
# Creates 300 train + 80 val + 40 test labeled aerial images
```

### Step 3 — Train the model
```bash
python src/model/train.py --epochs 30 --batch 8 --lr 0.001
# Saves best weights to: weights/aerial_detector.pth
```

### Step 4 — Evaluate
```bash
python src/model/evaluate.py
# Outputs: Precision, Recall, F1 per class + mAP
```

### Step 5 — Run the dashboard
```bash
python dashboard/app.py
# Open: http://localhost:5000
```

---

## ☁️ Google Colab (Recommended for GPU)

1. Upload the entire `aerial-surveillance/` folder to Google Drive
2. Open `notebooks/Aerial_Surveillance_Pipeline.ipynb` in Colab
3. Runtime → Change runtime type → **GPU**
4. Run cells 1–11 in order

---

## 🏷️ Detectable Object Classes

| ID | Class | Description |
|----|-------|-------------|
| 0  | person | Human entities (top-down view) |
| 1  | tarpaulin_gray | Gray camouflage tarpaulin |
| 2  | tarpaulin_green | Green camouflage tarpaulin |
| 3  | artificial_net_2d | Flat 2D camouflage net |
| 4  | artificial_net_3d | Raised 3D camouflage net |
| 5  | artificial_grass_mat | Artificial grass mat |
| 6  | artificial_hedge | Artificial hedge/shrub line |
| 7  | structure | Buildings / artificial structures |

---

## 🧠 Model Architecture

```
AerialDetector
  Input:  (B, 3, 416, 416)
  ↓ Backbone: 5-stage ConvBNReLU + ResBlocks → feature map (B, 512, 13, 13)
  ↓ Head: 1×1 Conv → 3×3 Conv → output (B, 5+8, 13, 13)
  Output: (B, 13, 13, 13)  [tx, ty, tw, th, obj_conf, class×8]
```

- **Parameters**: ~3.2M
- **Input**: 416×416 RGB
- **Grid**: 13×13 cells
- **Loss**: Custom YOLO-style detection loss

---

## 🔧 Image Enhancement Pipeline

1. **Non-local Means Denoising** — removes sensor noise
2. **CLAHE** — adaptive contrast enhancement per tile
3. **Gamma Correction** — brightness normalization
4. **Unsharp Masking** — edge sharpening for detail recovery

---

## 📊 Training Tips

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| LR Scheduler | Cosine Annealing |
| Batch Size | 8 (CPU) / 16 (GPU) |
| Epochs | 30–50 |
| Augmentation | Horizontal flip + Brightness jitter |

---

## 📝 Citation / Reference

If using this project in research:
```
Aerial Surveillance Using Satellite Images Using Machine Learning
BTech CSE Final Year Project
Government College of Engineering Nagpur
```
