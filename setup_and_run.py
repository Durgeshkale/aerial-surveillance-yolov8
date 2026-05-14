"""
AERIAL SURVEILLANCE — SETUP & RUN
Just run: python setup_and_run.py
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def banner(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] Failed: {cmd}")
        sys.exit(1)

# ── Step 1: Install dependencies ─────────────────────────────────────
banner("STEP 1/5 — Installing Dependencies")
packages = ["torch", "torchvision", "opencv-python", "Pillow", "flask", "numpy", "matplotlib"]
for pkg in packages:
    print(f"  Installing {pkg}...")
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"], cwd=ROOT)
print("  ✓ Done")

# ── Step 2: Generate dataset ──────────────────────────────────────────
banner("STEP 2/5 — Generating Synthetic Dataset")
if os.path.exists(os.path.join(ROOT, "dataset", "train", "images")):
    print("  Dataset already exists, skipping.")
else:
    run_cmd(f"py src/data/generate_dataset.py")

# ── Step 3: Train model ───────────────────────────────────────────────
banner("STEP 3/5 — Training Model (15 epochs)")
if os.path.exists(os.path.join(ROOT, "weights", "aerial_detector.pth")):
    print("  Weights already exist, skipping training.")
else:
    run_cmd(f"py src/model/train.py --epochs 15 --batch 8 --lr 0.001")

# ── Step 4: Evaluate ──────────────────────────────────────────────────
banner("STEP 4/5 — Evaluating Model")
run_cmd(f"py src/model/evaluate.py")

# ── Step 5: Launch dashboard ──────────────────────────────────────────
banner("STEP 5/5 — Launching Dashboard")
print("  Open your browser at: http://localhost:5000")
print("  Press Ctrl+C to stop\n")
run_cmd(f"py dashboard/app.py")
