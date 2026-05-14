"""
Aerial Surveillance Web Dashboard — Production
================================================
Run locally:  python dashboard/app.py
Production:   gunicorn --chdir dashboard app:app --bind 0.0.0.0:$PORT
"""

import os
import sys
import json
import base64
import datetime
import time

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.detector import (
    load_model, predict, draw_detections,
    compute_threat_assessment, CLASSES, CLASS_COLORS
)
from src.utils.enhance import enhance_aerial_image, compute_image_stats
from database import (
    save_detection, get_recent_detections,
    get_detection_by_id, get_stats_summary
)

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

WEIGHTS_PATH = os.path.join(ROOT, "weights", "aerial_yolov8.pt")
HF_MODEL_URL = "https://huggingface.co/Durgeshkale/aerial-surveillance-weights/resolve/main/aerial_yolov8.pt"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def download_weights():
    """Download weights from HuggingFace if not present."""
    if os.path.exists(WEIGHTS_PATH):
        print(f"  ✓ Weights found locally: {WEIGHTS_PATH}")
        return True

    print(f"  Weights not found locally. Downloading from HuggingFace...")
    print(f"  URL: {HF_MODEL_URL}")

    try:
        import urllib.request
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                print(f"  Downloading... {pct:.1f}%", end="\r")

        urllib.request.urlretrieve(HF_MODEL_URL, WEIGHTS_PATH, progress)
        print(f"\n  ✓ Weights downloaded → {WEIGHTS_PATH}")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to download weights: {e}")
        return False


# Download weights on startup
download_weights()

print(f"Loading model on {DEVICE}...")
model = load_model(WEIGHTS_PATH, DEVICE)
print("Model ready.\n")


def img_to_b64(img_bgr):
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def b64_to_img(b64_str):
    data = base64.b64decode(b64_str.split(",")[-1])
    arr  = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        img = b64_to_img(data["image"])
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400

        conf_thresh = float(data.get("conf_thresh", 0.25))
        do_enhance  = bool(data.get("enhance", True))
        do_denoise  = bool(data.get("denoise", True))
        filename    = data.get("filename", "unknown")

        t_start    = time.time()
        orig_stats = compute_image_stats(img)

        if do_enhance:
            img_proc  = enhance_aerial_image(
                img,
                apply_clahe=True,
                apply_denoise=do_denoise,
                apply_sharpen=True,
                apply_gamma=True,
            )
            enh_stats = compute_image_stats(img_proc)
        else:
            img_proc  = img.copy()
            enh_stats = orig_stats

        detections    = predict(model, img_proc, conf_thresh=conf_thresh)
        result_img    = draw_detections(img_proc, detections)
        threat        = compute_threat_assessment(detections)
        processing_ms = int((time.time() - t_start) * 1000)

        class_counts = {}
        for d in detections:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

        avg_conf = 0.0
        if detections:
            avg_conf = sum(d["confidence"] for d in detections) / len(detections)

        ts = datetime.datetime.now().strftime("%H:%M:%S")

        try:
            save_detection(
                filename=filename,
                detections=detections,
                threat=threat,
                enh_stats=enh_stats,
                conf_thresh=conf_thresh,
                processing_ms=processing_ms,
            )
        except Exception as db_err:
            print(f"[DB WARNING] {db_err}")

        return jsonify({
            "success":        True,
            "result_image":   img_to_b64(result_img),
            "enhanced_image": img_to_b64(img_proc) if do_enhance else None,
            "detections":     detections,
            "class_counts":   class_counts,
            "total_objects":  len(detections),
            "avg_confidence": round(avg_conf * 100, 1),
            "processing_ms":  processing_ms,
            "orig_stats":     orig_stats,
            "enh_stats":      enh_stats,
            "threat":         threat,
            "timestamp":      ts,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    try:
        rows    = get_recent_detections(limit=20)
        history = []
        for r in rows:
            history.append({
                "id":       r["id"],
                "time":     r["timestamp"],
                "filename": r["filename"],
                "total":    r["total_objects"],
                "threat":   r["threat_level"],
                "score":    r["threat_score"],
                "classes":  list(json.loads(r["class_counts"]).keys()),
                "avg_conf": round(float(r.get("avg_confidence", 0)) * 100, 1),
            })
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"history": [], "error": str(e)})


@app.route("/api/detection/<int:det_id>", methods=["GET"])
def get_detection(det_id):
    try:
        det = get_detection_by_id(det_id)
        if not det:
            return jsonify({"error": "Not found"}), 404
        return jsonify(det)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        return jsonify(get_stats_summary())
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/export_pdf", methods=["POST"])
def export_pdf():
    try:
        data          = request.get_json(force=True)
        detections    = data.get("detections", [])
        threat        = data.get("threat", {})
        enh_stats     = data.get("enh_stats", {})
        total         = data.get("total_objects", 0)
        avg_conf      = data.get("avg_confidence", 0)
        processing_ms = data.get("processing_ms", 0)
        timestamp     = data.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        result_b64    = data.get("result_image", "")

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image as RLImage
            import io

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=2*cm, rightMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            story  = []

            story.append(Paragraph("AERIAL SURVEILLANCE — DETECTION REPORT",
                ParagraphStyle("t", parent=styles["Title"], fontSize=18,
                               textColor=colors.HexColor("#00d4ff"), spaceAfter=6)))
            story.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
            story.append(Spacer(1, 0.5*cm))

            tc = {"CRITICAL":"#ff0000","HIGH":"#ff4444","MEDIUM":"#ffaa00",
                  "LOW":"#00cc44","CLEAR":"#00ff9d"}.get(threat.get("level","CLEAR"),"#888")
            story.append(Paragraph(
                f"THREAT: {threat.get('level','N/A')}  |  Score: {threat.get('score',0)}",
                ParagraphStyle("th", parent=styles["Normal"], fontSize=13,
                               textColor=colors.HexColor(tc), spaceBefore=4, spaceAfter=4)))
            story.append(Paragraph(threat.get("summary",""), styles["Normal"]))
            story.append(Spacer(1, 0.4*cm))

            stats_data = [
                ["Metric","Value"],
                ["Total Objects", str(total)],
                ["Avg Confidence", f"{avg_conf}%"],
                ["Processing Time", f"{processing_ms}ms"],
                ["High Threat", str(threat.get("high_count",0))],
                ["Medium Threat", str(threat.get("medium_count",0))],
                ["Camouflage", str(threat.get("camouflage_count",0))],
                ["Sharpness", str(enh_stats.get("sharpness_score","N/A"))],
                ["Brightness", str(enh_stats.get("mean_brightness","N/A"))],
            ]
            t = Table(stats_data, colWidths=[9*cm, 5*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0c1620")),
                ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#00d4ff")),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#111e2e"),colors.HexColor("#0c1620")]),
                ("TEXTCOLOR",(0,1),(-1,-1),colors.white),
                ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#1a3a5c")),
                ("PADDING",(0,0),(-1,-1),6),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.5*cm))

            story.append(Paragraph("Detection Details", styles["Heading2"]))
            if detections:
                dd = [["#","Class","Confidence","Threat","BBox"]]
                for i, d in enumerate(detections):
                    dd.append([str(i+1), d["class_name"].replace("_"," "),
                               f"{d['confidence']:.1%}", d.get("threat","LOW"), str(d["bbox"])])
                dt = Table(dd, colWidths=[1*cm,4*cm,2.5*cm,2*cm,6.5*cm])
                dt.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0c1620")),
                    ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#00d4ff")),
                    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#111e2e"),colors.HexColor("#0c1620")]),
                    ("TEXTCOLOR",(0,1),(-1,-1),colors.white),
                    ("FONTSIZE",(0,0),(-1,-1),8),
                    ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#1a3a5c")),
                    ("PADDING",(0,0),(-1,-1),4),
                ]))
                story.append(dt)
            else:
                story.append(Paragraph("No detections.", styles["Normal"]))

            if result_b64:
                story.append(Spacer(1,0.5*cm))
                story.append(Paragraph("Detection Output", styles["Heading2"]))
                try:
                    buf2 = io.BytesIO(base64.b64decode(result_b64.split(",")[-1]))
                    story.append(RLImage(buf2, width=15*cm, height=10*cm))
                except Exception:
                    pass

            doc.build(story)
            buf.seek(0)
            fname = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            return send_file(buf, mimetype="application/pdf",
                             as_attachment=True, download_name=fname)

        except ImportError:
            return jsonify({"error":"reportlab not installed","message":"pip install reportlab"}), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/model_info")
def model_info():
    return jsonify({
        "model":         "YOLOv8s — Aerial Surveillance",
        "classes":       CLASSES,
        "device":        DEVICE,
        "weights_exist": os.path.exists(WEIGHTS_PATH),
    })


if __name__ == "__main__":
    print("="*52)
    print("  Aerial Surveillance Dashboard")
    print("  → http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("="*52+"\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
