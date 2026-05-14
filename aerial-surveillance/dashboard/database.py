"""
Database Module — Aerial Surveillance
Auto-creates SQLite database on first run.
"""

import os
import sqlite3
import json
import datetime

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "surveillance.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            date             TEXT NOT NULL,
            filename         TEXT,
            total_objects    INTEGER DEFAULT 0,
            threat_level     TEXT DEFAULT 'CLEAR',
            threat_score     INTEGER DEFAULT 0,
            threat_summary   TEXT DEFAULT '',
            high_count       INTEGER DEFAULT 0,
            medium_count     INTEGER DEFAULT 0,
            low_count        INTEGER DEFAULT 0,
            camouflage_count INTEGER DEFAULT 0,
            avg_confidence   REAL DEFAULT 0,
            class_counts     TEXT DEFAULT '{}',
            image_stats      TEXT DEFAULT '{}',
            conf_thresh      REAL DEFAULT 0.25,
            processing_ms    INTEGER DEFAULT 0
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS detection_objects (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            class_name   TEXT,
            confidence   REAL,
            threat       TEXT,
            bbox_x1      INTEGER,
            bbox_y1      INTEGER,
            bbox_x2      INTEGER,
            bbox_y2      INTEGER,
            FOREIGN KEY (detection_id) REFERENCES detections(id)
        )
    """)
    conn.commit()
    conn.close()


def save_detection(filename, detections, threat, enh_stats,
                   conf_thresh=0.25, processing_ms=0):
    conn = get_connection()
    c    = conn.cursor()
    now  = datetime.datetime.now()
    ts   = now.strftime("%Y-%m-%d %H:%M:%S")
    date = now.strftime("%Y-%m-%d")

    class_counts = {}
    for d in detections:
        class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

    avg_conf = 0.0
    if detections:
        avg_conf = sum(d["confidence"] for d in detections) / len(detections)

    c.execute("""
        INSERT INTO detections
        (timestamp, date, filename, total_objects, threat_level, threat_score,
         threat_summary, high_count, medium_count, low_count, camouflage_count,
         avg_confidence, class_counts, image_stats, conf_thresh, processing_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        ts, date, filename, len(detections),
        threat.get("level","CLEAR"), threat.get("score",0),
        threat.get("summary",""),
        threat.get("high_count",0), threat.get("medium_count",0),
        threat.get("low_count",0), threat.get("camouflage_count",0),
        round(avg_conf, 4), json.dumps(class_counts),
        json.dumps(enh_stats), conf_thresh, processing_ms,
    ))

    det_id = c.lastrowid
    for d in detections:
        bbox = d.get("bbox", [0,0,0,0])
        c.execute("""
            INSERT INTO detection_objects
            (detection_id, class_name, confidence, threat,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (?,?,?,?,?,?,?,?)
        """, (det_id, d.get("class_name","unknown"), d.get("confidence",0),
              d.get("threat","LOW"), bbox[0], bbox[1], bbox[2], bbox[3]))

    conn.commit()
    conn.close()
    return det_id


def get_recent_detections(limit=20):
    conn = get_connection()
    c    = conn.cursor()
    c.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_detection_by_id(det_id):
    conn = get_connection()
    c    = conn.cursor()
    c.execute("SELECT * FROM detections WHERE id=?", (det_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    det = dict(row)
    c.execute("SELECT * FROM detection_objects WHERE detection_id=? ORDER BY confidence DESC", (det_id,))
    objects = []
    for obj in c.fetchall():
        o = dict(obj)
        objects.append({
            "class_name": o["class_name"],
            "confidence": round(o["confidence"] * 100, 1),
            "threat":     o["threat"],
            "bbox":       [o["bbox_x1"], o["bbox_y1"], o["bbox_x2"], o["bbox_y2"]],
        })
    det["objects"]      = objects
    det["class_counts"] = json.loads(det.get("class_counts", "{}"))
    det["image_stats"]  = json.loads(det.get("image_stats", "{}"))
    conn.close()
    return det


def get_stats_summary():
    conn = get_connection()
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) as total FROM detections")
    total_scans = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM detection_objects")
    total_objects = c.fetchone()["total"]
    c.execute("SELECT threat_level, COUNT(*) as cnt FROM detections GROUP BY threat_level")
    threat_dist = {r["threat_level"]: r["cnt"] for r in c.fetchall()}
    c.execute("SELECT AVG(confidence) as avg FROM detection_objects")
    row = c.fetchone()
    avg_conf = round(float(row["avg"]) * 100, 1) if row["avg"] else 0
    conn.close()
    return {
        "total_scans": total_scans, "total_objects": total_objects,
        "threat_dist": threat_dist, "avg_confidence": avg_conf,
    }


init_db()
