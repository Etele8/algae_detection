#!/usr/bin/env python3
"""
Compute bbox size stats as SEEN BY THE MODEL (after letterbox to 2024×2024).
- Reads YOLO txt labels (cx, cy, w, h normalized to original image).
- Reapplies the same scale+pad as your training cache (like read_yolo_remapped).
- Ignores lines with 3 values (points).
- Optionally skips raw classes (e.g. crowd).
"""

from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict

LABEL_DIR = Path("D:/intezet/Bogi/data/r-cnn/xml_labels")
IMAGE_DIR = Path("D:/intezet/Bogi/data/r-cnn/images")
TARGET_SIZE = 2024
EXCLUDE_RAW = [7]

def quantiles(x):
    if len(x) == 0:
        return {"min": np.nan, "p5": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan}
    x = np.asarray(x, dtype=np.float32)
    return {
        "min": float(np.min(x)),
        "p5":  float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }

def main():
    txts = sorted([p for p in LABEL_DIR.glob("*.txt")])
    if not txts:
        print(f"No .txt found in {LABEL_DIR}")
        return

    short_all, long_all, area_all = [], [], []
    per_cls = defaultdict(lambda: {"short": [], "long": [], "area": [], "count": 0})

    skipped_points, skipped_excluded, skipped_bad, missing_img = 0, 0, 0, 0

    for txt in txts:
        stem = txt.stem
        if stem.endswith("_og"):
            stem = stem[:-3]

        img_path = IMAGE_DIR / f"{stem}_red.png"
        if not img_path.exists():
            missing_img += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing_img += 1
            continue
        H0, W0 = img.shape[:2]

        # scale + pad (same as cache)
        s = TARGET_SIZE / max(H0, W0)
        new_w, new_h = int(round(W0 * s)), int(round(H0 * s))
        dx = (TARGET_SIZE - new_w) / 2.0
        dy = (TARGET_SIZE - new_h) / 2.0

        for ln in txt.read_text().splitlines():
            parts = ln.strip().split()
            if not parts:
                continue
            if len(parts) == 3:   # point
                skipped_points += 1
                continue
            if len(parts) != 5:
                skipped_bad += 1
                continue

            try:
                c_raw, cx, cy, bw, bh = map(float, parts)
            except Exception:
                skipped_bad += 1
                continue

            c_raw = int(c_raw)
            if c_raw in EXCLUDE_RAW:
                skipped_excluded += 1
                continue

            # denormalize in original coords
            x1o = (cx - bw/2) * W0
            y1o = (cy - bh/2) * H0
            x2o = (cx + bw/2) * W0
            y2o = (cy + bh/2) * H0

            # map into 2024×2024
            x1 = x1o * s + dx
            x2 = x2o * s + dx
            y1 = y1o * s + dy
            y2 = y2o * s + dy

            w_px = max(0.0, x2 - x1)
            h_px = max(0.0, y2 - y1)
            if w_px <= 0 or h_px <= 0:
                continue

            s_ = min(w_px, h_px)
            l_ = max(w_px, h_px)
            a_ = w_px * h_px

            short_all.append(s_)
            long_all.append(l_)
            area_all.append(a_)
            per_cls[c_raw]["short"].append(s_)
            per_cls[c_raw]["long"].append(l_)
            per_cls[c_raw]["area"].append(a_)
            per_cls[c_raw]["count"] += 1

    print("\n=== GLOBAL box size stats (px in 2024×2024 space) ===")
    for k,v in {"short_side": short_all, "long_side": long_all, "area": area_all}.items():
        print(f"\n{k}:")
        q = quantiles(v)
        for kk,vv in q.items():
            print(f"  {kk:>3}: {vv:.2f}")

    print("\n=== PER-CLASS stats (raw YOLO ids) ===")
    for cls_id in sorted(per_cls.keys()):
        d = per_cls[cls_id]
        print(f"\n[Class {cls_id}] count={d['count']}")
        for k,v in {"short_side": d["short"], "long_side": d["long"], "area": d["area"]}.items():
            q = quantiles(v)
            print(f"  {k}:")
            for kk,vv in q.items():
                print(f"    {kk:>3}: {vv:.2f}")

if __name__ == "__main__":
    main()
