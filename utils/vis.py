import cv2
import numpy as np
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
class Config:
    IMAGE_PATH = Path("D:/intezet/Bogi/data/r-cnn/images/Image_6588_og.png")
    LABEL_PATH = Path("D:/intezet/Bogi/data/r-cnn/labels/Image_6588_og.txt")
    OUT_PATH   = Path("D:/intezet/Bogi/data/label_testing/Image_6588_og_out.png")
    CLASS_NAMES = ["EUK", "FC", "FE", "EUK_col", "FC_col", "FE_col"]
    THICKNESS = 1
    MIN_BOX = 2

PALETTE = [
    (255, 0, 0), (0, 255, 255), (0, 0, 255), (0, 128, 255),
    (120, 120, 0), (0, 120, 24), (128, 128, 128), (255, 0, 255),
    (0, 255, 0), (255, 255, 0)
]

def yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w/2) * W
    y1 = (cy - h/2) * H
    x2 = (cx + w/2) * W
    y2 = (cy + h/2) * H
    x1, x2 = max(0, min(W-1, x1)), max(0, min(W-1, x2))
    y1, y2 = max(0, min(H-1, y1)), max(0, min(H-1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def visualize(image_path, label_path, out_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    H, W = img.shape[:2]

    labels = []
    if label_path.exists() and label_path.stat().st_size > 0:
        for ln in label_path.read_text().splitlines():
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            labels.append((int(cls), cx, cy, w, h))
    else:
        print(f"[warn] Label file missing/empty: {label_path}")

    drawn = 0
    for cls, cx, cy, w, h in labels:
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
        if (x2 - x1) < Config.MIN_BOX or (y2 - y1) < Config.MIN_BOX:
            continue
        color = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, Config.THICKNESS)
        label_text = (
            Config.CLASS_NAMES[cls] if 0 <= cls < len(Config.CLASS_NAMES) else str(cls)
        )
        cv2.putText(img, label_text, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        drawn += 1

    cv2.imwrite(str(out_path), img)
    print(f"[ok] Drew {drawn} boxes -> {out_path}")

if __name__ == "__main__":
    visualize(Config.IMAGE_PATH, Config.LABEL_PATH, Config.OUT_PATH)
