"""Stage 5b - Visualize: overlay detections back onto the original frames.

This is the QA view: for every frame, each detected organism's mask outline and
bounding box is drawn on the full image, coloured by its morphotype cluster (or
predicted label) and tagged with a short id. It answers the two questions that
matter for accuracy: *did detection find the organisms and nothing spurious?*
and *are look-alikes landing in the same group?*

Overlays are reconstructed from saved artifacts (objects.csv + per-object mask
PNGs), so this stage is cheap and never re-runs the segmenter.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import Config

# A fixed, high-contrast palette indexed by cluster id (BGR).
_PALETTE = [
    (66, 135, 245), (245, 145, 66), (66, 245, 111), (245, 66, 66),
    (200, 66, 245), (66, 245, 230), (245, 66, 167), (150, 245, 66),
    (245, 209, 66), (66, 170, 245), (180, 120, 70), (120, 200, 200),
]
_NOISE_COLOR = (140, 140, 140)  # grey for the HDBSCAN noise bucket (-1)


def _color_for(cluster: int) -> tuple[int, int, int]:
    if cluster < 0:
        return _NOISE_COLOR
    return _PALETTE[cluster % len(_PALETTE)]


def run_visualize(cfg: Config) -> Path:
    objects = pd.read_csv(cfg.outputs_dir / "objects.csv")
    if objects.empty:
        raise RuntimeError("No objects to visualize - run detection first.")

    # Attach a label per object: prefer predicted label, else cluster id, else none.
    label_col, color_key = None, None
    clusters_path = cfg.outputs_dir / "clusters.csv"
    preds_path = cfg.outputs_dir / "predictions.csv"
    if preds_path.exists():
        preds = pd.read_csv(preds_path)[["object_id", "label"]]
        objects = objects.merge(preds, on="object_id", how="left")
        label_col = "label"
    if clusters_path.exists():
        cl = pd.read_csv(clusters_path)[["object_id", "cluster"]]
        objects = objects.merge(cl, on="object_id", how="left")
        color_key = "cluster"

    pad = int(cfg["detect"]["crop_padding"])
    out_dir = cfg.outputs_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map non-numeric labels to stable colour indices for consistent colouring.
    label_to_idx: dict = {}

    n = 0
    for image_id, grp in objects.groupby("image_id"):
        src = grp["raw_path"].iloc[0]
        # Recover the original frame path from the manifest (crops live elsewhere).
        frame_path = _frame_path(cfg, image_id)
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        overlay = img.copy()

        for _, o in grp.iterrows():
            if color_key == "cluster" and not pd.isna(o.get("cluster")):
                color = _color_for(int(o["cluster"]))
            elif label_col and isinstance(o.get(label_col), str):
                idx = label_to_idx.setdefault(o[label_col], len(label_to_idx))
                color = _color_for(idx)
            else:
                color = (66, 135, 245)

            # Place the saved (clamped, padded) mask back at its origin.
            mask = cv2.imread(o["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                ox = max(0, int(o["x0"]) - pad)
                oy = max(0, int(o["y0"]) - pad)
                mh, mw = mask.shape[:2]
                mh, mw = min(mh, h - oy), min(mw, w - ox)
                region = (mask[:mh, :mw] > 127)
                sub = overlay[oy:oy + mh, ox:ox + mw]
                sub[region] = (0.45 * np.array(color) + 0.55 * sub[region]).astype(np.uint8)
                cnts, _ = cv2.findContours(region.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay[oy:oy + mh, ox:ox + mw], cnts, -1, color, 2)

            tag = ""
            if label_col and isinstance(o.get(label_col), str):
                tag = o[label_col]
            elif color_key == "cluster" and not pd.isna(o.get("cluster")):
                tag = f"c{int(o['cluster'])}"
            if tag:
                cv2.putText(overlay, tag, (int(o["x0"]), max(0, int(o["y0"]) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        blended = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
        cv2.imwrite(str(out_dir / f"{image_id}.png"), blended)
        n += 1

    print(f"Wrote {n} detection overlays -> {out_dir}")
    return out_dir


def _frame_path(cfg: Config, image_id: str) -> Path:
    """Resolve the original full frame for an image id via the manifest."""
    man = pd.read_csv(cfg.outputs_dir / "manifest.csv")
    row = man[man["image_id"] == image_id]
    if not row.empty:
        return Path(row.iloc[0]["path"])
    return cfg.data_dir / f"{image_id}.png"
