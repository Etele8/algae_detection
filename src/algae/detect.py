"""Stage 2 - Detection: find every organism in each frame (unsupervised).

We use a class-agnostic "segment everything" model (MobileSAM / FastSAM via
ultralytics) that needs no labels. Each returned mask is filtered to a
plausible organism size, the scale-bar corner is ignored, and a padded RGB
crop + binary mask is written to disk for the embedding stage.

A dependency-free ``classical`` OpenCV backend is provided as a fallback so the
pipeline can run even before the deep models are available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config, resolve_device
from .manifest import usable_images


@dataclass
class Detection:
    """A single detected organism within one image."""

    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive-exclusive)
    mask: np.ndarray                 # bool array, full image height x width
    score: float = 1.0


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class _UltralyticsSAM:
    """MobileSAM / FastSAM 'segment everything' via the ultralytics package."""

    def __init__(self, cfg: Config):
        d = cfg["detect"]
        self.backend = d["backend"]
        self.device = resolve_device(d["device"])
        self.imgsz = int(d["imgsz"])

        # Sensible default checkpoint per backend (ultralytics auto-downloads).
        defaults = {
            "mobile_sam": "mobile_sam.pt",   # fast, CPU-friendly
            "sam2": "sam2.1_b.pt",           # SAM 2.1; set sam2.1_l.pt for best quality on GPU
            "sam": "sam_b.pt",               # original SAM ViT-B
            "fastsam": "FastSAM-s.pt",
        }
        weights_name = d.get("weights") or defaults[self.backend]
        # Prefer a copy already cached in models/, else let ultralytics fetch it.
        weights_path = cfg.models_dir / weights_name
        weights_arg = str(weights_path) if weights_path.exists() else weights_name

        if self.backend == "fastsam":
            from ultralytics import FastSAM
            self.model = FastSAM(weights_arg)  # already segments everything
        else:  # mobile_sam / sam / sam2 all use the ultralytics SAM class
            from ultralytics import SAM
            self.model = SAM(weights_arg)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        results = self.model(
            image_bgr[:, :, ::-1],  # ultralytics expects RGB
            device=self.device,
            imgsz=self.imgsz,
            retina_masks=True,
            verbose=False,
        )
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.masks is None:
            return out
        masks = r.masks.data.cpu().numpy().astype(bool)  # (N, H, W)
        confs = (
            r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None
            else np.ones(len(masks))
        )
        for m, c in zip(masks, confs):
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            ys, xs = np.where(m)
            if xs.size == 0:
                continue
            out.append(Detection((int(xs.min()), int(ys.min()),
                                   int(xs.max()) + 1, int(ys.max()) + 1), m, float(c)))
        return out


class _ClassicalDetector:
    """Background-subtraction fallback: organisms differ from the uniform field.

    The slide background is a near-uniform light gray; organisms are darker,
    more saturated and/or greener. We score each pixel by deviation from a
    large-kernel background estimate, threshold, clean up with morphology and
    take connected components. Fast, CPU-only, no weights.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)

        # Large-kernel median ~ background illumination field.
        k = max(31, (min(h, w) // 8) | 1)
        bg = cv2.medianBlur(gray, k)
        darkness = cv2.GaussianBlur((bg.astype(np.float32) - gray.astype(np.float32)), (0, 0), 2)

        score = cv2.normalize(darkness, None, 0, 255, cv2.NORM_MINMAX)
        score = np.maximum(score, cv2.normalize(sat, None, 0, 255, cv2.NORM_MINMAX) * 0.6)
        score = score.astype(np.uint8)

        _, fg = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        out: list[Detection] = []
        for i in range(1, n):  # skip background label 0
            x, y, bw, bh, area = stats[i]
            mask = labels == i
            out.append(Detection((int(x), int(y), int(x + bw), int(y + bh)), mask, 1.0))
        return out


def load_detector(cfg: Config):
    backend = cfg["detect"]["backend"]
    if backend in ("mobile_sam", "sam2", "sam", "fastsam"):
        return _UltralyticsSAM(cfg)
    if backend == "classical":
        return _ClassicalDetector(cfg)
    raise ValueError(f"Unknown detect.backend: {backend!r}")


# --------------------------------------------------------------------------- #
# Filtering + crop export
# --------------------------------------------------------------------------- #
def _in_scalebar_corner(det: Detection, w: int, h: int, frac_w: float, frac_h: float) -> bool:
    """True if the detection center sits in the bottom-right scale-bar band."""
    cx = (det.bbox[0] + det.bbox[2]) / 2
    cy = (det.bbox[1] + det.bbox[3]) / 2
    return cx >= w * (1 - frac_w) and cy >= h * (1 - frac_h)


def _save_crop(image_bgr: np.ndarray, det: Detection, pad: int,
               raw_path: Path, mask_path: Path) -> None:
    h, w = image_bgr.shape[:2]
    x0 = max(0, det.bbox[0] - pad)
    y0 = max(0, det.bbox[1] - pad)
    x1 = min(w, det.bbox[2] + pad)
    y1 = min(h, det.bbox[3] + pad)
    crop = image_bgr[y0:y1, x0:x1]
    mask_crop = (det.mask[y0:y1, x0:x1].astype(np.uint8)) * 255
    cv2.imwrite(str(raw_path), crop)
    cv2.imwrite(str(mask_path), mask_crop)


def run_detection(cfg: Config) -> Path:
    """Detect organisms in every usable frame; write crops + an object table."""
    d = cfg["detect"]
    images = usable_images(cfg)
    if d.get("max_images"):
        images = images.head(int(d["max_images"]))
    detector = load_detector(cfg)

    raw_dir = cfg.outputs_dir / "crops" / "raw"
    mask_dir = cfg.outputs_dir / "crops" / "mask"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    min_area_frac = float(d["min_area_frac"])
    max_area_frac = float(d["max_area_frac"])
    min_crop_px = int(d["min_crop_px"])
    pad = int(d["crop_padding"])
    corner_w, corner_h = d["ignore_corner_frac"]

    rows = []
    for _, row in tqdm(list(images.iterrows()), desc=f"detect[{d['backend']}]"):
        image_bgr = cv2.imread(row["path"])
        if image_bgr is None:
            continue
        h, w = image_bgr.shape[:2]
        img_area = h * w
        dets = detector.detect(image_bgr)

        kept = 0
        for det in dets:
            area = int(det.mask.sum())
            frac = area / img_area
            bw = det.bbox[2] - det.bbox[0]
            bh = det.bbox[3] - det.bbox[1]
            if frac < min_area_frac or frac > max_area_frac:
                continue
            if bw < min_crop_px or bh < min_crop_px:
                continue
            if _in_scalebar_corner(det, w, h, corner_w, corner_h):
                continue
            object_id = f"{row['image_id']}__{kept:03d}"
            raw_path = raw_dir / f"{object_id}.png"
            mask_path = mask_dir / f"{object_id}.png"
            _save_crop(image_bgr, det, pad, raw_path, mask_path)
            rows.append(
                {
                    "object_id": object_id,
                    "image_id": row["image_id"],
                    "x0": det.bbox[0], "y0": det.bbox[1],
                    "x1": det.bbox[2], "y1": det.bbox[3],
                    "area_px": area, "area_frac": frac,
                    "score": det.score,
                    "raw_path": str(raw_path),
                    "mask_path": str(mask_path),
                }
            )
            kept += 1

    df = pd.DataFrame(rows)
    out = cfg.outputs_dir / "objects.csv"
    df.to_csv(out, index=False)
    print(f"Detected {len(df)} organisms across {images.shape[0]} frames -> {out}")
    return out
