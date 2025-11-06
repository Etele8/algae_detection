import random
import cv2
import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

def seed_everything(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class timer:
    def __init__(self, name=""):
        self.name=name; self.t0=None
    def __enter__(self): self.t0=time.time()
    def __exit__(self, *exc):
        dt=time.time()-self.t0
        print(f"[{self.name}] {dt*1000:.1f} ms")
        

def iter_stems(path) -> List[str]:
    return sorted([p.stem.replace("_og","") for p in path.glob("*_og.png")])

# letterbox sizing here

def _zscore(img_hw3: np.ndarray) -> np.ndarray:
    m = img_hw3.reshape(-1,3).mean(axis=0, keepdims=True)
    s = img_hw3.reshape(-1,3).std(axis=0, keepdims=True)
    s = np.maximum(s, 1e-6)
    out = (img_hw3.reshape(-1,3) - m) / s
    return out.reshape(img_hw3.shape).astype(np.float32)

def _neutralize_scale_bar(img_bgr_u8: np.ndarray, br_w=0.17, br_h=0.07) -> np.ndarray:
    h, w = img_bgr_u8.shape[:2]
    x0 = int(w * (1.0 - br_w)); y0 = int(h * (1.0 - br_h))
    if x0 < w and y0 < h:
        med = np.median(img_bgr_u8, axis=(0,1), keepdims=True).astype(img_bgr_u8.dtype)
        img_bgr_u8[y0:h, x0:w, :] = med
    return img_bgr_u8

def _find_pair_paths(stem: str, im_dir: Path,
                     exts: Sequence[str]=(".png",".jpg",".jpeg",".tif",".tiff")) -> Tuple[Path, Path]:
    # flexible matcher for *_og.* and *_red.* (case-insensitive)
    stem_lower = stem.lower()
    candidates = list(im_dir.glob(f"{stem}*"))
    og, red = None, None
    for p in candidates:
        if p.suffix.lower() not in exts: continue
        name = p.name.lower()
        if stem_lower in name and "_og" in name and og is None:
            og = p
        if stem_lower in name and "_red" in name and red is None:
            red = p
    if og is None or red is None:
        raise FileNotFoundError(f"Pair not found for '{stem}' in {im_dir}")
    return og, red

def letterbox_fused_from_paths(
    og_path: Path,
    red_path: Path,
    target: int = 2048,
    *,
    neutralize_scale_bar: bool = True,
    bar_w: float = 0.15,
    bar_h: float = 0.05,
    do_zscore: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a 6-ch letterboxed canvas (OG BGR + RED BGR) without touching disk.
    Returns:
      fused_6chw: float32 array in [0..1] if do_zscore=False, else z-scored
      meta: dict with orig_w, orig_h, scale, pad_l, pad_t, new_w, new_h
    """
    og  = cv2.imread(str(og_path),  cv2.IMREAD_COLOR)
    red = cv2.imread(str(red_path), cv2.IMREAD_COLOR)
    if og is None or red is None:
        raise FileNotFoundError(f"Failed to read: {og_path} / {red_path}")

    if neutralize_scale_bar:
        og  = _neutralize_scale_bar(og.copy(),  br_w=bar_w, br_h=bar_h)
        red = _neutralize_scale_bar(red.copy(), br_w=bar_w, br_h=bar_h)

    H0, W0 = og.shape[:2]
    scale = min(target / max(1, W0), target / max(1, H0))
    new_w = int(round(W0 * scale))
    new_h = int(round(H0 * scale))
    pad_l = int((target - new_w) // 2)
    pad_t = int((target - new_h) // 2)
    pad_r = target - new_w - pad_l
    pad_b = target - new_h - pad_t

    og_rs  = cv2.resize(og,  (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    red_rs = cv2.resize(red, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    og_lb  = cv2.copyMakeBorder(og_rs,  pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    red_lb = cv2.copyMakeBorder(red_rs, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)

    if do_zscore:
        og_lb  = _zscore(og_lb)
        red_lb = _zscore(red_lb)

    fused_hw6 = np.concatenate([og_lb, red_lb], axis=-1).astype(np.float32)  # (H,W,6)
    fused_6chw = fused_hw6.transpose(2,0,1).astype(np.float32)               # (6,H,W)

    meta = {
        "orig_w": float(W0), "orig_h": float(H0),
        "scale":  float(scale),
        "pad_l":  float(pad_l), "pad_t": float(pad_t),
        "new_w":  float(new_w), "new_h": float(new_h),
    }
    return fused_6chw, meta

def letterbox(
    stem: str,
    image_dir: Path,
    target: int = 2048,
    *,
    neutralize_scale_bar: bool = False,
    bar_w: float = 0.15,
    bar_h: float = 0.05,
    do_zscore: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    og_path, red_path = _find_pair_paths(stem, image_dir)
    return letterbox_fused_from_paths(
        og_path, red_path, target,
        neutralize_scale_bar=neutralize_scale_bar,
        bar_w=bar_w, bar_h=bar_h, do_zscore=do_zscore
    )


def compose_two_up(og, red):
    h, w = og.shape[:2]
    out = np.zeros((h, 2*w, 3), dtype=og.dtype)
    out[:, :w]  = og
    out[:,  w:] = red
    return out

def draw_box(panel, x1,y1,x2,y2, color, thick=2):
    cv2.rectangle(panel, (int(x1),int(y1)), (int(x2),int(y2)), color, thick)
    
def draw_vis(panel_bgr, boxes, scores, labels, palette: Dict[int,Tuple[int,int,int]], draw_scores=True):
    for i in range(labels.shape[0]):
        c=int(labels[i]); col = palette.get(c, (220,220,220))
        x1,y1,x2,y2 = [int(round(v)) for v in boxes[i].tolist()]
        cv2.rectangle(panel_bgr, (x1,y1), (x2,y2), col, 2)
        if draw_scores:
            txt = f"{c}:{float(scores[i]):.2f}"
            (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(panel_bgr, (x1, y1-th-6), (x1+tw+6, y1), col, -1)
            cv2.putText(panel_bgr, txt, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    return panel_bgr