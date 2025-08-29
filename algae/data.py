from typing import List
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import CFG
from .cache import load_or_create_cached_fused

class CachedPicoAlgaeDataset(Dataset):
    def __init__(self, stems: List[str], is_train: bool = False, aug=None):
        self.stems = list(stems); self.is_train = is_train; self.aug = aug
    def __len__(self): return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        fused = load_or_create_cached_fused(stem)             # (6,H,W) float32 [0,1]
        img = np.transpose(fused, (1,2,0))                    # (H,W,6)

        lblp = CFG.LABEL_DIR / f"{stem}_og.txt"
        W = H = CFG.IMAGE_SIZE
        bboxes_px, class_labels = [], []
        if lblp.exists() and lblp.stat().st_size > 0:
            for ln in lblp.read_text().splitlines():
                parts = ln.strip().split()
                if len(parts) != 5: continue
                c, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw/2)*W; y1 = (cy - bh/2)*H; x2 = (cx + bw/2)*W; y2 = (cy + bh/2)*H
                if x2 > x1 and y2 > y1:
                    bboxes_px.append([x1,y1,x2,y2]); class_labels.append(int(c) + 1)

        # augment (albumentations expects normalized xyxy)
        if self.is_train and self.aug is not None:
            if len(bboxes_px) > 0:
                H0,W0 = img.shape[:2]
                bnorm, kept = [], []
                for (x1,y1,x2,y2),lab in zip(bboxes_px,class_labels):
                    xn1,xn2 = float(np.clip(x1/W0,0,1)), float(np.clip(x2/W0,0,1))
                    yn1,yn2 = float(np.clip(y1/H0,0,1)), float(np.clip(y2/H0,0,1))
                    if xn2-xn1>0 and yn2-yn1>0: bnorm.append([xn1,yn1,xn2,yn2]); kept.append(lab)
                tr = self.aug(image=img, bboxes=bnorm, class_labels=kept) if bnorm else self.aug(image=img, bboxes=[], class_labels=[])
            else:
                tr = self.aug(image=img, bboxes=[], class_labels=[])
            img = tr["image"]; bnorm = tr.get("bboxes", []); class_labels = tr.get("class_labels", [])
            H2,W2 = img.shape[:2]
            bboxes_px, cleaned = [], []
            for (xn1,yn1,xn2,yn2),lab in zip(bnorm, class_labels):
                x1,y1,x2,y2 = xn1*W2, yn1*H2, xn2*W2, yn2*H2
                if x2-x1>1 and y2-y1>1: bboxes_px.append([x1,y1,x2,y2]); cleaned.append(lab)
            class_labels = cleaned

        image = torch.from_numpy(np.transpose(img,(2,0,1))).float().contiguous()
        if len(bboxes_px) == 0:
            boxes_t = torch.zeros((0,4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(bboxes_px, dtype=torch.float32)
            labels_t = torch.tensor(class_labels, dtype=torch.int64)
        return image, {"boxes": boxes_t, "labels": labels_t}

# ---------- dataset utilities ----------
def _read_yolo_labels(stem: str):
    p = CFG.LABEL_DIR / f"{stem}_og.txt"
    if not p.exists() or p.stat().st_size == 0: return []
    out = []
    for ln in p.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) != 5: continue
        cls0 = int(float(parts[0])); out.append(cls0 + 1)
    return out

def compute_class_counts(stems: List[str]) -> dict:
    counts = defaultdict(int)
    for s in stems:
        for c in _read_yolo_labels(s): counts[c] += 1
    return dict(counts)

def stem_has_colony(stem: str) -> bool:
    return any(c in CFG.COLONY_CLASS_IDS for c in _read_yolo_labels(stem))

def build_weighted_sampler(train_stems: List[str]):
    import torch
    weights = [CFG.COLONY_OVERSAMPLE_FACTOR if stem_has_colony(s) else 1.0 for s in train_stems]
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
