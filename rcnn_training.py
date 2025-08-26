import random
import types
import math
import contextlib
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List
from collections import defaultdict

from sklearn.metrics import precision_recall_curve, average_precision_score

import cv2
import numpy as np

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import clip_boxes_to_image

torch.backends.cudnn.benchmark = True

# =========================
# Config
# =========================
@dataclass
class Config:
    IMAGE_DIR: Path = Path("/workspace/data/images")   # expects <stem>_og.png and <stem>_red.png
    LABEL_DIR: Path = Path("/workspace/data/labels")   # expects <stem>.txt (YOLO: cls cx cy w h)
    CACHE_DIR: Path = Path("/workspace/data/cache_fused")

    IMAGE_SIZE: int = 2080
    NUM_CLASSES: int = 7
    BACKBONE: str = "resnet101"

    BATCH_SIZE: int = 4
    WORKERS: int = 2
    EPOCHS: int = 20
    LR: float = 1e-4
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VAL_FRACTION: float = 0.2
    SEED: int = 42

    USE_ZSCORE: bool = False
    SAVE_BEST_TO: Path = Path("best_frcnn_6ch.pth")

    # tensorboard
    LOGDIR = "runs/algae6"
    LOG_EVERY = 50          # batches
    PR_IOU = 0.5            # IoU to match detections to GT
    PR_SCORE_THRESH = 0.00  # keep all preds for PR curves
    CLASS_NAMES = ["EUK", "FC", "FE", "EUK colony", "FC colony", "FE colony"]
    
    CEN_F1_TOL_PX: int = 8          # distance tolerance in pixels
    CEN_F1_SCORE_THRESH: float = 0.30  # score threshold for counting F1
    
CFG = Config()


# ====================
# augmentations
# ====================


class ChannelDimming6(ImageOnlyTransform):
    def __init__(self, p=0.3, min_scale=0.3, max_scale=0.7):
        super().__init__(p=p)
        self.min_scale, self.max_scale = min_scale, max_scale
    def apply(self, img, **params):
        # img: H,W,6
        if np.random.rand() < 0.5:
            # dim OG (0..2)
            s = np.random.uniform(self.min_scale, self.max_scale)
            img[..., 0:3] *= s
        else:
            # dim RED (3..5)
            s = np.random.uniform(self.min_scale, self.max_scale)
            img[..., 3:6] *= s
        return img
    
    
class PoissonNoise(ImageOnlyTransform):
    def __init__(self, lam_scale=0.05, p=0.2):
        super().__init__(p=p)
        self.lam_scale = lam_scale
    def apply(self, img, **params):
        # assume [0,1], add Poisson-like shot noise per channel
        lam = np.clip(img * self.lam_scale, 0, 1)
        noise = np.random.poisson(lam * 255.0) / 255.0
        out = img + noise - lam
        return np.clip(out, 0.0, 1.0)


# USE only for EUK classes, for others it decreases the performance
class RedCLAHE(ImageOnlyTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, img, **params):
        # img: HxWx6 float32 [0,1]
        red = (img[..., 3:6] * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        for ch in range(3):
            red[..., ch] = clahe.apply(red[..., ch])
        img[..., 3:6] = red.astype(np.float32) / 255.0
        return img


def build_train_aug():
    return A.Compose([
        # --- geometric (bbox-aware) ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.25),

        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.30),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 1.0), p=0.20),
        A.MotionBlur(blur_limit=(5, 9), p=0.15),

        RedCLAHE(p=0.50),
        ChannelDimming6(p=0.30),
        PoissonNoise(p=0.20),
    ],
    bbox_params=A.BboxParams(
        format="albumentations",
        label_fields=["class_labels"],
        min_visibility=0.30
    ))




# =========================
# tesnorboard helpers
# =========================

@torch.no_grad()
def log_roi_embeddings(
    model,
    dataloader,
    writer: SummaryWriter,
    device,
    *,
    max_rois: int = 512,
    score_thresh: float = 0.2,
    tag: str = "embeddings/roi_head",
    global_step: int = 0,
):
    """
    Logs embeddings from the box head’s pooled features (before classification).
    Great to visualize clusters by predicted/gt class.
    """
    model.eval()
    feats = []
    thumbs = []
    meta = []
    total = 0

    for images, targets in dataloader:
        # prepare batch
        imgs = [img.to(device) for img in images]
        outs = model(imgs)  # full forward; uses RPN + heads

        for img6, out, tgt in zip(images, outs, targets):
            if total >= max_rois:
                break
            # Keep high-score detections
            keep = out["scores"] >= score_thresh
            boxes = out["boxes"][keep]
            labels = out["labels"][keep]
            scores = out["scores"][keep]

            if boxes.numel() == 0:
                continue

            # Re-run only the roi pooling + box head to get the per-ROI features
            # (This mirrors torchvision internals)
            with torch.no_grad():
                # get backbone features
                fdict = model.backbone(img6.unsqueeze(0).to(device))
                # roi align
                roi_pool = model.roi_heads.box_roi_pool(
                    fdict, [boxes.to(device)], [img6.shape[-2:]]
                )  # [R, C, 7, 7] default
                # head features
                box_head_feats = model.roi_heads.box_head(roi_pool)  # [R, H]
                # optional: L2 normalize for nicer projector geometry
                box_head_feats = torch.nn.functional.normalize(box_head_feats, dim=1)

            # thumbnails cropped from OG image for each box
            og_bgr = img6[:3].clamp(0,1).cpu().numpy()      # [3,H,W] BGR
            og_rgb = og_bgr[[2,1,0], :, :]                  # [3,H,W] RGB
            og = np.transpose(og_rgb, (1,2,0))
            H, W = og.shape[:2]
            for feat_vec, b, lab, sc in zip(box_head_feats.cpu(), boxes.cpu(), labels.cpu(), scores.cpu()):
                if total >= max_rois:
                    break
                x1,y1,x2,y2 = [int(v) for v in b.tolist()]
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(W-1,x2), min(H-1,y2)
                if x2<=x1 or y2<=y1:
                    continue
                
                # --- skip crops that intersect the BR scale-bar region ---
                BRW, BRH = 0.18, 0.12   # same as cache masking
                sx0, sy0 = int(W * (1.0 - BRW)), int(H * (1.0 - BRH))
                sx1, sy1 = W, H
                ix0, iy0 = max(x1, sx0), max(y1, sy0)
                ix1, iy1 = min(x2, sx1), min(y2, sy1)
                inter_w = max(0, ix1 - ix0)
                inter_h = max(0, iy1 - iy0)
                if inter_w * inter_h > 0:      # intersects the scale-bar area
                    continue

                crop = og[y1:y2, x1:x2, :]                      # Hc,Wc,3
                # resize to 160x160, pad if needed
                if crop.size == 0:
                    continue
                # simple keep-aspect resize and pad
                h,w = crop.shape[:2]
                scale = 160 / max(h,w)
                new = cv2.resize(crop, (int(w*scale), int(h*scale)))
                pad_h = 160 - new.shape[0]
                pad_w = 160 - new.shape[1]
                new = np.pad(new, ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0)
                new = np.transpose(new, (2,0,1))                # 3,H,W
                thumbs.append(torch.from_numpy(new).unsqueeze(0).float())  # [1,3,160,160]
                feats.append(feat_vec)                          # [H]
                meta.append(f"pred_c{int(lab)} s={sc:.2f}")

                total += 1

        if total >= max_rois:
            break

    if total == 0:
        print("[Embedding] No ROI embeddings collected.")
        return

    features = torch.stack(feats, dim=0)
    label_img = torch.cat(thumbs, dim=0).clamp(0,1)
    writer.add_embedding(mat=features, metadata=meta, label_img=label_img, tag=tag, global_step=global_step)
    writer.flush()
    print(f"[Embedding] Logged {total} ROI embeddings → {tag} @ step {global_step}")
    
    
def boxes_to_centroids_xy(boxes: torch.Tensor) -> torch.Tensor:
    # boxes: [N,4] xyxy  → [N,2] (cx,cy)
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 2))
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    return torch.stack([cx, cy], dim=1)

def match_by_centroid(dets_xy: torch.Tensor, det_scores: torch.Tensor,
                      gts_xy: torch.Tensor, tol_px: float):
    """
    Greedy one-to-one matching by Euclidean distance for a *single class* on a *single image*.
    dets_xy: [Nd,2], det_scores:[Nd], gts_xy:[Ng,2]
    Returns: tp_flags (bool[Nd]), num_gt
    """
    Nd = dets_xy.shape[0]
    Ng = gts_xy.shape[0]
    if Nd == 0:
        return torch.zeros(0, dtype=torch.bool), Ng
    if Ng == 0:
        return torch.zeros(Nd, dtype=torch.bool), 0

    order = torch.argsort(det_scores, descending=True)
    dets_xy = dets_xy[order]
    det_scores = det_scores[order]

    gt_taken = torch.zeros(Ng, dtype=torch.bool, device=dets_xy.device)
    tp = torch.zeros(Nd, dtype=torch.bool, device=dets_xy.device)

    for i in range(Nd):
        # compute distances to all GTs
        d = dets_xy[i].unsqueeze(0)            # [1,2]
        # squared L2 then sqrt
        diffs = gts_xy - d                     # [Ng,2]
        dist = torch.sqrt((diffs ** 2).sum(dim=1))  # [Ng]
        if dist.numel() == 0:
            continue
        best_dist, best_j = dist.min(dim=0)
        if best_dist.item() <= tol_px and not gt_taken[best_j]:
            tp[i] = True
            gt_taken[best_j] = True

    # return in original det order
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return tp[inv].cpu(), Ng

    
def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [Na,4], b:[Nb,4] in xyxy
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)

def match_dets_to_gts(dets, det_scores, gts, iou_thr=0.5):
    """
    Greedy one-to-one matching by IoU for a *single class* on a *single image*.
    dets: [Nd,4], det_scores:[Nd], gts:[Ng,4]
    Returns: tp_flags (bool[Nd]), num_gt
    """
    Nd = dets.shape[0]
    Ng = gts.shape[0]
    if Nd == 0:
        return torch.zeros(0, dtype=torch.bool), Ng
    order = torch.argsort(det_scores, descending=True)
    dets = dets[order]
    det_scores = det_scores[order]

    ious = box_iou_xyxy(dets, gts) if Ng > 0 else dets.new_zeros((Nd, 0))
    gt_taken = torch.zeros(Ng, dtype=torch.bool, device=dets.device)
    tp = torch.zeros(Nd, dtype=torch.bool, device=dets.device)

    for i in range(Nd):
        if Ng == 0:
            break
        # best GT for this det
        iou_row = ious[i]
        best_iou, best_j = (iou_row.max(), int(iou_row.argmax())) if iou_row.numel() > 0 else (0.0, -1)
        if best_iou >= iou_thr and not gt_taken[best_j]:
            tp[i] = True
            gt_taken[best_j] = True

    # return tp flags in *original* det order
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return tp[inv].cpu(), Ng



# =========================
# Utility: fused cache
# =========================
def zscore(img_f32: np.ndarray) -> np.ndarray:
    m = img_f32.mean(axis=(0, 1), keepdims=True)
    s = img_f32.std(axis=(0, 1), keepdims=True) + 1e-6
    return (img_f32 - m) / s

def build_cache_name(stem: str) -> Path:
    z = 1 if CFG.USE_ZSCORE else 0
    return CFG.CACHE_DIR / f"{stem}_sz{CFG.IMAGE_SIZE}_z{z}.npy"

def max_mtime(*paths: Path) -> float:
    return max(p.stat().st_mtime for p in paths if p.exists())


def load_or_create_cached_fused(stem: str) -> np.ndarray:
    """
    Returns fused (6, H, W) float32 from cache if fresh, else recomputes and caches.
    - Reads <stem>_og.png and <stem>_red.png (BGR)
    - Resizes both to (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)
    - Optionally z-scores per image
    - Concatenates into 6 channels (BGR + BGR)
    - **Neutralizes the bottom-right scale-bar region** before saving
    Cache keys: fused, imgsz, zscore, src_mtime
    """
    ogp  = CFG.IMAGE_DIR / f"{stem}_og.png"
    redp = CFG.IMAGE_DIR / f"{stem}_red.png"
    assert ogp.exists() and redp.exists(), f"Missing pair for {stem}"

    cache_path: Path = build_cache_name(stem).with_suffix(".npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    src_mtime = max(ogp.stat().st_mtime, redp.stat().st_mtime)

    def _save_npz(fused_arr: np.ndarray):
        np.savez_compressed(
            cache_path,
            fused=fused_arr.astype(np.float32),
            imgsz=np.int32(CFG.IMAGE_SIZE),
            zscore=np.int32(1 if CFG.USE_ZSCORE else 0),
            src_mtime=np.float64(src_mtime),
        )

    # Try load cache
    if cache_path.exists():
        try:
            npz = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if hasattr(npz, "files") and {"fused", "imgsz", "zscore", "src_mtime"} <= set(npz.files):
                if (int(npz["imgsz"]) == CFG.IMAGE_SIZE and
                    int(npz["zscore"]) == (1 if CFG.USE_ZSCORE else 0) and
                    float(npz["src_mtime"]) >= src_mtime):
                    return npz["fused"]
        except Exception:
            try:
                cache_path.unlink()
            except OSError:
                pass

    # Build cache
    og  = cv2.imread(str(ogp))   # BGR uint8
    red = cv2.imread(str(redp))  # BGR uint8
    if og is None or red is None:
        raise FileNotFoundError(f"Read failed for {stem}")

    # Resize to square (keeps your current behavior)
    og  = cv2.resize(og,  (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    red = cv2.resize(red, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    if CFG.USE_ZSCORE:
        og  = zscore(og)
        red = zscore(red)

    # BGR+BGR -> (H,W,6)
    fused_hw6 = np.concatenate([og, red], axis=-1).astype(np.float32)

    # ---- Neutralize bottom-right scale bar (percent-based window) ----
    H = W = CFG.IMAGE_SIZE
    BR_WIDTH  = 0.18   # rightmost 18% of width (tweak if needed)
    BR_HEIGHT = 0.12   # bottom 12% of height (tweak if needed)
    x0 = int(W * (1.0 - BR_WIDTH))
    y0 = int(H * (1.0 - BR_HEIGHT))
    x1 = W
    y1 = H

    # Fill with per-channel median to blend naturally
    patch = fused_hw6[y0:y1, x0:x1, :]
    if patch.size > 0:
        fill = np.median(fused_hw6, axis=(0, 1), keepdims=True)  # global median
        fused_hw6[y0:y1, x0:x1, :] = fill

    # (H,W,6) -> (6,H,W)
    fused = np.transpose(fused_hw6, (2, 0, 1)).astype(np.float32)

    _save_npz(fused)
    return fused



# =========================
# Dataset (cached fused)
# =========================
class CachedPicoAlgaeDataset(Dataset):
    def __init__(self, stems: List[str], is_train: bool = False, aug=None):
        self.stems = list(stems)
        self.is_train = is_train
        self.aug = aug

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
    
        # (6, H, W) float32 in [0,1]
        fused = load_or_create_cached_fused(stem)
        img = np.transpose(fused, (1, 2, 0))  # -> (H, W, 6)
    
        # Load YOLO labels and convert to pixel xyxy
        lblp = CFG.LABEL_DIR / f"{stem}_og.txt"
        W = H = CFG.IMAGE_SIZE
        bboxes_px, class_labels = [], []
        if lblp.exists() and lblp.stat().st_size > 0:
            for ln in lblp.read_text().splitlines():
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue
                c, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw / 2.0) * W
                y1 = (cy - bh / 2.0) * H
                x2 = (cx + bw / 2.0) * W
                y2 = (cy + bh / 2.0) * H
                if x2 > x1 and y2 > y1:
                    bboxes_px.append([x1, y1, x2, y2])
                    class_labels.append(int(c) + 1)  # shift: 1..N (0=background)
    
        # ---- Albumentations (normalized xyxy) ----
        if self.is_train and self.aug is not None:
            H0, W0 = img.shape[:2]
        
            if len(bboxes_px) > 0:
                # normalize to [0,1]
                bboxes_norm = []
                kept_labels = []
                for (x1, y1, x2, y2), lab in zip(bboxes_px, class_labels):
                    xn1 = x1 / W0; yn1 = y1 / H0; xn2 = x2 / W0; yn2 = y2 / H0
                    # pre‑clamp to [0,1] to satisfy Albumentations' checker
                    xn1 = float(np.clip(xn1, 0.0, 1.0))
                    yn1 = float(np.clip(yn1, 0.0, 1.0))
                    xn2 = float(np.clip(xn2, 0.0, 1.0))
                    yn2 = float(np.clip(yn2, 0.0, 1.0))
                    if xn2 - xn1 > 0 and yn2 - yn1 > 0:
                        bboxes_norm.append([xn1, yn1, xn2, yn2])
                        kept_labels.append(lab)
        
                if len(bboxes_norm) > 0:
                    transformed = self.aug(image=img, bboxes=bboxes_norm, class_labels=kept_labels)
                else:
                    # all boxes became invalid after clamp — run photometric-only
                    transformed = self.aug(image=img, bboxes=[], class_labels=[])
            else:
                transformed = self.aug(image=img, bboxes=[], class_labels=[])
        
            img = transformed["image"]
            bboxes_norm = transformed.get("bboxes", [])
            class_labels = transformed.get("class_labels", [])
        
            # post‑clamp + de‑normalize back to pixel xyxy
            H2, W2 = img.shape[:2]
            bboxes_px, cleaned_labels = [], []
            for (xn1, yn1, xn2, yn2), lab in zip(bboxes_norm, class_labels):
                xn1 = float(np.clip(xn1, 0.0, 1.0))
                yn1 = float(np.clip(yn1, 0.0, 1.0))
                xn2 = float(np.clip(xn2, 0.0, 1.0))
                yn2 = float(np.clip(yn2, 0.0, 1.0))
                x1, y1 = xn1 * W2, yn1 * H2
                x2, y2 = xn2 * W2, yn2 * H2
                if x2 - x1 > 1.0 and y2 - y1 > 1.0:
                    bboxes_px.append([x1, y1, x2, y2])
                    cleaned_labels.append(lab)
            class_labels = cleaned_labels

    
        # ---- to tensors ----
        image = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().contiguous()  # (6,H,W)
        if len(bboxes_px) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t  = torch.tensor(bboxes_px, dtype=torch.float32)
            labels_t = torch.tensor(class_labels, dtype=torch.int64)
    
        target = {"boxes": boxes_t, "labels": labels_t}
        return image, target



class ChannelMixer1x1(torch.nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.mix = torch.nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        torch.nn.init.eye_(self.mix.weight.data.view(in_ch, in_ch))  # near-identity
        torch.nn.init.zeros_(self.mix.bias)
    def forward(self, x):
        return self.mix(x)
    

class SELayer(torch.nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(ch, ch // r, 1, bias=True), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch // r, ch, 1, bias=True), torch.nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w
    

def inject_se_after_layer2(backbone_body):
    layer2 = backbone_body.layer2
    orig_forward = layer2.forward
    layer2.add_module("se_after", SELayer(512, r=16))
    def new_forward(x):
        y = orig_forward(x)
        return layer2.se_after(y)
    layer2.forward = new_forward



# --- Soft-NMS (Gaussian) ---
def soft_nms_gaussian(boxes, scores, iou_thr=0.5, sigma=0.5, score_thr=1e-3, max_keep=300):
    # boxes: [N,4], scores:[N]
    device = boxes.device
    idxs = torch.arange(boxes.size(0), device=device)
    keep = []
    while scores.numel() > 0:
        i = torch.argmax(scores)
        max_idx = idxs[i]
        max_box = boxes[i].unsqueeze(0)
        keep.append(int(max_idx))
        if scores.numel() == 1:
            break
        # remove chosen
        boxes_i = boxes[i+1:] if i>0 else boxes[1:]
        scores_i = scores[i+1:] if i>0 else scores[1:]
        idxs_i  = idxs[i+1:]  if i>0 else idxs[1:]
        # IoU to decay
        ious = box_iou_xyxy(max_box, boxes_i).squeeze(0)  # [M]
        decay = torch.exp(-(ious * ious) / sigma)
        scores_i = scores_i * decay
        # filter low scores
        keep_mask = scores_i >= score_thr
        boxes = boxes_i[keep_mask]
        scores = scores_i[keep_mask]
        idxs  = idxs_i[keep_mask]
        if len(keep) >= max_keep:
            break
    return torch.tensor(keep, device=device, dtype=torch.long)


def patch_postprocess_with_softnms(model, sigma=0.5):
    def postprocess_detections_softnms(self, class_logits, box_regression, proposals, image_shapes):
        # Handle API name differences across torchvision versions
        score_thresh = getattr(self, "box_score_thresh", getattr(self, "score_thresh", 0.05))
        nms_thresh   = getattr(self, "box_nms_thresh",   getattr(self, "nms_thresh",   0.5))
        dets_per_img = getattr(self, "box_detections_per_img", getattr(self, "detections_per_img", 100))

        num_classes    = class_logits.shape[-1]
        boxes_per_img  = [len(p) for p in proposals]

        # Decode class-specific boxes: (sumN, C*4) -> list[(Ni, C, 4)]
        pred_boxes_all  = self.box_coder.decode(box_regression, proposals)
        pred_boxes_all  = pred_boxes_all.reshape(-1, num_classes, 4).split(boxes_per_img, 0)
        pred_scores_all = torch.softmax(class_logits, -1).split(boxes_per_img, 0)

        # Older torchvision expects tuple of lists (boxes, scores, labels)
        boxes_out, scores_out, labels_out = [], [], []

        for boxes, scores, img_shape in zip(pred_boxes_all, pred_scores_all, image_shapes):
            Ni = boxes.shape[0]

            # clip to image
            boxes = clip_boxes_to_image(boxes.reshape(-1, 4), img_shape).reshape(Ni, num_classes, 4)

            # drop background (class 0)
            boxes  = boxes[:, 1:, :]   # (Ni, C-1, 4)
            scores = scores[:, 1:]     # (Ni, C-1)

            # flatten per-class
            flat_boxes  = boxes.reshape(-1, 4)
            flat_scores = scores.reshape(-1)
            flat_labels = torch.arange(1, num_classes, device=boxes.device).repeat(Ni)

            keep_boxes, keep_scores, keep_labels = [], [], []
            for cls_id in range(1, num_classes):
                m = (flat_labels == cls_id)
                b, s = flat_boxes[m], flat_scores[m]
                if b.numel() == 0:
                    continue
                # score filter
                keep1 = s >= score_thresh
                b, s = b[keep1], s[keep1]
                if b.numel() == 0:
                    continue
                # Soft-NMS
                keep2 = soft_nms_gaussian(
                    b, s,
                    iou_thr=nms_thresh,
                    sigma=sigma,
                    score_thr=score_thresh,
                    max_keep=dets_per_img * 2
                )
                if keep2.numel() == 0:
                    continue

                keep_boxes.append(b[keep2])
                keep_scores.append(s[keep2])
                keep_labels.append(torch.full((keep2.numel(),), cls_id, dtype=torch.int64, device=b.device))

            if keep_boxes:
                boxes_cat  = torch.cat(keep_boxes,  0)
                scores_cat = torch.cat(keep_scores, 0)
                labels_cat = torch.cat(keep_labels, 0)
                topk = min(dets_per_img, boxes_cat.size(0))
                order = torch.argsort(scores_cat, descending=True)[:topk]
                boxes_cat, scores_cat, labels_cat = boxes_cat[order], scores_cat[order], labels_cat[order]
            else:
                dev = class_logits.device
                boxes_cat  = torch.empty((0,4), device=dev)
                scores_cat = torch.empty((0,), device=dev)
                labels_cat = torch.empty((0,), dtype=torch.int64, device=dev)

            boxes_out.append(boxes_cat)
            scores_out.append(scores_cat)
            labels_out.append(labels_cat)

        # Return tuple to satisfy older ROIHeads.forward: boxes, scores, labels
        return boxes_out, scores_out, labels_out

    model.roi_heads.postprocess_detections = types.MethodType(postprocess_detections_softnms, model.roi_heads)


# =========================
# Model (6-channel conv1)
# =========================

def get_faster_rcnn_model(num_classes=7, image_size=(1024, 1024), backbone='resnet101') -> FasterRCNN:
    # Backbone + FPN
    backbone_fpn = resnet_fpn_backbone(backbone_name=backbone, weights="DEFAULT", returned_layers=[1,2,3,4], extra_blocks=None)
    # ---- 1×1 mixer BEFORE conv1, then 6-ch conv1 ----
    old_conv = backbone_fpn.body.conv1
    mixer = ChannelMixer1x1(in_ch=6)
    new_conv = torch.nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        # copy RGB weights into first 3 channels, avg into next 3
        new_conv.weight[:, :3] = old_conv.weight
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:] = mean_w.repeat(1, 3, 1, 1)
    backbone_fpn.body.conv1 = torch.nn.Sequential(mixer, new_conv)
    # ---- SE after stage2 (layer2) ----
    inject_se_after_layer2(backbone_fpn.body)

    # ROIAlign tuned for ~50 px objects
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],  # P2..P5
        output_size=7,
        sampling_ratio=2
    )

    # Anchors matched to ~50px objects (P2..P5 only; 4 tuples)
    anchor_generator = AnchorGenerator(
        sizes=((16, 24, 32),   # P2
               (32, 48, 64),   # P3  ← main scale
               (64, 96, 128),  # P4
               (128, 192, 256),# P5
               (256, 348, 512)
               ),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = FasterRCNN(
        backbone=backbone_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # --- sampler / assigner knobs ---
        rpn_fg_iou_thresh=0.5,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=512,
        rpn_positive_fraction=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.5,
        box_score_thresh=0.00,
        box_nms_thresh=0.5,
        box_detections_per_img=1000,
    )

    # Multi-scale train: short side randomly from these; keep a generous max
    model.transform = GeneralizedRCNNTransform(
        min_size=[1600, 1800, 2048],
        max_size=4096,
        image_mean=[0.0] * 6,
        image_std=[1.0] * 6,
    )

    # ---- Soft-NMS hook (replaces built-in hard NMS) ----
    patch_postprocess_with_softnms(model, sigma=0.5)
    return model


# =========================
# Metrics
# =========================


def weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.5, skip_box_thr=0.0):
    """
    Simple WBF for one image. boxes: list[tensor Nx4], scores: list[tensor N], labels:list[tensor N]
    Returns merged (boxes,scores,labels) tensors.
    """
    if len(boxes) == 1:
        return boxes[0], scores[0], labels[0]
    B = torch.cat(boxes, 0); S = torch.cat(scores, 0); L = torch.cat(labels, 0)
    order = torch.argsort(S, descending=True)
    B, S, L = B[order], S[order], L[order]
    keep_boxes, keep_scores, keep_labels = [], [], []
    used = torch.zeros(B.size(0), dtype=torch.bool, device=B.device)
    for i in range(B.size(0)):
        if used[i] or S[i] < skip_box_thr: 
            continue
        same = (L == L[i]) & (~used)
        ious = box_iou_xyxy(B[i:i+1], B[same]).squeeze(0)
        cluster_mask = same.clone()
        cluster_mask[same] = ious >= iou_thr
        cluster_idxs = torch.where(cluster_mask)[0]
        weights = S[cluster_idxs]
        bb = B[cluster_idxs]
        wsum = weights.sum().clamp(min=1e-8)
        merged = (bb * weights[:, None]).sum(0) / wsum
        keep_boxes.append(merged)
        keep_scores.append(weights.mean())  # or max
        keep_labels.append(L[i])
        used[cluster_idxs] = True
    if keep_boxes:
        return torch.stack(keep_boxes, 0), torch.stack(keep_scores, 0), torch.stack(keep_labels, 0)
    else:
        return torch.empty((0,4), device=B.device), torch.empty((0,), device=B.device), torch.empty((0,), dtype=torch.int64, device=B.device)


@contextlib.contextmanager
def _override_min_size(model, min_size_value: int):
    orig = model.transform.min_size
    model.transform.min_size = [int(min_size_value)]
    try:
        yield
    finally:
        model.transform.min_size = orig

@torch.no_grad()
def predict_one_with_tta(model, img6: torch.Tensor, device, min_sizes=(1600, 1800, 2048), do_flips=True):
    """
    Runs the model multiple times with different transform.min_size and flips.
    Boxes are already returned in the original img6 coordinates by torchvision.
    """
    _, H, W = img6.shape
    variants = []

    def run(min_size, flip=None):
        with _override_min_size(model, min_size):
            x = img6
            if flip == "h":
                x = torch.flip(x, dims=[2])
            elif flip == "v":
                x = torch.flip(x, dims=[1])
            with autocast("cuda"):
                out = model([x.to(device)])[0]
            b, s, l = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()
            # unflip back to original coords
            if flip == "h":
                b[:, [0, 2]] = W - b[:, [2, 0]]
            elif flip == "v":
                b[:, [1, 3]] = H - b[:, [3, 1]]
            variants.append((b, s, l))

    # identity at largest size
    # run each scale once
    for ms in sorted(set(min_sizes)):
        run(ms, None)

    if do_flips:
        run(max(min_sizes), "h")
        run(max(min_sizes), "v")

    boxes, scores, labels = zip(*variants)
    mb, ms, ml = weighted_boxes_fusion(list(boxes), list(scores), list(labels), iou_thr=0.5, skip_box_thr=0.0)
    return {"boxes": mb, "scores": ms, "labels": ml}


@torch.no_grad()
def evaluate_epoch(model, val_loader, writer: SummaryWriter, epoch: int,
                   class_names=CFG.CLASS_NAMES, iou_thr=CFG.PR_IOU,
                   score_thr=CFG.PR_SCORE_THRESH, device=CFG.DEVICE):

    model.eval()

    # --- For AP/PR (IoU-based) ---
    y_true = defaultdict(list)   # class -> [0/1 ...]
    y_score = defaultdict(list)  # class -> [score ...]
    gt_count = defaultdict(int)

    # --- For centroid-F1 (distance-based) ---
    cen_tp = defaultdict(int)
    cen_fp = defaultdict(int)
    cen_fn = defaultdict(int)
    tol = CFG.CEN_F1_TOL_PX
    cen_score_thr = CFG.CEN_F1_SCORE_THRESH

    for images, targets in val_loader:
        imgs = [im.to(device) for im in images]

        # TTA+WBF predictions per image (already in original img coords)
        outputs = []
        for img6 in imgs:
            outputs.append(predict_one_with_tta(
                model, img6, device=device,
                min_sizes=(1600, 1800, 2048),
                do_flips=True
            ))

        for out, tgt in zip(outputs, targets):
            det_boxes = out["boxes"].cpu()
            det_scores = out["scores"].cpu()
            det_labels = out["labels"].cpu()
            gt_boxes = tgt["boxes"].cpu()
            gt_labels = tgt["labels"].cpu()

            # IoU-based bookkeeping for AP
            for c in range(1, len(class_names) + 1):
                det_mask = det_labels == c
                gt_mask  = gt_labels == c

                det_b = det_boxes[det_mask]
                det_s = det_scores[det_mask]
                gt_b  = gt_boxes[gt_mask]

                keep = det_s >= score_thr
                det_b = det_b[keep]
                det_s = det_s[keep]

                tp_flags, num_gt = match_dets_to_gts(det_b, det_s, gt_b, iou_thr=iou_thr)
                gt_count[c] += int(num_gt)
                y_true[c].extend(tp_flags.numpy().astype(np.int32).tolist())
                y_score[c].extend(det_s.numpy().tolist())

            # Centroid-F1 bookkeeping
            for c in range(1, len(class_names) + 1):
                det_mask = (det_labels == c) & (det_scores >= cen_score_thr)
                gt_mask  = gt_labels == c

                det_b = det_boxes[det_mask]
                det_s = det_scores[det_mask]
                gt_b  = gt_boxes[gt_mask]

                det_xy = boxes_to_centroids_xy(det_b)
                gt_xy  = boxes_to_centroids_xy(gt_b)

                tp_flags, num_gt = match_by_centroid(det_xy, det_s, gt_xy, tol_px=tol)
                tp = int(tp_flags.sum().item())
                fp = int(det_xy.shape[0] - tp)
                fn = int(num_gt - tp)
                cen_tp[c] += tp
                cen_fp[c] += fp
                cen_fn[c] += fn

    # ---- Compute per-class AP (IoU-based) ----
    aps = []
    for c in range(1, len(class_names) + 1):
        name = class_names[c - 1]
        t = np.array(y_true[c], dtype=np.int32)
        s = np.array(y_score[c], dtype=np.float32)

        if gt_count[c] == 0:
            writer.add_scalar(f"val/AP/{name}", float('nan'), epoch)
            continue

        if len(s) == 0:
            writer.add_scalar(f"val/AP/{name}", 0.0, epoch)
            writer.add_pr_curve(f"val/PR/{name}",
                                labels=torch.tensor([0, 1]),
                                predictions=torch.tensor([0.0, 0.0]),
                                global_step=epoch)
            aps.append(0.0)
            continue

        ap = average_precision_score(t, s)
        aps.append(ap)
        writer.add_scalar(f"val/AP/{name}", ap, epoch)

        precision, recall, _ = precision_recall_curve(t, s)
        writer.add_pr_curve_raw(
            tag=f"val/PR/{name}",
            true_positive_counts=torch.tensor((precision * recall * gt_count[c]).clip(min=0)),
            false_positive_counts=torch.tensor(((1 - precision) * recall * gt_count[c]).clip(min=0)),
            true_negative_counts=torch.tensor(np.zeros_like(precision)),
            false_negative_counts=torch.tensor(((1 - recall) * gt_count[c]).clip(min=0)),
            precision=torch.tensor(precision),
            recall=torch.tensor(recall),
            global_step=epoch,
        )

    if aps:
        mAP = float(np.nanmean(aps))
        writer.add_scalar("val/mAP@0.5", mAP, epoch)
        print(f"[Val] Epoch {epoch}: mAP@0.5 = {mAP:.4f}")
    else:
        mAP = float("nan")
        print(f"[Val] Epoch {epoch + 1}: no classes with GT found for AP.")

    # ---- Compute per-class + micro Centroid-F1 ----
    micro_tp = micro_fp = micro_fn = 0
    for c in range(1, len(class_names) + 1):
        name = class_names[c - 1]
        tp, fp, fn = cen_tp[c], cen_fp[c], cen_fn[c]
        micro_tp += tp; micro_fp += fp; micro_fn += fn

        if (tp + fp + fn) == 0:
            # nothing for this class in the epoch
            writer.add_scalar(f"val/F1c/{name}@{tol}px_thr{int(cen_score_thr*100)}", float('nan'), epoch)
            continue

        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 2 * prec * rec / max(1e-8, prec + rec)

        writer.add_scalar(f"val/Precision_c/{name}@{tol}px", prec, epoch)
        writer.add_scalar(f"val/Recall_c/{name}@{tol}px", rec, epoch)
        writer.add_scalar(f"val/F1c/{name}@{tol}px_thr{int(cen_score_thr*100)}", f1, epoch)

    # micro-averaged across classes
    mprec = micro_tp / max(1, micro_tp + micro_fp)
    mrec  = micro_tp / max(1, micro_tp + micro_fn)
    mf1   = 2 * mprec * mrec / max(1e-8, mprec + mrec)
    writer.add_scalar(f"val/F1c_micro@{tol}px_thr{int(cen_score_thr*100)}", mf1, epoch)
    print(f"[Val] Epoch {epoch + 1}: Centroid-F1 (micro) @ {tol}px = {mf1:.4f}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mAP


# =========================
# Training
# =========================

def save_checkpoint(model, optimizer, scheduler, epoch, best_map, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_map": best_map,
        "cfg": CFG.__dict__,
    }, str(path))
    print(f"[Checkpoint] Saved: {path}")

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

class WarmupThenCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters, max_iters, min_lr=0.0, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        t = self.last_epoch + 1
        out = []
        for base_lr in self.base_lrs:
            if t <= self.warmup_iters:
                out.append(base_lr * t / max(1, self.warmup_iters))
            else:
                tt = (t - self.warmup_iters) / max(1, self.max_iters - self.warmup_iters)
                cos = 0.5 * (1 + np.cos(np.pi * tt))
                out.append(self.min_lr + (base_lr - self.min_lr) * cos)
        return out
    
    
def build_optimizer(model, base_lr_backbone=2e-3, lr_heads=1e-2, weight_decay=1e-4):
    # backbone (except conv1) → low LR; conv1 + FPN + RPN + ROI heads → higher LR
    params_backbone = []
    params_heads = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.body.conv1" in n:
            params_heads.append(p)  # 6-ch conv1
        elif n.startswith("backbone.body"):
            params_backbone.append(p)
        else:
            params_heads.append(p)  # fpn, rpn, roi_heads
    return torch.optim.SGD(
        [
            {"params": params_backbone, "lr": base_lr_backbone},
            {"params": params_heads, "lr": lr_heads},
        ],
        momentum=0.9, weight_decay=weight_decay, nesterov=True
    )



def train():
    writer = SummaryWriter(CFG.LOGDIR)
    best_map = -float("inf")
    
    seed_everything(CFG.SEED)
    CFG.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build stems from images present (*_og.png)
    all_stems = sorted([p.stem.replace("_og", "") for p in CFG.IMAGE_DIR.glob("*_og.png")])
    assert all_stems, "No *_og.png images found."

    # Split train/val
    val_count = max(1, int(len(all_stems) * CFG.VAL_FRACTION))
    train_count = len(all_stems) - val_count
    # deterministic split
    rng = random.Random(CFG.SEED)
    stems = all_stems.copy()
    rng.shuffle(stems)
    train_stems = stems[:train_count]
    val_stems   = stems[train_count:]
    
    train_steps = max(1, math.ceil(len(train_stems) / CFG.BATCH_SIZE))
    total_iters = CFG.EPOCHS * train_steps

    train_aug = build_train_aug()
    train_ds = CachedPicoAlgaeDataset(train_stems, is_train=True, aug=train_aug)
    val_ds   = CachedPicoAlgaeDataset(val_stems, is_train=False, aug=None)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4, collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=CFG.WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4, collate_fn=lambda x: tuple(zip(*x))
    )

    
    model = get_faster_rcnn_model(
        num_classes=CFG.NUM_CLASSES,
        image_size=(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
        backbone=CFG.BACKBONE
        ).to(CFG.DEVICE)
    
    model.roi_heads.score_thresh = 0.00
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 1000
    
    # lighter RPN/ROI for peak-mem & speed
    def set_rpn_topn(rpn, pre_train=1000, pre_test=600, post_train=500, post_test=300):
        # torchvision <= 0.13 style
        if hasattr(rpn, "pre_nms_top_n_train"):
            rpn.pre_nms_top_n_train = pre_train
            rpn.pre_nms_top_n_test  = pre_test
        # torchvision >= 0.14 style (dict config)
        elif hasattr(rpn, "pre_nms_top_n") and isinstance(rpn.pre_nms_top_n, dict):
            rpn.pre_nms_top_n["training"] = pre_train
            rpn.pre_nms_top_n["testing"]  = pre_test
    
        if hasattr(rpn, "post_nms_top_n_train"):
            rpn.post_nms_top_n_train = post_train
            rpn.post_nms_top_n_test  = post_test
        elif hasattr(rpn, "post_nms_top_n") and isinstance(rpn.post_nms_top_n, dict):
            rpn.post_nms_top_n["training"] = post_train
            rpn.post_nms_top_n["testing"]  = post_test

    set_rpn_topn(model.rpn)
    
    model.rpn.batch_size_per_image       = 256
    model.roi_heads.batch_size_per_image = 256
    model.roi_heads.detections_per_img   = 600

    # memory layout often speeds convs
    model = model.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(model,
        base_lr_backbone=2e-3,
        lr_heads=1e-2,
        weight_decay=1e-4
        )
    
    lr_scheduler = WarmupThenCosine(optimizer,
        warmup_iters=min(1000,total_iters//10),
        max_iters=total_iters,
        min_lr=1e-5)
    
    scaler = GradScaler()
    
    log_roi_embeddings(model, val_loader, writer, CFG.DEVICE, max_rois=400, tag="emb/roi", global_step=0)
    
    global_step = 0
    for epoch in range(CFG.EPOCHS):
        model.train()
        running = defaultdict(float)

        for i, (images, targets) in enumerate(train_loader, 0):
            images  = [img.to(CFG.DEVICE, non_blocking=True) for img in images]
            targets = [{k: v.to(CFG.DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            
            with autocast("cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            
            prev_scale = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.step(optimizer)          # may be skipped by scaler on overflow
            scaler.update()
            did_step = float(scaler.get_scale()) >= prev_scale  # if overflow, scale decreased → no step
    
            if did_step:
                lr_scheduler.step()
            
            # accumulate
            for k, v in loss_dict.items():
                running[k] += v.item()

            # per-batch logging every LOG_EVERY
            if (i + 1) % CFG.LOG_EVERY == 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
                writer.add_scalar("train/total_loss", loss.item(), global_step)
                # you can also log LR:
                for gi, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"train/lr/group{gi}", pg["lr"], global_step)
                global_step += 1

        # epoch-averaged loss (nice and smooth)
        num_batches = max(1, len(train_loader))
        for k in running:
            writer.add_scalar(f"epoch_avg/{k}", running[k] / num_batches, epoch)
        writer.add_scalar("epoch_avg/total_loss", sum(running.values()) / num_batches, epoch)

        # validation with PR curves and AP
        mAp = evaluate_epoch(model, val_loader, writer, epoch, class_names=CFG.CLASS_NAMES, iou_thr=CFG.PR_IOU, score_thr=CFG.PR_SCORE_THRESH, device=CFG.DEVICE)
        if not np.isnan(mAp) and mAp > best_map:
                best_map = mAp
                save_checkpoint(model, optimizer, lr_scheduler, epoch, best_map, CFG.SAVE_BEST_TO)

    writer.flush()
    writer.close()
    print("Training done.")


if __name__ == "__main__":
    train()