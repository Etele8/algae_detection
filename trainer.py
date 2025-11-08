import os
import random
import types
import math
import contextlib
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

from sklearn.metrics import average_precision_score

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
from torchvision.ops import nms as tv_nms
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
import torch.nn as nn

torch.backends.cudnn.benchmark = True

@dataclass
class Config:
    IMAGE_DIR: Path = Path("/workspace/data/images")
    LABEL_DIR: Path = Path("/workspace/data/labels")
    CACHE_DIR: Path = Path("/workspace/data/cache_fused")

    IMAGE_SIZE: int = 2048
    NUM_CLASSES: int = 6
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

    LOGDIR = "runs/algae"
    LOG_EVERY = 50
    PR_IOU = 0.5
    PR_SCORE_THRESH = 0.00
    CLASS_NAMES = ["EUK", "FC", "FE", "FC colony", "FE colony"]
    COLONY_CLASS_IDS: Tuple[int,...] = (4,5)
    SINGLE_CLASS_IDS: Tuple[int,...] = (1, 2, 3)


    OVERSAMPLE_COLONY: bool = True
    COLONY_OVERSAMPLE_FACTOR: float = 3.0
    RAW_TO_MODEL = {
    0: 1,  # EUK
    1: 2,  # FC
    2: 3,  # FE
    4: 4,  # FC colony
    5: 5,  # FE colony
    # 6 colony cell points
    # 3 (EUK_colony)
    # 7 (artifact/crowd)
    8: 4,
    }

    SAVE_VAL_VIS_EVERY: int = 10
    VIS_OUT_DIR: str = "val_vis"

CFG = Config()

class ChannelDimming6(ImageOnlyTransform):
    def __init__(self, p=0.3, min_scale=0.3, max_scale=0.7):
        super().__init__(p=p)
        self.min_scale, self.max_scale = min_scale, max_scale
    def apply(self, img, **params):
        if np.random.rand() < 0.5:
            s = np.random.uniform(self.min_scale, self.max_scale)
            img[..., 0:3] *= s
        else:
            s = np.random.uniform(self.min_scale, self.max_scale)
            img[..., 3:6] *= s
        return img

class PoissonNoise(ImageOnlyTransform):
    def __init__(self, lam_scale=0.05, p=0.2):
        super().__init__(p=p)
        self.lam_scale = lam_scale
    def apply(self, img, **params):
        lam = np.clip(img * self.lam_scale, 0, 1)
        noise = np.random.poisson(lam * 255.0) / 255.0
        out = img + noise - lam
        return np.clip(out, 0.0, 1.0)

class RedCLAHE(ImageOnlyTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    def apply(self, img, **params):
        red = (img[..., 3:6] * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        for ch in range(3):
            red[..., ch] = clahe.apply(red[..., ch])
        img[..., 3:6] = red.astype(np.float32) / 255.0
        return img

def build_train_aug():
    return A.Compose([
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

def _size_filter_xyxy(b: torch.Tensor, min_side=None, max_side=None):
    min_side = CFG.__dict__.get("MIN_SIDE_PX", 6) if min_side is None else min_side
    max_side = CFG.__dict__.get("MAX_SIDE_PX", 512) if max_side is None else max_side
    if b.numel() == 0:
        return b, torch.zeros(0, dtype=torch.bool, device=b.device)
    w = (b[:, 2] - b[:, 0])
    h = (b[:, 3] - b[:, 1])
    keep = (torch.minimum(w, h) >= min_side) & (torch.maximum(w, h) <= max_side)
    return b[keep], keep

def boxes_to_centroids_xy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 2))
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    return torch.stack([cx, cy], dim=1)

def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
    Nd = dets.shape[0]; Ng = gts.shape[0]
    if Nd == 0:
        return torch.zeros(0, dtype=torch.bool), Ng
    order = torch.argsort(det_scores, descending=True)
    dets = dets[order]; det_scores = det_scores[order]
    ious = box_iou_xyxy(dets, gts) if Ng > 0 else dets.new_zeros((Nd, 0))
    gt_taken = torch.zeros(Ng, dtype=torch.bool, device=dets.device)
    tp = torch.zeros(Nd, dtype=torch.bool, device=dets.device)
    for i in range(Nd):
        if Ng == 0: break
        iou_row = ious[i]
        best_iou, best_j = (iou_row.max(), int(iou_row.argmax())) if iou_row.numel() > 0 else (0.0, -1)
        if best_iou >= iou_thr and not gt_taken[best_j]:
            tp[i] = True; gt_taken[best_j] = True
    inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device=order.device)
    return tp[inv].cpu(), Ng

def match_dets_to_gts_with_indices(dets, det_scores, gts, iou_thr=0.5):
    Nd = dets.shape[0]; Ng = gts.shape[0]
    if Nd == 0:
        return torch.zeros(0, dtype=torch.bool), Ng, torch.full((0,), -1, dtype=torch.long)
    order = torch.argsort(det_scores, descending=True)
    dets = dets[order]; det_scores = det_scores[order]
    ious = box_iou_xyxy(dets, gts) if Ng > 0 else dets.new_zeros((Nd, 0))
    gt_taken = torch.zeros(Ng, dtype=torch.bool, device=dets.device)
    tp = torch.zeros(Nd, dtype=torch.bool, device=dets.device)
    match_j = torch.full((Nd,), -1, dtype=torch.long, device=dets.device)
    for i in range(Nd):
        if Ng == 0: break
        row = ious[i]
        if row.numel() == 0: continue
        best_iou, j = (row.max(), int(row.argmax()))
        if best_iou >= iou_thr and not gt_taken[j]:
            tp[i] = True; gt_taken[j] = True; match_j[i] = j
    inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device=order.device)
    return tp[inv].cpu(), Ng, match_j[inv].cpu()

def match_by_centroid(dets_xy: torch.Tensor, det_scores: torch.Tensor,
                      gts_xy: torch.Tensor, tol_px: float):
    Nd = dets_xy.shape[0]; Ng = gts_xy.shape[0]
    if Nd == 0:
        return torch.zeros(0, dtype=torch.bool), Ng
    if Ng == 0:
        return torch.zeros(Nd, dtype=torch.bool), 0
    order = torch.argsort(det_scores, descending=True)
    dets_xy = dets_xy[order]; det_scores = det_scores[order]
    gt_taken = torch.zeros(Ng, dtype=torch.bool, device=dets_xy.device)
    tp = torch.zeros(Nd, dtype=torch.bool, device=dets_xy.device)
    for i in range(Nd):
        d = dets_xy[i].unsqueeze(0)
        diffs = gts_xy - d
        dist = torch.sqrt((diffs ** 2).sum(dim=1))
        if dist.numel() == 0:
            continue
        best_dist, best_j = dist.min(dim=0)
        if best_dist.item() <= tol_px and not gt_taken[best_j]:
            tp[i] = True; gt_taken[best_j] = True
    inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device=order.device)
    return tp[inv].cpu(), Ng


class ChannelMixer1x1(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.mix = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.mix.bias)
            self.mix.weight.zero_()
            eye = torch.eye(in_ch)[:, :, None, None]
            self.mix.weight.copy_(eye)
    def forward(self, x): return self.mix(x)

class SELayer(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

def inject_se_after_layer2(backbone_body: nn.Module):
    layer2 = backbone_body.layer2
    orig_forward = layer2.forward
    layer2.add_module("se_after", SELayer(512, r=16))
    def new_forward(x):
        y = orig_forward(x)
        return layer2.se_after(y)
    layer2.forward = new_forward

def build_old_frcnn_6ch_se(num_classes=7) -> FasterRCNN:
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights="DEFAULT",                        
        returned_layers=[1,2,3,4],
        extra_blocks=LastLevelMaxPool()
    )

    old_conv = backbone.body.conv1
    mix = ChannelMixer1x1(in_ch=6)
    new_conv = nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        mean_w = old_conv.weight.mean(dim=1, keepdim=True) 
        new_conv.weight[:, 3:] = mean_w.repeat(1, 3, 1, 1)
    backbone.body.conv1 = nn.Sequential(mix, new_conv)   

    inject_se_after_layer2(backbone.body)

    sizes = (
        (40, 56, 72),     
        (80, 96, 112),     
        (160, 192, 224),    
        (256, 384, 512),  
        (768, 1024, 1280),
    )
    ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    anchor_gen = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)

    roi_pooler = MultiScaleRoIAlign(featmap_names=['0','1','2','3','pool'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=roi_pooler,
        rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=512, rpn_positive_fraction=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.5,
        box_score_thresh=0.0, box_nms_thresh=0.5, box_detections_per_img=1000
    )

    model.transform = GeneralizedRCNNTransform(
        min_size=[1600, 1800, 2048, 2300],
        max_size=4096,
        image_mean=[0.0]*6,
        image_std=[1.0]*6,
    )
    return model

def build_cache_name(stem: str) -> Path:
    z = 1 if CFG.USE_ZSCORE else 0
    return CFG.CACHE_DIR / f"{stem}_sz{CFG.IMAGE_SIZE}_z{z}_lb1.npz"

def load_or_create_cached_fused(stem: str, im_dir: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    ogp  = im_dir / f"{stem}_og.png"
    redp = im_dir / f"{stem}_red.png"
    assert ogp.exists() and redp.exists(), f"Missing pair for {stem}"
    cache_path: Path = build_cache_name(stem)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    src_mtime = max(ogp.stat().st_mtime, redp.stat().st_mtime)

    def _save_npz(fused_arr: np.ndarray, meta: Dict[str, float]):
        np.savez_compressed(
            cache_path,
            fused=fused_arr.astype(np.float32),
            imgsz=np.int32(CFG.IMAGE_SIZE),
            zscore=np.int32(1 if CFG.USE_ZSCORE else 0),
            src_mtime=np.float64(src_mtime),
            letterbox=np.int32(1),
            **{k: np.float32(v) for k, v in meta.items()},
        )

    if cache_path.exists():
        try:
            npz = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if {"fused","imgsz","zscore","src_mtime","letterbox"} <= set(npz.files):
                if (int(npz["imgsz"]) == CFG.IMAGE_SIZE and
                    int(npz["zscore"]) == (1 if CFG.USE_ZSCORE else 0) and
                    int(npz["letterbox"]) == 1 and
                    float(npz["src_mtime"]) >= src_mtime):
                    meta = {
                        "orig_w": float(npz["orig_w"]),
                        "orig_h": float(npz["orig_h"]),
                        "scale":  float(npz["scale"]),
                        "pad_l":  float(npz["pad_l"]),
                        "pad_t":  float(npz["pad_t"]),
                        "new_w":  float(npz["new_w"]),
                        "new_h":  float(npz["new_h"]),
                    }
                    return npz["fused"], meta
        except Exception:
            try: cache_path.unlink()
            except OSError: pass

    og  = cv2.imread(str(ogp))
    red = cv2.imread(str(redp))
    if og is None or red is None:
        raise FileNotFoundError(f"Read failed for {stem}")

    H0, W0 = og.shape[:2]
    target = CFG.IMAGE_SIZE

    def neutralize_br(img_bgr_uint8: np.ndarray, br_w=0.15, br_h=0.05) -> np.ndarray:
        h, w = img_bgr_uint8.shape[:2]
        x0 = int(w * (1.0 - br_w)); y0 = int(h * (1.0 - br_h))
        x1 = w; y1 = h
        if x1 > x0 and y1 > y0:
            med = np.median(img_bgr_uint8, axis=(0,1), keepdims=True).astype(img_bgr_uint8.dtype)
            img_bgr_uint8[y0:y1, x0:x1, :] = med
        return img_bgr_uint8

    og  = neutralize_br(og.copy())
    red = neutralize_br(red.copy())

    scale = min(target / max(1, W0), target / max(1, H0))
    new_w = int(round(W0 * scale))
    new_h = int(round(H0 * scale))
    pad_l = int((target - new_w) // 2)
    pad_t = int((target - new_h) // 2)
    pad_r = target - new_w - pad_l
    pad_b = target - new_h - pad_t

    og_rs  = cv2.resize(og,  (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    red_rs = cv2.resize(red, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    og_pad  = cv2.copyMakeBorder(og_rs,  pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    red_pad = cv2.copyMakeBorder(red_rs, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)

    fused_hw6 = np.concatenate([og_pad, red_pad], axis=-1).astype(np.float32)
    fused = np.transpose(fused_hw6, (2, 0, 1)).astype(np.float32)

    meta = {
        "orig_w": float(W0), "orig_h": float(H0),
        "scale": float(scale),
        "pad_l": float(pad_l), "pad_t": float(pad_t),
        "new_w": float(new_w), "new_h": float(new_h),
    }
    _save_npz(fused, meta)
    return fused, meta

def read_yolo_remapped(stem: str, meta: Dict[str, float]) -> Tuple[List[List[float]], List[int]]:

    p = CFG.LABEL_DIR / f"{stem}_og.txt"
    if not p.exists() or p.stat().st_size == 0:
        return [], [], [], []

    W0 = meta["orig_w"]; H0 = meta["orig_h"]
    s  = meta["scale"];  dx = meta["pad_l"]; dy = meta["pad_t"]

    bxs = []; labs = []; crowd_boxes = []; points = []
    for ln in p.read_text().splitlines():
        pr = ln.strip().split()
        if len(pr) == 3:
            c_raw, x, y = map(float, pr)
            
            xo = x * W0
            yo = y * H0
            
            x1 = xo * s + dx
            y1 = yo * s + dy
        elif len(pr) == 5:
            c_raw, cx, cy, bw, bh = map(float, pr)
            if c_raw == 8:
                c_raw = 4

            x1o = (cx - bw/2.0) * W0
            y1o = (cy - bh/2.0) * H0
            x2o = (cx + bw/2.0) * W0
            y2o = (cy + bh/2.0) * H0

            x1 = x1o * s + dx; x2 = x2o * s + dx
            y1 = y1o * s + dy; y2 = y2o * s + dy

        if c_raw == 6:
                points.append([x1, y1])

        elif x2 > x1 and y2 > y1:
            if c_raw == 7:
                crowd_boxes.append([float(x1), float(y1), float(x2), float(y2)])
            else:
                bxs.append([float(x1), float(y1), float(x2), float(y2)])
                labs.append(CFG.RAW_TO_MODEL[c_raw])
    return bxs, labs, crowd_boxes, points

class PicoAlgaeDataset(Dataset):
    def __init__(self, stems:List[str], is_train:bool=False, aug=None, return_stem: bool = False):
        self.stems=list(stems)
        self.is_train=is_train
        self.aug=aug
        self.return_stem = return_stem
    def __len__(self):
        return len(self.stems)
    
    def __getitem__(self, idx):
        stem = self.stems[idx]
        fused, meta = load_or_create_cached_fused(stem, CFG.IMAGE_DIR)   # (6,H,W) + letterbox params
        img = np.transpose(fused, (1, 2, 0))              # (H,W,6)

        bxs_px, clabs, crowd_boxes, points = read_yolo_remapped(stem, meta)

        if self.is_train and self.aug is not None:
            if len(bxs_px)>0:
                H0,W0=img.shape[:2]
                bnorm=[]; labs_keep=[]
                for (x1, y1, x2, y2), L in zip(bxs_px, clabs):
                    xn1 = float(np.clip(x1 / W0, 0.0, 1.0))
                    yn1 = float(np.clip(y1 / H0, 0.0, 1.0))
                    xn2 = float(np.clip(x2 / W0, 0.0, 1.0))
                    yn2 = float(np.clip(y2 / H0, 0.0, 1.0))
                    if xn2 - xn1 > 0 and yn2 - yn1 > 0:
                        bnorm.append([xn1, yn1, xn2, yn2])
                        labs_keep.append(L)
                if len(bnorm)>0:
                    tr=self.aug(image=img, bboxes=bnorm, class_labels=labs_keep)
                else:
                    tr=self.aug(image=img, bboxes=[], class_labels=[])
            else:
                tr=self.aug(image=img, bboxes=[], class_labels=[])
            img=tr["image"]
            bnorm=tr.get("bboxes",[])
            clabs=tr.get("class_labels",[])
            H2,W2=img.shape[:2]
            bxs_px=[]
            for (xn1, yn1, xn2, yn2), L in zip(bnorm, clabs):
                xn1 = float(np.clip(xn1, 0.0, 1.0))
                yn1 = float(np.clip(yn1, 0.0, 1.0))
                xn2 = float(np.clip(xn2, 0.0, 1.0))
                yn2 = float(np.clip(yn2, 0.0, 1.0))
                x1, y1 = xn1 * W2, yn1 * H2
                x2, y2 = xn2 * W2, yn2 * H2
                if x2 - x1 > 1 and y2 - y1 > 1:
                    bxs_px.append([x1, y1, x2, y2])

        image=torch.from_numpy(np.transpose(img,(2,0,1))).float().contiguous()
        if len(bxs_px) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(bxs_px, dtype=torch.float32)
            labels_t = torch.tensor(clabs, dtype=torch.int64)
        if len(crowd_boxes) == 0:
            crowd_t = torch.zeros((0, 4), dtype=torch.float32)
            iscrowd = torch.zeros((0,),    dtype=torch.uint8)
        else:
            crowd_t = torch.tensor(crowd_boxes, dtype=torch.float32)
            iscrowd = torch.ones((len(crowd_boxes),), dtype=torch.uint8)

        target = {"boxes": boxes_t, "labels": labels_t, "crowd_boxes": crowd_t, "iscrowd": iscrowd}
        if self.return_stem:
            target["stem"] = stem
        
        return image, target

def _area(b: torch.Tensor) -> torch.Tensor:
    return ((b[:, 2] - b[:, 0]).clamp_min(0) *
            (b[:, 3] - b[:, 1]).clamp_min(0))

def _centers(b: torch.Tensor) -> torch.Tensor:
    return torch.stack([(b[:, 0] + b[:, 2]) * 0.5,
                        (b[:, 1] + b[:, 3]) * 0.5], dim=1)

def _iou_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_a = (a[..., 2] - a[..., 0]).clamp_min(0) * (a[..., 3] - a[..., 1]).clamp_min(0)
    area_b = (b[..., 2] - b[..., 0]).clamp_min(0) * (b[..., 3] - b[..., 1]).clamp_min(0)
    return inter / (area_a + area_b - inter + 1e-6)

def _soft_nms_gaussian(boxes, scores, iou_thr=0.5, sigma=0.5, score_thr=1e-3):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    device = boxes.device
    idx = torch.arange(boxes.size(0), device=device)  # keep indices on same device

    keep = []
    b = boxes.clone(); s = scores.clone(); idc = idx.clone()
    while s.numel():
        i = torch.argmax(s)
        keep.append(int(idc[i]))
        if s.numel() == 1:
            break

        bi = b[i]
        bj = torch.cat([b[:i], b[i+1:]], dim=0)
        sj = torch.cat([s[:i], s[i+1:]], dim=0)
        ij = torch.cat([idc[:i], idc[i+1:]], dim=0)

        x1 = torch.maximum(bi[0], bj[:, 0]); y1 = torch.maximum(bi[1], bj[:, 1])
        x2 = torch.minimum(bi[2], bj[:, 2]); y2 = torch.minimum(bi[3], bj[:, 3])
        inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
        ai = (bi[2] - bi[0]).clamp_min(0) * (bi[3] - bi[1]).clamp_min(0)
        aj = (bj[:, 2] - bj[:, 0]).clamp_min(0) * (bj[:, 3] - bj[:, 1]).clamp_min(0)
        iou = inter / (ai + aj - inter + 1e-6)

        decay = torch.exp(-(iou * iou) / sigma)
        sj = sj * decay

        m = sj >= score_thr
        b, s, idc = bj[m], sj[m], ij[m]

    return torch.tensor(keep, dtype=torch.long, device=device)

def size_aware_softnms_per_class(boxes, scores, labels, split_area_px,
                                 iou_small, sigma_small,
                                 iou_big, sigma_big, score_floor):
    if boxes.numel() == 0:
        return boxes, scores, labels
    A = _area(boxes); outB = []; outS = []; outL = []
    for c in torch.unique(labels):
        m = (labels == c); bc, sc = boxes[m], scores[m]
        if bc.numel() == 0:
            continue
        small = (A[m] <= split_area_px)
        if small.any():
            ks = _soft_nms_gaussian(bc[small], sc[small], iou_thr=iou_small, sigma=sigma_small, score_thr=score_floor)
            outB.append(bc[small][ks]); outS.append(sc[small][ks])
            outL.append(torch.full((ks.numel(),), int(c), dtype=torch.long, device=labels.device))
        big = ~small
        if big.any():
            kb = _soft_nms_gaussian(bc[big], sc[big], iou_thr=iou_big, sigma=sigma_big, score_thr=score_floor)
            outB.append(bc[big][kb]); outS.append(sc[big][kb])
            outL.append(torch.full((kb.numel(),), int(c), dtype=torch.long, device=labels.device))
    if not outB:
        return boxes.new_zeros((0, 4)), scores.new_zeros((0,)), labels.new_zeros((0,), dtype=torch.long)
    B = torch.cat(outB, 0); S = torch.cat(outS, 0); L = torch.cat(outL, 0)
    return B, S, L


def parent_kill(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    colony_classes: tuple = (4, 5),
    tol_px: float = 4.0,
):
    """
    Drop larger 'parent' boxes that (approximately) contain an already-kept smaller box.

    Rule:
      - If the kept CHILD is a colony (label in `colony_classes`), it can kill parents
        of ANY class.
      - If the kept CHILD is a single cell, it only kills parents of the SAME class.

    Args:
      boxes, scores, labels: tensors from detector (XYXY)
      colony_classes: tuple of class IDs considered colonies (child can cross-class kill)
      tol_px: coordinate tolerance for approximate containment

    Returns:
      (boxes_kept, scores_kept, labels_kept) — in small→big area order.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    areas = (boxes[:,2]-boxes[:,0]).clamp_min(0) * (boxes[:,3]-boxes[:,1]).clamp_min(0)
    order = torch.argsort(areas, descending=False)
    B, S, L = boxes[order], scores[order], labels[order]

    colony_set = set(int(c) for c in colony_classes)
    keep = []

    for i in range(B.size(0)):
        bi = B[i]; li = int(L[i].item())
        drop = False
        for k in keep:
            bk = B[k]; lk = int(L[k].item())

            contains = (
                bk[0] >= bi[0] - tol_px and
                bk[1] >= bi[1] - tol_px and
                bk[2] <= bi[2] + tol_px and
                bk[3] <= bi[3] + tol_px
            )

            if not contains:
                continue

            if lk in colony_set:
                drop = True
                break
            else:

                if li == lk:
                    drop = True
                    break

        if not drop:
            keep.append(i)

    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return B[keep], S[keep], L[keep]




def _roi_redorange_stats_og(roi_hw6: np.ndarray) -> dict:
    """
    ROI: float32 HxWx6 in [0,1] (BGR OG + BGR RED). Uses OG only.
    Returns:
        {
          "bright_frac": fraction of pixels with mean(B,G,R) >= I_thr,
          "ro_frac_on_bright": fraction of *bright* pixels where R >= alpha*G and R >= alpha*B
        }
    """
    og = np.clip(roi_hw6[..., :3], 0.0, 1.0)   # B,G,R
    B, G, R = og[...,0], og[...,1], og[...,2]
    
    I_mean = float(((R + G + B) / 3.0).mean())

    return {"bright_frac": I_mean, "ro_frac_on_bright": R.mean()}


def drop_bright_green(hw6: np.ndarray,
                      boxes: torch.Tensor,
                      scores: torch.Tensor,
                      labels: torch.Tensor,
                      classes=(1, 2, 3),
                      *,
                      min_bright_frac: float = 0.05,
                      ro_min_frac: float = 0.20,
                      stem: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove detections (only for `classes`) that are:
      - sufficiently bright (bright_frac >= min_bright_frac)
      - AND have too little red/orange on bright pixels (ro_frac_on_bright < ro_min_frac)
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    H, W = hw6.shape[:2]
    keep = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)

    for i in range(labels.shape[0]):
        c = int(labels[i].item())
        if c not in classes:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in boxes[i].tolist()]
        x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            keep[i] = False
            continue

        roi = hw6[y1:y2, x1:x2, :]
        stats = _roi_redorange_stats_og(roi)
        if (stats["bright_frac"] >= min_bright_frac) and (stats["ro_frac_on_bright"] < ro_min_frac):
            keep[i] = False
            print(f"stem: {stem}, bright_frac: {stats['bright_frac']:.4f}, ro_frac_on_bright: {stats['ro_frac_on_bright']:.4f}")

    return boxes[keep], scores[keep], labels[keep]


class _DSU:
    """Disjoint-set (union-find) for clustering conflicts."""
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]: ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]: self.r[ra] += 1

@torch.no_grad()
def suppress_cross_class_conflicts(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    classes=(1, 2, 3),
    r_px=5,
    area_lo=0.8, area_hi=1.3, 
    iou_min=None,
    per_class_floor=None,
    margin=0.1,
    priority_order=(1, 3, 2),
    return_debug=False
):
    """
    Resolve cross-class duplicates among 'classes' that refer to the SAME physical cell.
    Keeps exactly one per conflict cluster based on:
      1) highest normalized score (score / floor[class]), with margin
      2) if within margin, prefer by 'priority_order'
      3) final tie → highest raw score, then smallest area.

    Same-object test between two boxes of DIFFERENT labels:
      - center distance <= r_px AND
      - area ratio in [area_lo, area_hi] AND
      - (optional) IoU >= iou_min

    Returns filtered (B,S,L) and optional debug list.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels, ([] if return_debug else None)

    dev = boxes.device
    C = set(int(c) for c in classes)
    idx = torch.nonzero(torch.isin(labels, torch.tensor(list(C), device=dev)), as_tuple=False).squeeze(1)
    if idx.numel() <= 1:
        return boxes, scores, labels, ([] if return_debug else None)

    b = boxes[idx]
    s = scores[idx]
    l = labels[idx]

    centers = _centers(b)
    areas = _area(b)

    N = b.size(0)
    dsu = _DSU(N)
    order = torch.argsort(centers[:, 0])
    cx_sorted = centers[order]
    for ii in range(N):
        i = order[ii].item()
        jj = ii + 1
        while jj < N and (cx_sorted[jj, 0] - cx_sorted[ii, 0]) <= r_px:
            j = order[jj].item()
            if l[i] != l[j]:
                if torch.linalg.vector_norm(centers[i] - centers[j]) <= r_px:
                    ai, aj = float(areas[i]), float(areas[j])
                    if ai > 0 and aj > 0:
                        ratio = ai / aj if ai > aj else aj / ai
                        if area_lo <= (1.0 / ratio) <= area_hi:
                            ok = True
                            if iou_min is not None:
                                if float(_iou_pair(b[i], b[j])) < float(iou_min):
                                    ok = False
                            if ok:
                                dsu.union(i, j)
            jj += 1

    clusters = {}
    for i in range(N):
        r = dsu.find(i)
        clusters.setdefault(r, []).append(i)

    keep_mask_local = torch.ones(N, dtype=torch.bool, device=dev)
    debug = []
    pr_rank = {int(c): k for k, c in enumerate(priority_order)}
    eps = 1e-8

    for comp in clusters.values():
        if len(comp) == 1:
            continue


        norm = []
        for i in comp:
            li = int(l[i].item())
            floor = float(per_class_floor.get(li, 1.0)) if per_class_floor else 1.0
            floor = max(floor, 1e-6)
            norm.append(float(s[i].item()) / floor)
        norm = torch.tensor(norm, device=b.device)

        order_local = torch.argsort(norm, descending=True)
        best_idx = comp[order_local[0].item()]
        winner_reason = "norm_score"
        if len(comp) >= 2:
            best_val = float(norm[order_local[0]])
            second_val = float(norm[order_local[1]])
            if best_val < margin + second_val:
                cand = sorted(comp, key=lambda i: pr_rank.get(int(l[i].item()), 999))
                best_idx = cand[0]
                winner_reason = "priority_fallback"

        ties = [i for i in comp if int(l[i].item()) == int(l[best_idx].item())]
        if len(ties) > 1:
            best_idx = max(ties, key=lambda i: float(s[i].item()))
            best_score = float(s[best_idx].item())
            eq = [i for i in ties if abs(float(s[i].item()) - best_score) <= eps]
            if len(eq) > 1:
                best_idx = min(eq, key=lambda i: float(areas[i].item()))
                winner_reason = "area_tiebreak"

        for i in comp:
            if i != best_idx:
                keep_mask_local[i] = False

        if return_debug:
            debug.append({
                "cluster_size": len(comp),
                "kept_global_idx": int(idx[best_idx].item()),
                "kept_label": int(l[best_idx].item()),
                "reason": winner_reason,
                "members_global_idx": [int(idx[i].item()) for i in comp],
                "members_labels": [int(l[i].item()) for i in comp],
            })

    keep_global = torch.ones(boxes.size(0), dtype=torch.bool, device=dev)
    keep_global[idx] = keep_mask_local

    outB, outS, outL = boxes[keep_global], scores[keep_global], labels[keep_global]
    if return_debug:
        return (outB, outS, outL, debug)
    return (outB, outS, outL)

def set_rpn_topn(rpn, pre_train=3000, pre_test=6000, post_train=3000, post_test=2000, nms_thresh=0.85, score_thresh=0.0):
    ok = False
    if hasattr(rpn, "pre_nms_top_n_train"):
        rpn.pre_nms_top_n_train = int(pre_train)
        rpn.pre_nms_top_n_test  = int(pre_test)
        ok = True
    if hasattr(rpn, "post_nms_top_n_train"):
        rpn.post_nms_top_n_train = int(post_train)
        rpn.post_nms_top_n_test  = int(post_test)
        ok = True

    if hasattr(rpn, "pre_nms_top_n") and isinstance(getattr(rpn, "pre_nms_top_n"), dict):
        rpn.pre_nms_top_n["training"] = int(pre_train)
        rpn.pre_nms_top_n["testing"]  = int(pre_test)
        ok = True
    if hasattr(rpn, "post_nms_top_n") and isinstance(getattr(rpn, "post_nms_top_n"), dict):
        rpn.post_nms_top_n["training"] = int(post_train)
        rpn.post_nms_top_n["testing"]  = int(post_test)
        ok = True

    if hasattr(rpn, "_pre_nms_top_n") and isinstance(getattr(rpn, "_pre_nms_top_n"), dict):
        rpn._pre_nms_top_n["training"] = int(pre_train)
        rpn._pre_nms_top_n["testing"]  = int(pre_test)
        ok = True
    if hasattr(rpn, "_post_nms_top_n") and isinstance(getattr(rpn, "_post_nms_top_n"), dict):
        rpn._post_nms_top_n["training"] = int(post_train)
        rpn._post_nms_top_n["testing"]  = int(post_test)
        ok = True

    if hasattr(rpn, "nms_thresh"):
        rpn.nms_thresh = float(nms_thresh)
    if hasattr(rpn, "score_thresh"):
        rpn.score_thresh = float(score_thresh)

    try:
        was_training = rpn.training
        rpn.train()
        tr_pre = rpn.pre_nms_top_n()
        tr_post = rpn.post_nms_top_n()
        rpn.eval()
        te_pre = rpn.pre_nms_top_n()
        te_post = rpn.post_nms_top_n()
        if was_training:
            rpn.train()
        print(f"[RPN] top-N set → train: pre={tr_pre} post={tr_post} | test: pre={te_pre} post={te_post}")
        print(f"[RPN] nms_thresh set to {getattr(rpn,'nms_thresh','?')}, score_thresh={getattr(rpn,'score_thresh','?')}")
    except Exception as e:
        print(f"[RPN] WARNING: could not confirm top-N: {e}")
    if not ok:
        print("[RPN] WARNING: No known fields matched to set pre/post NMS top-N (torchvision API mismatch).")

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
    params_backbone = []
    params_heads = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.body.conv1" in n:
            params_heads.append(p)
        elif n.startswith("backbone.body"):
            params_backbone.append(p)
        else:
            params_heads.append(p)
    return torch.optim.SGD(
        [
            {"params": params_backbone, "lr": base_lr_backbone},
            {"params": params_heads, "lr": lr_heads},
        ],
        momentum=0.9, weight_decay=weight_decay, nesterov=True
    )
    
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
    
def _read_yolo_labels(stem: str):
    p = CFG.LABEL_DIR / f"{stem}_og.txt"
    if not p.exists() or p.stat().st_size == 0:
        return []
    out = []
    for ln in p.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) != 5:
            continue
        cls0 = int(float(parts[0]))
        out.append(cls0 + 1)
    return out
    
def stem_has_colony(stem: str) -> bool:
    return any(c in CFG.COLONY_CLASS_IDS for c in _read_yolo_labels(stem))

def build_weighted_sampler(train_stems: List[str]):
    weights = [CFG.COLONY_OVERSAMPLE_FACTOR if stem_has_colony(s) else 1.0 for s in train_stems]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

@torch.no_grad()
def evaluate_epoch(
    model,
    val_loader,
    epoch: int,
    class_names: List[str],
    *,
    iou_thr: float = 0.50,
    score_thr: float = 0.00,
    device=CFG.DEVICE,
) -> float:
    """
    Eval with your full postprocessing pipeline BEFORE matching.
    Returns micro-F1 and logs per-class P/R/F1.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # classes 1..K
    K = max(1, getattr(CFG, "NUM_CLASSES", len(class_names)) - 1)
    classes = list(range(1, K + 1))
    singles = set(getattr(CFG, "SINGLE_CLASS_IDS", (1,2,3)))
    colonies = set(getattr(CFG, "COLONY_CLASS_IDS", (4,5)))

    per_cls = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}

    for images, targets in val_loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outs = model(images)

        for img_t, out, tgt in zip(images, outs, targets):
            gboxes  = tgt["boxes"].to(device)
            glabels = tgt["labels"].to(device)
            crowd   = tgt.get("crowd_boxes", None)
            crowd   = crowd.to(device) if (crowd is not None and crowd.numel() > 0) else None

            B = out["boxes"].to(device)
            S = out["scores"].to(device)
            L = out["labels"].to(device)

            B, S, L = B.cpu(), S.cpu(), L.cpu()
            gboxes  = gboxes.cpu()
            glabels = glabels.cpu()
            crowd   = crowd.cpu() if crowd is not None else None


            if B.numel() and score_thr > 0:
                m = S >= float(score_thr)
                B, S, L = B[m], S[m], L[m]

            if B.numel():
                H, W = img_t.shape[-2], img_t.shape[-1]
                B = clip_boxes_to_image(B, (H, W))

            
            hw6 = img_t.detach().permute(1,2,0).contiguous().float().cpu().numpy()

            floor = float(getattr(CFG, "SCORE_FLOOR", 0.0))
            class_thr = getattr(CFG, "CLASS_THR", None)
            if B.numel():
                if class_thr:
                    per = torch.tensor([class_thr.get(int(c), floor) for c in L], dtype=S.dtype, device=S.device)
                else:
                    per = torch.full_like(S, floor)
                keep = S >= per
                B, S, L = B[keep], S[keep], L[keep]
                

            if B.numel():
                B, S, L = suppress_cross_class_conflicts(
                    B, S, L,
                    classes=tuple(singles),
                    r_px=15,
                    area_lo=0.4,
                    area_hi=1.8,
                    iou_min=0.35,
                    per_class_floor={1:0.5, 2:0.15, 3:0.7},
                    margin=0.01,
                    priority_order=(1,3,2),
                    return_debug=False
                )
                

            if B.numel():
                B, S, L = parent_kill(
                    B, S, L,
                    tol_px=float(getattr(CFG, "PARENT_TOL_PX", 4.0))
                )
                
            if B.numel():
                B, S, L = size_aware_softnms_per_class(
                    B, S, L,
                    split_area_px=70*70,
                    iou_small=0.6,
                    sigma_small=0.05,
                    iou_big=0.6,
                    sigma_big=0.05,
                    score_floor=0.1
                )
            
            stem = tgt.get("stem", "val")
            if B.numel() and getattr(CFG, "RO_ENABLE", True):
                B,S,L = drop_bright_green(hw6, B, S, L, classes=(1,2,3), min_bright_frac=0.27, ro_min_frac=0.26, stem=stem)

            if (B.numel() > 0) and (crowd is not None) and (crowd.numel() > 0):
                ious_c = box_iou_xyxy(B, crowd)
                keep = (ious_c.max(dim=1).values < float(iou_thr))
                B, S, L = B[keep], S[keep], L[keep]

            for c in classes:
                pm = (L == c)
                gm = (glabels == c)
                p_b, p_s = B[pm], S[pm]
                g_b = gboxes[gm]

                tp_mask, Ng = match_dets_to_gts(p_b, p_s, g_b, iou_thr=float(iou_thr))
                tp = int(tp_mask.sum().item())
                fp = int(p_b.shape[0] - tp)
                fn = int(Ng - tp)

                per_cls[c]["tp"] += tp
                per_cls[c]["fp"] += fp
                per_cls[c]["fn"] += fn

    def _prf1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    per_class_metrics = {}
    for c in classes:
        tp, fp, fn = per_cls[c]["tp"], per_cls[c]["fp"], per_cls[c]["fn"]
        p, r, f1 = _prf1(tp, fp, fn)
        per_class_metrics[c] = (p, r, f1)

    TP = sum(per_cls[c]["tp"] for c in classes)
    FP = sum(per_cls[c]["fp"] for c in classes)
    FN = sum(per_cls[c]["fn"] for c in classes)
    P_micro, R_micro, F1_micro = _prf1(TP, FP, FN)

    print(f"\n[Val+Filters @ IoU={iou_thr:.2f}] Micro P={P_micro:.3f} R={R_micro:.3f} F1={F1_micro:.3f}")
    for c in classes:
        name = class_names[c-1] if (c-1) < len(class_names) else f"class_{c}"
        p, r, f1 = per_class_metrics[c]
        tpc, fpc, fnc = per_cls[c]["tp"], per_cls[c]["fp"], per_cls[c]["fn"]
        print(f"  - {name:>12s}: P={p:.3f} R={r:.3f} F1={f1:.3f} (tp={tpc}, fp={fpc}, fn={fnc})")

    return float(F1_micro)


def train():
    best_map = -float("inf")

    seed_everything(CFG.SEED)
    CFG.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_stems = sorted([p.stem.replace("_og", "") for p in CFG.IMAGE_DIR.glob("*_og.png")])
    assert all_stems, "No *_og.png images found."

    val_count = max(1, int(len(all_stems) * CFG.VAL_FRACTION))
    train_count = len(all_stems) - val_count
    rng = random.Random(CFG.SEED)
    stems = all_stems.copy()
    rng.shuffle(stems)
    train_stems = stems[:train_count]
    val_stems   = stems[train_count:]

    train_steps = max(1, math.ceil(len(train_stems) / CFG.BATCH_SIZE))
    total_iters = CFG.EPOCHS * train_steps

    train_aug = build_train_aug()
    train_ds = PicoAlgaeDataset(train_stems, is_train=True, aug=train_aug, return_stem=False)
    val_ds   = PicoAlgaeDataset(val_stems, is_train=False, aug=None, return_stem=True)

    if CFG.OVERSAMPLE_COLONY:
        sampler = build_weighted_sampler(train_stems)
        train_loader = DataLoader(
            train_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, sampler=sampler,
            num_workers=CFG.WORKERS, pin_memory=True, persistent_workers=True,
            prefetch_factor=4, collate_fn=lambda x: tuple(zip(*x))
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
            num_workers=CFG.WORKERS, pin_memory=True, persistent_workers=True,
            prefetch_factor=4, collate_fn=lambda x: tuple(zip(*x))
        )

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=CFG.WORKERS, pin_memory=True, persistent_workers=True,
        prefetch_factor=4, collate_fn=lambda x: tuple(zip(*x))
    )

    model = build_old_frcnn_6ch_se(
        num_classes=CFG.NUM_CLASSES
    ).to(CFG.DEVICE)

    set_rpn_topn(model.rpn, pre_train=3000, pre_test=6000, post_train=3000, post_test=3000, nms_thresh=0.85, score_thresh=0.0)

    model.roi_heads.score_thresh = 0.0
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 1000

    model.rpn.batch_size_per_image       = 256
    model.roi_heads.batch_size_per_image = 256
    
    model.roi_heads.positive_fraction = 0.33

    model = model.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(model, base_lr_backbone=2e-3, lr_heads=1e-2, weight_decay=1e-4)
    lr_scheduler = WarmupThenCosine(optimizer,
        warmup_iters=min(1000,total_iters//10),
        max_iters=total_iters,
        min_lr=1e-5)

    scaler = GradScaler()
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
            
            scaler.step(optimizer)
            scaler.update()
            did_step = float(scaler.get_scale()) >= prev_scale

            if did_step:
                lr_scheduler.step()

            for k, v in loss_dict.items():
                running[k] += v.item()

        mAp = evaluate_epoch(model, val_loader, epoch,
                             class_names=CFG.CLASS_NAMES, iou_thr=CFG.PR_IOU,
                             score_thr=CFG.PR_SCORE_THRESH, device=CFG.DEVICE)

        if not math.isnan(mAp) and mAp > best_map:
            best_map = mAp
            save_checkpoint(model, optimizer, lr_scheduler, epoch, best_map, CFG.SAVE_BEST_TO)
    print("Training done.")

if __name__ == "__main__":
    train()
