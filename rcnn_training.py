import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

from sklearn.metrics import precision_recall_curve, average_precision_score

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.ops import box_iou
from torchvision.ops import MultiScaleRoIAlign
import torchvision.transforms.functional as TF


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

    # RPN proposal counts
    RPN_PRE_NMS_TRAIN: int = 4000
    RPN_POST_NMS_TRAIN: int = 2000
    RPN_PRE_NMS_TEST: int = 2000
    RPN_POST_NMS_TEST: int = 1000

    # tensorboard
    LOGDIR = "runs/algae"
    LOG_EVERY = 50          # batches
    PR_IOU = 0.1            # IoU to match detections to GT
    PR_SCORE_THRESH = 0.00  # keep all preds for PR curves
    CLASS_NAMES = ["EUK", "FC", "FE", "EUK colony", "FC colony", "FE colony"]
    
CFG = Config()


# =========================
# tesnorboard helpers
# =========================

@torch.no_grad()
def log_image_embeddings_backbone(
    model,
    dataloader,
    writer: SummaryWriter,
    device,
    *,
    max_images: int = 5,
    tag: str = "embeddings/image_backbone",
    global_step: int = 0,
):
    """
    Logs 1 embedding per image using GAP over the lowest-level FPN feature (P2/'0').
    Assumes your dataset returns (6ch_tensor, target) where 6ch = [OG(3), RED(3)] in [0..1].
    Thumbnails show the OG RGB.
    """
    model.eval()
    feats = []
    thumbs = []
    meta = []
    n_collected = 0

    for images, targets in dataloader:
        # images is a tuple/list per custom collate_fn; keep 1 image at a time to keep it simple
        for img6, tgt in zip(images, targets):
            if n_collected >= max_images:
                break
            x = img6.unsqueeze(0).to(device)                 # [1, 6, H, W]
            # Run backbone only
            with torch.autocast(device.type if device.type != "cpu" else "cpu", enabled=False):
                feat_dict = model.backbone(x)                 # dict: {'0','1','2','3'} -> [1, C, h, w]
            # Take finest map '0' and global-average-pool to a vector
            f = feat_dict['0']                                # [1, C, h, w]
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)       # [1, C]
            feats.append(f.squeeze(0).cpu())

            # thumbnail from OG channels only (first 3), clamp to [0,1]
            og = img6[:3].clamp(0, 1).cpu()                  # [3,H,W]
            # downsample for projector thumbnails (e.g., 160x160)
            H, W = og.shape[1], og.shape[2]
            scale = 160 / max(H, W)
            og_small = TF.resize(og, [int(H*scale), int(W*scale)])
            # pad to a square for nicer grids
            pad_h = 160 - og_small.shape[1]
            pad_w = 160 - og_small.shape[2]
            og_small = TF.pad(og_small, [0,0,pad_w,pad_h])   # [3,160,160]
            thumbs.append(og_small.unsqueeze(0))             # [1,3,160,160]

            # metadata (e.g., count of boxes by class)
            if "labels" in tgt and tgt["labels"].numel() > 0:
                unique, counts = torch.unique(tgt["labels"], return_counts=True)
                summary = ", ".join([f"c{int(u.item())}:{int(c.item())}" for u,c in zip(unique,counts)])
            else:
                summary = "no_boxes"
            meta.append(summary)

            n_collected += 1
        if n_collected >= max_images:
            break

    if n_collected == 0:
        print("[Embedding] No samples found to log.")
        return

    features = torch.stack(feats, dim=0)                 # [N, D]
    label_img = torch.cat(thumbs, dim=0)                 # [N, 3, 160, 160]
    metadata = meta                                      # list[str] of length N

    writer.add_embedding(
        mat=features,
        metadata=metadata,
        label_img=label_img,
        tag=tag,
        global_step=global_step,
    )
    writer.flush()
    writer.close()
    print(f"[Embedding] Logged {len(metadata)} image embeddings → {tag} @ step {global_step}")


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
            og = img6[:3].clamp(0,1).cpu().numpy()  # [3,H,W]
            og = np.transpose(og, (1,2,0))          # HWC for cv2/np
            H, W = og.shape[:2]
            for feat_vec, b, lab, sc in zip(box_head_feats.cpu(), boxes.cpu(), labels.cpu(), scores.cpu()):
                if total >= max_rois:
                    break
                x1,y1,x2,y2 = [int(v) for v in b.tolist()]
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(W-1,x2), min(H-1,y2)
                if x2<=x1 or y2<=y1:
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

import numpy as np
import cv2
from pathlib import Path
import os

def load_or_create_cached_fused(stem: str) -> np.ndarray:
    """
    Returns fused (6,H,W) float32 from cache if fresh, else recomputes and caches.
    Cache is a compressed .npz with keys: fused, imgsz, zscore, src_mtime
    """
    ogp  = CFG.IMAGE_DIR / f"{stem}_og.png"
    redp = CFG.IMAGE_DIR / f"{stem}_red.png"
    assert ogp.exists() and redp.exists(), f"Missing pair for {stem}"

    # ⬇️ ensure .npz suffix to avoid ambiguity
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
            # Ensure it's an NPZ (has .files) and has expected keys
            if hasattr(npz, "files") and {"fused", "imgsz", "zscore", "src_mtime"} <= set(npz.files):
                if (int(npz["imgsz"]) == CFG.IMAGE_SIZE and
                    int(npz["zscore"]) == (1 if CFG.USE_ZSCORE else 0) and
                    float(npz["src_mtime"]) >= src_mtime):
                    return npz["fused"]
            # else: fall through to rebuild
        except Exception:
            # corrupted/stale -> remove and rebuild
            try:
                cache_path.unlink()
            except OSError:
                pass

    # build cache
    og  = cv2.imread(str(ogp))  # BGR uint8
    red = cv2.imread(str(redp))
    if og is None or red is None:
        raise FileNotFoundError(f"Read failed for {stem}")

    og  = cv2.resize(og,  (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    red = cv2.resize(red, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    if CFG.USE_ZSCORE:
        og  = zscore(og)
        red = zscore(red)

    # BGR+BGR → (H,W,6) → (6,H,W)
    fused = np.concatenate([og, red], axis=-1)
    fused = np.transpose(fused, (2, 0, 1)).astype(np.float32)

    _save_npz(fused)
    return fused


# =========================
# Dataset (cached fused)
# =========================
class CachedPicoAlgaeDataset(Dataset):
    def __init__(self, stems: List[str]):
        self.stems = list(stems)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        fused = load_or_create_cached_fused(stem)      # (6,H,W) float32
        image = torch.from_numpy(fused)                # torch.float32

        # labels
        lblp = CFG.LABEL_DIR / f"{stem}_og.txt"
        W = H = CFG.IMAGE_SIZE
        boxes, labels = [], []
        if lblp.exists() and lblp.stat().st_size > 0:
            for ln in lblp.read_text().splitlines():
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue
                c, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw/2) * W
                y1 = (cy - bh/2) * H
                x2 = (cx + bw/2) * W
                y2 = (cy + bh/2) * H
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(c) + 1)  # 1..N (0 is background)
        if len(boxes) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t  = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}
        return image, target


# =========================
# Model (6-channel conv1)
# =========================
def get_faster_rcnn_model(num_classes=7, image_size=(1024, 1024), backbone='resnet101') -> FasterRCNN:
    backbone_fpn = resnet_fpn_backbone(backbone_name=backbone, weights="DEFAULT")  # training from scratch (or load later)
    old_conv = backbone_fpn.body.conv1
    new_conv = torch.nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        # copy RGB weights into BGR+BGR slots
        new_conv.weight[:, :3] = old_conv.weight
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)    # avg of RGB
        new_conv.weight[:, 3:] = mean_w.repeat(1, 3, 1, 1)

    backbone_fpn.body.conv1 = new_conv

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=4
    )
    
   

    model = FasterRCNN(
        backbone=backbone_fpn,
        num_classes=num_classes,
        # rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    # Match training/eval transforms (no z-score here)
    model.transform = GeneralizedRCNNTransform(
        min_size=image_size[0],
        max_size=image_size[1],
        image_mean=[0.0] * 6,
        image_std=[1.0] * 6,
    )

    # RPN proposal counts
    # model.rpn.pre_nms_top_n = lambda training: CFG.RPN_PRE_NMS_TRAIN if training else CFG.RPN_PRE_NMS_TEST
    # model.rpn.post_nms_top_n = lambda training: CFG.RPN_POST_NMS_TRAIN if training else CFG.RPN_PRE_NMS_TEST

    return model


# =========================
# Metrics
# =========================
@torch.no_grad()
def evaluate_epoch(model, val_loader, writer: SummaryWriter, epoch: int,
                   class_names=CFG.CLASS_NAMES, iou_thr=CFG.PR_IOU, score_thr=CFG.PR_SCORE_THRESH, device=CFG.DEVICE):
    model.eval()
    # Collect per-class data
    # y_true_scores[c] -> list of (is_tp:0/1, score:float)
    y_true = defaultdict(list)
    y_score = defaultdict(list)
    gt_count = defaultdict(int)

    for images, targets in val_loader:
        imgs = [im.to(device) for im in images]
        outputs = model(imgs)  # post-NMS predictions from torchvision

        for out, tgt in zip(outputs, targets):
            # move to cpu for bookkeeping
            det_boxes = out["boxes"].cpu()
            det_scores = out["scores"].cpu()
            det_labels = out["labels"].cpu()      # 1..6 if you used +1 shift; adjust if needed
            gt_boxes = tgt["boxes"].cpu()
            gt_labels = tgt["labels"].cpu()

            # iterate over classes
            for c in range(1, len(class_names)+1):   # assuming 1..6 in your pipeline
                det_mask = det_labels == c
                gt_mask  = gt_labels  == c

                det_b = det_boxes[det_mask]
                det_s = det_scores[det_mask]
                gt_b  = gt_boxes[gt_mask]

                # keep all scores (score_thr can be >0 if you want)
                keep = det_s >= score_thr
                det_b = det_b[keep]
                det_s = det_s[keep]

                tp_flags, num_gt = match_dets_to_gts(det_b, det_s, gt_b, iou_thr=iou_thr)
                gt_count[c] += int(num_gt)

                # append detections: 1 for TP, 0 for FP
                y_true[c].extend(tp_flags.numpy().astype(np.int32).tolist())
                y_score[c].extend(det_s.numpy().tolist())

    # Compute per-class PR curves and AP
    aps = []
    for c in range(1, len(class_names)+1):
        name = class_names[c-1]
        t = np.array(y_true[c], dtype=np.int32)
        s = np.array(y_score[c], dtype=np.float32)

        if gt_count[c] == 0:
            # no GT for this class in val set
            writer.add_scalar(f"val/AP/{name}", float('nan'), epoch)
            continue

        if len(s) == 0:
            # no predictions for this class
            writer.add_scalar(f"val/AP/{name}", 0.0, epoch)
            # log a flat PR curve (all zeros)
            writer.add_pr_curve(f"val/PR/{name}",
                                labels=torch.tensor([0, 1]),
                                predictions=torch.tensor([0.0, 0.0]),
                                global_step=epoch)
            aps.append(0.0)
            continue

        # AP (area under PR curve)
        ap = average_precision_score(t, s)  # sklearn treats 1=positive
        aps.append(ap)
        writer.add_scalar(f"val/AP/{name}", ap, epoch)

        # PR curve
        precision, recall, _ = precision_recall_curve(t, s)
        # TensorBoard expects tensors
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

    # mAP over classes with GT
    if aps:
        mAP = float(np.nanmean(aps))
        writer.add_scalar("val/mAP@0.5", mAP, epoch)
        print(f"[Val] Epoch {epoch}: mAP@0.5 = {mAP:.4f}")
    else:
        print(f"[Val] Epoch {epoch}: no classes with GT found.")


# =========================
# Training
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train():
    writer = SummaryWriter(CFG.LOGDIR)
    
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

    train_ds = CachedPicoAlgaeDataset(train_stems)
    val_ds   = CachedPicoAlgaeDataset(val_stems)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.WORKERS, pin_memory=True, collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.WORKERS, pin_memory=True, collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_faster_rcnn_model(
        num_classes=CFG.NUM_CLASSES,
        image_size=(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
        backbone=CFG.BACKBONE
    ).to(CFG.DEVICE)

    optimizer = SGD(
        model.parameters(),
        lr=CFG.LR,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    lr_scheduler = StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    best_f1 = -1.0
    
    log_image_embeddings_backbone(model, val_loader, writer, CFG.DEVICE, max_images=5, tag="emb/img", global_step=0)
    log_roi_embeddings(model, val_loader, writer, CFG.DEVICE, max_rois=400, tag="emb/roi", global_step=0)
    
    global_step = 0
    for epoch in range(CFG.EPOCHS):
        model.train()
        running = defaultdict(float)

        for i, (images, targets) in enumerate(train_loader, 0):
            images = [img.to(CFG.DEVICE) for img in images]
            targets = [{k: v.to(CFG.DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            
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

        lr_scheduler.step()
        # epoch-averaged loss (nice and smooth)
        num_batches = max(1, len(train_loader))
        for k in running:
            writer.add_scalar(f"epoch_avg/{k}", running[k] / num_batches, epoch)
        writer.add_scalar("epoch_avg/total_loss", sum(running.values()) / num_batches, epoch)

        # validation with PR curves and AP
        evaluate_epoch(model, val_loader, writer, epoch, class_names=CFG.CLASS_NAMES, iou_thr=CFG.PR_IOU, score_thr=CFG.PR_SCORE_THRESH, device=CFG.DEVICE)

    writer.flush()
    writer.close()
    print("Training done.")


if __name__ == "__main__":
    train()