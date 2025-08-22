# -*- coding: utf-8 -*-
"""
Faster R-CNN (6-channel) training pipeline for picoalgae detection.
- Adds per-channel z-score normalization (configurable)
- Uses small anchors for tiny/faint objects
- Keeps original logic/structure otherwise
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.anchor_utils import AnchorGenerator

import albumentations as A
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.utils.tensorboard import SummaryWriter


# ==============================
# Config
# ==============================

class Config:
    # Data
    IMAGE_DIR: Path = Path("/workspace/data/images")
    LABEL_DIR: Path = Path("/workspace/data/labels")
    IMG_SUFFIX_OG: str = "_og.png"
    IMG_SUFFIX_RED: str = "_red.png"
    LABEL_SUFFIX: str = ".txt"

    # Model / Training
    NUM_CLASSES: int = 7  # 6 classes + 1 background
    BACKBONE: str = "resnet101"
    IMAGE_SIZE: int = 2560  # min_size=max_size here to keep square resize
    LR: float = 1e-4
    EPOCHS: int = 15
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Augmentation
    AUGMENT: bool = False  # preserve your default

    # Normalization
    USE_ZSCORE: bool = True  # <— per-channel z-score normalization

    # RPN / Detection settings for small objects
    ANCHOR_SIZES: Tuple[Tuple[int, ...], ...] = ((8,), (16,), (32,), (64,), (128,))
    ANCHOR_RATIOS: Tuple[Tuple[float, ...], ...] = ((1.0,), (1.0,), (1.0,), (1.0,), (1.0,))
    RPN_PRE_NMS_TOPN_TRAIN: int = 4000
    RPN_POST_NMS_TOPN_TRAIN: int = 2000
    RPN_PRE_NMS_TOPN_TEST: int = 2000
    RPN_POST_NMS_TOPN_TEST: int = 1000
    RPN_NMS_THRESH: float = 0.7         # keep as default-ish
    BOX_NMS_THRESH: float = 0.5         # can try 0.6 if many close blobs
    BOX_DETS_PER_IMG: int = 1000
    BOX_SCORE_THRESH: float = 0.02      # let faint ones through; filter later if needed

    # Logging / Output
    TB_LOGDIR: str = "runs/algae_debug"
    CKPT_PATH: str = "fasterrcnn_6ch_picoalgae.pth"


# ==============================
# Utilities
# ==============================

def zscore_per_channel(img: np.ndarray) -> np.ndarray:
    """
    Per-image, per-channel z-score normalization.
    Expects img in float32 range (0..1), shape (H, W, C).
    """
    m = img.mean(axis=(0, 1), keepdims=True)
    s = img.std(axis=(0, 1), keepdims=True) + 1e-6
    return (img - m) / s


# ==============================
# Dataset
# ==============================

class PicoAlgaeDataset(Dataset):
    """
    Dataloader for paired images (original & red) fused into a 6-channel tensor.
    - Inputs are resized to (imgsz, imgsz)
    - Labels are YOLO (cx, cy, w, h) normalized; converted to Pascal VOC (x1,y1,x2,y2) in pixels
    - Class ids are shifted by +1 to reserve 0 for background
    """

    def __init__(
        self,
        image_pairs: List[Tuple[Path, Path]],
        label_dir: Path,
        imgsz: int = 640,
        augment: bool = False,
    ) -> None:
        self.label_dir = Path(label_dir)
        self.samples: List[Tuple[Tensor, Dict[str, Tensor]]] = []
        self.imgsz = imgsz

        # Albumentations expects (H, W, C) images and pascal_voc boxes
        self.transform = None
        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=10,
                        p=0.25,
                        border_mode=cv2.BORDER_REFLECT,
                    ),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.05),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc", label_fields=["category_ids"], min_visibility=0.0
                ),
            )

        for i, (img1_path, img2_path) in enumerate(image_pairs):
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            if img1 is None or img2 is None:
                print(f"Skipping pair: {img1_path}, {img2_path}")
                continue

            # Normalize to [0, 1] and resize
            img1 = cv2.resize(img1.astype(np.float32) / 255.0, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2.astype(np.float32) / 255.0, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)

            # Optional per-channel z-score (helps faint/small contrast targets)
            if Config.USE_ZSCORE:
                img1 = zscore_per_channel(img1)
                img2 = zscore_per_channel(img2)

            fused = np.concatenate([img1, img2], axis=-1)  # (H, W, 6)
            h, w, _ = fused.shape

            # Load labels (YOLO normalized cx,cy,w,h -> Pascal VOC in pixels)
            label_path = self.label_dir / (img1_path.stem + Config.LABEL_SUFFIX)
            boxes: List[List[float]] = []
            labels: List[int] = []

            if label_path.exists() and label_path.stat().st_size > 0:
                with open(label_path) as f:
                    for line in f:
                        cls, cx, cy, bw, bh = map(float, line.strip().split())
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h

                        # Clip to image
                        x1 = np.clip(x1, 0, w)
                        y1 = np.clip(y1, 0, h)
                        x2 = np.clip(x2, 0, w)
                        y2 = np.clip(y2, 0, h)

                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(int(cls) + 1)  # shift by +1 for background=0

            # Optional augmentation
            boxes_before = len(boxes)
            if self.transform and len(boxes) > 0:
                aug = self.transform(image=fused, bboxes=boxes, category_ids=labels)
                fused = aug["image"]
                boxes = aug["bboxes"]
                labels = aug["category_ids"]

            boxes_after = len(boxes)
            if boxes_after != boxes_before:
                print(f"[{i+1}/{len(image_pairs)}] {img1_path.name}: {boxes_before} -> {boxes_after} boxes after aug")

            if len(boxes) == 0:
                print(f"Skipping {img1_path.name}: all boxes removed by aug")
                continue

            # To tensors
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            fused_chw = np.transpose(fused, (2, 0, 1))  # (6, H, W)
            image_tensor = torch.tensor(fused_chw, dtype=torch.float32)

            target = {"boxes": boxes_t, "labels": labels_t}
            self.samples.append((image_tensor, target))
            print(f"[{i + 1} / {len(image_pairs)}]")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        return self.samples[idx]


# ==============================
# Model
# ==============================

def get_faster_rcnn_model(
    num_classes: int = Config.NUM_CLASSES,
    image_size: Tuple[int, int] = (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    backbone: str = Config.BACKBONE,
) -> FasterRCNN:
    """
    Build Faster R-CNN with an FPN backbone and a 6-channel first conv.
    Adds small anchors + relaxed thresholds for tiny/faint objects.
    """
    fpn = resnet_fpn_backbone(backbone_name=backbone, weights=None)
    old_conv = fpn.body.conv1
    new_conv = torch.nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        # Copy first 3 channels; init extra 3 with mean of old weights for stability
        new_conv.weight[:, :3] = old_conv.weight
        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:] = mean_weight.repeat(1, 3, 1, 1)

    fpn.body.conv1 = new_conv

    # Small anchors & generous proposal counts
    anchor_gen = AnchorGenerator(
        sizes=Config.ANCHOR_SIZES,
        aspect_ratios=Config.ANCHOR_RATIOS,
    )

    model = FasterRCNN(
        backbone=fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        rpn_pre_nms_top_n_train=Config.RPN_PRE_NMS_TOPN_TRAIN,
        rpn_post_nms_top_n_train=Config.RPN_POST_NMS_TOPN_TRAIN,
        rpn_pre_nms_top_n_test=Config.RPN_PRE_NMS_TOPN_TEST,
        rpn_post_nms_top_n_test=Config.RPN_POST_NMS_TOPN_TEST,
        rpn_nms_thresh=Config.RPN_NMS_THRESH,
        box_nms_thresh=Config.BOX_NMS_THRESH,
        box_detections_per_img=Config.BOX_DETS_PER_IMG,
        box_score_thresh=Config.BOX_SCORE_THRESH,
    )

    # Custom transform for 6-channel input
    model.transform = GeneralizedRCNNTransform(
        min_size=image_size[0],
        max_size=image_size[1],
        image_mean=[0.0] * 6,
        image_std=[1.0] * 6,
    )

    print("model loaded")
    return model


# ==============================
# Training
# ==============================

def collate_fn(batch: List[Tuple[Tensor, Dict[str, Tensor]]]) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
    """Standard detection collate: tuple(zip(*batch))."""
    return tuple(zip(*batch))  # type: ignore[return-value]


def train(
    model: FasterRCNN,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = Config.EPOCHS,
    lr: float = Config.LR,
) -> None:
    """
    Train loop.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=Config.TB_LOGDIR)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        nan_batches = 0
        empty_batches = 0

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip if no boxes in batch
            num_boxes = [t["boxes"].shape[0] for t in targets]
            if sum(num_boxes) == 0:
                print(f"Batch {batch_idx}: Skipped (no boxes)")
                empty_batches += 1
                continue

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses):
                print(f"\nBatch {batch_idx}: NaN loss! Skipping batch.")
                print("Loss dict:", loss_dict)
                print("Num boxes:", num_boxes)
                print("Boxes:", [t["boxes"] for t in targets])
                nan_batches += 1
                continue

            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Total Loss: {epoch_loss:.4f}, NaN batches: {nan_batches}, Empty batches: {empty_batches}"
        )

    torch.save(model.state_dict(), Config.CKPT_PATH)
    print(f"Model saved to {Config.CKPT_PATH}")
    writer.close()


# ==============================
# Main
# ==============================

def main() -> None:
    # Build paired list from *_og.png → *_red.png
    image_files = sorted([f for f in Config.IMAGE_DIR.glob(f"*{Config.IMG_SUFFIX_OG}")])
    image_pairs: List[Tuple[Path, Path]] = [
        (f, Config.IMAGE_DIR / f.name.replace(Config.IMG_SUFFIX_OG, Config.IMG_SUFFIX_RED)) for f in image_files
    ]

    dataset = PicoAlgaeDataset(
        image_pairs=image_pairs,
        label_dir=Config.LABEL_DIR,
        imgsz=Config.IMAGE_SIZE,
        augment=Config.AUGMENT,
    )
    print(f"Loaded {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_fn,
    )

    model = get_faster_rcnn_model(
        num_classes=Config.NUM_CLASSES,
        image_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        backbone=Config.BACKBONE,
    )

    device = Config.DEVICE
    train(model, dataloader, device, num_epochs=Config.EPOCHS, lr=Config.LR)


if __name__ == "__main__":
    main()
