"""Stage 3 - Embedding: turn each organism crop into a feature vector.

We use DINOv2 (a self-supervised ViT) via HuggingFace ``transformers``. No
labels and no fine-tuning are needed: the pretrained features already separate
shapes/textures well, which is exactly what morphotype clustering needs.

Each crop is background-neutralized (using its SAM mask), padded to a square to
preserve aspect ratio (filaments are long and thin!), resized, and run through
the ViT. The CLS token is used as the object descriptor.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config, resolve_device


def _prep_crop(raw_path: str, mask_path: str, size: int) -> np.ndarray:
    """Load a crop, neutralize background, square-pad, resize -> RGB uint8."""
    bgr = cv2.imread(raw_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if bgr is None:
        raise FileNotFoundError(raw_path)
    if mask is not None and mask.shape == bgr.shape[:2]:
        # Fill background with the crop's own median colour so the ViT focuses
        # on the organism rather than on a hard cut-out edge.
        m = mask > 127
        if m.any():
            bg_color = np.median(bgr[~m], axis=0) if (~m).any() else np.array([128, 128, 128])
            neutral = np.zeros_like(bgr)
            neutral[:] = bg_color
            bgr = np.where(m[:, :, None], bgr, neutral)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Square pad (preserve aspect ratio) then resize.
    h, w = rgb.shape[:2]
    side = max(h, w)
    pad_color = tuple(int(v) for v in np.median(rgb.reshape(-1, 3), axis=0))
    square = np.full((side, side, 3), pad_color, dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    square[y0:y0 + h, x0:x0 + w] = rgb
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


class DinoV2Embedder:
    """Wraps a pretrained DINOv2 model for batched crop embedding."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        e = cfg["embed"]
        self.torch = torch
        self.device = resolve_device(e["device"])
        self.pooling = e["pooling"]
        self.size = int(e["image_size"])
        self.processor = AutoImageProcessor.from_pretrained(e["model_name"])
        self.model = AutoModel.from_pretrained(e["model_name"]).to(self.device).eval()

    def embed_batch(self, imgs: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        last = out.last_hidden_state          # (B, 1 + n_patches, D)
        if self.pooling == "cls":
            feats = last[:, 0]                # CLS token
        else:
            feats = last[:, 1:].mean(dim=1)   # mean over patch tokens
        return feats.cpu().numpy().astype(np.float32)


def run_embedding(cfg: Config) -> Path:
    """Embed every detected crop; save an (N, D) matrix + aligned object ids."""
    objects = pd.read_csv(cfg.outputs_dir / "objects.csv")
    if objects.empty:
        raise RuntimeError("No objects to embed - run detection first.")

    embedder = DinoV2Embedder(cfg)
    bs = int(cfg["embed"]["batch_size"])
    size = embedder.size

    feats: list[np.ndarray] = []
    ids: list[str] = []
    batch_imgs: list[np.ndarray] = []
    batch_ids: list[str] = []

    def flush():
        if batch_imgs:
            feats.append(embedder.embed_batch(batch_imgs))
            ids.extend(batch_ids)
            batch_imgs.clear()
            batch_ids.clear()

    for _, row in tqdm(list(objects.iterrows()), desc="embed[dinov2]"):
        batch_imgs.append(_prep_crop(row["raw_path"], row["mask_path"], size))
        batch_ids.append(row["object_id"])
        if len(batch_imgs) >= bs:
            flush()
    flush()

    matrix = np.concatenate(feats, axis=0)
    np.save(cfg.outputs_dir / "embeddings.npy", matrix)
    pd.DataFrame({"object_id": ids}).to_csv(
        cfg.outputs_dir / "embeddings_index.csv", index=False
    )
    print(f"Embedded {matrix.shape[0]} crops -> {matrix.shape[1]}-d vectors")
    return cfg.outputs_dir / "embeddings.npy"
