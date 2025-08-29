import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import CFG
from .aug import build_train_aug
from .data import CachedPicoAlgaeDataset, compute_class_counts, build_weighted_sampler
from .modeling import get_faster_rcnn_model
from .eval import evaluate_epoch, dump_val_visuals, log_roi_embeddings
from .utils import seed_everything

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

class WarmupThenCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters, max_iters, min_lr=0.0, last_epoch=-1):
        self.warmup_iters = warmup_iters; self.max_iters = max_iters; self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        import numpy as np
        t = self.last_epoch + 1; out = []
        for base_lr in self.base_lrs:
            if t <= self.warmup_iters:
                out.append(base_lr * t / max(1, self.warmup_iters))
            else:
                tt = (t - self.warmup_iters) / max(1, self.max_iters - self.warmup_iters)
                cos = 0.5 * (1 + np.cos(np.pi * tt))
                out.append(self.min_lr + (base_lr - self.min_lr) * cos)
        return out

def build_optimizer(model, base_lr_backbone=2e-3, lr_heads=1e-2, weight_decay=1e-4):
    params_backbone, params_heads = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if "backbone.body.conv1" in n:
            params_heads.append(p)
        elif n.startswith("backbone.body"):
            params_backbone.append(p)
        else:
            params_heads.append(p)
    return torch.optim.SGD(
        [{"params": params_backbone, "lr": base_lr_backbone},
         {"params": params_heads,    "lr": lr_heads}],
        momentum=0.9, weight_decay=weight_decay, nesterov=True
    )

def _set_rpn_topn(rpn, pre_train=1000, pre_test=2000, post_train=500, post_test=1000):
    if hasattr(rpn, "pre_nms_top_n_train"):
        rpn.pre_nms_top_n_train = pre_train; rpn.pre_nms_top_n_test  = pre_test
    elif hasattr(rpn, "pre_nms_top_n") and isinstance(rpn.pre_nms_top_n, dict):
        rpn.pre_nms_top_n["training"] = pre_train; rpn.pre_nms_top_n["testing"]  = pre_test
    if hasattr(rpn, "post_nms_top_n_train"):
        rpn.post_nms_top_n_train = post_train; rpn.post_nms_top_n_test  = post_test
    elif hasattr(rpn, "post_nms_top_n") and isinstance(rpn.post_nms_top_n, dict):
        rpn.post_nms_top_n["training"] = post_train; rpn.post_nms_top_n["testing"]  = post_test

def train():
    writer = SummaryWriter(CFG.LOGDIR)
    best_map = -float("inf")

    seed_everything(CFG.SEED)
    CFG.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_stems = sorted([p.stem.replace("_og","") for p in CFG.IMAGE_DIR.glob("*_og.png")])
    assert all_stems, "No *_og.png images found."

    val_count  = max(1, int(len(all_stems) * CFG.VAL_FRACTION))
    train_stems = all_stems[:-val_count]; val_stems = all_stems[-val_count:]

    tr_counts = compute_class_counts(train_stems); va_counts = compute_class_counts(val_stems)
    print("[Counts] Train:", tr_counts); print("[Counts] Val  :", va_counts)
    for c in range(1, CFG.NUM_CLASSES):
        name = CFG.CLASS_NAMES[c-1] if c-1 < len(CFG.CLASS_NAMES) else f"cls{c}"
        writer.add_scalar(f"sanity/ds_counts/train/{name}", tr_counts.get(c,0), 0)
        writer.add_scalar(f"sanity/ds_counts/val/{name}",   va_counts.get(c,0), 0)

    train_steps = max(1, math.ceil(len(train_stems) / CFG.BATCH_SIZE))
    total_iters = CFG.EPOCHS * train_steps

    train_ds = CachedPicoAlgaeDataset(train_stems, is_train=True,  aug=build_train_aug())
    val_ds   = CachedPicoAlgaeDataset(val_stems,   is_train=False, aug=None)

    if CFG.OVERSAMPLE_COLONY:
        sampler = build_weighted_sampler(train_stems)
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, sampler=sampler,
                                  num_workers=CFG.WORKERS, pin_memory=True, persistent_workers=True,
                                  prefetch_factor=4, collate_fn=lambda x: tuple(zip(*x)))
    else:
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                                  num_workers=CFG.WORKERS, pin_memory=True, persistent_workers=True,
                                  prefetch_factor=4, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=CFG.WORKERS, pin_memory=True, persistent_workers=True,
                            prefetch_factor=4, collate_fn=lambda x: tuple(zip(*x)))

    model = get_faster_rcnn_model(num_classes=CFG.NUM_CLASSES, image_size=(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
                                  backbone=CFG.BACKBONE).to(CFG.DEVICE)
    model.roi_heads.score_thresh = 0.00
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 1000
    _set_rpn_topn(model.rpn, pre_train=1000, pre_test=2000, post_train=500, post_test=1000)
    model.rpn.batch_size_per_image       = 256
    model.roi_heads.batch_size_per_image = 256
    model.roi_heads.detections_per_img   = 600
    model = model.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(model, base_lr_backbone=2e-3, lr_heads=1e-2, weight_decay=1e-4)
    lr_scheduler = WarmupThenCosine(optimizer,
        warmup_iters=min(1000,total_iters//10),
        max_iters=total_iters, min_lr=1e-5)
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
            scaler.step(optimizer)
            scaler.update()
            did_step = float(scaler.get_scale()) >= prev_scale
            if did_step: lr_scheduler.step()

            for k, v in loss_dict.items():
                running[k] += v.item()
            if (i + 1) % CFG.LOG_EVERY == 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
                writer.add_scalar("train/total_loss", loss.item(), global_step)
                for gi, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"train/lr/group{gi}", pg["lr"], global_step)
                global_step += 1

        num_batches = max(1, len(train_loader))
        for k in running:
            writer.add_scalar(f"epoch_avg/{k}", running[k] / num_batches, epoch)
        writer.add_scalar("epoch_avg/total_loss", sum(running.values()) / num_batches, epoch)

        mAp = evaluate_epoch(model, val_loader, writer, epoch,
                             class_names=CFG.CLASS_NAMES, iou_thr=CFG.PR_IOU,
                             score_thr=CFG.PR_SCORE_THRESH, device=CFG.DEVICE)

        if CFG.SAVE_VAL_VIS_EVERY and ((epoch + 1) % CFG.SAVE_VAL_VIS_EVERY == 0):
            dump_val_visuals(model, val_loader, out_dir=CFG.VIS_OUT_DIR, max_images=6, device=CFG.DEVICE)

        if not np.isnan(mAp) and mAp > best_map:
            best_map = mAp
            save_checkpoint(model, optimizer, lr_scheduler, epoch, best_map, CFG.SAVE_BEST_TO)

    writer.flush(); writer.close()
    print("Training done.")
