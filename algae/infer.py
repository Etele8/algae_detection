from contextlib import contextmanager
import torch
from torch.amp import autocast
from torchvision.ops import nms as tv_nms

from .config import CFG
from .metrics import box_iou_xyxy
from .utils import _get_class_thr, _size_filter_xyxy

@contextmanager
def _override_min_size(model, min_size_value: int):
    orig = model.transform.min_size
    model.transform.min_size = [int(min_size_value)]
    try: yield
    finally: model.transform.min_size = orig

def weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.5, skip_box_thr=0.0):
    if len(boxes) == 1: return boxes[0], scores[0], labels[0]
    B = torch.cat(boxes, 0); S = torch.cat(scores, 0); L = torch.cat(labels, 0)
    order = torch.argsort(S, descending=True); B,S,L = B[order],S[order],L[order]
    keep_boxes, keep_scores, keep_labels = [], [], []
    used = torch.zeros(B.size(0), dtype=torch.bool, device=B.device)
    for i in range(B.size(0)):
        if used[i] or S[i] < skip_box_thr: continue
        same = (L == L[i]) & (~used)
        ious = box_iou_xyxy(B[i:i+1], B[same]).squeeze(0)
        cluster_mask = same.clone(); cluster_mask[same] = ious >= iou_thr
        idxs = torch.where(cluster_mask)[0]; weights = S[idxs]; bb = B[idxs]
        merged = (bb * weights[:, None]).sum(0) / weights.sum().clamp(min=1e-8)
        keep_boxes.append(merged); keep_scores.append(weights.mean()); keep_labels.append(L[i])
        used[idxs] = True
    if keep_boxes:
        return torch.stack(keep_boxes,0), torch.stack(keep_scores,0), torch.stack(keep_labels,0)
    dev = B.device
    return (torch.empty((0,4), device=dev), torch.empty((0,), device=dev), torch.empty((0,), dtype=torch.int64, device=dev))

@torch.no_grad()
def fuse_tta_by_nms(boxes, scores, labels, iou=0.60, topk=1000):
    if len(boxes) == 1: return boxes[0], scores[0], labels[0]
    B = torch.cat(boxes,0); S = torch.cat(scores,0); L = torch.cat(labels,0)
    keepB, keepS, keepL = [], [], []
    for c in torch.unique(L):
        c = int(c)
        if c == 0: continue
        m = (L==c); b,s = B[m], S[m]
        if b.numel()==0: continue
        keep = tv_nms(b, s, iou)
        keepB.append(b[keep]); keepS.append(s[keep])
        keepL.append(torch.full((keep.numel(),), c, dtype=torch.int64, device=b.device))
    if keepB:
        Bc = torch.cat(keepB,0); Sc = torch.cat(keepS,0); Lc = torch.cat(keepL,0)
        order = torch.argsort(Sc, descending=True)[:topk]
        return Bc[order], Sc[order], Lc[order]
    dev = B.device
    return (torch.empty((0,4), device=dev), torch.empty((0,), device=dev), torch.empty((0,), dtype=torch.int64, device=dev))

@torch.no_grad()
def predict_one_with_tta(model, img6: torch.Tensor, device, min_sizes=(1600,1800,2048), do_flips=True, fuse="nms"):
    _, H, W = img6.shape
    variants = []
    def run(min_size, flip=None):
        with _override_min_size(model, min_size):
            x = img6
            if flip == "h": x = torch.flip(x, dims=[2])
            elif flip == "v": x = torch.flip(x, dims=[1])
            with autocast("cuda"):
                out = model([x.to(device)])[0]
            b, s, l = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()
            if flip == "h": b[:,[0,2]] = W - b[:,[2,0]]
            elif flip == "v": b[:,[1,3]] = H - b[:,[3,1]]
            variants.append((b,s,l))
    for ms in sorted(set(min_sizes)): run(ms, None)
    if do_flips:
        run(max(min_sizes), "h"); run(max(min_sizes), "v")
    boxes, scores, labels = zip(*variants)
    if fuse == "nms":
        mb, ms, ml = fuse_tta_by_nms(list(boxes), list(scores), list(labels), iou=0.60, topk=1000)
        return {"boxes": mb, "scores": ms, "labels": ml}
    if fuse == "wbf":
        mb, ms, ml = weighted_boxes_fusion(list(boxes), list(scores), list(labels), iou_thr=0.55, skip_box_thr=0.05)
        return {"boxes": mb, "scores": ms, "labels": ml}
    B = torch.cat(boxes); S = torch.cat(scores); L = torch.cat(labels)
    keep = tv_nms(B,S,0.60)
    return {"boxes": B[keep], "scores": S[keep], "labels": L[keep]}

@torch.no_grad()
def predict_filtered(model, img6: torch.Tensor, device=CFG.DEVICE):
    out = predict_one_with_tta(model, img6.to(device), device=device, fuse="nms")
    B,S,L = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()
    keep = torch.zeros(len(B), dtype=torch.bool)
    for c in torch.unique(L):
        if int(c)==0: continue
        thr = _get_class_thr(int(c))
        keep |= (L==c) & (S>=thr)
    B,S,L = B[keep], S[keep], L[keep]
    B,k2 = _size_filter_xyxy(B, min_side=None, max_side=None)
    S,L = S[k2], L[k2]
    return {"boxes": B, "scores": S, "labels": L}
