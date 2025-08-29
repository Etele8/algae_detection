import torch
from collections import defaultdict

def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[...,0] * wh[...,1]
    area_a = (a[:,2]-a[:,0])*(a[:,3]-a[:,1])
    area_b = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    union = area_a[:,None] + area_b[None,:] - inter
    return inter / union.clamp(min=1e-6)

def boxes_to_centroids_xy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0: return boxes.new_zeros((0,2))
    cx = 0.5*(boxes[:,0]+boxes[:,2]); cy = 0.5*(boxes[:,1]+boxes[:,3])
    return torch.stack([cx,cy], dim=1)

def match_dets_to_gts(dets, det_scores, gts, iou_thr=0.5):
    Nd = dets.shape[0]; Ng = gts.shape[0]
    if Nd == 0: return torch.zeros(0, dtype=torch.bool), Ng
    order = torch.argsort(det_scores, descending=True)
    dets = dets[order]; det_scores = det_scores[order]
    ious = box_iou_xyxy(dets, gts) if Ng>0 else dets.new_zeros((Nd,0))
    gt_taken = torch.zeros(Ng, dtype=torch.bool, device=dets.device)
    tp = torch.zeros(Nd, dtype=torch.bool, device=dets.device)
    for i in range(Nd):
        if Ng == 0: break
        row = ious[i]
        if row.numel()==0: continue
        best_iou, best_j = (row.max(), int(row.argmax()))
        if best_iou >= iou_thr and not gt_taken[best_j]:
            tp[i] = True; gt_taken[best_j] = True
    inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device=order.device)
    return tp[inv].cpu(), Ng

def match_by_centroid(dets_xy: torch.Tensor, det_scores: torch.Tensor, gts_xy: torch.Tensor, tol_px: float):
    Nd = dets_xy.shape[0]; Ng = gts_xy.shape[0]
    if Nd == 0: return torch.zeros(0, dtype=torch.bool), Ng
    if Ng == 0: return torch.zeros(Nd, dtype=torch.bool), 0
    order = torch.argsort(det_scores, descending=True)
    dets_xy = dets_xy[order]
    gt_taken = torch.zeros(Ng, dtype=torch.bool, device=dets_xy.device)
    tp = torch.zeros(Nd, dtype=torch.bool, device=dets_xy.device)
    for i in range(Nd):
        d = dets_xy[i].unsqueeze(0)
        diffs = gts_xy - d
        dist = torch.sqrt((diffs**2).sum(dim=1))
        if dist.numel()==0: continue
        best_dist, best_j = dist.min(dim=0)
        if best_dist.item() <= tol_px and not gt_taken[best_j]:
            tp[i] = True; gt_taken[best_j] = True
    inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device=order.device)
    return tp[inv].cpu(), Ng
