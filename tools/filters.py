from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import cv2
import torch
import numpy as np

from scipy.optimize import linear_sum_assignment

def _area(b: torch.Tensor) -> torch.Tensor:
    return ((b[:, 2] - b[:, 0]).clamp_min(0) *
            (b[:, 3] - b[:, 1]).clamp_min(0))

def _centers(b: torch.Tensor) -> torch.Tensor:
    return torch.stack([(b[:, 0] + b[:, 2]) * 0.5,
                        (b[:, 1] + b[:, 3]) * 0.5], dim=1)

def _iou_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a, b: (...,4) xyxy
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_a = (a[..., 2] - a[..., 0]).clamp_min(0) * (a[..., 3] - a[..., 1]).clamp_min(0)
    area_b = (b[..., 2] - b[..., 0]).clamp_min(0) * (b[..., 3] - b[..., 1]).clamp_min(0)
    return inter / (area_a + area_b - inter + 1e-6)

def _soft_nms_gaussian(boxes, scores, iou_thr=0.5, sigma=0.5, score_thr=1e-3):
    if boxes.numel()==0: return torch.empty(0, dtype=torch.long)
    idx = torch.arange(boxes.size(0))
    keep=[]
    b=boxes.clone(); s=scores.clone(); idc=idx.clone()
    while s.numel():
        i = torch.argmax(s)
        keep.append(int(idc[i]))
        if s.numel()==1: break
        bi=b[i]; bj=torch.cat([b[:i],b[i+1:]])
        sj=torch.cat([s[:i],s[i+1:]])
        ij=torch.cat([idc[:i],idc[i+1:]])
        # IoU
        x1=torch.maximum(bi[0], bj[:,0]); y1=torch.maximum(bi[1], bj[:,1])
        x2=torch.minimum(bi[2], bj[:,2]); y2=torch.minimum(bi[3], bj[:,3])
        inter=(x2-x1).clamp_min(0)*(y2-y1).clamp_min(0)
        ai=(bi[2]-bi[0]).clamp_min(0)*(bi[3]-bi[1]).clamp_min(0)
        aj=(bj[:,2]-bj[:,0]).clamp_min(0)*(bj[:,3]-bj[:,1]).clamp_min(0)
        iou=inter/(ai+aj-inter+1e-6)
        decay=torch.exp(-(iou*iou)/sigma)
        sj = sj*decay
        m = sj>=score_thr
        b,s,idc = bj[m], sj[m], ij[m]
    return torch.tensor(keep, dtype=torch.long)

def size_aware_softnms_per_class(boxes, scores, labels, split_area_px,
                                 iou_small, sigma_small,
                                 iou_big, sigma_big, score_floor):
    if boxes.numel()==0: return boxes, scores, labels
    A=_area(boxes); outB=[]; outS=[]; outL=[]
    for c in torch.unique(labels):
        m=(labels==c); bc,sc = boxes[m], scores[m]
        if bc.numel()==0: continue
        small=(A[m] <= split_area_px)
        if small.any():
            ks=_soft_nms_gaussian(bc[small], sc[small], iou_thr=iou_small, sigma=sigma_small, score_thr=score_floor)
            outB.append(bc[small][ks]); outS.append(sc[small][ks]); outL.append(torch.full((ks.numel(),), int(c), dtype=torch.long))
        big=~small
        if big.any():
            kb=_soft_nms_gaussian(bc[big], sc[big], iou_thr=iou_big, sigma=sigma_big, score_thr=score_floor)
            outB.append(bc[big][kb]); outS.append(sc[big][kb]); outL.append(torch.full((kb.numel(),), int(c), dtype=torch.long))
    if not outB:
        dev=boxes.device
        return boxes.new_zeros((0,4)), scores.new_zeros((0,)), labels.new_zeros((0,), dtype=torch.long)
    B=torch.cat(outB,0); S=torch.cat(outS,0); L=torch.cat(outL,0)
    return B,S,L


def parent_kill(boxes, scores, labels, tol_px: float = 4.0):
    if boxes.numel() == 0:
        return boxes, scores, labels

    # sort small -> big so children are kept first
    areas = (boxes[:,2]-boxes[:,0]).clamp_min(0) * (boxes[:,3]-boxes[:,1]).clamp_min(0)
    order = torch.argsort(areas, descending=False)
    B, S, L = boxes[order], scores[order], labels[order]

    keep = []
    for i in range(B.size(0)):
        bi = B[i]
        li = int(L[i].item())
        drop = False
        for k in keep:
            # only compare with SAME-CLASS kept boxes
            if int(L[k].item()) != li:
                continue
            bk = B[k]  # smaller, already kept
            if (bk[0] >= bi[0] - tol_px and
                bk[1] >= bi[1] - tol_px and
                bk[2] <= bi[2] + tol_px and
                bk[3] <= bi[3] + tol_px):
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
    keep = torch.ones(labels.shape[0], dtype=torch.bool)

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
    classes=(1, 2, 3),           # which classes to compare (EUK/FC/FE)
    r_px=5,                      # center distance gate (pixels)
    area_lo=0.8, area_hi=1.3,    # area ratio gate
    iou_min=None,                # optional IoU gate (e.g., 0.4) or None
    per_class_floor=None,        # e.g., {1:0.05, 2:0.15, 3:0.05}
    margin=0.1,                  # winner must beat runner-up by 10
    priority_order=(1, 3, 2),    # fallback preference (e.g., EUK > FE > FC)
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

    # Pairwise conflict graph (only across different labels)
    N = b.size(0)
    dsu = _DSU(N)
    # Cheap spatial prefilter: sort by x center, only compare neighbors within r_px
    order = torch.argsort(centers[:, 0])
    cx_sorted = centers[order]
    for ii in range(N):
        i = order[ii].item()
        # walk right while cx diff <= r_px
        jj = ii + 1
        while jj < N and (cx_sorted[jj, 0] - cx_sorted[ii, 0]) <= r_px:
            j = order[jj].item()
            if l[i] != l[j]:
                # quick gates
                # center distance
                if torch.linalg.vector_norm(centers[i] - centers[j]) <= r_px:
                    # area ratio
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

    # Build clusters
    clusters = {}
    for i in range(N):
        r = dsu.find(i)
        clusters.setdefault(r, []).append(i)

    # Decide winners
    keep_mask_local = torch.ones(N, dtype=torch.bool, device=dev)
    debug = []
    pr_rank = {int(c): k for k, c in enumerate(priority_order)}
    eps = 1e-8

    for comp in clusters.values():
        if len(comp) == 1:
            continue  # no conflict

        # Scores normalized by per-class floors (or 1.0 if not provided)
        norm = []
        for i in comp:
            li = int(l[i].item())
            floor = float(per_class_floor.get(li, 1.0)) if per_class_floor else 1.0
            floor = max(floor, 1e-6)
            norm.append(float(s[i].item()) / floor)
        norm = torch.tensor(norm)

        # Best vs second-best by normalized score
        order_local = torch.argsort(norm, descending=True)
        best_idx = comp[order_local[0].item()]
        winner_reason = "norm_score"
        if len(comp) >= 2:
            best_val = float(norm[order_local[0]])
            second_val = float(norm[order_local[1]])
            if best_val < margin + second_val:
                # within margin -> use priority order
                # choose the class with highest priority present
                cand = sorted(comp, key=lambda i: pr_rank.get(int(l[i].item()), 999))
                best_idx = cand[0]
                winner_reason = "priority_fallback"

        # Final tiebreaks inside cluster for identical label & near-identical norm
        ties = [i for i in comp if int(l[i].item()) == int(l[best_idx].item())]
        if len(ties) > 1:
            # pick highest raw score
            best_idx = max(ties, key=lambda i: float(s[i].item()))
            # still a tie? pick smallest area
            best_score = float(s[best_idx].item())
            eq = [i for i in ties if abs(float(s[i].item()) - best_score) <= eps]
            if len(eq) > 1:
                best_idx = min(eq, key=lambda i: float(areas[i].item()))
                winner_reason = "area_tiebreak"

        # Drop all others in the component
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

    # Lift local mask back to full set
    keep_global = torch.ones(boxes.size(0), dtype=torch.bool, device=dev)
    keep_global[idx] = keep_mask_local

    outB, outS, outL = boxes[keep_global], scores[keep_global], labels[keep_global]
    if return_debug:
        return (outB, outS, outL, debug)
    return (outB, outS, outL)

def center_dedup(boxes, scores, labels, r_px=6, area_lo=0.6, area_hi=1.8):
    """
    params:
      boxes, scores, labels: torch.Tensors of detections
        r_px: center distance threshold in pixels
        area_lo, area_hi: area ratio bounds for deduplication
    returns:
      filtered boxes, scores, labels with center-distance deduplication
    """
    if boxes.numel()==0: return boxes, scores, labels
    order=torch.argsort(scores, descending=True)
    B,S,L=boxes[order], scores[order], labels[order]
    keep=[]; areas=_area(B)
    for i in range(B.size(0)):
        bi=B[i]; ai=areas[i]
        cx_i=0.5*(bi[0]+bi[2]); cy_i=0.5*(bi[1]+bi[3])
        drop=False
        for k in keep:
            bk=B[k]; ak=areas[k]
            cx_k=0.5*(bk[0]+bk[2]); cy_k=0.5*(bk[1]+bk[3])
            if torch.hypot(cx_i-cx_k, cy_i-cy_k) <= r_px and L[i]==L[k]:
                ratio=float(ai/max(ak,1e-6))
                if area_lo <= ratio <= area_hi:
                    drop=True; break
        if not drop: keep.append(i)
    keep=torch.tensor(keep, dtype=torch.long)
    return B[keep], S[keep], L[keep]

@torch.no_grad()
def enforce_colony_rules(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    colony_classes=(4, 5),
    tol_px: float = 2.0,
    mode: str = "center",   # kept for API; we use both center & full robustly
    iou_thr: float = 0.30,  # extra safety: kill if IoU with same-type colony ≥ iou_thr
    debug: bool = False,
):
    """
    Inference-only.

    Rules:
      - class 4 (FC colony) removes ONLY class 2 (FC single)
      - class 5 (FE colony) removes ONLY class 3 (FE single)

    A single is killed if ANY of these holds against a same-type colony:
      (A) its center is inside colony (with tol_px)
      (B) its full box is inside colony (with tol_px)
      (C) IoU(single, colony) ≥ iou_thr  (helps when coords are slightly off)

    Returns:
      boxes_f, scores_f, labels_f, colony_count
    """
    if boxes.numel() == 0 or labels.numel() == 0:
        return boxes, scores, labels, 0

    dev = boxes.device
    labels = labels.to(dev)

    # masks
    is_colony = torch.isin(labels, torch.tensor(list(colony_classes), device=dev))
    if not is_colony.any():
        return boxes, scores, labels, 0

    Bc = boxes[is_colony]   # colonies
    Lc = labels[is_colony]
    Bn = boxes[~is_colony]  # non-colonies
    Ln = labels[~is_colony]

    if Bn.numel() == 0:
        return boxes, scores, labels, int(is_colony.sum().item())

    # expand colonies by tolerance
    x1c = (Bc[:, 0] - tol_px).unsqueeze(0)  # [1,C]
    y1c = (Bc[:, 1] - tol_px).unsqueeze(0)
    x2c = (Bc[:, 2] + tol_px).unsqueeze(0)
    y2c = (Bc[:, 3] + tol_px).unsqueeze(0)

    # centers of singles
    cx = ((Bn[:, 0] + Bn[:, 2]) * 0.5).unsqueeze(1)  # [N,1]
    cy = ((Bn[:, 1] + Bn[:, 3]) * 0.5).unsqueeze(1)

    inside_center = (cx >= x1c) & (cx <= x2c) & (cy >= y1c) & (cy <= y2c)  # [N,C]

    # full containment
    x1n = Bn[:, 0].unsqueeze(1); y1n = Bn[:, 1].unsqueeze(1)
    x2n = Bn[:, 2].unsqueeze(1); y2n = Bn[:, 3].unsqueeze(1)
    inside_full = (x1n >= x1c) & (y1n >= y1c) & (x2n <= x2c) & (y2n <= y2c)  # [N,C]

    # pairwise IoU (vectorized)
    def _pair_iou(a, b):
        # a: [N,4], b: [C,4] -> [N,C]
        x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
        y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
        x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
        y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
        inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
        area_a = ((a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0))[:, None]
        area_b = ((b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0))[None, :]
        return inter / (area_a + area_b - inter + 1e-6)

    iou_nc = _pair_iou(Bn, Bc)  # [N,C]

    # class-specific kill map
    kill_map = {4: 2, 5: 3}

    keep_non = torch.ones(Ln.shape[0], dtype=torch.bool, device=dev)
    removed_debug = []

    for col_cls, single_cls in kill_map.items():
        # which colony columns belong to this class?
        col_cols = (Lc == col_cls)  # [C]
        if not col_cols.any():
            continue
        # which non-colony rows are target singles?
        single_rows = (Ln == single_cls)  # [N]
        if not single_rows.any():
            continue

        # restrict matrices to rows/cols
        ic = inside_center[single_rows][:, col_cols]  # [Ns, Cc]
        ifl = inside_full[single_rows][:, col_cols]
        io = iou_nc[single_rows][:, col_cols] >= iou_thr

        kill = (ic | ifl | io).any(dim=1)  # [Ns]
        if kill.any():
            idx_rows = torch.nonzero(single_rows, as_tuple=False).squeeze(1)
            dead = idx_rows[kill]
            keep_non[dead] = False
            if debug:
                removed_debug += [(int(r.item()), int(single_cls)) for r in dead]

    # stitch back
    idx_all = torch.arange(labels.numel(), device=dev)
    idx_non = idx_all[~is_colony]
    idx_col = idx_all[is_colony]
    keep_all = torch.zeros_like(idx_all, dtype=torch.bool, device=dev)
    keep_all[idx_col] = True
    keep_all[idx_non] = keep_non

    out_b, out_s, out_l = boxes[keep_all], scores[keep_all], labels[keep_all]
    if debug:
        # quick one-line summary to help you confirm it’s working
        before_fc = int((labels == 2).sum().item()); before_fe = int((labels == 3).sum().item())
        after_fc  = int((out_l == 2).sum().item());  after_fe  = int((out_l == 3).sum().item())
        print(f"[enforce_colony_rules] colonies={int(is_colony.sum().item())} | "
              f"FC singles: {before_fc}->{after_fc} | FE singles: {before_fe}->{after_fe} | "
              f"removed={len(removed_debug)}")

    return out_b, out_s, out_l, int(is_colony.sum().item())

import numpy as np
import torch

@torch.no_grad()
def drop_green_dominant_artifacts(
    hw6: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    classes=(1, 2, 3),          # singles only
    # Pixel-level gates
    I_min: float = 0.25,        # require some brightness to consider “green dominance”
    g_over_rb_min: float = 1.18,# G must exceed max(R,B) by this ratio for a pixel to count
    exg_min: float = 0.06,      # mean(ExG) >= this; ExG := G - 0.5*(R+B)
    frac_green_min: float = 0.20,# ≥ this fraction of pixels must be green-dominant
    # Local-background contrast gates (annulus around box)
    ring_px: int = 6,           # background ring thickness
    dG_bg_min: float = 0.06,    # mean(G_roi) - mean(G_bg) must be ≥ this to flag as artifact
    exg_bg_min: float = 0.04,   # mean(ExG_roi) - mean(ExG_bg) ≥ this to flag as artifact
    stem: str = "val",
):
    """
    Reject (drop) ROIs that look like greenish false positives on the OG image.

    A ROI is dropped if BOTH:
      (A) It shows green-dominant statistics:
          - mean(ExG) >= exg_min
          - and fraction of “green-dominant” pixels >= frac_green_min
            where pixel is green-dominant if (G>=I_min) and (G / (max(R,B)+1e-6) >= g_over_rb_min)
      (B) Its green dominance exceeds the *local background* in a ring by:
          - mean(G_roi) - mean(G_bg) >= dG_bg_min
          - and mean(ExG_roi) - mean(ExG_bg) >= exg_bg_min

    Returns filtered (boxes, scores, labels).
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    H, W = hw6.shape[:2]
    og_bgr = hw6[..., 0:3].clip(0.0, 1.0)  # B,G,R in [0,1]
    Bch, Gch, Rch = og_bgr[...,0], og_bgr[...,1], og_bgr[...,2]
    ExG = (Gch - 0.5*(Rch + Bch)).astype(np.float32)  # per-pixel ExG

    keep = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)

    for i in range(labels.shape[0]):
        cls = int(labels[i].item())
        if cls not in classes:
            continue

        # ROI bounds
        x1, y1, x2, y2 = [int(round(v)) for v in boxes[i].tolist()]
        x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
        y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
        if x2 <= x1 or y2 <= y1:
            keep[i] = False
            continue

        roi_G   = Gch[y1:y2, x1:x2]
        roi_R   = Rch[y1:y2, x1:x2]
        roi_B   = Bch[y1:y2, x1:x2]
        roi_ExG = ExG[y1:y2, x1:x2]

        if roi_G.size == 0:
            keep[i] = False
            continue

        # --- A) green-dominant stats inside ROI ---
        bright = roi_G >= I_min
        # ratio vs the stronger of R/B
        denom = np.maximum(roi_R, roi_B) + 1e-6
        g_over_rb = roi_G / denom
        green_dom_px = bright & (g_over_rb >= g_over_rb_min)

        frac_green = float(green_dom_px.mean())
        mean_exg   = float(roi_ExG.mean())

        greenish_roi = (frac_green >= frac_green_min) and (mean_exg >= exg_min)
        if not greenish_roi:
            # Not greenish enough → keep
            continue

        # --- B) local background contrast (annulus) ---
        xa1 = max(0, x1 - ring_px); ya1 = max(0, y1 - ring_px)
        xa2 = min(W, x2 + ring_px); ya2 = min(H, y2 + ring_px)

        # background ring mask
        bg_mask = np.ones((ya2 - ya1, xa2 - xa1), dtype=bool)
        bg_mask[(y1 - ya1):(y2 - ya1), (x1 - xa1):(x2 - xa1)] = False

        if bg_mask.sum() <= 0:
            # No ring → fall back to ROI-only decision (already greenish)
            keep[i] = False
            continue

        bg_G   = Gch[ya1:ya2, xa1:xa2][bg_mask]
        bg_ExG = ExG[ya1:ya2, xa1:xa2][bg_mask]

        dG   = float(roi_G.mean()   - bg_G.mean())
        dExG = float(roi_ExG.mean() - bg_ExG.mean())

        # If ROI is clearly greener than the immediately surrounding background, drop it
        if (dG >= dG_bg_min) and (dExG >= exg_bg_min):
            keep[i] = False
        # else keep true

    return boxes[keep], scores[keep], labels[keep]

@torch.no_grad()
def dedup_same_class_by_center(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    r_px: float = 8.0,          # center distance to consider “same object”
    iou_min: float = 0.20,      # or strong overlap means duplicate
    area_lo: float = 0.5,       # area ratio gate (0.5..2.0 ~ similar size)
    area_hi: float = 2.0,
):
    """
    Greedy same-class de-dup: for each class, sort by score desc and keep the first
    box in any r_px / IoU / area-similarity neighborhood; drop the rest.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    dev = boxes.device
    keep = torch.ones(labels.size(0), dtype=torch.bool, device=dev)
    
    def _center(b):
        return 0.5 * (b[:2] + b[2:])

    def _area(b):
        return (b[2]-b[0]).clamp_min(0) * (b[3]-b[1]).clamp_min(0)

    def _iou(a, b):
        x1 = torch.maximum(a[0], b[0]); y1 = torch.maximum(a[1], b[1])
        x2 = torch.minimum(a[2], b[2]); y2 = torch.minimum(a[3], b[3])
        inter = (x2-x1).clamp_min(0) * (y2-y1).clamp_min(0)
        aa = _area(a); ab = _area(b)
        return inter / (aa + ab - inter + 1e-6)

    for c in torch.unique(labels):
        idx = torch.nonzero(labels == c, as_tuple=False).squeeze(1)
        if idx.numel() <= 1:
            continue
        # highest score first
        order = idx[torch.argsort(scores[idx], descending=True)]

        kept = []
        for i in order:
            if not keep[i]:
                continue
            bi = boxes[i]
            ci = _center(bi)
            ai = _area(bi)

            duplicate = False
            for k in kept:
                bk = boxes[k]
                # center proximity
                ck = _center(bk)
                if torch.linalg.vector_norm(ci - ck) <= r_px:
                    # either they overlap well OR are similarly sized
                    iou = _iou(bi, bk)
                    ak = _area(bk)
                    ratio = (ai / (ak + 1e-6)).item()
                    similar_area = (area_lo <= ratio <= area_hi) or (area_lo <= 1/ratio <= area_hi)
                    if (iou >= iou_min) or similar_area:
                        duplicate = True
                        break
            if duplicate:
                keep[i] = False
            else:
                kept.append(i)

    return boxes[keep], scores[keep], labels[keep]
