from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score

from .config import CFG
from .infer import predict_one_with_tta
from .metrics import box_iou_xyxy, boxes_to_centroids_xy, match_dets_to_gts, match_by_centroid
from .utils import _get_class_thr, _get_class_tol, _size_filter_xyxy

# --- probes (RPN proposal counts) ---
def attach_rpn_probe(model):
    rpn = model.rpn; rpn._probe_counts = []
    if not hasattr(rpn, "_orig_filter_proposals"):
        rpn._orig_filter_proposals = rpn.filter_proposals
        def _wrapped(self, proposals, objectness, image_shapes, num_anchors_per_level):
            boxes, scores = self._orig_filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level)
            try: self._probe_counts.extend([int(b.shape[0]) for b in boxes])
            except Exception: pass
            return boxes, scores
        rpn.filter_proposals = type(rpn._orig_filter_proposals)(_wrapped.__code__, {}, "wrapped",
                                                                (_wrapped.__defaults__ or ()), (), rpn._orig_filter_proposals.__closure__)
        rpn.filter_proposals = _wrapped.__get__(rpn, type(rpn))

def pop_rpn_probe_stats(model):
    rpn = model.rpn; counts = getattr(rpn, "_probe_counts", [])
    rpn._probe_counts = []; return counts

@torch.no_grad()
def draw_boxes(img_rgb, boxes, color, text=None):
    import cv2
    img = img_rgb.copy()
    for i,b in enumerate(boxes):
        x1,y1,x2,y2 = [int(v) for v in b.tolist()]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        if text is not None:
            cv2.putText(img, str(text[i]), (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

@torch.no_grad()
def dump_val_visuals(model, val_loader, out_dir="val_vis", max_images=6, device=CFG.DEVICE):
    import os, cv2
    os.makedirs(out_dir, exist_ok=True)
    model.eval(); saved = 0
    for images, targets in val_loader:
        img6 = images[0].to(device)
        out  = predict_one_with_tta(model, img6, device=device, min_sizes=(1600,1800,2048), do_flips=True, fuse="nms")
        B,S,L = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()
        gB,gL  = targets[0]["boxes"].cpu(), targets[0]["labels"].cpu()

        keep = torch.zeros(len(B), dtype=torch.bool)
        for c in torch.unique(L):
            if int(c)==0: continue
            m = (L==c) & (S>=_get_class_thr(int(c)))
            keep |= m
        Bk,Sk,Lk = B[keep], S[keep], L[keep]
        Bk, keep2 = _size_filter_xyxy(Bk, min_side=6, max_side=160)
        Sk, Lk = Sk[keep2], Lk[keep2]

        dXY = boxes_to_centroids_xy(Bk); gXY = boxes_to_centroids_xy(gB)
        tp_flags = torch.zeros(Bk.size(0), dtype=torch.bool)
        for c in torch.unique(Lk):
            if int(c)==0: continue
            mc = (Lk==c); tol_c = _get_class_tol(int(c))
            tp_c, _ = match_by_centroid(dXY[mc], Sk[mc], gXY[gL==c], tol_px=tol_c)
            tp_flags[mc] = tp_c

        img_bgr = (images[0][:3].clamp(0,1).numpy().transpose(1,2,0) * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img1 = draw_boxes(img_rgb, gB, (0,255,0))
        img2 = draw_boxes(img1, Bk[~tp_flags], (255,0,0), text=[f"{s:.2f}" for s in Sk[~tp_flags]])
        img3 = draw_boxes(img2, Bk[tp_flags], (0,128,255), text=[f"{s:.2f}" for s in Sk[tp_flags]])
        cv2.imwrite(f"{out_dir}/vis_{saved:03d}.png", cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
        saved += 1
        if saved >= max_images: break
    print(f"[VIS] Saved {saved} images to {out_dir}")

@torch.no_grad()
def log_roi_embeddings(model, dataloader, writer: SummaryWriter, device, *, max_rois=512, score_thresh=0.2, tag="emb/roi", global_step=0):
    import torch, numpy as np, cv2
    model.eval(); feats=[]; thumbs=[]; meta=[]; total=0
    outs_all = []
    for images, _ in dataloader:
        imgs = [img.to(device) for img in images]
        outs_all.extend(model(imgs))
        if len(outs_all) >= max_rois: break
        for img6, out in zip(images, outs_all[-len(images):]):
            keep = out["scores"] >= score_thresh
            boxes = out["boxes"][keep]; labels = out["labels"][keep]; scores = out["scores"][keep]
            if boxes.numel()==0: continue
            fdict = model.backbone(img6.unsqueeze(0).to(device))
            roi_pool = model.roi_heads.box_roi_pool(fdict, [boxes.to(device)], [img6.shape[-2:]])
            box_head_feats = model.roi_heads.box_head(roi_pool)
            box_head_feats = torch.nn.functional.normalize(box_head_feats, dim=1)

            og_bgr = img6[:3].clamp(0,1).cpu().numpy(); og_rgb = og_bgr[[2,1,0],:,:]; og = np.transpose(og_rgb,(1,2,0))
            H,W = og.shape[:2]
            for feat_vec, b, lab, sc in zip(box_head_feats.cpu(), boxes.cpu(), labels.cpu(), scores.cpu()):
                if total >= max_rois: break
                x1,y1,x2,y2 = [int(v) for v in b.tolist()]
                x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(W-1,x2), min(H-1,y2)
                if x2<=x1 or y2<=y1: continue
                BRW,BRH = 0.18,0.12; sx0,sy0 = int(W*(1-BRW)), int(H*(1-BRH))
                ix0,iy0 = max(x1,sx0), max(y1,sy0); ix1,iy1 = min(x2,W), min(y2,H)
                if max(0,ix1-ix0) * max(0,iy1-iy0) > 0:  # intersects scale bar
                    continue
                crop = og[y1:y2, x1:x2, :]
                if crop.size == 0: continue
                h,w = crop.shape[:2]; scale = 160 / max(h,w)
                new = cv2.resize(crop, (int(w*scale), int(h*scale)))
                pad_h, pad_w = 160-new.shape[0], 160-new.shape[1]
                new = np.pad(new, ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0)
                new = np.transpose(new, (2,0,1))
                thumbs.append(torch.from_numpy(new).unsqueeze(0).float())
                feats.append(feat_vec); meta.append(f"pred_c{int(lab)} s={sc:.2f}")
                total += 1
            if total >= max_rois: break
        if total >= max_rois: break
    if total == 0:
        print("[Embedding] No ROI embeddings collected."); return
    features = torch.stack(feats, 0); label_img = torch.cat(thumbs, 0).clamp(0,1)
    writer.add_embedding(mat=features, metadata=meta, label_img=label_img, tag=tag, global_step=global_step)
    writer.flush()
    print(f"[Embedding] Logged {total} ROI embeddings → {tag} @ step {global_step}")

@torch.no_grad()
def evaluate_epoch(model, val_loader, writer: SummaryWriter, epoch: int,
                   class_names=CFG.CLASS_NAMES, iou_thr=CFG.PR_IOU, score_thr=CFG.PR_SCORE_THRESH, device=CFG.DEVICE):
    model.roi_heads.score_thresh = 0.05
    model.eval()
    attach_rpn_probe(model)

    y_true = defaultdict(list); y_score = defaultdict(list); gt_count = defaultdict(int)
    cen_tp = defaultdict(int);  cen_fp = defaultdict(int);  cen_fn = defaultdict(int)
    det_counts = []
    gt_min_sides, gt_max_sides = [], []
    offs_all = []; offs_per_cls = defaultdict(list)

    # offsets for IoU-TPs (diagnostic)
    def _offsets_for_iou_tps(dB, dS, gB, thr=iou_thr):
        if dB.numel()==0 or gB.numel()==0: return []
        order = torch.argsort(dS, descending=True); dB = dB[order]
        ious = box_iou_xyxy(dB, gB)
        taken = torch.zeros(gB.size(0), dtype=torch.bool, device=dB.device)
        offs = []; dC = boxes_to_centroids_xy(dB); gC = boxes_to_centroids_xy(gB)
        for i in range(dB.size(0)):
            row = ious[i]; 
            if row.numel()==0: break
            best_iou,j = row.max(0); j=int(j)
            if best_iou >= thr and not taken[j]:
                taken[j]=True; offs.append(float(torch.norm(dC[i]-gC[j]).item()))
        return offs

    for images, targets in val_loader:
        outputs = [predict_one_with_tta(model, img.to(device), device=device,
                                        min_sizes=(1600,1800,2048), do_flips=True, fuse="nms")
                   for img in images]
        for out, tgt in zip(outputs, targets):
            det_boxes, det_scores, det_labels = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()
            gt_boxes,  gt_labels  = tgt["boxes"].cpu(), tgt["labels"].cpu()

            det_counts.append(int(det_boxes.shape[0]))
            if gt_boxes.numel() > 0:
                w = (gt_boxes[:,2]-gt_boxes[:,0]).numpy()
                h = (gt_boxes[:,3]-gt_boxes[:,1]).numpy()
                gt_min_sides.extend(np.minimum(w,h).tolist())
                gt_max_sides.extend(np.maximum(w,h).tolist())

            # IoU-based bookkeeping + offsets diag
            for c in range(1, len(class_names)+1):
                m_det = (det_labels == c); m_gt = (gt_labels == c)
                dB, dS = det_boxes[m_det], det_scores[m_det]; gB = gt_boxes[m_gt]
                keep = dS >= float(score_thr); dB, dS = dB[keep], dS[keep]
                tp_flags, num_gt = match_dets_to_gts(dB, dS, gB, iou_thr=iou_thr)
                gt_count[c] += int(num_gt)
                y_true[c].extend(tp_flags.numpy().astype(np.int32).tolist())
                y_score[c].extend(dS.numpy().tolist())
                offs = _offsets_for_iou_tps(dB, dS, gB, thr=iou_thr)
                offs_all.extend(offs); 
                if offs: offs_per_cls[c].extend(offs)

            # Centroid-F1 bookkeeping (per-class tol & thr)
            for c in range(1, len(class_names)+1):
                thr_c = _get_class_thr(c); tol_c = _get_class_tol(c)
                m_det = (det_labels==c) & (det_scores>=thr_c); m_gt = (gt_labels==c)
                dB, dS = det_boxes[m_det], det_scores[m_det]; gB = gt_boxes[m_gt]
                dB, keep_mask = _size_filter_xyxy(dB, min_side=None, max_side=None); dS = dS[keep_mask]
                tp_flags, num_gt = match_by_centroid(boxes_to_centroids_xy(dB), dS,
                                                     boxes_to_centroids_xy(gB), tol_px=tol_c)
                tp = int(tp_flags.sum().item()); fp = int(dB.shape[0]-tp); fn = int(num_gt-tp)
                cen_tp[c]+=tp; cen_fp[c]+=fp; cen_fn[c]+=fn

    # AP/mAP
    aps = {}
    for c in range(1, len(class_names)+1):
        name = class_names[c-1]; t = np.array(y_true[c], dtype=np.int32); s = np.array(y_score[c], dtype=np.float32)
        if gt_count[c] == 0: ap = float("nan")
        elif len(s) == 0:
            ap = 0.0
            writer.add_pr_curve(f"val/PR/{name}", labels=torch.tensor([0,1]), predictions=torch.tensor([0.0,0.0]), global_step=epoch)
        else:
            ap = average_precision_score(t, s)
            writer.add_pr_curve(f"val/PR/{name}", labels=torch.tensor(t), predictions=torch.tensor(s), global_step=epoch)
        aps[c] = ap
        writer.add_scalar(f"val/AP/{name}", 0.0 if np.isnan(ap) else ap, epoch)

    primary_ids = [1,2,3]
    mAP_primary = float(np.nanmean([aps[c] for c in primary_ids]))
    mAP_all     = float(np.nanmean([aps[c] for c in aps]))
    writer.add_scalar("val/mAP@0.5_primary3", mAP_primary, epoch)
    writer.add_scalar("val/mAP@0.5_all6",     mAP_all,     epoch)
    print(f"[Val] Epoch {epoch}: mAP@0.5 primary3={mAP_primary:.4f} | all6={mAP_all:.4f}")

    def _micro(tp, fp, fn):
        p = tp / max(1, tp+fp); r = tp / max(1, tp+fn)
        return 2*p*r / max(1e-8, p+r)

    micro_tp_all = sum(cen_tp.values()); micro_fp_all = sum(cen_fp.values()); micro_fn_all = sum(cen_fn.values())
    micro_tp_pri = sum(cen_tp[c] for c in primary_ids)
    micro_fp_pri = sum(cen_fp[c] for c in primary_ids)
    micro_fn_pri = sum(cen_fn[c] for c in primary_ids)

    mf1_all = _micro(micro_tp_all, micro_fp_all, micro_fn_all)
    mf1_pri = _micro(micro_tp_pri, micro_fp_pri, micro_fn_pri)
    writer.add_scalar("val/F1c_micro_all6@autoTol", mf1_all, epoch)
    writer.add_scalar("val/F1c_micro_primary3@autoTol", mf1_pri, epoch)
    print(f"[Val] Epoch {epoch}: F1c_micro_all6@autoTol = {mf1_all:.4f}  |  F1c_micro_primary3@autoTol = {mf1_pri:.4f}")

    # diag: center offsets (IoU-TPs)
    if len(offs_all) > 0:
        p50,p90,p95 = np.percentile(offs_all, [50,90,95])
        print(f"[Diag] Center offset (IoU-TPs): p50={p50:.1f}px p90={p90:.1f}px p95={p95:.1f}px")
        for c in sorted(offs_per_cls.keys()):
            if len(offs_per_cls[c]) == 0: continue
            pc50,pc90,pc95 = np.percentile(offs_per_cls[c], [50,90,95])
            print(f"[Diag]  class {c}: p50={pc50:.1f} p90={pc90:.1f} p95={pc95:.1f}")

    # sanity: RPN & final detection counts
    rpn_counts = pop_rpn_probe_stats(model)
    if len(rpn_counts)>0:
        rpn_mean = float(np.mean(rpn_counts))
        rpn_p5,rpn_p50,rpn_p95 = [float(np.percentile(rpn_counts,p)) for p in (5,50,95)]
        writer.add_scalar("sanity/rpn_props_per_img_mean", rpn_mean, epoch)
        writer.add_scalar("sanity/rpn_props_per_img_p05",  rpn_p5,   epoch)
        writer.add_scalar("sanity/rpn_props_per_img_p50",  rpn_p50,  epoch)
        writer.add_scalar("sanity/rpn_props_per_img_p95",  rpn_p95,  epoch)
        print(f"[Sanity] RPN proposals/img — mean={rpn_mean:.1f}, p05={rpn_p5:.0f}, p50={rpn_p50:.0f}, p95={rpn_p95:.0f}")
    if len(det_counts)>0:
        det_mean = float(np.mean(det_counts))
        det_p5,det_p50,det_p95 = [float(np.percentile(det_counts,p)) for p in (5,50,95)]
        writer.add_scalar("sanity/final_dets_per_img_mean", det_mean, epoch)
        writer.add_scalar("sanity/final_dets_per_img_p05",  det_p5,   epoch)
        writer.add_scalar("sanity/final_dets_per_img_p50",  det_p50,  epoch)
        writer.add_scalar("sanity/final_dets_per_img_p95",  det_p95,  epoch)
        print(f"[Sanity] Final detections/img — mean={det_mean:.1f}, p05={det_p5:.0f}, p50={det_p50:.0f}, p95={det_p95:.0f}")

    if len(gt_min_sides)>0:
        gmin_p5,gmin_p50,gmin_p95 = [float(np.percentile(gt_min_sides,p)) for p in (5,50,95)]
        gmax_p5,gmax_p50,gmax_p95 = [float(np.percentile(gt_max_sides,p)) for p in (5,50,95)]
        from .config import CFG as _CFG
        _CFG.MIN_SIDE_PX = max(3, int(gmin_p5) - 2)
        _CFG.MAX_SIDE_PX = int(gmax_p95)
        writer.add_scalar("sanity/gt_min_side_px_p05", gmin_p5, epoch)
        writer.add_scalar("sanity/gt_min_side_px_p50", gmin_p50, epoch)
        writer.add_scalar("sanity/gt_min_side_px_p95", gmin_p95, epoch)
        writer.add_scalar("sanity/gt_max_side_px_p05", gmax_p5, epoch)
        writer.add_scalar("sanity/gt_max_side_px_p50", gmax_p50, epoch)
        writer.add_scalar("sanity/gt_max_side_px_p95", gmax_p95, epoch)
        print(f"[Sanity] GT min-side px — p05={gmin_p5:.1f}, p50={gmin_p50:.1f}, p95={gmin_p95:.1f} | "
              f"GT max-side px — p05={gmax_p5:.1f}, p50={gmax_p50:.1f}, p95={gmax_p95:.1f}")

    return mAP_primary
