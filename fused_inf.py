from __future__ import annotations
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2, torch, numpy as np
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import torch.nn as nn
from torch.amp import autocast

from tools.models import load_colony_model, load_single_model
from tools.counting_hook import DenseGate, load_density_model
from tools.utils import (seed_everything,
    letterbox,
    timer,
    iter_stems,
    compose_two_up,
    draw_vis,
    density_gate_by_area,
    )
from tools.infer import infer_singles, infer_colonies
from tools.filters import (
    suppress_cross_class_conflicts,
    parent_kill, size_aware_softnms_per_class,
    drop_bright_green,
    drop_green_dominant_artifacts,
    enforce_colony_rules
)

PALETTE = {
    1: (255,0,  0),
    2: (  0,255,255),
    3: ( 0,0, 255),
    4: (128,128,128),
    5: (255,255,255),
}


singles_path = Path("models/best_frcnn_6ch12_state.pth")
colony_path  = Path("models/fold3_best_state.pth")
density_path = Path("models/best_roi_count_state.pth")
dense_gate_path = Path("models/gate_dense_lr.npz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

seed_everything(42)

@dataclass
class FUSECFG:
    IMAGE_DIR: Path = Path("data/test")
    OUT_DIR:   Path = Path("infer_fused_out")
    VISUALS:   bool = True

    COLONY_CKPT:  Path = colony_path
    SINGLE_CKPT:  Path = singles_path
    OLD_WARP_SZ: int = 2080
    LB_SIZE:     int = 2048

    SINGLE_CLASS_IDS: Tuple[int,...] = (1,2,3)
    COLONY_CLASS_IDS: Tuple[int,...] = (4,5)

    SCORE_FLOOR: float = 0.10
    CLASS_THR: Dict[int,float] = field(default_factory=lambda: {
        1: 0.13, 2: 0.10, 3: 0.30, 4: 0.40, 5: 0.40
    })

CFG = FUSECFG()
CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
if CFG.VISUALS:
    (CFG.OUT_DIR/"vis").mkdir(parents=True, exist_ok=True)

gate = DenseGate(
    npz_path=dense_gate_path,
    rule_params={"edge_thr":0.12, "fill_thr":0.65, "lvar_thr":35.0}
)
density_net, density_dev = load_density_model(density_path, DEVICE)


def run():
    print("[load] new(letterbox colonies) + old(warp singles) ...")
    single_model = load_single_model(CFG.SINGLE_CKPT, DEVICE, CFG.OLD_WARP_SZ)
    colony_model = load_colony_model(CFG.COLONY_CKPT, DEVICE, min_size=CFG.LB_SIZE)

    stems = iter_stems(CFG.IMAGE_DIR)
    rows=[]
    for i, stem in enumerate(stems, 1):
        og_path  = CFG.IMAGE_DIR / f"{stem}_og.png"
        red_path = CFG.IMAGE_DIR / f"{stem}_red.png"

        fused_6chw, meta = letterbox(stem, CFG.IMAGE_DIR, target=CFG.LB_SIZE)
        hw6_lb = fused_6chw.transpose(1,2,0).copy()
        H, W = hw6_lb.shape[:2]

        with timer("colonies"):
            Bn, Sn, Ln = infer_colonies(colony_model, fused_6chw, DEVICE)
        with timer("singles"):
            Bo_lb, So, Lo = infer_singles(single_model, og_path, red_path, meta, CFG.OLD_WARP_SZ, DEVICE)

        keep_old = torch.isin(Lo, torch.tensor(CFG.SINGLE_CLASS_IDS, device=Lo.device))
        keep_new = torch.isin(Ln, torch.tensor(CFG.COLONY_CLASS_IDS, device=Ln.device))
        B = torch.cat([Bo_lb[keep_old], Bn[keep_new]], 0)
        S = torch.cat([So[keep_old],    Sn[keep_new]], 0)
        L = torch.cat([Lo[keep_old],    Ln[keep_new]], 0)

        if B.numel():
            B, S, L = B.cpu(), S.cpu(), L.cpu()
            
        mode, dbg = density_gate_by_area(B, S, L, H=int(H), W=int(W), score_min=0.10)
        print(f"[gate] {stem}: {mode} (n={dbg['n']}")
        
        class_thr = dict(CFG.CLASS_THR)
        score_floor = CFG.SCORE_FLOOR
        
        if mode == "sparse":
            score_floor = max(score_floor, 0.18)
            class_thr[1] = max(class_thr.get(1, 0.0), 0.5)
            class_thr[2] = max(class_thr.get(2, 0.0), 0.5)
            class_thr[3] = max(class_thr.get(3, 0.0), 0.5)

        if B.numel():
            per = torch.tensor([class_thr.get(int(c), score_floor) for c in L], dtype=S.dtype)
            m = S >= per
            B, S, L = B[m], S[m], L[m]
    
        if B.numel():
            B, S, L = suppress_cross_class_conflicts(
                B, S, L,
                classes=(1, 2, 3),
                r_px=15.0,
                area_lo=0.4,
                area_hi=1.8,
                iou_min=0.60,
                per_class_floor=CFG.CLASS_THR,
                margin=0.01,
                priority_order=(1, 3, 2),
                return_debug=False,
            )

        if B.numel():
            B, S, L = parent_kill(B, S, L, tol_px=4.0)

        if B.numel():
            B, S, L = size_aware_softnms_per_class(
                B, S, L,
                split_area_px=70*70,
                iou_small=0.6, sigma_small=0.05,
                iou_big=0.6,    sigma_big=0.05,
                score_floor=CFG.SCORE_FLOOR,
            )


        if B.numel():
            # B, S, L = drop_bright_green(
            #     hw6_lb, B, S, L,
            #     classes=(1, 2,),
            #     min_bright_frac=0.23,
            #     ro_min_frac=0.29,
            #     stem=stem,
            # )
            B, S, L = drop_green_dominant_artifacts(
                hw6_lb, B, S, L,
                classes=(1,2,),
                I_min=0.25,
                g_over_rb_min=1.28,
                exg_min=0.11,
                frac_green_min=0.20,
                ring_px=10,
                dG_bg_min=0.06,
                exg_bg_min=0.04,
                stem=stem,
            )
            
        if B.numel():
            B, S, L, _ = enforce_colony_rules(
                B, S, L,
                colony_classes=(4, 5),
                tol_px=0.0,
                mode="center",
            )

        euk = int((L == 1).sum().item())
        fc  = int((L == 2).sum().item())
        fe  = int((L == 3).sum().item())
        flagged_colony = bool(((L == 4).any().item()) or ((L == 5).any().item()))

        lines = [
            (f"EUK: {euk}", PALETTE[1]),
            (f"FC:  {fc}",  PALETTE[2]),
            (f"FE:  {fe}",  PALETTE[3]),
        ]
        if flagged_colony:
            lines.append(("Colony present", (255, 255, 255)))

        y = 22
        x0 = 12
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thick = 2
        line_h = 22
        
        og  = (hw6_lb[..., 0:3].clip(0, 1) * 255).astype(np.uint8)
        red = (hw6_lb[..., 3:6].clip(0, 1) * 255).astype(np.uint8)
        panel = compose_two_up(og.copy(), red.copy())
        panel = draw_vis(panel, B, S, L, PALETTE, draw_scores=True)
        for txt, color in lines:
            cv2.putText(panel, txt, (x0, y), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(panel, txt, (x0, y), font, scale, color, thick, cv2.LINE_AA)
            y += line_h
        vis_dir = CFG.OUT_DIR / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / f"{stem}.png"), panel)

        if i % 25 == 0:
            print(f"[{i}/{len(stems)}] done")

    csv_path = CFG.OUT_DIR/"counts.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(set().union(*[r.keys() for r in rows])))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[OK] CSV → {csv_path}")
    if CFG.VISUALS: print(f"[OK] Visuals → {CFG.OUT_DIR/'vis'}")

if __name__ == "__main__":
    run()