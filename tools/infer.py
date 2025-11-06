import torch
import numpy as np
from typing import Tuple, Dict
import cv2
from pathlib import Path
from torchvision.ops.boxes import clip_boxes_to_image
from tools.utils import _neutralize_scale_bar

@torch.no_grad()
def infer_colonies(model, fused_6chw: np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Input is 6×H×W letterboxed float32 [0..1]; returns LB coords."""
    x = torch.from_numpy(fused_6chw).unsqueeze(0).to(device)
    out = model(x)[0]
    return out["boxes"].detach().cpu(), out["scores"].detach().cpu(), out["labels"].detach().cpu()

def build_warp_from_original(og_path: Path, red_path: Path, S: int) -> torch.Tensor:
    """Create 6×S×S simple-resized tensor for the old model (BGR+BGR in [0..1])."""
    og  = cv2.imread(str(og_path), cv2.IMREAD_COLOR).astype(np.float32)/255.0
    red = cv2.imread(str(red_path), cv2.IMREAD_COLOR).astype(np.float32)/255.0
    og = _neutralize_scale_bar(og, br_w=0.11, br_h=0.03)
    red = _neutralize_scale_bar(red, br_w=0.11, br_h=0.03)
    ogS  = cv2.resize(og,  (S,S), interpolation=cv2.INTER_AREA)
    redS = cv2.resize(red, (S,S), interpolation=cv2.INTER_AREA)
    fused_hw6 = np.concatenate([ogS, redS], axis=-1).astype(np.float32)
    return torch.from_numpy(fused_hw6.transpose(2,0,1)).unsqueeze(0)  # 1×6×S×S

@torch.no_grad()
def infer_singles(model, og_path: Path, red_path: Path, meta: Dict[str,float], S: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Old model: run on S×S warp, map boxes warp→native→letterbox using meta.
    """
    x = build_warp_from_original(og_path, red_path, S).to(device)
    out = model(x)[0]
    bS = out["boxes"].detach().cpu()
    s  = out["scores"].detach().cpu()
    l  = out["labels"].detach().cpu()
    # warp S→native
    H0, W0 = int(meta["orig_h"]), int(meta["orig_w"])
    b_nat = warp_to_native_xyxy(bS, H0=H0, W0=W0, S=S)
    # native→letterbox
    b_lb  = native_to_lb_xyxy(b_nat, meta)
    return b_lb, s, l

def native_to_lb_xyxy(boxes: torch.Tensor, meta: Dict[str,float]) -> torch.Tensor:
    """
    boxes in native/original image coords -> letterbox coords using meta from your cache:
    meta = {scale, pad_l, pad_t, orig_w, orig_h, new_w, new_h}
    """
    if boxes.numel()==0: return boxes
    b = boxes.clone()
    b[:, [0,2]] = b[:, [0,2]] * float(meta["scale"]) + float(meta["pad_l"])
    b[:, [1,3]] = b[:, [1,3]] * float(meta["scale"]) + float(meta["pad_t"])
    Ht, Wt = int(max(meta["new_h"] + 2*meta["pad_t"], 0)), int(max(meta["new_w"] + 2*meta["pad_l"], 0))  # canvas
    return clip_boxes_to_image(b, (int(meta["orig_h"]*meta["scale"]+2*meta["pad_t"]),
                                   int(meta["orig_w"]*meta["scale"]+2*meta["pad_l"])))

def lb_to_native_xyxy(boxes: torch.Tensor, meta: Dict[str,float]) -> torch.Tensor:
    """letterbox → native/original coords"""
    if boxes.numel()==0: return boxes
    b = boxes.clone()
    b[:, [0,2]] = (b[:, [0,2]] - float(meta["pad_l"])) / max(1e-6, float(meta["scale"]))
    b[:, [1,3]] = (b[:, [1,3]] - float(meta["pad_t"])) / max(1e-6, float(meta["scale"]))
    return clip_boxes_to_image(b, (int(meta["orig_h"]), int(meta["orig_w"])))

def warp_to_native_xyxy(boxes_S: torch.Tensor, H0:int, W0:int, S:int) -> torch.Tensor:
    """simple-resize (S×S) → native coords"""
    if boxes_S.numel()==0: return boxes_S
    b = boxes_S.clone()
    b[:, [0,2]] *= (W0 / float(S))
    b[:, [1,3]] *= (H0 / float(S))
    return clip_boxes_to_image(b, (H0, W0))