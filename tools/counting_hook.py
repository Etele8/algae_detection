import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from pathlib import Path

TYPE_MAP = {
    4: 0,
    5: 1,
}

def _og_gray_u8_from_roi(roi_hw6: np.ndarray) -> np.ndarray:
    og = (roi_hw6[..., 0:3].clip(0,1) * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def _edge_density_scharr(gray_u8: np.ndarray, mag_rel_thresh: float = 0.12) -> float:
    gx = cv2.Scharr(gray_u8, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray_u8, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mmax = float(mag.max()) if mag.size else 0.0
    if mmax <= 1e-6: return 0.0
    return float((mag >= (mag_rel_thresh * mmax)).mean())

def _fill_ratio_otsu(gray_u8: np.ndarray, invert: bool = False) -> float:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, th = cv2.threshold(gray_u8, 0, 255, mode + cv2.THRESH_OTSU)
    return float((th > 0).mean())

def _laplacian_var(gray_u8: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    return float(lap.var())

def roi_features_og(roi_hw6: np.ndarray, edge_rel_thr: float) -> np.ndarray:
    gray = _og_gray_u8_from_roi(roi_hw6)
    ed   = _edge_density_scharr(gray, mag_rel_thresh=edge_rel_thr)
    fill = _fill_ratio_otsu(gray, invert=False)
    lvar = _laplacian_var(gray)
    m = float(gray.mean()); s = float(gray.std())
    return np.array([ed, fill, lvar, 1.0, m, s], dtype=np.float32)

class CountHead(nn.Module):
    def __init__(self, in_ch=6, base=32, num_types=1, use_class_emb=True, emb_dim=8):
        super().__init__()
        self.use_class_emb = use_class_emb and (num_types > 1)
        self.emb_dim = emb_dim if self.use_class_emb else 0

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*4, base*4, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        in_feat = base*4 + self.emb_dim + 1
        if self.use_class_emb:
            self.class_emb = nn.Embedding(num_types, self.emb_dim)

        self.mu = nn.Sequential(
            nn.Linear(in_feat, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.r = nn.Sequential(
            nn.Linear(in_feat, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.mu[-1].bias.fill_(0.1)

    def forward(self, x: torch.Tensor, type_idx: torch.Tensor, box_area_px: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        z = self.gap(e3).flatten(1)

        feats = [z]
        if self.use_class_emb:
            ce = self.class_emb(type_idx)
            feats.append(ce)
        log_area = torch.log1p(box_area_px).unsqueeze(1)
        feats.append(log_area)

        h = torch.cat(feats, dim=1)


        mu = F.softplus(self.mu(h)) + 1e-6
        r  = F.softplus(self.r(h)) + 1e-6
        return mu.squeeze(1), r.squeeze(1)
            
def load_density_model(path, device) -> tuple[torch.nn.Module, torch.device]:
    device = torch.device(device)
    net = CountHead(in_ch=6, base=32, num_types=2,
                    use_class_emb=True, emb_dim=8).to(device)
    if path and path.exists():
        ck = torch.load(path, map_location="cpu")
        state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"[Density] Loaded {path.name}: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[Density] WARNING: checkpoint not found — using random init.")
    net.eval()
    return net, device

class DenseGate:
    def __init__(self, npz_path: Path | None, rule_params: dict):
        self.mode = "rule" if (npz_path is None or not Path(npz_path).exists()) else "lr"
        self.edge_thr = rule_params.get("edge_thr", 0.12)
        self.fill_thr = rule_params.get("fill_thr", 0.65)
        self.lvar_thr = rule_params.get("lvar_thr", 35.0)
        if self.mode == "lr":
            pack = np.load(str(npz_path), allow_pickle=True)
            self.mu  = pack["scaler_mean"]
            self.sd  = pack["scaler_scale"]
            self.coef = pack["coef"]     
            self.bias = pack["intercept"]
            cfg = pack["config"].item() if isinstance(pack["config"], np.ndarray) else pack["config"]
            self.edge_thr = float(cfg.get("EDGE_REL_THR", self.edge_thr))

    def predict_dense(self, roi_hw6: np.ndarray) -> tuple[bool, float, dict]:
        """
        Returns: (is_dense, prob_dense, feats_dict)
        """
        feats = roi_features_og(roi_hw6, self.edge_thr)
        if self.mode == "lr":
            x = (feats - self.mu) / np.maximum(self.sd, 1e-6)
            z = float(x @ self.coef.reshape(-1) + self.bias.reshape(()))
            p_dense = 1.0 / (1.0 + np.exp(-z))
            return (p_dense >= 0.5), p_dense, {
                "edge": float(feats[0]), "fill": float(feats[1]), "lap_var": float(feats[2]),
                "og_mean": float(feats[4]), "og_std": float(feats[5])
            }
        else:
            # rule fallback
            ed, fill, lvar = float(feats[0]), float(feats[1]), float(feats[2])
            is_dense = (ed < self.edge_thr) and (fill > self.fill_thr) and (lvar < self.lvar_thr)
            return is_dense, float(is_dense), {
                "edge": ed, "fill": fill, "lap_var": lvar,
                "og_mean": float(feats[4]), "og_std": float(feats[5])
            }
            
def count_basic_method(roi_hw6: np.ndarray, cls_id: int,
                       density_net: torch.nn.Module, density_dev: torch.device) -> int:
    """
    Colony cell counting with your ROI count head (scalar μ per ROI).
    - roi_hw6: float32 HxWx6 in [0,1] (cache-space crop)
    - cls_id: detector's model-space class id (e.g., CFG.RAW_TO_MODEL[4] or [5])
    - returns integer count (rounded from μ; keep μ as float if you prefer)
    """
    Hc, Wc, C = roi_hw6.shape
    assert C == 6, "ROI must be 6 channels (OG+RED)."

    S = 384
    roi_resized = cv2.resize(roi_hw6, (S, S), interpolation=cv2.INTER_LINEAR)


    x = torch.from_numpy(roi_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(density_dev)   
    type_idx = torch.tensor([TYPE_MAP.get(int(cls_id), 0)], dtype=torch.long, device=density_dev)  
    box_area_px = torch.tensor([float(Hc * Wc)], dtype=torch.float32, device=density_dev)         

    with torch.no_grad():
        mu, r = density_net(x, type_idx, box_area_px)  
        count_float = float(mu[0].item())

    return int(round(max(0.0, count_float)))

 
def count_image(hw6_lb: np.ndarray, boxes, scores, labels, cfg: object, gate, density_net, density_dev) -> tuple[dict, dict, list, int]:
    H,W = hw6_lb.shape[:2]
    singles = {c:0 for c in cfg.SINGLE_CLASS_IDS}
    colony_count = 0

    for i in range(labels.shape[0]):
        c = int(labels[i].item())
        if (c not in cfg.SINGLE_CLASS_IDS) and (c not in cfg.COLONY_CLASS_IDS):
            continue
        x1,y1,x2,y2 = [int(round(v)) for v in boxes[i].tolist()]
        x1=max(0,min(x1,W-1)); x2=max(0,min(x2,W-1))
        y1=max(0,min(y1,H-1)); y2=max(0,min(y2,H-1))
        if x2<=x1 or y2<=y1: continue
        roi = hw6_lb[y1:y2, x1:x2, :]

        if c in cfg.SINGLE_CLASS_IDS:
            singles[c] += 1
            continue

        colony_count += 1

    total = sum(singles.values()) + colony_count
    return singles, colony_count, total