import random, numpy as np, torch
from .config import CFG

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _coerce_cen_thr_mapping(obj, default=0.30):
    if isinstance(obj, (int,float)): return float(obj)
    if isinstance(obj, dict):
        out = {}
        for k,v in obj.items():
            try: ck = int(k)
            except Exception: continue
            if isinstance(v, (int,float)): out[ck] = float(v)
            elif isinstance(v, dict):
                for key in ("thr","threshold","score_thr","score","value"):
                    if key in v and isinstance(v[key], (int,float)):
                        out[ck] = float(v[key]); break
        if out: return out
    return float(default)

def _get_class_thr(c: int) -> float:
    thr = CFG.CEN_F1_SCORE_THRESH
    if isinstance(thr, (int,float)): return float(thr)
    if isinstance(thr, dict):
        v = thr.get(c, thr.get(str(c), None))
        if isinstance(v, (int,float)): return float(v)
        if isinstance(v, dict):
            for key in ("thr","threshold","score_thr","score","value"):
                if key in v and isinstance(v[key], (int,float)): return float(v[key])
    return 0.30

def _get_class_tol(c: int) -> int:
    tol = CFG.CEN_F1_TOL_PX
    if isinstance(tol, (int,float)): return int(tol)
    if isinstance(tol, dict):
        v = tol.get(c, tol.get(str(c), None))
        if isinstance(v, (int,float)): return int(v)
    return 10

def _size_filter_xyxy(b: torch.Tensor, min_side=None, max_side=None):
    min_side = CFG.__dict__.get("MIN_SIDE_PX", 6) if min_side is None else min_side
    max_side = CFG.__dict__.get("MAX_SIDE_PX", 512) if max_side is None else max_side
    if b.numel()==0:
        return b, torch.zeros(0, dtype=torch.bool, device=b.device)
    w = (b[:,2]-b[:,0]); h = (b[:,3]-b[:,1])
    keep = (torch.minimum(w,h) >= min_side) & (torch.maximum(w,h) <= max_side)
    return b[keep], keep
