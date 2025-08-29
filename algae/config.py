from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    IMAGE_DIR: Path = Path("/workspace/data/images")   # expects <stem>_og.png & <stem>_red.png
    LABEL_DIR: Path = Path("/workspace/data/labels")   # expects <stem>_og.txt (YOLO: cls cx cy w h)
    CACHE_DIR: Path = Path("/workspace/data/cache_fused")

    IMAGE_SIZE: int = 2080
    NUM_CLASSES: int = 7
    BACKBONE: str = "resnet101"

    BATCH_SIZE: int = 4
    WORKERS: int = 2
    EPOCHS: int = 20
    LR: float = 1e-4
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VAL_FRACTION: float = 0.2
    SEED: int = 42

    USE_ZSCORE: bool = False
    SAVE_BEST_TO: Path = Path("best_frcnn_6ch.pth")

    # tensorboard / metrics
    LOGDIR = "runs/algae11"
    LOG_EVERY = 50
    PR_IOU = 0.5
    PR_SCORE_THRESH = 0.00
    CLASS_NAMES = ["EUK", "FC", "FE", "EUK colony", "FC colony", "FE colony"]

    # centroid-F1 (per-class tolerance in px; per-class thresholds allowed too)
    CEN_F1_TOL_PX = {1:16, 2:16, 3:16, 4:24, 5:28, 6:28}
    CEN_F1_SCORE_THRESH: float | dict = 0.30

    # updated each eval from GT size stats (used by size filter)
    MIN_SIDE_PX = 0
    MAX_SIDE_PX = 0

    PRIMARY_CLASS_IDS: tuple = (1, 2, 3)
    COLONY_CLASS_IDS:  tuple = (4, 5, 6)
    SELECT_ON_PRIMARY_MAP: bool = True

    OVERSAMPLE_COLONY: bool = False
    COLONY_OVERSAMPLE_FACTOR: float = 3.0

    SAVE_VAL_VIS_EVERY: int = 4
    VIS_OUT_DIR: str = "val_vis"

CFG = Config()
