from __future__ import annotations
import os, sys
from pathlib import Path

def _ensure_streams():
    # Choose a per-user writable log location
    base = Path(os.getenv("LOCALAPPDATA") or Path.home() / ".algaecounter")
    log_dir = base / "AlgaeCounter" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "stdout_stderr.log"
    f = open(log_path, "a", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f

_ensure_streams()

import csv
import shutil
import contextlib
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import numpy as np
import cv2
import torch

# ----- Qt -----
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QProgressBar
)

# Kill foreign Qt paths that poison the import (robust packaging)
for var in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_API", "PYQTGRAPH_QT_LIB"):
    os.environ.pop(var, None)

# Put PySide6 folder (with Qt6*.dll) first on PATH so the right DLLs load
pyside_dir = Path(sys.executable).parent / "Lib" / "site-packages" / "PySide6"
if pyside_dir.exists():
    os.environ["PATH"] = str(pyside_dir) + os.pathsep + os.environ.get("PATH", "")

# Optional: prioritize this Python's DLLs
dlls_dir = Path(sys.base_prefix) / "DLLs"
if dlls_dir.exists():
    os.environ["PATH"] = str(dlls_dir) + os.pathsep + os.environ["PATH"]

# ====== REUSED IMPORTS (no re-declarations) ======
from tools.models import load_colony_model, load_single_model
from tools.utils import seed_everything, letterbox, compose_two_up, draw_vis
from tools.infer import infer_singles, infer_colonies
from tools.filters import (
    size_aware_softnms_per_class,
    drop_bright_green,
    parent_kill,
    suppress_cross_class_conflicts,
    enforce_colony_rules,
    drop_green_dominant_artifacts,
)

import numpy as _np, cv2 as _cv2, os as _os

def _safe_imread(path, flags=_cv2.IMREAD_COLOR):
    p = _os.fspath(path)
    try:
        # np.fromfile handles Unicode paths on Windows
        data = _np.fromfile(p, dtype=_np.uint8)
        if data.size == 0:
            return None
        img = _cv2.imdecode(data, flags)
        return img
    except Exception:
        return None

# Monkey-patch globally so tools.utils.letterbox also benefits
_cv2.imread = _safe_imread

PALETTE = {
    1: (255,   0,   0),
    2: (  0, 255, 255),
    3: (  0,   0, 255),
    4: (128, 128, 128),
    5: (255, 255, 255),
}

def resolve_data_path(rel: Path) -> Path:
    # 1) onefile extracted location
    if hasattr(sys, "_MEIPASS"):
        p = Path(sys._MEIPASS) / rel
        if p.exists():
            return p
    # 2) onedir (next to the executable)
    p = Path(sys.executable).parent / rel
    if p.exists():
        return p
    # 3) running from source
    p = Path(__file__).resolve().parent / rel
    if p.exists():
        return p
    raise FileNotFoundError(f"Bundled data not found: {rel}")


def APP_BASE() -> Path:
    # When frozen by PyInstaller, files live under sys._MEIPASS
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

# ---- crash logger (very early) ----
import traceback, datetime
LOG_PATH = Path(os.getcwd()) / "algae_gui_error.log"

def _excepthook(exc_type, exc, tb):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] {exc_type.__name__}: {exc}\n{''.join(traceback.format_exception(exc_type, exc, tb))}\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg)
    except Exception:
        pass
    try:
        QMessageBox.critical(None, "Fatal error",
                             f"{exc_type.__name__}: {exc}\n\nSee log:\n{LOG_PATH}")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc, tb)

sys.excepthook = _excepthook
# -----------------------------------

@dataclass
class FUSECFG:
    # Hard-wired model locations inside the package (PyInstaller)
    MODEL_DIR: Path = APP_BASE() / "models"
    SINGLE_FILENAME: str = "best_frcnn_6ch12_state.pth"
    COLONY_FILENAME: str = "fold3_best_state.pth"

    OLD_WARP_SZ: int = 2080
    LB_SIZE: int = 2048

    SINGLE_CLASS_IDS: Tuple[int, ...] = (1, 2, 3)   # EUK, FC, FE
    COLONY_CLASS_IDS: Tuple[int, ...] = (4, 5)

    SCORE_FLOOR: float = 0.1
    CLASS_THR: Dict[int, float] = field(default_factory=lambda: {
        1: 0.13, 2: 0.1, 3: 0.3, 4: 0.4, 5: 0.4
    })

    SPLIT_AREA_PX: int = 70 * 70
    SOFTNMS_IOU_SMALL: float = 0.6
    SOFTNMS_SIGMA_SMALL: float = 0.05
    SOFTNMS_IOU_BIG: float = 0.6
    SOFTNMS_SIGMA_BIG: float = 0.05

    XCLS_R_PX: int = 15
    XCLS_AREA_LO: float = 0.4
    XCLS_AREA_HI: float = 1.8
    XCLS_IOU_MIN: float = 0.35
    XCLS_MARGIN: float = 0.01
    XCLS_PRIORITY: Tuple[int, int, int] = (1, 3, 2)

    RO_ENABLE: bool = True
    RO_MIN_BRIGHT: float = 0.27
    RO_MIN_ROFRAC: float = 0.26

CFG = FUSECFG()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
seed_everything(42)

def _natural_key(p: Path):
    import re
    s = p.stem
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def pair_pngs_in_order(folder: Path) -> List[Tuple[Path, Path, str]]:
    """
    Returns list of (og_path, red_path, stem_for_outputs).
    Assumes strict order og, red, og, red, … after natural sort.
    """
    pngs = sorted([p for p in folder.glob("*.png") if p.is_file()], key=_natural_key)
    if len(pngs) < 2:
        return []
    if len(pngs) % 2 != 0:
        pngs = pngs[:-1]
    pairs = []
    for i in range(0, len(pngs), 2):
        og_path = pngs[i]
        red_path = pngs[i + 1]
        stem = og_path.stem
        pairs.append((og_path, red_path, stem))
    return pairs

def process_pair(
    og_path: Path,
    red_path: Path,
    stem: str,
    temp_pair_dir: Path,
    out_dir: Path,
    save_visuals: bool,
    single_model,
    colony_model,
):
    """
    Reuse pipeline. Copy to temp with names {stem}_og.png / {stem}_red.png
    so your letterbox(stem, dir, target=...) works unchanged.
    """
    temp_pair_dir.mkdir(parents=True, exist_ok=True)
    tmp_og = temp_pair_dir / f"{stem}_og.png"
    tmp_red = temp_pair_dir / f"{stem}_red.png"
    shutil.copy2(str(og_path), str(tmp_og))
    shutil.copy2(str(red_path), str(tmp_red))

    # 1) letterbox fused + meta
    fused_6chw, meta = letterbox(stem, temp_pair_dir, target=CFG.LB_SIZE)
    hw6_lb = fused_6chw.transpose(1, 2, 0).copy()  # HxWx6 float

    # 2) run both models
    Bn, Sn, Ln = infer_colonies(colony_model, fused_6chw, DEVICE)                 # LB
    Bo_lb, So, Lo = infer_singles(single_model, tmp_og, tmp_red, meta, CFG.OLD_WARP_SZ, DEVICE)

    # 3) fuse (singles from old, colonies from new)
    keep_old = torch.isin(Lo, torch.tensor(CFG.SINGLE_CLASS_IDS))
    keep_new = torch.isin(Ln, torch.tensor(CFG.COLONY_CLASS_IDS))
    B = torch.cat([Bo_lb[keep_old], Bn[keep_new]], 0)
    S = torch.cat([So[keep_old],     Sn[keep_new]], 0)
    L = torch.cat([Lo[keep_old],     Ln[keep_new]], 0)

    if B.numel():
        B, S, L = B.cpu(), S.cpu(), L.cpu()

    # 4) per-class floors
    if B.numel():
        floor = CFG.SCORE_FLOOR
        if CFG.CLASS_THR:
            per = torch.tensor([CFG.CLASS_THR.get(int(c), floor) for c in L], dtype=S.dtype)
        else:
            per = torch.full_like(S, floor)
        m = S >= per
        B, S, L = B[m], S[m], L[m]

    # 5) cross-class conflict cleanup (EUK/FC/FE)
    if B.numel():
        B, S, L = suppress_cross_class_conflicts(
            B, S, L,
            classes=(1, 2, 3),
            r_px=CFG.XCLS_R_PX,
            area_lo=CFG.XCLS_AREA_LO,
            area_hi=CFG.XCLS_AREA_HI,
            iou_min=CFG.XCLS_IOU_MIN,
            per_class_floor=CFG.CLASS_THR,
            margin=CFG.XCLS_MARGIN,
            priority_order=CFG.XCLS_PRIORITY,
            return_debug=False,
        )

    # 6) parent kill
    if B.numel():
        B, S, L = parent_kill(B, S, L, tol_px=4.0)

    # 7) size-aware soft-NMS per class
    if B.numel():
        B, S, L = size_aware_softnms_per_class(
            B, S, L,
            split_area_px=CFG.SPLIT_AREA_PX,
            iou_small=CFG.SOFTNMS_IOU_SMALL, sigma_small=CFG.SOFTNMS_SIGMA_SMALL,
            iou_big=CFG.SOFTNMS_IOU_BIG,     sigma_big=CFG.SOFTNMS_SIGMA_BIG,
            score_floor=CFG.SCORE_FLOOR,
        )

    # 8) color gate on singles
    if B.numel() and CFG.RO_ENABLE:
        B, S, L = drop_bright_green(
            hw6_lb, B, S, L,
            classes=(1, 2,),
            min_bright_frac=CFG.RO_MIN_BRIGHT,
            ro_min_frac=CFG.RO_MIN_ROFRAC,
            stem=stem,
        )
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

    # 9) enforce colony rules (4 kills 2s inside; 5 kills 3s inside)
    if B.numel():
        B, S, L, _ = enforce_colony_rules(
            B, S, L,
            colony_classes=(4, 5),
            tol_px=0.0,
            mode="center",
        )

    # Counts (singles only) + flag
    euk = int((L == 1).sum().item())
    fc  = int((L == 2).sum().item())
    fe  = int((L == 3).sum().item())
    flagged_colony = bool(((L == 4).any().item()) or ((L == 5).any().item()))

    # Visuals (optional) -> <output>/vis/
    if save_visuals:
        og  = (hw6_lb[..., 0:3].clip(0, 1) * 255).astype(np.uint8)
        red = (hw6_lb[..., 3:6].clip(0, 1) * 255).astype(np.uint8)
        panel = compose_two_up(og.copy(), red.copy())
        panel = draw_vis(panel, B, S, L, PALETTE, draw_scores=False)
        vis_dir = out_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / f"{stem}.png"), panel)

    with contextlib.suppress(Exception):
        (temp_pair_dir / f"{stem}_og.png").unlink(missing_ok=True)
        (temp_pair_dir / f"{stem}_red.png").unlink(missing_ok=True)

    return {"image": stem, "EUK": euk, "FC": fc, "FE": fe, "flagged_colony": flagged_colony}

# ----------------------------
# Worker Thread
# ----------------------------
class InferenceWorker(QThread):
    progress = Signal(int, int, str)     # i, n, stem
    status   = Signal(str)
    done     = Signal(str)               # csv path
    error    = Signal(str)

    def __init__(self, in_folder: Path, out_folder: Path, save_visuals: bool, parent=None):
        super().__init__(parent)
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.save_visuals = save_visuals
        self.single_model = None
        self.colony_model = None

    def run(self):
        try:
            # Validate bundled models exist
            single_path = CFG.MODEL_DIR / CFG.SINGLE_FILENAME
            colony_path = CFG.MODEL_DIR / CFG.COLONY_FILENAME
            missing = []
            if not single_path.is_file(): missing.append(str(single_path))
            if not colony_path.is_file(): missing.append(str(colony_path))
            if missing:
                raise FileNotFoundError(
                    "Bundled model file(s) not found:\n" + "\n".join(missing) +
                    "\n\nMake sure the installer includes a 'models' folder next to the executable."
                )

            self.status.emit("Pairing .png files...")
            pairs = pair_pngs_in_order(self.in_folder)
            if not pairs:
                raise RuntimeError("No .png pairs found (need at least two files, og/red order).")

            out_dir = self.out_folder
            out_dir.mkdir(parents=True, exist_ok=True)
            temp_pair_dir = out_dir / ".tmp_pairs"
            temp_pair_dir.mkdir(parents=True, exist_ok=True)

            self.status.emit("Loading models...")
            if self.single_model is None:
                self.single_model = load_single_model(single_path, DEVICE, CFG.OLD_WARP_SZ)
            if self.colony_model is None:
                self.colony_model = load_colony_model(colony_path, DEVICE, min_size=CFG.LB_SIZE)

            n = len(pairs)
            rows = []
            for i, (og_p, red_p, stem) in enumerate(pairs, 1):
                self.status.emit(f"Processing {i}/{n}: {stem}")
                rec = process_pair(
                    og_p, red_p, stem,
                    temp_pair_dir=temp_pair_dir,
                    out_dir=out_dir,
                    save_visuals=self.save_visuals,
                    single_model=self.single_model,
                    colony_model=self.colony_model,
                )
                rows.append(rec)
                self.progress.emit(i, n, stem)

            # CSV -> out_dir/counts.csv
            csv_path = out_dir / "counts.csv"
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["image", "EUK", "FC", "FE", "flagged_colony"])
                w.writeheader()
                for r in rows:
                    w.writerow(r)

            with contextlib.suppress(Exception):
                shutil.rmtree(temp_pair_dir, ignore_errors=True)

            self.done.emit(str(csv_path))
        except Exception as e:
            self.error.emit(str(e))

# ----------------------------
# GUI
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Algae Inference")
        self.setMinimumSize(760, 240)

        # Widgets
        self.in_edit = QLineEdit()
        self.in_browse = QPushButton("Browse…")
        self.out_edit = QLineEdit()
        self.out_browse = QPushButton("Browse…")
        self.save_vis_cb = QCheckBox("Save visuals (saved to <output>/vis)")
        self.save_vis_cb.setChecked(True)
        self.run_btn = QPushButton("Run inference")
        self.progress = QProgressBar()
        self.status_lbl = QLabel("Ready")

        # Layout
        row1a = QHBoxLayout()
        row1a.addWidget(QLabel("Input folder with .png files (og, red, og, red, …):"))
        row1b = QHBoxLayout()
        row1b.addWidget(self.in_edit, 1)
        row1b.addWidget(self.in_browse)

        row2a = QHBoxLayout()
        row2a.addWidget(QLabel("Output folder (CSV here; visuals optional in /vis):"))
        row2b = QHBoxLayout()
        row2b.addWidget(self.out_edit, 1)
        row2b.addWidget(self.out_browse)

        row3 = QHBoxLayout()
        row3.addWidget(self.save_vis_cb)
        row4 = QHBoxLayout()
        row4.addWidget(self.run_btn)
        row5 = QHBoxLayout()
        row5.addWidget(self.progress)
        row6 = QHBoxLayout()
        row6.addWidget(self.status_lbl)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addLayout(row1a)
        lay.addLayout(row1b)
        lay.addLayout(row2a)
        lay.addLayout(row2b)
        lay.addLayout(row3)
        lay.addLayout(row4)
        lay.addLayout(row5)
        lay.addLayout(row6)
        self.setCentralWidget(root)

        # Signals
        self.in_browse.clicked.connect(self._browse_in)
        self.out_browse.clicked.connect(self._browse_out)
        self.run_btn.clicked.connect(self._run)

        self.worker: InferenceWorker | None = None

    def _browse_in(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder")
        if d:
            self.in_edit.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_edit.setText(d)

    def _set_busy(self, busy: bool):
        # Disable real widgets (not methods)
        widgets = [
            self.in_browse, self.out_browse, self.run_btn,
            self.save_vis_cb, self.in_edit, self.out_edit
        ]
        for w in widgets:
            w.setEnabled(not busy)
        self.progress.setRange(0, 0 if busy else 100)
        QApplication.processEvents()

    def _run(self):
        in_dir = Path(self.in_edit.text().strip())
        out_dir = Path(self.out_edit.text().strip())
        if not in_dir.is_dir():
            QMessageBox.critical(self, "Error", "Please choose a valid **input** folder.")
            return
        if not out_dir.is_dir():
            QMessageBox.critical(self, "Error", "Please choose a valid **output** folder.")
            return

        # Early check for bundled models to avoid silent fail
        single_path = CFG.MODEL_DIR / CFG.SINGLE_FILENAME
        colony_path = CFG.MODEL_DIR / CFG.COLONY_FILENAME
        if not (single_path.is_file() and colony_path.is_file()):
            QMessageBox.critical(
                self, "Models missing",
                "Bundled models were not found at:\n"
                f"{single_path}\n{colony_path}\n\n"
                "Ensure your installer bundles a 'models' folder next to the app."
            )
            return

        self.worker = InferenceWorker(in_dir, out_dir, self.save_vis_cb.isChecked(), self)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self._on_status)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)

        self.progress.setValue(0)
        self.progress.setMaximum(0)  # busy
        self._set_busy(True)
        self.status_lbl.setText("Starting…")
        self.worker.start()

    def _on_progress(self, i: int, n: int, stem: str):
        self.progress.setMaximum(n)
        self.progress.setValue(i)
        self.status_lbl.setText(f"Processed {i}/{n}: {stem}")

    def _on_status(self, msg: str):
        self.status_lbl.setText(msg)

    def _on_done(self, csv_path: str):
        self._set_busy(False)
        self.status_lbl.setText(f"Done. CSV → {csv_path}")
        QMessageBox.information(self, "Finished", f"Done.\nCSV saved to:\n{csv_path}")

    def _on_error(self, err: str):
        self._set_busy(False)
        self.status_lbl.setText(f"Error: {err}")
        QMessageBox.critical(self, "Error", err)

def main():
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication([])
    w = MainWindow()
    # Helpful early warning if models are missing
    sp = CFG.MODEL_DIR / CFG.SINGLE_FILENAME
    cp = CFG.MODEL_DIR / CFG.COLONY_FILENAME
    if not (sp.is_file() and cp.is_file()):
        QMessageBox.warning(
            w, "Models not found",
            "Default models were not found in the bundled 'models' folder.\n"
            "Please rebuild your installer with the models included."
        )
    w.show()
    app.exec()

if __name__ == "__main__":
    # Windows-safe entry for PyInstaller
    from multiprocessing import freeze_support
    freeze_support()
    main()
