# Algae_detection

Pico-algae detection and counting with Faster R-CNN on a **dual-image / 6-channel** input (RGB + RED). The repo contains training, fused inference (singles + colonies + density gate), visualization, and a minimal GUI for batch processing.

> Target objects: **EUK**, **FC**, **FE** single cells (primary); colony classes are flagged and used for gating and reporting.

---

## Features

- **6-channel Faster R-CNN** (concatenated original + red-filtered images).
- **Two-stage inference**: singles detector + colony detector, with optional **density-based gate** for sparse vs. dense images.
- **Centroid-F1 evaluation** for count-centric metrics (tolerant to box size mismatch).
- **Visualization**: side-by-side OG/RED panels with boxes, labels, and per-class counts; auto export.
- **Simple GUI** (`algae_gui.py`) for local batch runs.

---

## Repository structure

```
algae_detection/
├─ trainer.py                # Train Faster R-CNN on 6-channel inputs
├─ fused_inf.py              # Two-stage inference + gating + exports
├─ algae_gui.py              # Minimal GUI for local processing
├─ run_debug.bat             # Example Windows entry point
└─ tools/
   ├─ models.py              # Model builders/loaders (singles, colonies, density)
   ├─ infer.py               # Inference utilities
   ├─ counting_hook.py       # Density model, gates, centroid counting
   ├─ filters.py             # NMS/Soft-NMS, WBF, center-dedup, class rules
   └─ utils.py               # IO, transforms, drawing, splits, timers, etc.
```

---

## Installation

### 1) Create environment

```bash
# Conda (CUDA 12.x example — adjust to your setup)
conda create -n algae python=3.10 -y
conda activate algae
conda install -c pytorch pytorch torchvision pytorch-cuda=12.1 -c nvidia -y
pip install -r requirements.txt  # if present; otherwise install your usual stack
```

### 2) Verify PyTorch + CUDA

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Data format

Images come in *pairs* per sample:

```
images/
  train/
    left/   <stem>_og.png   # original image (RGB)
    right/  <stem>_red.png  # red-filtered image (RGB)
  val/
    left/   ...
    right/  ...
```

Labels are COCO-style (or your existing JSON) with class ids:

- **1 = EUK**, **2 = FC**, **3 = FE** (primary counting classes)
- **4, 5 = colony / flagged** (used for gating/flags; not primary count target)
- Adjust to preference, may include other classes

> During preprocessing, OG and RED images are concatenated → **6 channels**.

---

## Quick start

### Train

Typical knobs:

- **Min/Max size** for FPN,
- **Anchor scales/ratios** tuned for ~40–50 px objects,
- **Augmentations** (keep RED channel consistent).

### Inference (fused pipeline)

What it does:

1) Runs singles detector.
2) Optionally runs colony detector.
3) Applies **density / sparsity gate** to adapt thresholds (e.g., higher conf for sparse).
4) Applies **parent-kill → center-dedup → Soft-NMS** (order tunable).
5) Exports:
   - `vis/*.png` side-by-side panels (OG/RED) with boxes, labels, **per-class counts**,
   - `preds.csv` with per-image totals and flags.

### GUI

Select an image folder, the checkpoints, and an output directory, then run.

---

## Evaluation (Centroid-F1)

Counting is evaluated via **centroid matching** with a tolerance radius `r_px` (e.g., 6–12 px):

- A detection is a TP if its **center** falls within `r_px` of a GT center (per class).
- FP and FN are derived from unmatched detections/GT points.
- Report **Precision/Recall/F1** per class and overall, plus **MAE/MAPE** for counts.

---

## Tuning tips

- **Sparse images**: raise the class-wise score thresholds and keep minimal Soft-NMS to avoid duplicate non-overlapping boxes on the same object.
- **Dense images**: lower thresholds a bit; use **center-dedup within r_px** before Soft-NMS to avoid collapsing neighbors.
- Prefer **centroid-F1** for model selection when **counting** is the goal (box IoU may be misleading on tight clusters).

---

## Outputs

- `out/vis/<stem>.png` — OG/RED panel with boxes and a legend (EUK/FC/FE counts in the top-left).
- `out/preds.csv` — Per-image counts and flags:
  - `euk`, `fc`, `fe`,
  - `flagged_colony` (true if any class 4 or 5 detection is present).

---

## Troubleshooting

- **CUDA mismatch / binary incompatibility**: Ensure PyTorch, TorchVision, and CUDA versions match. Recreate the env if needed.
- **Too many small duplicates on sparse images**: increase score thresholds and/or tighten **center-dedup** (`r_px`) before Soft-NMS.
- **Close neighbors merged in dense patches**: reduce Soft-NMS suppression and slightly lower thresholds; consider smaller anchor scales.

**Notes for contributors**

- Keep new scripts inside `tools/` if they are helpers; top-level for entry points.
- Prefer dataclass configs and deterministic seeds.
- When changing post-processing, **log both mAP and centroid-F1**, but optimize for centroid-F1 if counting is the objective.
