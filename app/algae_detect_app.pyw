
import argparse
import csv
from pathlib import Path
import numpy as np
import cv2
import torch

# ---------------------------
# User-editable: class names & colors
# ---------------------------
CLASS_NAMES = [
    "background",
    "class1",
    "class2",
    "class3",
    "class4",
    "class5",
    "class6",
]

COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (0, 255, 0),
    (120, 120, 0),
    (0, 120, 24),
    (128, 128, 128),
]

ALLOWED_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform

def get_faster_rcnn_model(num_classes=7, image_size=(2560, 2560), backbone='resnet101'):
    backbone = resnet_fpn_backbone(backbone_name=backbone, weights=None)
    old_conv = backbone.body.conv1
    new_conv = torch.nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:] = mean_weight.repeat(1, 3, 1, 1)
    backbone.body.conv1 = new_conv
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    # Loosen NMS / increase proposals for skinny objects
    try:
        model.rpn.pre_nms_top_n_test  = 6000
        model.rpn.post_nms_top_n_test = 2000
        model.rpn.nms_thresh          = 0.8
        model.rpn.score_thresh        = 0.0
        model.roi_heads.nms_thresh        = 0.6
        model.roi_heads.detections_per_img = 300
    except Exception:
        pass

    # Custom transform for 6-channel input
    model.transform = GeneralizedRCNNTransform(
        min_size=image_size[0],
        max_size=image_size[1],
        image_mean=[0.0] * 6,
        image_std=[1.0] * 6
    )
    return model

# ---------------------------
# Pairing helpers
# ---------------------------
def _is_img(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTS

def _list_images(images_dir: Path, recursive: bool):
    if recursive:
        return sorted([p for p in images_dir.rglob("*") if p.is_file() and _is_img(p)])
    else:
        return sorted([p for p in images_dir.glob("*") if p.is_file() and _is_img(p)])

def _pair_iterator_suffix(images_dir: Path, recursive: bool):
    """Original mode: find *_og.* and match to *_red.* in the same folder."""
    patt = "**/*_og.*" if recursive else "*_og.*"
    for og_path in sorted(images_dir.glob(patt)):
        if not _is_img(og_path):
            continue
        red_path = og_path.parent / og_path.name.replace("_og", "_red")
        if red_path.exists() and _is_img(red_path):
            yield og_path, red_path
        else:
            print("[WARN] Missing _red pair for:", og_path.name, "(skipping)", flush=True)

def _redness_score(img_float01: np.ndarray) -> float:
    """Heuristic 'redness' score using resized float image in [0,1]."""
    if img_float01.ndim != 3 or img_float01.shape[2] < 3:
        return 0.0
    R = img_float01[..., 2]
    G = img_float01[..., 1]
    B = img_float01[..., 0]
    denom = (R.mean() + G.mean() + B.mean() + 1e-8)
    return float(R.mean() / denom)

def _pair_iterator_order(images_dir: Path, recursive: bool, first_is_og: bool, autodetect: bool, imgsz: int):
    """
    Sequential mode: sort all images and pair them (0,1), (2,3), ...
    If autodetect=True, try to detect which file in each pair is the 'red' image.
    Otherwise, use first_is_og to decide order.
    """
    files = _list_images(images_dir, recursive)
    files = [p for p in files if ("_og" not in p.stem and "_red" not in p.stem)]
    if len(files) % 2 == 1:
        print(f"[WARN] Found an odd number of images ({len(files)}). The last one will be ignored.", flush=True)
        files = files[:-1]
    for i in range(0, len(files), 2):
        a, b = files[i], files[i+1]
        if autodetect:
            imgA = cv2.imread(str(a)).astype(np.float32) / 255.0
            imgB = cv2.imread(str(b)).astype(np.float32) / 255.0
            imgA = cv2.resize(imgA, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            imgB = cv2.resize(imgB, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            sA = _redness_score(imgA)
            sB = _redness_score(imgB)
            margin = 0.02  # minimal difference to swap confidently
            if sA > sB + margin:
                og_path, red_path = b, a    # A is "redder"
            elif sB > sA + margin:
                og_path, red_path = a, b
            else:
                og_path, red_path = (a, b) if first_is_og else (b, a)
            yield og_path, red_path
        else:
            og_path, red_path = (a, b) if first_is_og else (b, a)
            yield og_path, red_path

def build_pairs(images_dir: Path, recursive: bool, pairing: str, first_is_og: bool, autodetect_order: bool, imgsz: int):
    """Return a concrete list of (og_path, red_path) pairs for progress tracking."""
    if pairing == "order":
        return list(_pair_iterator_order(images_dir, recursive, first_is_og, autodetect_order, imgsz))
    else:
        return list(_pair_iterator_suffix(images_dir, recursive))

# ---------------------------
# Image prep & drawing
# ---------------------------
def _prepare_six_channel(og_path: Path, red_path: Path, imgsz: int):
    img1 = cv2.imread(str(og_path)).astype(np.float32) / 255.0
    img2 = cv2.imread(str(red_path)).astype(np.float32) / 255.0
    img1 = cv2.resize(img1, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    fused = np.concatenate([img1, img2], axis=-1)              # (H, W, 6)
    fused_t = torch.from_numpy(np.transpose(fused, (2, 0, 1)))  # (6, H, W)
    return fused, fused_t, img1  # visualize on OG image by convention

def draw_detections(img_vis_uint8, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        li = int(label.item())
        color = COLORS[li % len(COLORS)]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_vis_uint8, (x1, y1), (x2, y2), color, 2)
        name = CLASS_NAMES[li] if li < len(CLASS_NAMES) else f"class{li}"
        cv2.putText(
            img_vis_uint8, f"{name} ({score:.2f})",
            (x1, max(y1 - 7, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return img_vis_uint8

# ---------------------------
# EUK stitcher (merge nearby same-class boxes)
# ---------------------------
def merge_euk_segments(boxes, labels, scores, euk_id=6, gap_px=40, iou_thr=0.05):
    """
    Merge EUK boxes that overlap (IoU>iou_thr) or are within gap_px by center distance.
    Other classes are untouched.
    """
    if boxes.numel() == 0:
        return boxes, labels, scores

    idx = (labels == euk_id).nonzero(as_tuple=True)[0]
    if idx.numel() < 2:
        return boxes, labels, scores

    e_boxes = boxes[idx].clone()
    e_scores = scores[idx].clone()

    changed = True
    while changed:
        changed = False
        N = e_boxes.shape[0]
        i = 0
        while i < N:
            j = i + 1
            merged = False
            while j < N:
                ax1, ay1, ax2, ay2 = e_boxes[i]
                bx1, by1, bx2, by2 = e_boxes[j]
                # IoU numerator (intersection area)
                inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
                inter = inter_w * inter_h
                # Union (for threshold only)
                if inter > iou_thr * 1.0:
                    merge = True
                else:
                    # center gap
                    cx1, cy1 = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
                    cx2, cy2 = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
                    center_dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                    merge = center_dist < gap_px
                if merge:
                    u = torch.tensor([min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)],
                                     dtype=e_boxes.dtype, device=e_boxes.device)
                    e_boxes[i] = u
                    e_scores[i] = max(e_scores[i], e_scores[j])
                    e_boxes = torch.cat([e_boxes[:j], e_boxes[j+1:]], dim=0)
                    e_scores = torch.cat([e_scores[:j], e_scores[j+1:]], dim=0)
                    N -= 1
                    merged = True
                else:
                    j += 1
            if not merged:
                i += 1
            changed |= merged

    # rebuild tensors
    other = (labels != euk_id).nonzero(as_tuple=True)[0]
    boxes2  = torch.cat([boxes[other], e_boxes], dim=0)
    labels2 = torch.cat([labels[other], torch.full((len(e_boxes),), euk_id, dtype=labels.dtype)], dim=0)
    scores2 = torch.cat([scores[other], e_scores], dim=0)
    return boxes2, labels2, scores2

# ---------------------------
# Main inference
# ---------------------------
def run_inference(
    images_dir: Path,
    model_path: Path,
    output_dir: Path,
    backbone: str = "resnet101",
    imgsz: int = 2560,
    conf_thresh: float = 0.10,
    euk_thresh: float = 0.20,
    euk_id: int = 6,
    stitch_euk: bool = True,
    stitch_gap_px: int = 40,
    save_viz: bool = True,
    recursive: bool = False,
    device: str = "auto",
    pairing: str = "suffix",         # 'suffix' or 'order'
    first_is_og: bool = True,        # used in 'order' mode
    autodetect_order: bool = False,  # used in 'order' mode
    pairs=None,                      # optional prebuilt list of (og, red)
    progress_cb=None,                # optional callback: progress_cb(done, total, current_name)
):
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "viz"
    if save_viz:
        viz_dir.mkdir(exist_ok=True)
    log_path = output_dir / "run_log.txt"

    # Load model
    model = get_faster_rcnn_model(num_classes=len(CLASS_NAMES), image_size=(imgsz, imgsz), backbone=backbone)
    model.load_state_dict(torch.load(str(model_path), map_location=device_t))
    model.to(device_t).eval()

    per_image_csv = output_dir / "per_image_counts.csv"
    summary_csv = output_dir / "summary_counts.csv"
    detections_csv = output_dir / "detections_full.csv"

    total_counts = {i: 0 for i in range(1, len(CLASS_NAMES))}
    per_image_rows = []
    det_rows = []

    # Build pairs if not supplied (GUI prebuilds to show progress)
    if pairs is None:
        if pairing == "order":
            iterator = _pair_iterator_order(images_dir, recursive, first_is_og, autodetect_order, imgsz)
            pairs = list(iterator)
            pairing_desc = f"order (first_is_og={first_is_og}, autodetect={autodetect_order})"
        else:
            iterator = _pair_iterator_suffix(images_dir, recursive)
            pairs = list(iterator)
            pairing_desc = "suffix (_og/_red)"
    else:
        pairing_desc = "prebuilt"

    total = len(pairs)

    # Logging header
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("Images dir : " + str(images_dir) + "\n")
        lf.write("Pairing    : " + pairing_desc + "\n")
        lf.write("Backbone   : " + str(backbone) + "\n")
        lf.write("Image size : " + str(imgsz) + "\n")
        lf.write("Conf/EUK   : " + str(conf_thresh) + "/" + str(euk_thresh) + "\n")
        lf.write("Total pairs: " + str(total) + "\n")

    done = 0
    for og_path, red_path in pairs:
        fused, fused_t, img1_float = _prepare_six_channel(og_path, red_path, imgsz)
        image_tensor = fused_t.unsqueeze(0).to(device_t, dtype=torch.float32)
        with torch.no_grad():
            pred = model(image_tensor)[0]
        boxes = pred["boxes"].detach().cpu()
        labels = pred["labels"].detach().cpu()
        scores = pred["scores"].detach().cpu()

        # Per-class score threshold only (no extra NMS; model already did NMS)
        keep_idx = []
        for i, (c, s) in enumerate(zip(labels.tolist(), scores.tolist())):
            thr = euk_thresh if c == euk_id else conf_thresh
            if s >= thr:
                keep_idx.append(i)
        if keep_idx:
            keep_idx = torch.tensor(keep_idx, dtype=torch.long)
            boxes, labels, scores = boxes[keep_idx], labels[keep_idx], scores[keep_idx]
        else:
            boxes = boxes[:0]; labels = labels[:0]; scores = scores[:0]

        # EUK-only stitching
        if stitch_euk:
            boxes, labels, scores = merge_euk_segments(boxes, labels, scores, euk_id=euk_id, gap_px=stitch_gap_px)

        # Counting and CSV rows
        image_counts = {i: 0 for i in range(1, len(CLASS_NAMES))}
        for li in labels.tolist():
            if li == 0:
                continue
            image_counts[li] = image_counts.get(li, 0) + 1
            total_counts[li] = total_counts.get(li, 0) + 1

        row = {"image": og_path.name, "og_path": str(og_path), "red_path": str(red_path), "total_detections": int(len(labels))}
        for i in range(1, len(CLASS_NAMES)):
            key = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class{i}"
            row[key] = image_counts.get(i, 0)
        per_image_rows.append(row)

        for (x1, y1, x2, y2), li, sc in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            cname = CLASS_NAMES[li] if li < len(CLASS_NAMES) else f"class{li}"
            det_rows.append({
                "image": og_path.name,
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "score": float(sc),
                "label_id": int(li),
                "class_name": cname,
            })

        if save_viz:
            img_vis = (img1_float * 255).astype(np.uint8).copy()
            img_vis = draw_detections(img_vis, boxes, labels, scores)
            out_path = viz_dir / og_path.name
            cv2.imwrite(str(out_path), img_vis)

        done += 1
        if progress_cb is not None:
            try:
                progress_cb(done, total, og_path.name)
            except Exception:
                pass

    # write CSVs
    if per_image_rows:
        cols = ["image", "og_path", "red_path", "total_detections"] + [
            (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class{i}") for i in range(1, len(CLASS_NAMES))
        ]
        with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(per_image_rows)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "count"])
        for i in range(1, len(CLASS_NAMES)):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class{i}"
            writer.writerow([i, name, total_counts.get(i, 0)])

    if det_rows:
        cols = ["image", "x1", "y1", "x2", "y2", "score", "label_id", "class_name"]
        with open(detections_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(det_rows)

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\nProcessed pairs: " + str(done) + " / " + str(total) + "\n")
        lf.write("Outputs:\n  " + str(per_image_csv) + "\n  " + str(summary_csv) + "\n")
        if det_rows:
            lf.write("  " + str(detections_csv) + "\n")
        if save_viz:
            lf.write("  " + str(viz_dir) + "\n")

# ---- GUI (with progress bar + EUK settings) ----
def launch_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as e:
        print("Tkinter not available:", e)
        return

    # DPI scaling on Windows
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    import threading

    root = tk.Tk()
    root.title("Picoalgae Detector")
    root.geometry("980x600")
    root.minsize(860, 520)
    root.resizable(True, True)

    style = ttk.Style()
    try:
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    pad = {"padx": 10, "pady": 8}
    root.columnconfigure(1, weight=1)

    images_var = tk.StringVar()
    model_var = tk.StringVar()
    out_var = tk.StringVar(value=str(Path("./inference_results").resolve()))
    save_viz_var = tk.BooleanVar(value=True)
    conf_var = tk.DoubleVar(value=0.10)
    iou_dummy = tk.DoubleVar(value=0.10)  # kept for layout compatibility
    recursive_var = tk.BooleanVar(value=False)

    pairing_var = tk.StringVar(value="suffix")
    first_is_og_var = tk.BooleanVar(value=True)
    autodetect_var = tk.BooleanVar(value=True)

    # EUK controls
    euk_thresh_var = tk.DoubleVar(value=0.20)
    euk_gap_var = tk.IntVar(value=40)
    euk_stitch_var = tk.BooleanVar(value=True)

    # Row 0..2: paths
    ttk.Label(root, text="Image folder:").grid(row=0, column=0, sticky="e", **pad)
    entry_images = ttk.Entry(root, textvariable=images_var)
    entry_images.grid(row=0, column=1, sticky="ew", **pad)
    ttk.Button(root, text="Browse…", command=lambda: images_var.set(filedialog.askdirectory(title="Select image folder") or images_var.get())).grid(row=0, column=2, **pad)

    ttk.Label(root, text="Model .pth:").grid(row=1, column=0, sticky="e", **pad)
    entry_model = ttk.Entry(root, textvariable=model_var)
    entry_model.grid(row=1, column=1, sticky="ew", **pad)
    ttk.Button(root, text="Browse…", command=lambda: model_var.set(filedialog.askopenfilename(title="Select model .pth file", filetypes=[("PyTorch weights", "*.pth")]) or model_var.get())).grid(row=1, column=2, **pad)

    ttk.Label(root, text="Output folder:").grid(row=2, column=0, sticky="e", **pad)
    entry_out = ttk.Entry(root, textvariable=out_var)
    entry_out.grid(row=2, column=1, sticky="ew", **pad)
    ttk.Button(root, text="Browse…", command=lambda: out_var.set(filedialog.askdirectory(title="Select output folder (will be created)") or out_var.get())).grid(row=2, column=2, **pad)

    # Pairing frame
    pairing_frame = ttk.LabelFrame(root, text="Pairing mode", padding=10)
    pairing_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=6)
    pairing_frame.columnconfigure(0, weight=1)
    pairing_frame.columnconfigure(1, weight=1)

    ttk.Radiobutton(pairing_frame, text="By file names (_og/_red)", variable=pairing_var, value="suffix").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Radiobutton(pairing_frame, text="Sequential pairs (Image_0001, Image_0002, …)", variable=pairing_var, value="order").grid(row=0, column=1, sticky="w", pady=2)

    first_cb = ttk.Checkbutton(pairing_frame, text="In order mode: FIRST is the OG image", variable=first_is_og_var)
    autod_cb = ttk.Checkbutton(pairing_frame, text="Auto-detect RED vs OG in each pair (recommended)", variable=autodetect_var)
    first_cb.grid(row=1, column=0, sticky="w", pady=2)
    autod_cb.grid(row=1, column=1, sticky="w", pady=2)

    def on_pairing_change(*_):
        is_order = pairing_var.get() == "order"
        if is_order:
            if autodetect_var.get():
                first_cb.state(["disabled"])
            else:
                first_cb.state(["!disabled"])
            autod_cb.state(["!disabled"])
        else:
            first_cb.state(["disabled"])
            autod_cb.state(["disabled"])
    pairing_var.trace_add("write", on_pairing_change)
    autodetect_var.trace_add("write", on_pairing_change)
    on_pairing_change()

    # Options frame
    opts = ttk.LabelFrame(root, text="Options", padding=10)
    opts.grid(row=4, column=0, columnspan=3, sticky="ew", padx=10, pady=6)
    for c in range(6):
        opts.columnconfigure(c, weight=1)

    ttk.Label(opts, text="Global confidence threshold:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
    ttk.Entry(opts, textvariable=conf_var, width=10).grid(row=0, column=1, sticky="w", padx=4, pady=2)
    ttk.Label(opts, text="Search subfolders:").grid(row=0, column=2, sticky="e", padx=4, pady=2)
    ttk.Checkbutton(opts, variable=recursive_var).grid(row=0, column=3, sticky="w", padx=4, pady=2)
    ttk.Label(opts, text="Save annotated images:").grid(row=0, column=4, sticky="e", padx=4, pady=2)
    ttk.Checkbutton(opts, variable=save_viz_var).grid(row=0, column=5, sticky="w", padx=4, pady=2)

    # EUK-specific options
    euk = ttk.LabelFrame(root, text="EUK-specific (optional)", padding=10)
    euk.grid(row=5, column=0, columnspan=3, sticky="ew", padx=10, pady=6)
    euk.columnconfigure(1, weight=1)
    euk_names = [name for name in CLASS_NAMES[1:]]  # exclude background
    euk_var = tk.StringVar(value=(euk_names[-1] if euk_names else ""))  # default last
    ttk.Label(euk, text="EUK class:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
    euk_combo = ttk.Combobox(euk, textvariable=euk_var, values=euk_names, state="readonly")
    euk_combo.grid(row=0, column=1, sticky="w", padx=4, pady=2)

    ttk.Label(euk, text="EUK score threshold:").grid(row=0, column=2, sticky="e", padx=4, pady=2)
    ttk.Entry(euk, textvariable=euk_thresh_var, width=10).grid(row=0, column=3, sticky="w", padx=4, pady=2)

    ttk.Checkbutton(euk, text="Stitch EUK segments", variable=euk_stitch_var).grid(row=1, column=0, sticky="w", padx=4, pady=2)
    ttk.Label(euk, text="Stitch gap (px):").grid(row=1, column=2, sticky="e", padx=4, pady=2)
    ttk.Entry(euk, textvariable=euk_gap_var, width=10).grid(row=1, column=3, sticky="w", padx=4, pady=2)

    # Bottom: progress + run
    bottom = ttk.Frame(root)
    bottom.grid(row=6, column=0, columnspan=3, sticky="ew", padx=10, pady=6)
    bottom.columnconfigure(0, weight=1)

    progress = ttk.Progressbar(bottom, orient="horizontal", mode="determinate", length=400, maximum=100)
    progress.grid(row=0, column=0, sticky="ew", padx=6)
    pct_var = tk.StringVar(value="0%")
    ttk.Label(bottom, textvariable=pct_var, width=6, anchor="e").grid(row=0, column=1, sticky="e")

    status_var = tk.StringVar(value="Ready")
    ttk.Label(bottom, textvariable=status_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(4,0))

    run_btn = ttk.Button(bottom, text="Run")
    close_btn = ttk.Button(bottom, text="Close", command=root.destroy)
    run_btn.grid(row=0, column=2, padx=6, rowspan=2, sticky="ns")

    def set_running(running: bool):
        state = "disabled" if running else "!disabled"
        try:
            if running:
                run_btn.state(["disabled"])
            else:
                run_btn.state(["!disabled"])
        except Exception:
            pass

    def run_now():
        if not images_var.get() or not model_var.get():
            messagebox.showerror("Missing inputs", "Please select both image folder and model file.")
            return

        # resolve EUK id from name
        euk_name = euk_var.get()
        euk_id = None
        for i, name in enumerate(CLASS_NAMES):
            if i == 0:
                continue
            if name == euk_name:
                euk_id = i
                break
        if euk_id is None:
            # fallback: last class
            euk_id = len(CLASS_NAMES) - 1

        # Build pairs for progress
        try:
            status_var.set("Scanning images and building pairs…")
            root.update_idletasks()
            pairs = build_pairs(
                images_dir=Path(images_var.get()),
                recursive=bool(recursive_var.get()),
                pairing=pairing_var.get(),
                first_is_og=bool(first_is_og_var.get()),
                autodetect_order=bool(autodetect_var.get()),
                imgsz=2560,
            )
        except Exception as e:
            messagebox.showerror("Error while scanning images", str(e))
            return

        total = len(pairs)
        if total == 0:
            messagebox.showwarning("No pairs found", "No valid image pairs were found with the current settings.")
            return

        # Init progress
        progress.configure(maximum=total, value=0)
        pct_var.set("0%")
        status_var.set(f"0 / {total} starting…")
        set_running(True)

        def progress_cb(done, total_, current_name):
            def _upd():
                progress["value"] = done
                pct = int((done / max(1, total_)) * 100)
                pct_var.set(f"{pct}%")
                status_var.set(f"{done} / {total_} — {current_name}")
            root.after(0, _upd)

        def worker():
            try:
                run_inference(
                    images_dir=Path(images_var.get()),
                    model_path=Path(model_var.get()),
                    output_dir=Path(out_var.get()),
                    conf_thresh=float(conf_var.get()),
                    euk_thresh=float(euk_thresh_var.get()),
                    euk_id=int(euk_id),
                    stitch_euk=bool(euk_stitch_var.get()),
                    stitch_gap_px=int(euk_gap_var.get()),
                    save_viz=bool(save_viz_var.get()),
                    recursive=bool(recursive_var.get()),
                    pairing=pairing_var.get(),
                    first_is_og=bool(first_is_og_var.get()),
                    autodetect_order=bool(autodetect_var.get()),
                    pairs=pairs,
                    progress_cb=progress_cb,
                )
                def done_ok():
                    set_running(False)
                    progress["value"] = total
                    pct_var.set("100%")
                    status_var.set("Done. See output folder for CSVs and (optional) viz.")
                    messagebox.showinfo("Done", "Inference complete.")
                root.after(0, done_ok)
            except Exception as e:
                def done_err():
                    set_running(False)
                    messagebox.showerror("Error", str(e))
                root.after(0, done_err)

        threading.Thread(target=worker, daemon=True).start()

    run_btn.configure(command=run_now)
    root.mainloop()

def main():
    import sys
    if len(sys.argv) == 1:
        launch_gui()
        return
    p = argparse.ArgumentParser(description="Picoalgae detector GUI/CLI")
    p.add_argument("--images", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path, default=Path("./inference_results"))
    p.add_argument("--imgsz", type=int, default=2560)
    p.add_argument("--backbone", type=str, default="resnet101")
    p.add_argument("--conf-thresh", type=float, default=0.10)
    p.add_argument("--euk-thresh", type=float, default=0.20)
    p.add_argument("--euk-id", type=int, default=6)
    p.add_argument("--stitch-euk", action="store_true")
    p.add_argument("--stitch-gap-px", type=int, default=40)
    p.add_argument("--no-viz", dest="save_viz", action="store_false")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--pairing", type=str, choices=["suffix", "order"], default="suffix")
    p.add_argument("--first-is-og", action="store_true", default=True)
    p.add_argument("--autodetect-order", action="store_true", default=False)
    args = p.parse_args()
    if not args.images or not args.model:
        print("Images folder and model path are required.")
        return
    run_inference(
        images_dir=args.images,
        model_path=args.model,
        output_dir=args.output,
        backbone=args.backbone,
        imgsz=args.imgsz,
        conf_thresh=args.conf_thresh,
        euk_thresh=args.euk_thresh,
        euk_id=args.euk_id,
        stitch_euk=args.stitch_euk,
        stitch_gap_px=args.stitch_gap_px,
        save_viz=args.save_viz,
        recursive=args.recursive,
        device=args.device,
        pairing=args.pairing,
        first_is_og=args.first_is_og,
        autodetect_order=args.autodetect_order,
    )

if __name__ == "__main__":
    main()
