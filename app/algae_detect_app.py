
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from torchvision.ops import nms

# ---------------------------
# User-editable: class names & colors
# ---------------------------
# Index 0 is background; model predicts 1..N for real classes.
CLASS_NAMES = [
    "background",
    "class1",
    "class2",
    "class3",
    "class4",
    "class5",
    "class6",
]

# BGR colors for drawing (OpenCV). Will wrap if you have > len(COLORS) classes.
COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (0, 255, 0),
    (120, 120, 0),
    (0, 120, 24),
    (128, 128, 128),
]

# ---------------------------
# Model: must match your training setup (6-channel input)
# ---------------------------
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform

def get_faster_rcnn_model(num_classes=7, image_size=(2560, 2560), backbone='resnet101'):
    backbone = resnet_fpn_backbone(backbone_name=backbone, weights=None)
    old_conv = backbone.body.conv1
    new_conv = torch.nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        # Copy pretrained weights for the first 3 channels
        new_conv.weight[:, :3] = old_conv.weight
        # Initialize extra 3 channels as mean of pretrained weights
        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:] = mean_weight.repeat(1, 3, 1, 1)

    backbone.body.conv1 = new_conv

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes
    )
    # Custom transform for 6-channel input (keep consistent with training)
    model.transform = GeneralizedRCNNTransform(
        min_size=image_size[0],
        max_size=image_size[1],
        image_mean=[0.0] * 6,
        image_std=[1.0] * 6
    )
    return model

# ---------------------------
# Inference utils
# ---------------------------
def _pair_iterator(images_dir: Path, recursive: bool):
    pat = "**/*_og.png" if recursive else "*_og.png"
    for og_path in sorted(images_dir.glob(pat)):
        red_path = og_path.parent / og_path.name.replace("_og", "_red")
        if red_path.exists():
            yield og_path, red_path
        else:
            print(f"[WARN] Missing _red pair for: {og_path.name} (skipping)", flush=True)

def _prepare_six_channel(og_path: Path, red_path: Path, imgsz: int):
    img1 = cv2.imread(str(og_path)).astype(np.float32) / 255.0
    img2 = cv2.imread(str(red_path)).astype(np.float32) / 255.0
    img1 = cv2.resize(img1, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    fused = np.concatenate([img1, img2], axis=-1)              # (H, W, 6)
    fused_t = torch.from_numpy(np.transpose(fused, (2, 0, 1)))  # (6, H, W)
    return fused, fused_t, img1  # return original (float01) for visualization

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

def run_inference(
    images_dir: Path,
    model_path: Path,
    output_dir: Path,
    backbone: str = "resnet101",
    imgsz: int = 2560,
    conf_thresh: float = 0.10,
    iou_thresh: float = 0.10,
    save_viz: bool = True,
    recursive: bool = False,
    device: str = "auto",
):
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "viz"
    if save_viz:
        viz_dir.mkdir(exist_ok=True)

    # Load model
    model = get_faster_rcnn_model(num_classes=len(CLASS_NAMES), image_size=(imgsz, imgsz), backbone=backbone)
    model.load_state_dict(torch.load(str(model_path), map_location=device_t))
    model.to(device_t).eval()

    # CSV outputs
    per_image_csv = output_dir / "per_image_counts.csv"
    summary_csv = output_dir / "summary_counts.csv"
    detections_csv = output_dir / "detections_full.csv"

    # Accumulators
    total_counts = {i: 0 for i in range(1, len(CLASS_NAMES))}  # ignore background index 0
    per_image_rows = []
    det_rows = []

    num_images = 0
    for og_path, red_path in _pair_iterator(images_dir, recursive):
        num_images += 1
        fused, fused_t, img1_float = _prepare_six_channel(og_path, red_path, imgsz)
        image_tensor = fused_t.unsqueeze(0).to(device_t, dtype=torch.float32)
        with torch.no_grad():
            pred = model(image_tensor)[0]
        boxes = pred["boxes"].detach().cpu()
        labels = pred["labels"].detach().cpu()
        scores = pred["scores"].detach().cpu()

        # NMS + thresholds
        keep = nms(boxes, scores, iou_thresh)
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        keep2 = scores > conf_thresh
        boxes, labels, scores = boxes[keep2], labels[keep2], scores[keep2]

        # Count per image
        image_counts = {i: 0 for i in range(1, len(CLASS_NAMES))}
        for li in labels.tolist():
            if li == 0:  # background shouldn't appear, but just in case
                continue
            if li not in image_counts:
                image_counts[li] = 0
            image_counts[li] += 1
            if li not in total_counts:
                total_counts[li] = 0
            total_counts[li] += 1

        # Save per-image row
        row = {"image": og_path.name, "og_path": str(og_path), "red_path": str(red_path), "total_detections": int(len(labels))}
        for i in range(1, len(CLASS_NAMES)):
            row[CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class{i}"] = image_counts.get(i, 0)
        per_image_rows.append(row)

        # Save detections rows (one per box)
        for (x1, y1, x2, y2), li, sc in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            cname = CLASS_NAMES[li] if li < len(CLASS_NAMES) else f"class{li}"
            det_rows.append({
                "image": og_path.name,
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "score": float(sc),
                "label_id": int(li),
                "class_name": cname,
            })

        # Visualization
        if save_viz:
            img_vis = (img1_float * 255).astype(np.uint8).copy()  # visualize on the original _og image
            img_vis = draw_detections(img_vis, boxes, labels, scores)
            out_path = viz_dir / og_path.name
            cv2.imwrite(str(out_path), img_vis)

        print(f"[{num_images}] {og_path.name} -> {len(labels)} detections", flush=True)

    # Write CSVs
    # Per-image counts
    if per_image_rows:
        cols = ["image", "og_path", "red_path", "total_detections"] + [
            (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class{i}") for i in range(1, len(CLASS_NAMES))
        ]
        with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in per_image_rows:
                writer.writerow(r)

    # Summary counts
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "count"])
        for i in range(1, len(CLASS_NAMES)):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class{i}"
            writer.writerow([i, name, total_counts.get(i, 0)])
        writer.writerow([])
        writer.writerow(["total_images", num_images])
        writer.writerow(["conf_thresh", f"{conf_thresh:.2f}"])
        writer.writerow(["iou_thresh", f"{iou_thresh:.2f}"])

    # Full detections
    if det_rows:
        cols = ["image", "x1", "y1", "x2", "y2", "score", "label_id", "class_name"]
        with open(detections_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in det_rows:
                writer.writerow(r)

    print("\nDone.")
    print(f"Per-image counts  : {per_image_csv}")
    print(f"Summary counts    : {summary_csv}")
    if det_rows:
        print(f"Detections (boxes): {detections_csv}")
    if save_viz:
        print(f"Annotated images  : {viz_dir}")

# ---------------------------
# GUI helper (Tkinter) — so colleagues can run without terminal args
# ---------------------------
def _maybe_gui(args):
    # If any critical arg is missing, pop a minimal GUI to collect them.
    need_gui = not args.images or not args.model
    if not need_gui:
        return args

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        print("[WARN] Tkinter not available; falling back to CLI args only.")
        return args

    root = tk.Tk()
    root.title("Picoalgae Detector")
    root.geometry("520x220")
    root.resizable(False, False)

    # Simple labels + buttons
    images_var = tk.StringVar(value=str(args.images) if args.images else "")
    model_var = tk.StringVar(value=str(args.model) if args.model else "")
    out_var = tk.StringVar(value=str(args.output) if args.output else "")
    save_viz_var = tk.BooleanVar(value=bool(args.save_viz))
    conf_var = tk.DoubleVar(value=float(args.conf_thresh))
    iou_var = tk.DoubleVar(value=float(args.iou_thresh))
    recursive_var = tk.BooleanVar(value=bool(args.recursive))

    def pick_images():
        d = filedialog.askdirectory(title="Select image folder (with *_og.png & *_red.png)")
        if d:
            images_var.set(d)
    def pick_model():
        p = filedialog.askopenfilename(title="Select model .pth file", filetypes=[("PyTorch weights", "*.pth")])
        if p:
            model_var.set(p)
    def pick_out():
        d = filedialog.askdirectory(title="Select output folder (will be created if missing)")
        if d:
            out_var.set(d)
    def run_now():
        if not images_var.get() or not model_var.get():
            messagebox.showerror("Missing inputs", "Please select both image folder and model file.")
            return
        root.destroy()

    # Layout
    import tkinter as tk
    row = 0
    tk.Label(root, text="Image folder:").grid(row=row, column=0, sticky="e", padx=6, pady=6)
    tk.Entry(root, textvariable=images_var, width=44).grid(row=row, column=1, padx=6)
    tk.Button(root, text="Browse", command=pick_images).grid(row=row, column=2, padx=6)
    row += 1

    tk.Label(root, text="Model .pth:").grid(row=row, column=0, sticky="e", padx=6, pady=6)
    tk.Entry(root, textvariable=model_var, width=44).grid(row=row, column=1, padx=6)
    tk.Button(root, text="Browse", command=pick_model).grid(row=row, column=2, padx=6)
    row += 1

    tk.Label(root, text="Output folder:").grid(row=row, column=0, sticky="e", padx=6, pady=6)
    tk.Entry(root, textvariable=out_var, width=44).grid(row=row, column=1, padx=6)
    tk.Button(root, text="Browse", command=pick_out).grid(row=row, column=2, padx=6)
    row += 1

    tk.Label(root, text="Conf thresh:").grid(row=row, column=0, sticky="e", padx=6, pady=6)
    tk.Entry(root, textvariable=conf_var, width=10).grid(row=row, column=1, sticky="w", padx=6)
    tk.Label(root, text="IoU thresh:").grid(row=row, column=1, sticky="e", padx=110)
    tk.Entry(root, textvariable=iou_var, width=10).grid(row=row, column=2, sticky="w")
    row += 1

    tk.Checkbutton(root, text="Save annotated images", variable=save_viz_var).grid(row=row, column=1, sticky="w", padx=6)
    tk.Checkbutton(root, text="Search subfolders", variable=recursive_var).grid(row=row, column=2, sticky="w")
    row += 1

    tk.Button(root, text="Run", command=run_now, width=12).grid(row=row, column=1, pady=12)
    tk.Button(root, text="Cancel", command=root.destroy, width=12).grid(row=row, column=2, pady=12)

    root.mainloop()

    # Transfer back
    args.images = Path(images_var.get()) if images_var.get() else None
    args.model = Path(model_var.get()) if model_var.get() else None
    args.output = Path(out_var.get()) if out_var.get() else None
    args.save_viz = bool(save_viz_var.get())
    args.conf_thresh = float(conf_var.get())
    args.iou_thresh = float(iou_var.get())
    args.recursive = bool(recursive_var.get())
    return args

# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser(description="Picoalgae Faster R-CNN detector — batch count + annotated images")
    p.add_argument("--images", type=Path, help="Folder containing *_og.png and *_red.png pairs")
    p.add_argument("--model", type=Path, help="Path to trained .pth weights")
    p.add_argument("--output", type=Path, default=Path("./inference_results"), help="Output folder")
    p.add_argument("--imgsz", type=int, default=2560, help="Resize square side, must match training")
    p.add_argument("--backbone", type=str, default="resnet101", help="resnet34/resnet50/resnet101 (must match training)")
    p.add_argument("--conf-thresh", type=float, default=0.10, help="Confidence threshold")
    p.add_argument("--iou-thresh", type=float, default=0.10, help="IoU threshold for NMS")
    p.add_argument("--no-viz", dest="save_viz", action="store_false", help="Disable saving annotated images")
    p.add_argument("--recursive", action="store_true", help="Also scan subfolders for *_og.png")
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'")
    args = p.parse_args()

    # Pop a tiny GUI if key args are missing
    args = _maybe_gui(args)

    if not args.images or not args.model:
        print("Images folder and model path are required. (Provide via CLI or the small GUI.)")
        sys.exit(2)

    run_inference(
        images_dir=args.images,
        model_path=args.model,
        output_dir=args.output,
        backbone=args.backbone,
        imgsz=args.imgsz,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        save_viz=args.save_viz,
        recursive=args.recursive,
        device=args.device,
    )

if __name__ == "__main__":
    main()
