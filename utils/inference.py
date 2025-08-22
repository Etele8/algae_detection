import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision.ops import nms

# ---- Config ----
image_dir = Path("/workspace/data/test")  # folder with *_og.png and *_red.png
output_dir = Path("/workspace/data/inference_results_101_2560_no_alb")
model_path = "/workspace/data/101_2560_no_alb.pth"
imgsz = 2560
backbone = 'resnet101'
conf_thresh = 0.1
iou_thresh = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir.mkdir(parents=True, exist_ok=True)

# ---- Load Model ----
model = get_faster_rcnn_model(num_classes=7, image_size=(imgsz, imgsz), backbone=backbone)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ---- Inference Loop ----
image_files = sorted(image_dir.glob("*_og.png"))

for img1_path in image_files:
    img2_path = image_dir / img1_path.name.replace("_og", "_red")
    if not img2_path.exists():
        print(f"Missing red image for {img1_path.name}, skipping.")
        continue

    # Load and prepare 6-channel input
    img1 = cv2.imread(str(img1_path)).astype(np.float32) / 255.0
    img2 = cv2.imread(str(img2_path)).astype(np.float32) / 255.0

    img1 = cv2.resize(img1, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (imgsz, imgsz), interpolation=cv2.INTER_AREA)

    fused = np.concatenate([img1, img2], axis=-1)  # (H, W, 6)
    fused = np.transpose(fused, (2, 0, 1))  # (6, H, W)
    image_tensor = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    boxes = prediction['boxes'].cpu()
    labels = prediction['labels'].cpu()
    scores = prediction['scores'].cpu()

    # NMS + confidence threshold
    keep = nms(boxes, scores, iou_thresh)
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    keep = scores > conf_thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    # Draw boxes on original image
    img_vis = (img1 * 255).astype(np.uint8)

    COLORS = [(0,0,255), (255,0,0), (0,255,255), (0,0,255), (120,120,0), (0,120,24), (128,128,128)]
    for box, label, score in zip(boxes, labels, scores):
        color = COLORS[label.item() % len(COLORS)]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img_vis, f"{label.item()} ({score:.2f})",
            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )


    # Save result
    out_path = output_dir / img1_path.name
    cv2.imwrite(str(out_path), img_vis)
    print(f"Saved: {out_path.name}")