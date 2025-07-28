import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import numpy as np
import cv2
from pathlib import Path

# ---------------------------
# Dataset
# ---------------------------

class PicoAlgaeDataset(Dataset):
    def __init__(self, image_pairs, label_dir, imgsz=640):
        self.label_dir = Path(label_dir)
        self.samples = []
        self.imgsz = imgsz

        for img1_path, img2_path in image_pairs:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))

            if img1 is None or img2 is None:
                print(f"Skipping pair: {img1_path}, {img2_path}")
                continue

            img1 = img1.astype(np.float32) / 255.0
            img2 = img2.astype(np.float32) / 255.0

            img1 = cv2.resize(img1, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
            
            fused = np.concatenate([img1, img2], axis=-1)
            fused = np.transpose(fused, (2, 0, 1))  # (6, H, W)
            image_tensor = torch.tensor(fused, dtype=torch.float32)

            h, w = image_tensor.shape[1:]
            label_path = self.label_dir / (img1_path.stem + '.txt')

            boxes, labels = [], []
            if label_path.exists() and label_path.stat().st_size > 0:
                with open(label_path) as f:
                    for line in f:
                        cls, cx, cy, bw, bh = map(float, line.strip().split())
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls))

            if boxes:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels}
            self.samples.append((image_tensor, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------
# Model
# ---------------------------

def get_faster_rcnn_model(num_classes=4, image_size=(512, 512)):
    backbone = resnet_fpn_backbone(backbone_name='resnet18', weights=None)
    old_conv = backbone.body.conv1
    new_conv = torch.nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight

    backbone.body.conv1 = new_conv

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes
    )

    # ✅ Custom transform for 6-channel input
    model.transform = GeneralizedRCNNTransform(
        min_size=image_size[0],
        max_size=image_size[1],
        image_mean=[0.0] * 6,
        image_std=[1.0] * 6
    )

    return model

# ---------------------------
# Training
# ---------------------------

def train(model, dataloader, device, num_epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in dataloader:
            print("Loaded batch")
            images = [img.to(device) for img in images]
            print("Images moved to device")
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            print("Targets moved to device")

            loss_dict = model(images, targets)
            print("Loss computed")
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            print("Backward done")
            optimizer.step()
            print("Step done")

            epoch_loss += losses.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

# ---------------------------
# Main
# ---------------------------

def main():
    image_dir = Path('D:/intezet/Bogi/Yolo/data_rcnn/images')
    label_dir = Path('D:/intezet/Bogi/Yolo/data_rcnn/labels')

    image_files = sorted([f for f in image_dir.glob("*_og.png")])
    image_pairs = [(f, image_dir / f.name.replace('_og', '_red')) for f in image_files]

    dataset = PicoAlgaeDataset(image_pairs, label_dir, imgsz=640)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))

    model = get_faster_rcnn_model(num_classes=4, image_size=(640, 640))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(model, dataloader, device)

if __name__ == "__main__":
    main()