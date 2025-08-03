import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import numpy as np
import cv2
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

# ---------------------------
# Config
# ---------------------------
num_classes = 7  # <-- 6 classes + 1 background!

# ---------------------------
# Dataset
# ---------------------------

class PicoAlgaeDataset(Dataset):
    def __init__(self, image_pairs, label_dir, imgsz=640):
        self.label_dir = Path(label_dir)
        self.samples = []
        self.imgsz = imgsz
        self.i = 0
        for img1_path, img2_path in image_pairs:
            self.i += 1
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
                        labels.append(int(cls) + 1)

            if boxes:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels}
            self.samples.append((image_tensor, target))
            print(f"{self.i}/{len(image_pairs)} are done")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    # def __getitem__(self, idx):
    #     image, target = self.samples[idx]
    #     print("labels in sample:", target["labels"].tolist())
    #     return image, target


# ---------------------------
# Model
# ---------------------------

def get_faster_rcnn_model(num_classes=7, image_size=(512, 512), backbone='resnet34'):
    backbone = resnet_fpn_backbone(backbone_name=backbone, weights=None)
    old_conv = backbone.body.conv1
    new_conv = torch.nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight
        # mean_weight = old_conv.weight.mean(dim=1, keepdim=True)  # (out_channels, 1, 7, 7)
        # new_conv.weight = mean_weight.repeat(1, 6, 1, 1)  # (out_channels, 6, 7, 7)

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

    print("model loaded")
    return model

# ---------------------------
# Training
# ---------------------------

def train(model, dataloader, device, num_epochs=10, lr=1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    writer = SummaryWriter(log_dir="runs/algae_detection")
    scaler = GradScaler('cuda')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()
            # writer.add_scalar("Loss/train", epoch_loss, epoch)
            # writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)
            # Visualize only 3 channels for TensorBoard
            # img_vis = images[0][:3, :, :].unsqueeze(0)
            # writer.add_images("SampleInput", img_vis, epoch)

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    # Save after training
    torch.save(model.state_dict(), "fasterrcnn_6ch_picoalgae.pth")
    print("Model saved to fasterrcnn_6ch_picoalgae.pth")
    writer.close()

# ---------------------------
# Main
# ---------------------------

def main():
    image_dir = Path('/workspace/data/images')
    label_dir = Path('/workspace/data/labels')

    image_files = sorted([f for f in image_dir.glob("*_og.png")])
    image_pairs = [(f, image_dir / f.name.replace('_og', '_red')) for f in image_files]

    dataset = PicoAlgaeDataset(image_pairs, label_dir, imgsz=3150)
    print(f"Loaded {len(dataset)} samples")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, collate_fn=lambda x: tuple(zip(*x)))

    model = get_faster_rcnn_model(num_classes=num_classes, image_size=(3150, 3150), backbone='resnet50')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(model, dataloader, device, num_epochs=8, lr=1e-4)

if __name__ == "__main__":
    main()