import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2


class YOLO6ChannelDataset(Dataset):
    def __init__(self, left_img_dir, right_img_dir, label_dir, img_size=(640, 640)):
        self.left_img_dir = left_img_dir
        self.right_img_dir = right_img_dir
        self.label_dir = label_dir
        self.img_size = img_size

        # Only collect files that exist in all three folders
        self.files = sorted([
            f for f in os.listdir(left_img_dir)
            if os.path.isfile(os.path.join(right_img_dir, f)) and
               os.path.isfile(os.path.join(label_dir, f.replace(".png", ".txt")))
        ])

        self.resize = transforms.Resize(img_size)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def load_yolo_labels(self, label_path):
        labels = []
        if not os.path.exists(label_path):
            return torch.zeros((0, 5))
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts)
                # These are already normalized (0-1), keep as is
                labels.append([class_id, x_center, y_center, w, h])
        return torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        # print(f"[Debug] Loading item {idx}")

        filename = self.files[idx]

        # Load and resize images
        left_img = Image.open(os.path.join(self.left_img_dir, filename)).convert('RGB')
        right_img = Image.open(os.path.join(self.right_img_dir, filename)).convert('RGB')
        left_img = self.resize(left_img)
        right_img = self.resize(right_img)

        left_tensor = self.to_tensor(left_img)
        right_tensor = self.to_tensor(right_img)

        # Combine into 6 channels
        img_6ch = torch.cat([left_tensor, right_tensor], dim=0)

        # Load labels
        label_path = os.path.join(self.label_dir, filename.replace(".png", ".txt"))
        targets = self.load_yolo_labels(label_path)

        return img_6ch, targets