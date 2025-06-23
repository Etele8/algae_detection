import cv2
import numpy as np
import os
from ultralytics.data.dataset import BaseDataset
from ultralytics.utils import LOGGER

class TwoChannelDataset(BaseDataset):
    def __init__(self, img_path, imgsz, cache=False, *args, **kwargs):
        # Extract custom config without passing to BaseDataset
        self.data_config = kwargs.pop('data', {})
        self.original_dir = self.data_config.get('original_dir', 'original')
        self.enhanced_dir = self.data_config.get('enhanced_dir', 'enhanced')
        print(f"Initializing dataset with img_path: {img_path}, imgsz: {imgsz}")
        print(f"Original dir: {self.original_dir}, Enhanced dir: {self.enhanced_dir}")

        # Use img_path as the base for original images
        full_img_path = os.path.join(img_path, self.original_dir)
        print(f"Full image path: {full_img_path}")

        # Initialize BaseDataset with minimal args
        super().__init__(img_path=full_img_path, imgsz=imgsz, cache=cache, *args, **kwargs)
        self.img_files = [f for f in self.img_files if os.path.exists(f)]
        print(f"Filtered image files: {self.img_files}")

    def __getitem__(self, index):
        print(f"Processing index: {index}")
        img_path = self.img_files[index]
        print(f"Processing image: {img_path}")
        original_img = cv2.imread(img_path)
        enhanced_path = img_path.replace(self.original_dir, self.enhanced_dir)
        print(f"Enhanced path: {enhanced_path}")
        enhanced_img = cv2.imread(enhanced_path)

        if original_img is None or enhanced_img is None:
            LOGGER.error(f"Failed to load {img_path} or {enhanced_path}")
            raise FileNotFoundError(f"Failed to load {img_path} or {enhanced_path}")

        img_size = self.imgsz
        original_img = cv2.resize(original_img, (img_size, img_size))
        enhanced_img = cv2.resize(enhanced_img, (img_size, img_size))

        if len(original_img.shape) == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        if len(enhanced_img.shape) == 2:
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)

        img = np.stack((original_img, enhanced_img), axis=0)
        img = img.transpose((1, 2, 0, 3))  # Shape: (height, width, 2, 3)
        img = img.reshape(img_size, img_size, 6)  # 6 channels

        label = self.labels[index].copy()
        print(f"Labels for {img_path}: {label}")
        if label.size > 0:
            label[:, 1:] = self.xywhn2xyxy(label[:, 1:], img.shape[1], img.shape[0], img_size, img_size)
        else:
            print(f"No labels found for {img_path}")

        return img, label, self.img_files[index]

    def __len__(self):
        return len(self.img_files)

    def xywhn2xyxy(self, x, w=640, h=640, padw=0, padh=0):
        y = x.copy()
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
        return y