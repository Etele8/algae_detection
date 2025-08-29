import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

class ChannelDimming6(ImageOnlyTransform):
    def __init__(self, p=0.3, min_scale=0.3, max_scale=0.7):
        super().__init__(p=p)
        self.min_scale, self.max_scale = min_scale, max_scale
    def apply(self, img, **params):
        if np.random.rand() < 0.5:
            s = np.random.uniform(self.min_scale, self.max_scale)
            img[..., 0:3] *= s
        else:
            s = np.random.uniform(self.min_scale, self.max_scale)
            img[..., 3:6] *= s
        return img

class PoissonNoise(ImageOnlyTransform):
    def __init__(self, lam_scale=0.05, p=0.2):
        super().__init__(p=p); self.lam_scale = lam_scale
    def apply(self, img, **params):
        lam = np.clip(img * self.lam_scale, 0, 1)
        noise = np.random.poisson(lam * 255.0) / 255.0
        out = img + noise - lam
        return np.clip(out, 0.0, 1.0)

class RedCLAHE(ImageOnlyTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.clip_limit = clip_limit; self.tile_grid_size = tile_grid_size
    def apply(self, img, **params):
        red = (img[..., 3:6] * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        for ch in range(3): red[..., ch] = clahe.apply(red[..., ch])
        img[..., 3:6] = red.astype(np.float32) / 255.0
        return img

def build_train_aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.25),

        A.RandomBrightnessContrast(0.15, 0.15, p=0.30),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 1.0), p=0.20),
        A.MotionBlur(blur_limit=(5, 9), p=0.15),

        RedCLAHE(p=0.50),
        ChannelDimming6(p=0.30),
        PoissonNoise(p=0.20),
    ],
    bbox_params=A.BboxParams(format="albumentations", label_fields=["class_labels"], min_visibility=0.30))
