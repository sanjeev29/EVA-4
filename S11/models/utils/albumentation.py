import albumentations as A
import numpy as np
import cv2

from albumentations.pytorch import ToTensor

mean = [0.4914*255, 0.4822*255, 0.4465*255]


class CustomAlbumentation():
    """
    Custom albumentation class
    """
    def __init__(self, mean, std):
        self.transform = A.Compose([
                A.Normalize(
                    mean=mean,
                    std=std,
                    always_apply=True
                ),
                A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_REPLICATE, p=1),
                A.RandomCrop(height=32, width=32, p=1),
                A.HorizontalFlip(p=1),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, fill_value=mean, p=1),
                ToTensor(),
            ])
        
    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image = augmented['image']
        return image