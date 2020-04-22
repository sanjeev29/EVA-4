import albumentations as A
import numpy as np
import cv2

from albumentations.pytorch import ToTensor

fill_value = (0.485*255, 0.456*255, 0.406*255)


class CustomAlbumentation():
    """
    Custom albumentation class
    """
    def __init__(self, mean, std):
        self.transform = A.Compose([
                A.PadIfNeeded(min_height=80, min_width=80, border_mode=4, value=None, p=1.0),
                A.RandomCrop(height=64, width=64, always_apply=True),
                A.HorizontalFlip(),
                A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, p=1.0),
                A.Normalize(mean=mean, std=std, always_apply=True),
                ToTensor(),
            ])
        
    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image = augmented['image']
        return image