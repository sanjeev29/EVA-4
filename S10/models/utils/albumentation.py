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
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=mean,
                    std=std,
                    always_apply=True
                ),
#                 A.PadIfNeeded(min_height=32, min_width=32, border_mode=cv2.BORDER_REPLICATE, p=1),
#                 A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, p=0.5),
                A.CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=mean, p=1),
                ToTensor(),
            ])
        
    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image = augmented['image']
        return image