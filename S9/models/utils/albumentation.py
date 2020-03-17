class CustomAlbumentation():
    """
    Custom albumentation class
    """
    def __init__(self):
        self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=mean,
                    std=std,
                    always_apply=True
                ),
                A.PadIfNeeded(min_height=32, min_width=32, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, p=0.5),
                ToTensor(),
            ])
        
    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image = augmented['image']
        return image