import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, image_data, image_labels, transform=None):
        self.transform = transform
        self.image_data = image_data
        self.image_labels = image_labels

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_data[idx]))
        label = self.image_labels[idx]

        if self.is_grayscale_image(image):
            image = np.stack((image,) * 3, axis=-1)

        if self.transform:
            image = self.transform(Image.fromarray(image))

        return image, label

    def is_grayscale_image(self, image):
        return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)