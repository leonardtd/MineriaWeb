import albumentations
import torch
import numpy as np
import cv2


class PlazaVeaDataset:
    def __init__(self, image_paths, descriptions, prices, targets, resize=None):

        self.image_paths = image_paths
        self.descriptions = descriptions
        self.prices = prices
        self.targets = targets
        self.resize = resize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        descriptions = self.descriptions[item]
        prices = self.prices[item]
        targets = self.targets[item]

        if self.resize is not None:
            image = cv2.resize(image, (self.resize[1], self.resize[0]))

        image = cv2.normalize(image, None, alpha=0, beta=1,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        image = np.array(image)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "descriptions": torch.tensor(descriptions, dtype=torch.float),
            "prices": torch.tensor(prices, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
