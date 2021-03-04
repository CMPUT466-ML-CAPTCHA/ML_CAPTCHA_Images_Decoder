from PIL import Image
import torch
import cv2
import numpy as np


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images,
                 transform=None,
                 target_transform=None,
                 height=50,
                 width=200):
        self.transform = transform
        self.num = len(images)
        self.target_transform = target_transform

        self.images = np.zeros((self.num, height, width), dtype=np.float32)
        self.labels = [0] * self.num

        for i in range(self.num):
            img = cv2.imread(str(images[i]))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            img = cv2.resize(img, (width, height))
            self.labels[i] = images[i].name.split("_")[0]
            self.images[i, :, :] = img

        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.images[index]
        if self.transform != None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.num
