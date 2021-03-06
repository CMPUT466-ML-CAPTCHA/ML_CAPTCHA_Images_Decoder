from PIL import Image
import torch
import cv2
import numpy as np

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
TABLE = NUMBER+ALPHABET


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None, target_transform=None, height=50, width=200):
        self.transform = transform
        self.target_transform = target_transform
        self.images = images
        self.width = width
        self.height = height

    def __getitem__(self, index):
        image = cv2.imread(str(self.images[index]))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        image = cv2.resize(image, (self.width, self.height))
        _, image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
        assert len(self.images[index].name.split("_")[0].upper()) == 6
        label = captcha_to_vector(
            self.images[index].name.split("_")[0])

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


def captcha_to_vector(captcha_str):
    captcha_str = captcha_str.upper()
    vector = np.zeros(36*6, dtype=np.float32)
    for i, char in enumerate(captcha_str):
        ascii = ord(char)
        if 48 <= ascii <= 57:
            index = ascii-48
        elif 65 <= ascii <= 90:
            index = ascii-ord('A')+10
        vector[i*36+index] = 1.0
    return vector


def vector_to_captcha(vector):
    pass
