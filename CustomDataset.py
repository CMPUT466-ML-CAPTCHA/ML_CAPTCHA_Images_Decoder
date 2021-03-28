from PIL import Image
import torch
import cv2
import numpy as np

NUMBER = ["{}".format(x) for x in range(10)]
ALPHABET = list(string.ascii_uppercase)
TABLE = NUMBER+ALPHABET

# Custom dataset
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

# Convert the CAPTCHA into the (6*36,) vector (6 characters, 10 numbers + 26 uppercase/capital characters)
# 1 means the CAPTCHA image contains this character in TABLE, 0 means otherwise
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

# Convert the vector to the CAPTCHA (the input vector is different from the vector above)
# Example: input: [1,2,34,2,6,7]; output: "23Y378"
def vector_to_captcha(vector):
    captcha_str = ""
    for i in vector:
        captcha_str += TABLE[i]
    return captcha_str