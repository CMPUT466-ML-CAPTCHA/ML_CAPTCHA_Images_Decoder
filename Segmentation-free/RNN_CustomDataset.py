from torchvision import transforms
from PIL import Image
import torch
from torch.utils import data

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

TABLE = NUMBER+ALPHABET  # The table for the captcha


class CustomDataset(data.Dataset):
    def __init__(self, dataset):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 3 chanels
        ])
        self.imgs = []
        self.labels = []
        self.imgs = dataset
        for img in self.imgs:
            self.labels.append(img.name.split("_")[0])
        self.l2 = TABLE

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        img = self.transform(img)
        label = self.labels[index]
        label = self.one_hot(label)
        return img, label

    def one_hot(self, x):
        z = torch.zeros(6, 36)
        for i in range(6):
            index = self.l2.index(x[i].upper())
            z[i][index] = 1
        return z
