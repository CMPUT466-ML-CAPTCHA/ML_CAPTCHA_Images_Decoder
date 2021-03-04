from PIL import Image
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self, transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, label = self.data[index]


# IN PROCESSING