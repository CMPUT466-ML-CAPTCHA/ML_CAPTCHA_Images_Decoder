# The pure version of the CNN model
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image
from CustomDataset import CustomDataset
from torchvision import transforms
import FunctionTest

# Load the data from the Google Drive
# data_dir = Path("/content/drive/MyDrive/Data")

# path of data set for local
data_dir = Path("./dataset")

# images: the list contain the path of each images
images = list(data_dir.glob("*.jpg"))
print("Number of images found: ", len(images))

random.shuffle(images)
print("Dataset size:", len(images))
# Split the data set

# test data
test_data = images[8000:]  # 2000 for test

# the part for training
training = images[:8000]
valid_data = training[6000:]  # 2000 for validation
train_data = training[:6000]  # 6000 for train

print("test set size:", len(test_data))
print("validation set size:", len(valid_data))
print("train set size:", len(train_data))

train_set = CustomDataset(train_data, transform=transforms.ToTensor)
valid_set = CustomDataset(valid_data, transform=transforms.ToTensor)
test_set = CustomDataset(test_data, transform=transforms.ToTensor)

# CNN Model
# Coming soon