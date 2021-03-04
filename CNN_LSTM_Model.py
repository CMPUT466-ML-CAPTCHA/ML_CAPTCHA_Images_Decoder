import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image

# Load the data from the Google Drive
# data_dir = Path("/content/drive/MyDrive/Data")

data_dir = Path("./dataset")  # for local
images = list(data_dir.glob("*.jpg"))  #the size of dataset
print("Number of images found: ", len(images))

# Some sample in the data
sample_images = images[:4]
_, ax = plt.subplots(2, 2, figsize=(5, 3))

image = cv2.imread(str(sample_images[0]))
print("image shape", image.shape)

for i in range(4):
    img = cv2.imread(str(sample_images[i]))
    # print("Shape of image: ", img.shape)
    ax[i // 2, i % 2].imshow(img)
    ax[i // 2, i % 2].axis('off')
# plt.show()

# make up the data set
dataset = []
for image in images:
    label = image.name.split("_")[0]
    dataset.append((str(image), label))

random.shuffle(dataset)
print("Dataset size:", len(dataset))
# Split the data set

# test data
test_data = dataset[8000:]  # 2000 for test

# the part for training
training = dataset[:8000]
valid_data = training[6000:]  # 2000 for validation
train_data = training[:6000]  # 6000 for train

print("test data size:", len(test_data))
print("validation data size:", len(valid_data))
print("train data size:", len(train_data))

# convert the image into np.array, convert the whole data set
# Also convet into the numpy type


def convert_image(dataset: list, height=50, width=200):
    num = len(dataset)
    images = np.zeros((num, height, width), dtype=np.float32)
    labels = [0] * num

    for i in range(num):
        img = cv2.imread(dataset[i][0])
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (width, height))
        # img = (img / 255.).astype(np.float32)
        # print(img.shape)
        # print(type(img))  # out: numpy.ndarray
        # print(img.dtype)  # out: dtype('uint8')
        # print(img)  # BGR
        labels[i] = dataset[i][1]
        images[i, :, :] = img

    return images, np.array(labels)


# convetion
train_data, train_labels = convert_image(train_data)
test_data, test_labels = convert_image(test_data)
valid_data, valid_labels = convert_image(valid_data)

# IN PROCESS