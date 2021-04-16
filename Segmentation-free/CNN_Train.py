# The pure version of the CNN model
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image
import torch
from CustomDataset import CustomDataset, vector_to_captcha
from torchvision import transforms
from torch.utils.data import DataLoader
from CNN_Model import CNN
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

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

train_set = CustomDataset(train_data, transform=transforms.ToTensor())
valid_set = CustomDataset(valid_data, transform=transforms.ToTensor())
test_set = CustomDataset(test_data, transform=transforms.ToTensor())

train_dataloader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)
test_dataloader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_set, batch_size=1, shuffle=True)

model = CNN().to(device)


def train(model, train_dataloader, valid_dataloader, device):
    model.train()
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(15):
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            predict = model(images)
            optimizer.zero_grad()
            loss = criterion(predict, labels)
            loss.backward()
            optimizer.step()
        accuracy = valid(model, valid_dataloader, device)
        print("epoch: {} loss: {:.10f} accuracy: {:.4f}".format(
            (epoch+1), loss.item(), accuracy))


def valid(model, valid_dataloader, device):
    num_correct = 0  # the counter for the correct items
    num_total = len(valid_dataloader)  # the counter for the total items
    model.eval()  # set the evaluation state of the model
    with torch.no_grad():
        for _, (images, labels) in enumerate(valid_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            labels = labels.reshape(6, 36)
            output = output.reshape(6, 36)
            # get the captcha character index
            labels = torch.argmax(labels, dim=1)
            # get the predict character index
            output = torch.argmax(output, dim=1)
            num_correct += ((labels == output).sum() == 6).sum().item()
        accuracy = num_correct / num_total * 100
        return accuracy


def test(model, test_dataloader, device):
    num_correct = 0  # the counter for the correct items
    num_total = len(test_dataloader)  # the counter for the total items
    model.eval()  # set the evaluation state of the model

    with torch.no_grad():
        for _, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            labels = labels.reshape(6, 36)
            output = output.reshape(6, 36)
            # get the captcha character index
            labels = torch.argmax(labels, dim=1)
            # get the predict character index
            output = torch.argmax(output, dim=1)
            num_correct += ((labels == output).sum() == 6).sum().item()
        accuracy = num_correct / num_total * 100
        return accuracy


print("\nTraining")
train(model, train_dataloader, valid_dataloader, device)

print("\nTesting")
accuracy = test(model, test_dataloader, device)
print("Accuracy: {}".format(accuracy))

# Sample

model.eval()  # set the evaluation state of the model
_, ax = plt.subplots(2, 3, figsize=(30, 15))
with torch.no_grad():
    for i in range(6):
        image, label = next(iter(test_dataloader))

        image = image.to(device)
        label = label.to(device)
        output = model(image)
        label = label.reshape(6, 36)
        output = output.reshape(6, 36)
        label = torch.argmax(label, dim=1)
        output = torch.argmax(output, dim=1)
        origin = vector_to_captcha(label)
        predict = vector_to_captcha(output)
        print("Origin: "+origin+"   Predict: "+predict)
