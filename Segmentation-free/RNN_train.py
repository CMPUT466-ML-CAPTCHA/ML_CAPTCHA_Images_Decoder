# The pure version of the CNN model
import itertools
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
import torch
from RNN_CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from RNN import RNN
import torch.nn as nn

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
TABLE = NUMBER+ALPHABET  # The table for the captcha
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

# Load the data from the Google Drive
# data_dir = Path("/content/drive/MyDrive/Data")

# path of data set for local
# data_dir = Path("./archive")
data_dir = Path("D:/Task/FS")

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

train_set = CustomDataset(train_data)
valid_set = CustomDataset(valid_data)
test_set = CustomDataset(test_data)

train_dataloader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)
test_dataloader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_set, batch_size=1, shuffle=True)

model = RNN().to(device)


def update_matrix(preds, label, conf_matrix):
    for p, t in zip(preds, label):
        conf_matrix[TABLE.index(p), TABLE.index(t)] += 1
    return conf_matrix


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
    captcha_str = ""
    for i in vector:
        captcha_str += TABLE[i]
    return captcha_str


def train(model, train_dataloader, valid_dataloader, device):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(60):
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_dataloader)):
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


def test(model, test_dataloader, device, conf_matrix):
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
            conf_matrix = update_matrix(vector_to_captcha(output), vector_to_captcha(labels), conf_matrix)
        accuracy = num_correct / num_total * 100
        return accuracy


print("\nTraining")
train(model, train_dataloader, valid_dataloader, device)

conf_matrix = torch.zeros(36, 36)
print("\nTesting")
accuracy = test(model, test_dataloader, device, conf_matrix)
print("Accuracy: {}".format(accuracy))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


plot_confusion_matrix(conf_matrix.numpy(), TABLE, title="Confusion matrix for RNN Model")
