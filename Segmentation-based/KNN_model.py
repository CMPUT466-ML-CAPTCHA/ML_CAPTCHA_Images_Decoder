import numpy as np
import cv2
import pathlib
import matplotlib.pyplot as plt
import scipy.ndimage
import math
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import string

# Parameters:
NUMBERS = list(string.digits)
ALPHABET = list(string.ascii_uppercase)
TABLE = NUMBERS + ALPHABET # The table for CAPTCHA
LEN_OF_TABLE = len(TABLE) # in total 10+26 alphanumeric characters
BATCH_SIZE = 100
LEN_OF_CAPTCHA = 6 # each picture contains 6 characters
LEARNING_RATE = 0.001 # η

# Convert the CAPTCHA into the (6*36,) vector (6 characters, 10 numbers + 26 uppercase/capital characters)
# 1 means the CAPTCHA image contains this character in TABLE, 0 means otherwise
def captcha_to_vector(captcha_str):
    captcha_str = captcha_str.upper()
    vector = np.zeros(36*6, dtype=np.float32)
    for i, char in enumerate(captcha_str):
        ascii = ord(char) # Convert char into ASCII code
        if 48 <= ascii <= 57:   # for digits
            index = ascii - 48
        elif 65 <= ascii <= 90: # for Latin letters
            index = ascii - ord('A') + 10
        vector[i*LEN_OF_TABLE+index] = 1.0
    return vector

def vector_to_captcha(vector):
    captcha_str = ""
    for i in vector:
        captcha_str += TABLE[i]
    return captcha_str

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None, target_transform=None, height=50, width=200):
        self.transform = transform
        self.target_transform = target_transform
        self.images = images
        self.width = width
        self.height = height

    def noise_remover(self, image):   # get the image with path
        # increase contrast: segmentation-based so the preprocessing is more complicated
        image = cv2.convertScaleAbs(image, alpha=3, beta=40)
        # Erode noise
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        # convert the image into grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        # resize the image to ensure the size
        image = cv2.resize(image, (self.width, self.height))
        # Binarization of images
        _, image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        # Method from dsp.stackexchange.com/questions/52089/removing-noisy-lines-from-image-opencv-python
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        # Shear transformation, from thepythoncode.com/article/image-transformations-using-opencv-in-python#Image_Shearing
        M = np.float32([[1, -0.5, 0],
             	          [0,    1, 0],
            	          [0,    0, 1]])
        rows, cols = image.shape #(50, 200)
        image = cv2.warpPerspective(image,M,(int(cols),int(rows)), cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        # horizontal stretch
        M = np.float32([[1.2, 0, 0],
             	          [0,   1, 0],
            	          [0,   0, 1]])
        rows, cols = image.shape #(50, 200)
        image = cv2.warpPerspective(image,M,(int(cols),int(rows)), cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        return image

    def __getitem__(self, index):
          # get the image with path
        image = cv2.imread(str(self.images[index]))
        image = self.noise_remover(image)

        label = captcha_to_vector(self.images[index].name.split("_")[0])
        img_seg_list = []
        label_lst = []
        # segmentation [image[:,:50], image[:,50:75], image[:,75:100], image[:,100:125], image[:,125:150], image[:,150:]]
        for j in range(6):
          left = (j+1)*25
          right = (j+2)*25
          im_seg = image[:, left:right]
          if self.transform is not None:
            img_seg_list.append(self.transform(im_seg))
          else:
            img_seg_list.append(im_seg)
          label_lst.append(label[j*36:(j+1)*36])
        return img_seg_list, label_lst

    def __len__(self):
        return len(self.images)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Path to the local dataset
data_dir = pathlib.Path("../dataset")

images = list(data_dir.glob("*.jpg")) # dataset as a list
print("Number of images found:", len(images)) # size of the dataset

train_data, test_data = train_test_split(images, test_size = 0.2, random_state = 6, shuffle = True)

train_set = CustomDataset(train_data, transform=transforms.ToTensor())
test_set  = CustomDataset(test_data,  transform=transforms.ToTensor())

train_dataloader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
test_dataloader  = DataLoader(dataset=test_set,  batch_size=1, shuffle=True)

def get_data(dataloader):
  X = []
  Y = []
  n = len(dataloader)
  for z in range(n):
    i, l = next(iter(dataloader))
    for image, label in zip(i, l):
      image = image.to(device)
      label = label.to(device)
      label = label.reshape(1, 36)
      label = torch.argmax(label, dim=1)
      label = vector_to_captcha(label)
      image = image.reshape(image.shape[2], image.shape[3]).cpu()
      X.append(image.flatten().tolist())
      Y.append(label)
  new_Y = []
  for j in range(len(Y)):
    new_Y.append(Y[j][0])
  return X, Y


# Groups the original and the predicted characters together to into CAPTCHAs
def group(lst):
  n = len(lst)
  i = 0
  new_list = []
  while i < n:
    captcha = lst[i:i+6]
    new_list.append(''.join(captcha))
    i += 6
  return new_list

train_x,train_y = get_data(train_dataloader)
test_x, test_y = get_data(test_dataloader)
test_y = group(test_y)

for i in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors = i, weights = "distance")
    knn.fit(train_x, train_y)
    y_predict = knn.predict(test_x)
    y_predict = group(y_predict)

    accuracy=metrics.accuracy_score(test_y, y_predict)
    print("# neighbors = {}, Accuracy: {:.2f}%".format(i, accuracy*100))