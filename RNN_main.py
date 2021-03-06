from numpy.core.fromnumeric import size
import tensorflow as tf
from CustomDataset import CustomDataset
from pathlib import Path
import random
from torchvision import transforms

from RNN_model import RNN

# Load the data from the Google Drive
# data_dir = Path("/content/drive/MyDrive/Data")

# path of data set for local
dataDir = Path("./archive")

# images: the list contain the path of each images
images = list(dataDir.glob("*.jpg"))
random.shuffle(images)

# single validation
# first 0.8 of all images for training, the other 0.2 for testing
# first 0.8 of the training set size for train, the other 0.2 for validation
SIZE = len(images)
TRAINING_SIZE = int(0.8*SIZE)        
TRAIN_SIZE = int(0.8*TRAINING_SIZE)

trainingData = images[:TRAINING_SIZE]

testData = images[TRAINING_SIZE:]
trainData = trainingData[:TRAIN_SIZE]
validationData = trainingData[TRAIN_SIZE:]

# init database
testSet = CustomDataset(testData, transform=transforms.ToTensor)
trainSet = CustomDataset(trainData, transform=transforms.ToTensor)
validationSet = CustomDataset(validationData, transform=transforms.ToTensor)

# RNN
