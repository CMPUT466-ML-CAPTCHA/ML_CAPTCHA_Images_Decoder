"""
Support vector machine (SVM) model for CAPTCHA
`Completely Automated Public Turing test to tell Computers and Humans Apart`
1) Data Preprocessing
"""

import os
import cv2
import torch
import string
import random
import pathlib
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from CustomDatasetForSementationBased import CustomDataset, captcha_to_vector, vector_to_captcha, get_data, group

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        
    # Path to the local dataset
    data_dir = pathlib.Path("../dataset")
    images = list(data_dir.glob("*.jpg")) # dataset as a list
    print("Number of images found:", len(images)) # size of the dataset
    if len(images) == 0: 
        print("No images found.")
        return
    
    """Split and create datasets"""
    random.shuffle(images)
    NUMBER_Images = len(images)
    # test data
    test_data = images[int(0.8*NUMBER_Images):]     # last 2k images (20%) in dataset are for testing
    
    # training data
    training = images[:int(0.8*NUMBER_Images)]      # first 8k (80%) images in dataset are for training
    train_data = training
    
    print("Training set size:\t", len(train_data))
    print("Test set size:\t\t", len(test_data))
    
    train_set = CustomDataset(train_data, transform=transforms.ToTensor())
    test_set  = CustomDataset(test_data,  transform=transforms.ToTensor())
    
    train_dataloader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)#BATCH_SIZE=1
    test_dataloader  = DataLoader(dataset=test_set,  batch_size=1, shuffle=True)
    
    """2) `SVM` Model** (a segmentation-based algorithm)"""
    """Training the support vector machine"""
    # SVC Code from kaggle.com/sanesanyo/digit-recognition-using-svm-with-98-accuracy
    # Splitting the data into test and training set for our first simple linear SVM testing
    # Creating our linear SVM object
    from sklearn.svm import SVC
    # Initialise the SVM model
    clf = SVC(C=1, kernel="linear")
    # Get the training data and corresponding labels ready
    train_x, train_y = get_data(train_dataloader)
    # Training the SVM model
    clf.fit(train_x, train_y)
    
    """Use the SVM model to recognise new `CAPTCHA`"""
    # Code for prediction and accuracy modified from the same Kaggle source
    # Saving the predictions on the test set 
    test_x, test_y = get_data(test_dataloader)
    # Use the SVM model to predict
    y_predict = clf.predict(test_x)
    # Group the original and the predicted characters together to a CAPTCHA
    test_y = group(test_y)
    y_predict = group(y_predict)
    # Measuring the accuracy of our predictions
    from sklearn import metrics
    accuracy = metrics.accuracy_score(test_y, y_predict)
    print("Accuracy: {}%".format(accuracy*100)) 
    # Accuracy for recognising single alphanumeric character images is around 76%
    
    # Finally display some sample results
    _, ax = plt.subplots(2, 3, figsize=(20,5))
    for p in range(6):
        i, l = next(iter(test_dataloader))
        img = np.zeros((50,150))
        j = 0
        predicts = []
        origins = []
        for image, label in zip(i, l):
            image = image.to(device)
            image = image.reshape(1, image.shape[2]*image.shape[3]).cpu()
            predict = clf.predict(image)
            predicts.append(predict[0])
            img[:,j:j+25] = image.reshape(50,25)
            j += 25
            label = label.to(device)
            label = label.reshape(1, 36)
            label = torch.argmax(label, dim=1)
            origin = vector_to_captcha(label)
            origins.append(origin)
        ax[p // 3, p % 3].imshow(img,cmap=plt.cm.gray)
        ax[p // 3, p % 3].title.set_text("Original: "+"".join(origins)+"    Predict: "+"".join(predicts))
    plt.show()
    
main()