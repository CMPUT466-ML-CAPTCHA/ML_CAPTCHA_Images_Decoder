"""
Support vector machine model for
# `Completely Automated Public Turing test to tell Computers and Humans Apart`
## **1) Data Preprocessing**
Import the libraries
"""

import os
import cv2
import torch
import beepy
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
    
    """Load the dataset, please modify the path `data_dir` **accordingly**"""
    # Path to the local dataset
    data_dir = pathlib.Path("./dataset")
    images = list(data_dir.glob("*.jpg")) # dataset as a list
    print("Number of images found:", len(images)) # size of the dataset
    if len(images) == 0: 
        print("No images found.")
        return
    
    """### Split and create datasets""" 
    random.shuffle(images)
    NUMBER_Images = len(images)
    # test data
    test_data = images[int(0.8*NUMBER_Images):]     # last 2k images (20%) in dataset are for testing
    
    # the part for training
    training = images[:int(0.8*NUMBER_Images)]      # first 8k (80%) images in dataset are for training
    train_data = training
    
    print("Training set size:\t", len(train_data))
    print("Test set size:\t\t", len(test_data))
    
    train_set = CustomDataset(train_data, transform=transforms.ToTensor())
    test_set  = CustomDataset(test_data,  transform=transforms.ToTensor())
    
    train_dataloader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)#BATCH_SIZE=1
    test_dataloader  = DataLoader(dataset=test_set,  batch_size=1, shuffle=True)
    
    """## **2) `SVM` Model** (segmentation-based algorithms)"""    
    """**Training the support vector machine**"""
    # SVC Code from kaggle.com/sanesanyo/digit-recognition-using-svm-with-98-accuracy
    # Splitting the data into test and training set for our first simple linear SVM testing
    # Creating our linear SVM object
    from sklearn.svm import SVC
    regularization = [i for i in range(1,5,1)]
    kernels = ['linear', 'poly', 'rbf']
    train_x, train_y = get_data(train_dataloader)
    test_x,  test_y  = get_data(test_dataloader)
    for c in regularization:
        for k in kernels:
            clf = SVC(C=c, kernel=k)
            clf.fit(train_x, train_y)
            
            """Use the **`SVM`** to recognise new `CAPTCHA`"""
            # Code for prediction and accuracy modified from the same Kaggle source
            # Saving the predictions on the test set 
            y_predict = clf.predict(test_x)
            # Group the original and the predicted characters together to a CAPTCHA
            test_y_ = group(test_y)
            y_predict = group(y_predict)
            # Measuring the accuracy of our predictions
            from sklearn import metrics
            accuracy = metrics.accuracy_score(test_y_, y_predict)
            print("Accuracy for SVM with C={} and kernel={}: {:.2f}%".format(c,k,accuracy*100)) 
    for i in range(1,8):
        beepy.beep(sound=i) # integer as argument

main()