#!env python3

""" Load and process mnist dataset
    
    Author: (March) Jiaolin Luo

"""

import numpy as np
import pandas as pd
import os
import sys


def prediction_accuracy(y_predicted, y_truth):
    return np.mean(y_predicted == y_truth)


# width and length
N_IMAGE_SIZE = 28
N_LABELS = 10
N_IMAGE_PIXELS = N_IMAGE_SIZE * N_IMAGE_SIZE

def mnist_dataset_load(dataset_path):
    #dataset_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep \
    #                + ".." + os.path.sep + ".." + os.path.sep + "dataset" + os.path.sep

    print("Loading training and testing dataset ...")

    train_data = np.loadtxt(dataset_path + "mnist_train.csv", delimiter=",")

    test_data = np.loadtxt(dataset_path + "mnist_test.csv", delimiter=",") 

    print("Training and testing datasets are loaded.")


    scale = 0.99 / 255

    X_train = np.asfarray(train_data[:, 1:]) * scale + 0.01
    X_test = np.asfarray(test_data[:, 1:]) * scale + 0.01

    y_train = np.asfarray(train_data[:, :1])
    y_test = np.asfarray(test_data[:, :1])

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print("X_train.shape=", X_train.shape)
    print("X_test.shape=", X_test.shape)

    print("y_train.shape=", y_train.shape)
    print("y_test.shape=", y_test.shape)

    #print(y_train)
    #exit()

    return X_train, X_train, X_test, y_test
