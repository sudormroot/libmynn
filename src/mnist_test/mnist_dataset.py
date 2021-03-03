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

    print("Loading training and testing dataset ...")


    walk = os.walk(dataset_path)  

    data = None

    for path,dir_list,file_list in walk:  

        for filename in file_list:  
        
            fullpath = os.path.join(path, filename) 

            if filename.endswith(".txt"):

                print(f"Loading f{filename}")

                ds = np.loadtxt(fullpath, delimiter=",")

                if data is None:
                    data = ds
                else:
                    data = np.concatenate([data, ds])


    print("Total {len(data)} data are loaded")

    indices = np.arange(len(data))

    np.random.shuffle(indices)

    N_TRAIN_DATA = 3000
    N_TEST_DATA = 100

    train_indices = np.random.choice(indices, N_TRAIN_DATA)
    test_indices = np.random.choice(indices, N_TEST_DATA)

    train_data = data[train_indices]
    test_data = data[test_indices]

    print("train_data length: ", len(train_data))
    print("test_data length: ", len(test_data))

    #train_data = np.loadtxt(dataset_path + "mnist_train.csv", delimiter=",")
    #test_data = np.loadtxt(dataset_path + "mnist_test.csv", delimiter=",") 

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

    return X_train, y_train, X_test, y_test
