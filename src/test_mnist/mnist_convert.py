#!env python3

""" Load and process mnist dataset
    
    Author: (March) Jiaolin Luo

"""

import numpy as np
import pandas as pd
import os
import sys

dataset_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep \
                    + ".." + os.path.sep + ".." + os.path.sep + "dataset" + os.path.sep

train_data = np.loadtxt(dataset_path + "mnist_train.csv", delimiter=",")

test_data = np.loadtxt(dataset_path + "mnist_test.csv", delimiter=",") 

print("Training and testing datasets are loaded.")


print("train_data length: ", len(train_data))
print("test_data length: ", len(test_data))


data = np.concatenate((train_data, test_data), axis = 0)

print("data length: ", len(data))

size = 700

for i in range(len(data) // size):

    start = i * size
    end = start + size
    
    filepath = dataset_path + "mnist_data/" + f'mnist_data_{i}.txt'
    
    print(f"output {filepath}")

    np.savetxt(filepath, data[start:end,:], fmt='%d', delimiter=',') 


