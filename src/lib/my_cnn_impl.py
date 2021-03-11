""" 
Course:        CT4101-Machine_Learning (2020-2021)
Student Name:  Jiaolin Luo
Student ID:    20230436
Student email: j.luo2@nuigalway.ie

This is a simple implementation for Multiple Layer Perceptron Classifier (MLPC),
which only depends on numpy.

The training uses SGD with batch size by 1.

"""


import numpy as np
import random
import pickle 

class MyConvLayer:
    def __init__(   self, 
                    *,
                    name,
                    pooling = "max", # max or mean
                    activation = "relu",
                    learning_rate = 0.5,
                    kernel_size = (5, 5, 3), # 5x5x3
                    n_kernels = 3
                ):
        pass

    def relu(self, x):
        pass
    def drelu(self, x):
        pass

    def pooling_max(self, x):
        pass

    def dpooling_max(self, x):
        pass

    def forward(self, x):
        pass
    def backward(self, grad):
        pass



