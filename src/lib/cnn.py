#!env python3

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
                    name = None,
                    pooling = "max", # max or mean
                    activation = "relu",
                    learning_rate = 0.5,
                    kernel_size = (5, 5, 3), # 5x5x3
                    n_kernels = 3
                ):

        self.pooling = pooling
        self.activation = activation
        self.learning_rate = learning_rate
        self.name = name
        self.kernel_size = kernel_size
        sigma = 0.001
        self.kernels = np.random.uniform(-sigma, sigma, (n_kernels, *self.kernel_size))

        pass

    def relu_forward(self, x):
        pass

    def relu_backward(self, x):
        pass

    def pooling_max_backward(self, x):
        pass

    def pooling_max_forward(self, x):
        pass

    def conv_forward(self, x):

        #w, h, c, n = self.kernels.shape

        print("self.kernels.shape = ", self.kernels.shape)

        y = np.zeros()


        pass

    def conv_backward(self, x):
        pass

    def forward(self, x):
        pass

    def backward(self, grad):
        pass


if __name__ == "__main__":
    import os
    import sys
    
    conv = MyConvLayer()
    
    conv.conv_forward(None)

    libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep + "mnist_test"
    sys.path.append(libpath)

    from mnist_dataset import mnist_dataset_load
    from mnist_dataset import N_IMAGE_SIZE
    from mnist_dataset import N_LABELS
    from mnist_dataset import N_IMAGE_PIXELS
    from mnist_dataset import prediction_accuracy


    dataset_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep \
                    + ".." + os.path.sep + ".." + os.path.sep + "dataset" + os.path.sep + "mnist_data"

    #X_train, y_train, X_test, y_test = mnist_dataset_load(dataset_path, n = 1)

    pass
