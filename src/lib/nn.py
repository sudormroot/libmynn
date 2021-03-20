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
import json

import os
import sys
libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep + "lib"
sys.path.append(libpath)

from softmax import MySoftMaxLayer
from optimizer import *



MYMLPC_VERSION="1.3"


#def prediction_accuracy(y_predicted, y_truth):
#    return np.mean(y_predicted == y_truth)



""" We implement a NN layer here.

"""

class MyNNLayer:

    """ Activation functions and their derivative forms.
        We allow users to choose 'sigmoid', 'tanh' and 'relu'
        activation functions

    """

    def noact(self, x):
        return x

    def dnoact(self, x):
        return np.ones(x.shape).reshape(-1, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        y = self.sigmoid(x)
        return y * (1 - y)


    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        y = self.tanh(x)
        return 1.0 - y ** 2

    def relu(self, x):
        y = x.copy()
        y[y < 0] = 0
        #print("y=",y)
        
        return y

    def drelu(self, x):
        return 1. * (x > 0)


    # used for initializing weights.
    def init_weights(self):

        #self.W = np.random.uniform(-1, 1, (self.n_neurons, self.n_input))
        #self.b = np.random.uniform(-1, 1, (self.n_neurons, 1))

        #sigma = 0.001
        #sigma = 0.5
        sigma = 1
        #mean = 0

        self.W = np.random.uniform(-sigma, sigma, (self.n_neurons, self.n_input))
        self.b = np.random.uniform(-sigma, sigma, (self.n_neurons, 1))

        """ #for Adam optimiser
        self.VdW = np.zeros((self.n_neurons, self.n_input))
        self.Vdb = np.zeros((self.n_neurons, 1))

        self.SdW = np.zeros((self.n_neurons, self.n_input))
        self.Sdb = np.zeros((self.n_neurons, 1))

        self.VCdW = self.VdW
        self.VCdb = self.Vdb

        self.SCdW = self.SdW
        self.SCdb = self.Sdb
        """

        # normalize weights again to prevent overflow errors for exp().
        #self.W = self.W / self.n_neurons
        #self.b = self.b / self.n_neurons

        #if self.name == 'input':
        #    self.W = np.zeros((self.n_neurons, self.n_input))
        #    self.b = np.zeros((self.n_neurons, 1))

        #Gaussian distribution
        #self.W = np.random.normal(-0.5, 0.5, (self.n_neurons, self.n_input))
        #self.b = np.random.normal(0.5, 0.5, self.n_neurons)

    # used for loading model from file.
    #def set_weights(self, W, b):
    #    self.W = W
    #    self.b = b

    def __init__(   self, 
                    *, 
                    name, # the name of this layer
                    n_input, # the dimension of inputs
                    n_neurons = 11, # the number of neurons
                    random_seed = 0,  # we enable to configure the random seed
                    learning_rate = 0.5, # learning rate
                    batch_size = 1, # batch size used for mini batch training
                    activation = 'sigmoid', # activation function
                    alpha = 0.0001, #regularization factor
                    v_gamma = 0.9, #factor for Adam optimiser
                    s_gamma = 0.999, #factor for Adam optimiser
                    W = None, # Used for loading model from file
                    b = None, # Used for loading model from file
                    optimizer = "simple", 
                    optimizer_options = {},
                    debug = False # debug flag
                    ):

        # We keep all parameters here for later use.
        self.learning_rate = learning_rate
        
        """ self.v_gamma = v_gamma
        self.s_gamma = s_gamma

        self.v_gamma_at_t = v_gamma
        self.s_gamma_at_t = s_gamma
        """

        activations = { 'sigmoid':  (self.sigmoid,  self.dsigmoid),
                        'tanh':     (self.tanh,     self.dtanh),
                        'relu':     (self.relu,     self.drelu),
                        'none':     (self.noact,    self.dnoact),
                        #'softmax':  (self.softmax,  self.dsoftmax)
                        }

        optimizers = {  'simple':   MyOptimizerNaive,
                        'adam':     MyOptimizerAdam}


        assert optimizer in optimizers

        self.optimizer = optimizers[optimizer]( n_input = n_input, 
                                                n_output = n_neurons,
                                                *optimizer_options)

        # Check
        # assert activation in activations

        # set activation function
        if activation:
            self.f = activations[activation][0]
            self.df = activations[activation][1]

        self.activation = activation
        self.name = name
        self.batch_size = batch_size
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.alpha = alpha

        if random_seed > 0:
            np.random.seed(random_seed)
 
        self.W = None
        self.b = None

        self.init_weights()

        self.debug = debug


        # x is the input from prior layer.
        # y = wx + b
        # z = f(y)

        self.x = None 
        self.y = None
        self.z = None

        # The gradients of W and b
        self.dW = None
        self.db = None


    """ forward propagation implementation for one layer.
        x is the inputs from prior layer.

        z = f(w*x + b)

    """

    def forward(self, x):

        #if self.name == "input":
        #    print("x.shape=",x.shape)
        #    return x

        # Keep a private copy
        self.x = x.copy()

        # Compute y = w*x + b
        self.y = self.W.dot(self.x) + self.b

        # Compute z = f(y)
        self.z = self.f(self.y)
        
        return self.z

    """ backward propagation implementation for one layer

        grad is the gradient from next layer.

    """

    def backward(self, grad):

        #if self.name == 'input':
        #    return grad

        # Keep a private copy of dL / dz
        dLdz = grad.copy()


        #print("dLdz.shape=", dLdz.shape)


        # We compute the value of the derivative on z.
        # The value is dz / dy
 
        #print("self.x.shape=", self.x.shape)

        dzdy = self.df(self.y)

        #print("dzdy.shape=", dzdy.shape)

        #dLdy = dLdz * dzdy 
        dLdy = dLdz * dzdy 

        # Compute the gradients of W and b
        #self.db = dLdy
        #self.dW = dLdy.reshape(-1, 1).dot(self.x.reshape(1, -1))

        self.dW = dLdy.dot(self.x.T) / self.batch_size

        self.db = np.mean(dLdy, axis=1).reshape(-1, 1)

        # regularisation terms
        self.dW += self.alpha * self.W
        self.db += self.alpha * self.b

        # Compute the output gradients for prior layer.
        #grad_next = dLdy.dot(self.W)
        grad_next = dLdy.T.dot(self.W)


        grad_next = np.sum(grad_next, axis = 0) / self.batch_size

        grad_next = grad_next.reshape(-1, 1)


        # We can adjust weights now.
        """
        #self.W = self.W - self.learning_rate * self.dW
        #self.b = self.b - self.learning_rate * self.db

        # Adam implementation
        VdW_new = (1 - self.v_gamma) * self.dW + self.v_gamma * self.VdW
        Vdb_new = (1 - self.v_gamma) * self.db + self.v_gamma * self.Vdb

        self.VdW = VdW_new
        self.Vdb = Vdb_new

        SdW_new = (1 - self.s_gamma) * (self.dW ** 2) + self.s_gamma * self.SdW
        Sdb_new = (1 - self.s_gamma) * (self.db ** 2) + self.s_gamma * self.Sdb

        self.SdW = SdW_new
        self.Sdb = Sdb_new


        VCdW_new = self.VdW / (1 - self.v_gamma_at_t)
        VCdb_new = self.Vdb / (1 - self.v_gamma_at_t)

        self.VCdW = VCdW_new
        self.VCdb = VCdb_new
        

        SCdW_new = self.SdW / (1 - self.s_gamma_at_t)
        SCdb_new = self.Sdb / (1 - self.s_gamma_at_t)

        self.SCdW = SCdW_new
        self.SCdb = SCdb_new

        self.v_gamma_at_t *= self.v_gamma
        self.s_gamma_at_t *= self.s_gamma

        epsilon = 1e-8

        self.W = self.W - self.learning_rate * (self.VCdW / (np.sqrt(self.SCdW) + epsilon))
        self.b = self.b - self.learning_rate * (self.VCdb / (np.sqrt(self.SCdb) + epsilon))

        #self.W = self.W - self.learning_rate * self.dW
        #self.b = self.b - self.learning_rate * self.db
        """

        dW_new,db_new = self.optimizer.backward(self.dW, self.db)

        self.W = self.W - self.learning_rate * dW_new
        self.b = self.b - self.learning_rate * db_new

        self.dW = dW_new
        self.db = db_new

        return grad_next



