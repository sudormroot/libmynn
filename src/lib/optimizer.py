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


""" Naive optimiser for gradient descending

"""

class MyOptimizerNaive:
    def __init__(   self,
                    *,
                    n_input, # the dimension of inputs
                    n_output # the number of neurons
                    ):
        pass

    def backward(self, dW, db):
        return dW, db
                    

""" Adam optimiser algorithm, refered to Dr. Michael Madden's lectures.

"""

class MyOptimizerAdam:

    def __init__(   self, 
                    *, 
                    n_input, # the dimension of inputs
                    n_output, # the number of neurons
                    v_gamma = 0.9, #factor for Adam optimiser
                    s_gamma = 0.999 #factor for Adam optimiser
                    ):
 
        self.n_input = n_input
        self.n_output = n_output

        self.v_gamma = v_gamma
        self.s_gamma = s_gamma

        self.v_gamma_at_t = self.v_gamma
        self.s_gamma_at_t = self.s_gamma

        #for Adam optimiser
        self.VdW = np.zeros((self.n_output, self.n_input))
        self.Vdb = np.zeros((self.n_output, 1))

        self.SdW = np.zeros((self.n_output, self.n_input))
        self.Sdb = np.zeros((self.n_output, 1))

        self.VCdW = self.VdW
        self.VCdb = self.Vdb

        self.SCdW = self.SdW
        self.SCdb = self.Sdb

    def backward(self, dW, db):

        # Adam implementation
        VdW_new = (1 - self.v_gamma) * dW + self.v_gamma * VdW
        Vdb_new = (1 - self.v_gamma) * db + self.v_gamma * Vdb

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

        dW_new = self.VCdW / (np.sqrt(self.SCdW) + epsilon)
        db_new = self.VCdb / (np.sqrt(self.SCdb) + epsilon)

        return dW_new,db_new
