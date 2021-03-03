#!env python3

""" Train a handwriting classification

"""

import numpy as np
import pandas as pd
import os
import sys

parent_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".."
sys.path.append(parent_path)

# This our implementation
from mymlpc_impl import MyMLPClassifier

from mnist_dataset import mnist_dataset_load
from mnist_dataset import N_IMAGE_SIZE
from mnist_dataset import N_LABELS
from mnist_dataset import N_IMAGE_PIXELS
from mnist_dataset import prediction_accuracy


dataset_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep \
                    + ".." + os.path.sep + ".." + os.path.sep + "dataset" + os.path.sep + "mnist_data"

X_train, y_train, X_test, y_test = mnist_dataset_load(dataset_path)


#X_train = X_train[:2000]
#y_train = y_train[:2000]

#X_test = X_test[:500]
#y_test = y_test[:500]

clf = MyMLPClassifier( n_input = N_IMAGE_PIXELS, 
                       n_output = N_LABELS, 
                       hidden_sizes = (56, 28,), #define hidden layers
                       learning_rate = 0.05, 
                       n_epochs = 1000, 
                       batch_size = 6,
                       alpha = 0.001,
                       #random_seed = 1,
                       activation = 'relu',
                       print_per_epoch = 1,
                       debug = True)

 
clf.fit(X_train, y_train)


resultdir = "results"

modelfile = resultdir + os.path.sep + "handwriting_mymlpc.model"

if not os.path.exists(resultdir):
    os.makedirs(resultdir)

# save model
clf.save(modelfile)

y_predicted = clf.predict(X_test)
test_accuracy = prediction_accuracy(y_predicted, y_test)
print("Testing data set accuracy: ", test_accuracy)

y_predicted = clf.predict(X_train)
train_accuracy = prediction_accuracy(y_predicted, y_train)
print("Training data set accuracy: ", train_accuracy)


