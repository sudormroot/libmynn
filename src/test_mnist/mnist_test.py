#!env python3

""" Test for handwriting classification

"""

import numpy as np
import pandas as pd
import os
import sys

parent_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep + "lib"
sys.path.append(parent_path)

# This our implementation
from mlpc import MyMLPClassifier

from mnist_dataset import mnist_dataset_load
from mnist_dataset import N_IMAGE_SIZE
from mnist_dataset import N_LABELS
from mnist_dataset import N_IMAGE_PIXELS
from mnist_dataset import prediction_accuracy


dataset_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep \
                    + ".." + os.path.sep + ".." + os.path.sep + "dataset" + os.path.sep + "mnist_data"

X_train, y_train, X_test, y_test = mnist_dataset_load(dataset_path)

resultdir = "results"
modelfile = resultdir + os.path.sep + "handwriting_mymlpc.model"


clf = MyMLPClassifier(modelfile = modelfile)

y_predicted = clf.predict(X_test)
test_accuracy = prediction_accuracy(y_predicted, y_test)
print("Testing data set accuracy: ", test_accuracy)

y_predicted = clf.predict(X_train)
train_accuracy = prediction_accuracy(y_predicted, y_train)
print("Training data set accuracy: ", train_accuracy)


