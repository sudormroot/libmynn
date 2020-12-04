#!env python3

""" Test for handwriting classification

"""

import numpy as np
import pandas as pd
import os

# This our implementation
from mymlpc_impl import MyMLPClassifier


def prediction_accuracy(y_predicted, y_truth):
    return np.mean(y_predicted == y_truth)


# width and length
N_IMAGE_SIZE = 28
N_LABELS = 10
N_IMAGE_PIXELS = N_IMAGE_SIZE * N_IMAGE_SIZE

dataset_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep \
                    + ".." + os.path.sep + "dataset" + os.path.sep

print("Loading training and testing dataset ...")

train_data = np.loadtxt(dataset_path + "mnist_train.csv", delimiter=",")

test_data = np.loadtxt(dataset_path + "mnist_test.csv", delimiter=",") 

print("Training and testing datasets are loaded.")


scale = 0.99 / 255

train_imgs = np.asfarray(train_data[:, 1:]) * scale + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * scale + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

train_labels = train_labels.flatten()
test_labels = test_labels.flatten()

print("train_imgs.shape=", train_imgs.shape)
print("test_imgs.shape=", test_imgs.shape)

print("train_labels.shape=", train_labels.shape)
print("test_labels.shape=", test_labels.shape)

#print(train_labels)
#exit()

clf = MyMLPClassifier( n_input = N_IMAGE_PIXELS, 
                       n_output = N_LABELS, 
                       hidden_sizes = (28,28), #define hidden layers
                       learning_rate = 0.001, 
                       n_epochs = 1000, 
                       batch_size = 8,
                       alpha = 0.0001,
                       #random_seed = 1,
                       activation = 'relu',
                       debug = True)



 
clf.fit(train_imgs, train_labels)


modelfile = "results" + os.path.sep + "handwriting_mymlpc.model"

# save model
clf.save(modelfile)



predicted_labels = clf.predict(test_imgs)
test_accuracy = prediction_accuracy(predicted_labels, test_labels)
print("Testing data set accuracy: ", test_accuracy)

predicted_labels = clf.predict(train_imgs)
train_accuracy = prediction_accuracy(predicted_labels, train_labels)
print("Training data set accuracy: ", train_accuracy)


