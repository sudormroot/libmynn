#!env python3


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

import os

# This our implementation
from mymlpc_impl import MyMLPClassifier


# import helper functions
from evaluation import prepare_beer_dataset
from evaluation import split_dataset
from evaluation import evaluate_mymlpc
from evaluation import print_results
from evaluation import save_results



dataset_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep \
                    + ".." + os.path.sep + "dataset" + os.path.sep  +"beer.txt"

df = prepare_beer_dataset(dataset_path)
    

#
# test saving and loading model to/from file
#
print("")
print("")
print("Evaluating for saving and loading model ...")
print("")
#X_train, y_train, X_test, y_test = split_dataset(df, "style")
#test_load_and_save_model(X_train, y_train, X_test, y_test)
print("")



X_train, y_train, X_test, y_test = split_dataset(df, "style")

# We evaluate our MLPC first.
myclf, train_acc, test_acc = evaluate_mymlpc(X_train, y_train, X_test, y_test)


