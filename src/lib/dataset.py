#!env python3

""" 
Course:        CT4101-Machine_Learning (2020-2021)
Student Name:  Jiaolin Luo
Student ID:    20230436
Student email: j.luo2@nuigalway.ie

This is an evaluation program for evaluating the MLPC algorithm we implement.
We will use the beer dataset to evaluate it.

"""

import numpy as np
import pandas as pd


def normalise_data_minmax(df, columns):

    for name in columns:
        x = np.array(df[name])

        a = min(x)
        b = max(x)

        x = ((x - a) / (b - a) - 0.5) * 2

        df[name] = x


""" Split the dataset for training and testing

    RETURNS:
        X_train, y_train, X_test, y_test
"""

def split_dataset(df, y_name):

    # The columns for input (X)
    x_names = ["calorific_value", "nitrogen", "turbidity", "alcohol", "sugars", "bitterness", "beer_id", "colour", "degree_of_fermentation"]

    # The columns for label (y)
    #y_name = 'style'

    # reshuffle the dataset
    df = df.sample(frac = 1) 

    #   
    # Split dataset into training and testing datasets
    #

    L = len(df)

    # The factor controling the splitting ratio
    K = 1./3.

    # We reserve 30% of the data as testing dataset
    n = K * L
    n = int(n)

    # split data set into training and testing datasets.
    df_dataset_test = df[0:n]
    df_dataset_train = df[n:L]

    # training dataset
    X_train = df_dataset_train[x_names]
    y_train = df_dataset_train[y_name]

    X_train = np.array(X_train)
    y_train = list(y_train)
    y_train = np.array(y_train)
    y_train = y_train.T

    # test dataset
    X_test = df_dataset_test[x_names]
    y_test = df_dataset_test[y_name]

    X_test = np.array(X_test)
    y_test = list(y_test)
    y_test = np.array(y_test)
    y_test = y_test.T

    return X_train, y_train, X_test, y_test

""" Measure accuracy for classifier
"""

def prediction_accuracy(y_predicted, y_truth):
    return np.mean(y_predicted == y_truth)


