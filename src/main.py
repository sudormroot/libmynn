#!env python3

""" 
Course:        CT4101-Machine_Learning (2020-2021)
Student Name:  Jiaolin Luo
Student ID:    20230436
Student email: j.luo2@nuigalway.ie

This is the main file responsible for evaluating our implementation.

"""

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

import os

# This our implementation
from mymlpc_impl import MyMLPClassifier


# import helper functions
from evaluation import prepare_beer_dataset
from evaluation import split_dataset
from evaluation import evaluate_mymlpc
from evaluation import evaluate_skmlpc
from evaluation import print_results
from evaluation import save_results
from evaluation import draw_loss
from evaluation import test_load_and_save_model
from evaluation import MAX_ITERS


# We run 10 times to measure the accuracy
N = 3


""" We start evaluation here.
"""

if __name__ == '__main__':

    if not os.path.exists("results"):
        os.makedirs("results")

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
    X_train, y_train, X_test, y_test = split_dataset(df, "style")
    test_load_and_save_model(X_train, y_train, X_test, y_test)
    print("")


    #
    # Comparing our classifier and sklearn baseline
    #
    print("")
    print("")
    print("Evaluating performance and comparing with sk-learn MLPC baseline classifier ...")
    print("")
    print("")

    my_test_accs = []
    my_train_accs = []
    
    sk_test_accs = []
    sk_train_accs = []
    
    my_loss_hists = []
    sk_loss_hists = []


    for i in range(N):
        
        print(f"--- {i} ---")
        #print("")

        X_train, y_train, X_test, y_test = split_dataset(df, "style")

        # We evaluate our MLPC first.
        myclf, train_acc, test_acc = evaluate_mymlpc(X_train, y_train, X_test, y_test)

        my_train_accs.append(train_acc)
        my_test_accs.append(test_acc)

        # Retrieve the loss history.
        loss_hist = myclf.loss_history()
        my_loss_hists.append(loss_hist)

        # We evaluate the sklearn MLPC.
        skclf, train_acc, test_acc = evaluate_skmlpc(X_train, y_train, X_test, y_test)
        
        sk_train_accs.append(train_acc)
        sk_test_accs.append(test_acc)

        loss_hist = skclf.loss_curve_
        sk_loss_hists.append(loss_hist)


    #
    # Compute the mean and std of accuracies on training and testing datasets
    #

    results = {}


    results["mymlpc"] = {}
    results["skmlpc"] = {}

    results["mymlpc"]["test_accs"] = my_test_accs
    results["mymlpc"]["train_accs"] = my_train_accs

    results["skmlpc"]["test_accs"] = sk_test_accs
    results["skmlpc"]["train_accs"] = sk_train_accs


    results["mymlpc"]["test_mean"] = np.mean(my_test_accs)
    results["mymlpc"]["test_std"] = np.std(my_test_accs)

    results["mymlpc"]["train_mean"] = np.mean(my_train_accs)
    results["mymlpc"]["train_std"] = np.std(my_train_accs)

    results["skmlpc"]["test_mean"] = np.mean(sk_test_accs)
    results["skmlpc"]["test_std"] = np.std(sk_test_accs)

    results["skmlpc"]["train_mean"] = np.mean(sk_train_accs)
    results["skmlpc"]["train_std"] = np.std(sk_train_accs)

    # Compute mean loss for all runs.
    my_loss_hists = np.array(my_loss_hists)
    my_loss_hist = np.mean(my_loss_hists, axis = 0)

    for i, loss_hist in enumerate(sk_loss_hists):
        if len(loss_hist) < MAX_ITERS:
            loss_hist += [0] * (MAX_ITERS - len(loss_hist))
            sk_loss_hists[i] = loss_hist

    sk_loss_hists = np.array(sk_loss_hists)
    sk_loss_hist = np.mean(sk_loss_hists, axis = 0)

    # present results
    print_results(results)
    save_results(results)
    draw_loss(my_loss_hist, sk_loss_hist)


