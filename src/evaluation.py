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
import sklearn as sk
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

import os

# This our implementation
from mymlpc_impl import MyMLPClassifier


MAX_ITERS=101
LEARNING_RATE=0.1


""" Load beer data set as DataFrame

"""

def load_beer_dataset(txtfile):

    print(f"load file: {txtfile}")

    names = [ "calorific_value", "nitrogen", 
              "turbidity",       "style", 
              "alcohol",         "sugars", 
              "bitterness",      "beer_id", 
              "colour",          "degree_of_fermentation"]

    df = pd.read_csv(txtfile, sep = "\t", names = names)

    return df

""" max-min normalisation, the data will be scaled into [-1, 1]
"""

def normalise_data(df, columns):

    for name in columns:
        x = np.array(df[name])

        a = min(x)
        b = max(x)

        x = ((x - a) / (b - a) - 0.5) * 2

        df[name] = x



""" This function is used to pre-process the beer dataset for:
    1. Normalise the dataset properly.
    2. Transform the label into one-hot coding.
    3. Split the dataset

"""

def prepare_beer_dataset(txtfile):
 
    X_train = None
    y_train = None

    X_test = None
    y_test = None

    #
    # We load beer data set as our training and testing datasets
    #
    df_dataset = load_beer_dataset(txtfile)

    # reshuffle the dataset
    df_dataset = df_dataset.sample(frac = 1) 

    # The columns for input (X)
    x_names = ["calorific_value", "nitrogen", "turbidity", "alcohol", "sugars", "bitterness", "beer_id", "colour", "degree_of_fermentation"]

    # The columns for label (y)
    y_name = 'style'
    #y_name_onehot = 'style_onehot'

    # We normalise the dataset
    normalise_data(df_dataset, x_names)


    return df_dataset



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


""" Test for saving and loading model

"""
def test_load_and_save_model(X_train, y_train, X_test, y_test):

    print("Training a model ...")

    n_input = X_train.shape[1]
    n_output = len(set(y_train))

    clf1 = MyMLPClassifier( n_input = n_input, 
                        n_output = n_output, 
                        hidden_sizes = (13,), #define hidden layers
                        learning_rate = LEARNING_RATE, 
                        n_epochs = MAX_ITERS, 
                        batch_size = 1,
                        alpha = 0.0001,
                        #random_seed = 1,
                        activation = 'relu',
                        print_per_epoch = 10,
                        debug = True)

    
    clf1.fit(X_train, y_train)
    y_predicted = clf1.predict(X_test)

    test_accuracy = prediction_accuracy(y_predicted, y_test)

    y_predicted = clf1.predict(X_train)
    train_accuracy = prediction_accuracy(y_predicted, y_train)

    print("Testing data set accuracy: ", test_accuracy)
    print("Training data set accuracy: ", train_accuracy)

    modelfile = "results" + os.path.sep + "beer_mymlpc.model"

    # save model
    clf1.save(modelfile)

    print("")
    print("Loading model from file ...")
    clf2 = MyMLPClassifier(modelfile = modelfile)

    y_predicted = clf2.predict(X_test)

    test_accuracy = prediction_accuracy(y_predicted, y_test)

    y_predicted = clf2.predict(X_train)
    train_accuracy = prediction_accuracy(y_predicted, y_train)

    print("Testing data set accuracy: ", test_accuracy)
    print("Training data set accuracy: ", train_accuracy)




""" The evaluation function of my MLPC

"""

def evaluate_mymlpc(X_train, y_train, X_test, y_test):

    #print("")
    #print("----------- Evaluation of my MLPC algorithm -----------")

    n_input = X_train.shape[1]
    n_output = len(set(y_train))

    clf = MyMLPClassifier( n_input = n_input, 
                        n_output = n_output, 
                        hidden_sizes = (7,), #define hidden layers
                        learning_rate = LEARNING_RATE, 
                        n_epochs = MAX_ITERS, 
                        batch_size = 1,
                        alpha = 0.0001,
                        #random_seed = 1,
                        activation = 'relu',
                        print_per_epoch = 10,
                        debug = True)

    
    clf.fit(X_train, y_train)


    # for debug purposes
    # clf.print_weights()

    y_predicted = clf.predict(X_test)
    

    test_accuracy = prediction_accuracy(y_predicted, y_test)
    #print("Testing data set accuracy: ", accuracy)

    y_predicted = clf.predict(X_train)
    train_accuracy = prediction_accuracy(y_predicted, y_train)
    #print("Training data set accuracy: ", accuracy)

    
    # we are required to log the prediction into a file
    f = open("results" + os.path.sep + "mymlpc_prediction.txt", "at")
    f.write("\n")
    f.write(f"y_test={y_test}\n")
    f.write("\n")
    f.write(f"y_predicted={y_predicted}\n")
    f.write("\n")
    f.write(f"test_accuracy={test_accuracy:.2f}\n")
    f.write(f"train_accuracy={train_accuracy:.2f}\n")
    f.write("\n")
    f.close()

    #print("----------- Finished -----------")

    return clf, train_accuracy, test_accuracy


""" The evaluation of sklearn MLPC algorithm
"""

def evaluate_skmlpc(X_train, y_train, X_test, y_test):
    
    #print("")
    #print("----------- Evaluation of sklearn MLPC algorithm -----------")

    clf = MLPClassifier(  solver = 'sgd', 
                          alpha = 0.0001,
                          activation = 'relu', 
                          learning_rate_init = LEARNING_RATE,
                          hidden_layer_sizes = (7,), 
                          batch_size = 1,
                          #random_state = 1,
                          max_iter = MAX_ITERS
                          )


    clf.fit(X_train, y_train.T)

    y_predicted = clf.predict(X_test)

    y_predicted = y_predicted.T

    test_accuracy = prediction_accuracy(y_predicted, y_test)
    #print("Testing data set accuracy: ", accuracy)

    y_predicted = clf.predict(X_train)
    
    y_predicted = y_predicted.T

    train_accuracy = prediction_accuracy(y_predicted, y_train)
    #print("Training data set accuracy: ", accuracy)

    #print("----------- Finished -----------")

    return clf, train_accuracy, test_accuracy


""" Draw loss

"""

def draw_loss(my_loss_hist, sk_loss_hist):

    # We draw loss-iteration figure for our MLPC

    # normalize loss
    my_loss_hist = my_loss_hist / max(my_loss_hist)
    sk_loss_hist = sk_loss_hist / max(sk_loss_hist)

    # for general paper purpose, the width is around 7.5
    fig_width = 7.5 
    fig_height = fig_width * 0.5

    fig = plt.figure(figsize = (fig_width, fig_height))

    plt.plot(my_loss_hist, color = 'coral', label = 'MY-MLPC')
    plt.plot(sk_loss_hist, color = 'turquoise', label = 'SK-MLPC')

    plt.xlabel("Iterations", fontsize = 12)
    plt.ylabel("Normalized Loss", fontsize = 12)

    assert len(my_loss_hist) == len(sk_loss_hist)

    plt.xlim(0, len(my_loss_hist) - 1)
    plt.ylim(0, 1)

    plt.legend( loc = 'upper center',
                fontsize = 12,
                ncol = 2,
                frameon = False)   

    resultdir = "results"

    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    figurefile = resultdir + os.path.sep + 'fig_loss.pdf'

    plt.savefig(figurefile, dpi = 600, format = 'pdf')

    plt.show()


""" Print results

"""
def print_results(results):

    print("")
    print("My MLPC:")
    print("Average accuracy on testing dataset: ", results["mymlpc"]["test_mean"])
    print("Standard deviation on testing dataset: ", results["mymlpc"]["test_std"])
    print("")
    
    print("Average accuracy on training dataset: ", results["mymlpc"]["train_mean"])
    print("Standard deviation on training dataset: ", results["mymlpc"]["train_std"])
    print("")

    print("sklearn MLPC:")
    print("Average accuracy on testing dataset: ", results["skmlpc"]["test_mean"])
    print("Standard deviation on testing dataset: ", results["skmlpc"]["test_std"])
    print("")
    
    print("Average accuracy on training dataset: ", results["skmlpc"]["train_mean"])
    print("Standard deviation on training dataset: ", results["skmlpc"]["train_std"])
    print("")



""" Save results to file

"""

def save_results(results):

    # Keep results into file for report writing

    resultdir = "results"

    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    f = open(resultdir + os.path.sep + 'results.txt', 'wt')
    

    f.write("MyMLPC accuracy:\n")
    f.write(f"test  mean : {results['mymlpc']['test_mean']:.6f}\n")
    f.write(f"test  std  : {results['mymlpc']['test_std']:.6f}\n")
    f.write(f"train mean : {results['mymlpc']['train_mean']:.6f}\n")
    f.write(f"train std  : {results['mymlpc']['train_std']:.6f}\n")

    f.write("\n")
    f.write("train_accurary\ttest_accurary\n")

    for i, (a1, a2) in enumerate(zip(results["mymlpc"]["train_accs"] , results["mymlpc"]["test_accs"] )):
        f.write(f"{i+1}\t{a1:.6f}\t{a2:.6f}\n")

    f.write("\n\n\n")

    f.write("sk-MLPC accuracy:\n")
    f.write(f"test mean  : {results['skmlpc']['test_mean']:.6f}\n")
    f.write(f"test std   : {results['skmlpc']['test_std']:.6f}\n")
    f.write(f"train mean : {results['skmlpc']['train_mean']:.6f}\n")
    f.write(f"train std  : {results['skmlpc']['train_std']:.6f}\n")

    f.write("\n")
    f.write("train_accurary\ttest_accurary\n")

    for i, (a1, a2) in enumerate(zip(results["skmlpc"]["train_accs"], results["skmlpc"]["test_accs"])):
        f.write(f"{i+1}\t{a1:.6f}\t{a2:.6f}\n")


    # generate LaTex code for report writing.
    f.write("\n\n\n\n")
    f.write("LaTex\n")

    for i, (a1, a2, a3, a4) in enumerate(zip(   results["mymlpc"]["train_accs"], results["mymlpc"]["test_accs"], 
                                                results["skmlpc"]["train_accs"], results["skmlpc"]["test_accs"])):
        f.write(f"{i+1} & {a1:.6f} & {a2:.6f} & {a3:.6f} & {a4:.6f} \\\\ \n")

    f.write(f"\\textbf{{mean}} & {results['mymlpc']['train_mean']:.6f} & {results['mymlpc']['test_mean']:.6f} & {results['skmlpc']['train_mean']:.6f} & {results['skmlpc']['test_mean']:.6f} \\\\ \n")
    f.write(f"\\textbf{{std}} & {results['mymlpc']['train_std']:.6f} & {results['mymlpc']['test_std']:.6f} & {results['skmlpc']['train_std']:.6f} & {results['skmlpc']['test_std']:.6f} \\\\ \n")
    f.write("\n\n\n\n")

    f.close()




