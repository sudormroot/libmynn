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

# This our implementation
from mymlpc import MyMLPClassifier


# We run 10 times to measure the accuracy
N = 10



""" A implementation of one-hot coding.
    
    INPUTS:
        labels: The list of labels
        name:   The label name for encoding

    RETURNS:
        The encoded one-hot code.

"""

def one_hot_encode(labels, name):

    sorted_labels = labels.copy()

    sorted_labels = list(sorted_labels)

    sorted_labels = sorted(sorted_labels)

    if name not in sorted_labels:
        raise KeyError

    idx = sorted_labels.index(name)

    n_labels = len(sorted_labels)

    I = np.eye(n_labels, dtype = np.double)

    y = I[idx]

    return list(y)


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

    # We normalise the dataset
    normalise_data(df_dataset, x_names)

    # Prepare label set for one-hot coding.
    y_labels = list(df_dataset[y_name])
    y_labels = list(set(y_labels))


    #
    # Convert y labels to onehot coding.
    #
    y_onehot = [] 
    
    for i, label in enumerate(df_dataset[y_name]):
        onehot = one_hot_encode(y_labels, label)
        y_onehot.append(onehot)

    df_dataset[y_name] = y_onehot

    return df_dataset



""" Split the dataset for training and testing

    RETURNS:
        X_train, y_train, X_test, y_test
"""

def split_dataset(df):

    # The columns for input (X)
    x_names = ["calorific_value", "nitrogen", "turbidity", "alcohol", "sugars", "bitterness", "beer_id", "colour", "degree_of_fermentation"]

    # The columns for label (y)
    y_name = 'style'

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

""" The evaluation function of my MLPC

"""

def evaluate_mymlpc(X_train, y_train, X_test, y_test):

    #print("")
    #print("----------- Evaluation of my MLPC algorithm -----------")

    n_input = X_train.shape[1]
    n_output = y_train.shape[0]

    clf = MyMLPClassifier( n_input = n_input, 
                        n_output = n_output, 
                        n_hiddens = 1, 
                        n_neurons = 7, 
                        learning_rate = 0.1, 
                        n_epochs = 100, 
                        batch_size = 1,
                        random_seed = 1,
                        activation = 'relu',
                        debug = False)


    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    test_accuracy = prediction_accuracy(y_predicted, y_test)
    #print("Testing data set accuracy: ", accuracy)

    y_predicted = clf.predict(X_train)
    train_accuracy = prediction_accuracy(y_predicted, y_train)
    #print("Training data set accuracy: ", accuracy)


    #print("----------- Finished -----------")

    return clf, train_accuracy, test_accuracy


""" The evaluation of sklearn MLPC algorithm
"""

def evaluate_skmlpc(X_train, y_train, X_test, y_test):
    
    #print("")
    #print("----------- Evaluation of sklearn MLPC algorithm -----------")

    clf = MLPClassifier(  solver = 'sgd', 
                          activation = 'relu', 
                          alpha = 1e-5,
                          learning_rate_init = 0.1,
                          hidden_layer_sizes = (7,), 
                          random_state = 1,
                          max_iter = 100)


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


""" We start evaluation here.
"""

if __name__ == '__main__':

    df = prepare_beer_dataset('beer.txt')
    
    my_test_accs = []
    my_train_accs = []

    sk_test_accs = []
    sk_train_accs = []

    loss_hists = []

    for i in range(N):
        
        #print(f"--- {i} ---")

        X_train, y_train, X_test, y_test = split_dataset(df)

        # We evaluate our MLPC first.
        myclf, train_acc, test_acc = evaluate_mymlpc(X_train, y_train, X_test, y_test)

        my_train_accs.append(train_acc)
        my_test_accs.append(test_acc)

        # Retrieve the loss history.
        loss_hist = myclf.loss_history()
        loss_hists.append(loss_hist)


        # We evaluate the sklearn MLPC.
        skclf, train_acc, test_acc = evaluate_skmlpc(X_train, y_train, X_test, y_test)
        
        sk_train_accs.append(train_acc)
        sk_test_accs.append(test_acc)

    
    my_test_acc_mean = np.mean(my_test_accs)
    my_test_acc_std = np.std(my_test_accs)

    my_train_acc_mean = np.mean(my_train_accs)
    my_train_acc_std = np.std(my_train_accs)

    sk_test_acc_mean = np.mean(sk_test_accs)
    sk_test_acc_std = np.std(sk_test_accs)

    sk_train_acc_mean = np.mean(sk_train_accs)
    sk_train_acc_std = np.std(sk_train_accs)

    print("")
    print("My MLPC:")
    print("Average accuracy on testing dataset: ", my_test_acc_mean)
    print("Standard deviation on testing dataset: ", my_test_acc_std)
    print("")
    
    print("Average accuracy on training dataset: ", my_train_acc_mean)
    print("Standard deviation on training dataset: ", my_train_acc_std)
    print("")

    print("sklearn MLPC:")
    print("Average accuracy on testing dataset: ", sk_test_acc_mean)
    print("Standard deviation on testing dataset: ", sk_test_acc_std)
    print("")
    
    print("Average accuracy on training dataset: ", sk_train_acc_mean)
    print("Standard deviation on training dataset: ", sk_train_acc_std)
    print("")


    f = open('result.txt', 'w')
    
    f.close()

    # We draw loss-iteration figure for our MLPC

    loss_hists = np.array(loss_hists)
    loss_hist = np.mean(loss_hists, axis = 0)

    plt.plot(loss_hist, color = 'coral')

    plt.xlabel("Iterations", fontsize = 12)
    plt.ylabel("MSE Loss", fontsize = 12)
    plt.xlim(0, len(loss_hist))
    plt.ylim(0, np.max(loss_hist))

    figurefile = 'fig_mymlpc_loss.pdf'
    plt.savefig(figurefile, dpi=600, format='pdf')

    plt.show()
