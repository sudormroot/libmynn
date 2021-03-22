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

from nn import *




""" The class of the implementation of a simple Multiple Layer Perceptron Classifier

"""

class MyMLPClassifier:

    # '*' indicates keyword only parameters
    def __init__(   self, 
                    *, 
                    modelfile = None, # Load a model from given filename
                    n_input = None, # The dimension of inputs
                    n_output = None, # The dimension of output
                    hidden_sizes = (13, 13), #hidden layer sizes
                    learning_rate = 0.005, # The learning rate
                    batch_size = 200, # The batch size for mini batch training
                    n_epochs = 30,  # The number of epochs
                    n_samples_per_epoch = -1, # The number of samples per epoch
                    threshold = 0.5, # The threshold for prediction
                    hidden_activation = 'relu', # activation function for input and hidden layers
                    output_activation = 'sigmoid',
                    random_seed = 0, # random seed
                    loss = 'MSE',
                    alpha = 0.0001, # regularization factor
                    print_per_epoch = 10, # printing per epoch
                    optimizer = "simple",
                    optimizer_options = {},
                    debug = False
                    ):



        loss_functions = {
                            'MSE':(self.MSELoss, self.dMSELoss)
                        }

        self.signature = __class__.__name__ + "_version_" + MYMLPC_VERSION

        # We keep the parameters here for later uses.
        self.model = {}

        if modelfile:
            # load model from a file
            self.load(modelfile)
            loss = self.model['loss']
            # check signature
            if self.model['signature'] != self.signature:
                print("model['signature']: ", model['signature'], " not machined with signature: ", self.signature)
                raise ValueError
        else:
            # initialize a new model
            self.model['signature'] = self.signature
            self.model['batch_size'] = batch_size
            self.model['n_epochs'] = n_epochs
            self.model['n_samples_per_epoch'] = n_samples_per_epoch
            self.model['threshold'] = threshold
            self.model['n_input'] = n_input
            self.model['n_output'] = n_output
            self.model['hidden_sizes'] = hidden_sizes
            self.model['learning_rate'] = learning_rate
            self.model['hidden_activation'] = hidden_activation
            self.model['output_activation'] = output_activation
            self.model['random_seed'] = random_seed
            self.model['print_per_epoch'] = print_per_epoch
            self.model['debug'] = debug
            self.model['alpha'] = alpha
            self.model['sorted_labels'] = [None] * self.model['n_output']
            self.model['loss'] = loss
            self.model['weights'] = []
            self.model['onehot_to_label'] = {}
            self.model['label_to_onehot'] = {}
            self.model['optimizer'] = optimizer
            self.model['optimizer_options'] = optimizer_options


        # Check parameters
        assert self.model['n_epochs'] >= 1
        assert self.model['threshold'] > 0.
        assert self.model['n_input'] >= 1
        assert self.model['n_output'] >= 1
        assert len(self.model['hidden_sizes']) >= 1
        assert self.model['learning_rate'] >= 0.
        assert self.model['batch_size'] >= 1
        assert self.model['loss'] in loss_functions
        assert self.model['alpha'] >= 0.

        # Set loss function
        self.loss = loss_functions[loss][0]
        self.dloss = loss_functions[loss][1]

        self.loss_hist_ = []

        """ We define the network structure by using our MyNNLayer class as building blocks.
            
        """

        # Keep network structure as a list
        self.net = []

        """
        # The input layer
        layer_input = MyNNLayer( 
                                name = "input", 
                                n_input = self.model['n_input'], #n_input, 
                                n_neurons = self.model['hidden_sizes'][0], 
                                batch_size = self.model['batch_size'], #batch_size,
                                random_seed = self.model['random_seed'],
                                activation = self.model['activation'],
                                learning_rate = self.model['learning_rate'],
                                alpha = self.model['alpha'],
                                debug = self.model['debug']
                                )

        self.net.append(layer_input)
        
        n_neurons = self.model['hidden_sizes'][0]
        """
        n_neurons = self.model['n_input']

        # Hidden layers
        for i in range(len(self.model['hidden_sizes'])):
            layer_hidden = MyNNLayer(    
                                        name = f"hidden_{i}", 
                                        n_input = n_neurons, 
                                        n_neurons = self.model['hidden_sizes'][i],
                                        batch_size = self.model['batch_size'], #batch_size,
                                        random_seed = self.model['random_seed'],
                                        activation = self.model['hidden_activation'],
                                        learning_rate = self.model['learning_rate'],
                                        alpha = self.model['alpha'],
                                        optimizer = self.model['optimizer'],
                                        optimizer_options = self.model['optimizer_options'],
                                        debug = self.model['debug']
                                        )
            
            n_neurons = self.model['hidden_sizes'][i]

            self.net.append(layer_hidden)


        # output layer
        # We use softmax activation for last layer to score into [0, 1]        

        layer_output = MyNNLayer(    
                                    name = "output", 
                                    n_input = self.model['hidden_sizes'][-1], 
                                    n_neurons = self.model['n_output'], 
                                    batch_size = self.model['batch_size'], #batch_size,
                                    random_seed = self.model['random_seed'],
                                    activation = self.model['output_activation'],
                                    learning_rate = self.model['learning_rate'],
                                    alpha = self.model['alpha'],
                                    optimizer = self.model['optimizer'],
                                    optimizer_options = self.model['optimizer_options'],
                                    debug = self.model['debug']
                                    )
        
        self.net.append(layer_output)
        
        
        #layer_softmax = MySoftMaxLayer()
        #self.net.append(layer_softmax)

        # set weights if loading from file
        if modelfile:
            for (W, b), layer in zip(self.model['weights'], self.net):
                #layer.set_weights(W, b)
                layer.W = W
                layer.b = b


    """ Initialize weights again.

    """

    def init_weights(self):
        for layer in self.net:
            layer.init_weights()

    """ Backward propagation for all layers.
        
        grad is the initial gradients from dL/dz

    """

    def backward_propagation(self, grad):

        grad = grad.copy()

        for layer in self.net[::-1]:
            grad = layer.backward(grad)


    """ Forward propagation for all layers.

        x is the inputs (row) vector.

    """

    def forward_propagation(self, x):


        z = x.T
        
        for layer in self.net:
            z = layer.forward(z)

        return z


    """ Get train labels.

    """
    
    def get_labels(self, y_train):

        labels = list(y_train)
        labels = list(set(labels))

        sorted_labels = sorted(labels)

        return sorted_labels



    """ Print weights
    """

    def print_weights(self):

        for layer in self.net:
            print(layer.name)
            print("W:")
            print(layer.W)
            print("b:")
            print(layer.b)


    """ Regularization
    """

    """
    def regularization(self):
    
        r = 0.0
    
        for layer in self.net:
            r += np.sum(np.abs(layer.W)) + np.sum(np.abs(layer.b))

        r *= self.alpha

        return r
    """

    """ We use MSE loss
    """

    def MSELoss(self, y_predicted, y_truth):

        loss =  0.5 * np.power((y_predicted - y_truth), 2)
        loss = np.sum(loss, axis = 0)  
        loss = np.mean(loss)

        return loss

    """ The derivative of MSE loss.
        
        dL / dz

    """

    def dMSELoss(self, y_predicted, y_truth):
        dloss = y_predicted - y_truth
        #dloss = np.mean(dloss, axis = 1)
        return dloss


    """ We compute accuracy here.
    """

    def accuracy(self, y_predicted, y_truth):
        return np.mean(y_predicted == y_truth)


    

    """ Training our neural networks.
        We implement a naive SGD (Stochastic Gradient Descending) algorithm here.
        
        X_train is in row
        y_train is in column

    """

    """ def fit(self, X_train, y_train, *, shuffle = True, kfold = 1, validation_ratio = 0.1):
        
        # Re-initialize weights again.
        #self.init_weights()

        # Create labels
        self.model['sorted_labels'] = self.get_labels(y_train)

        if self.model['n_output'] > len(self.model['sorted_labels']):
            self.model['sorted_labels'] += [None] * (self.model['n_output'] - len(self.model['sorted_labels']))


        # label-to-onehot
        n_labels = len(self.model['sorted_labels'])
        onehot_I = np.eye(n_labels, dtype = np.double)
        self.model['label_to_onehot'] = {k:tuple(onehot_I[i]) for i, k in enumerate(self.model['sorted_labels'])}
        #print(self.model['label_to_onehot'].items())

        # onehot-to-label
        self.model['onehot_to_label'] = {v:k for k,v in self.model['label_to_onehot'].items()}
        #print(self.model['onehot_to_label'])

        # one-hot encoding for labels
        y_train_onehot = [list(self.model['label_to_onehot'][y]) for y in y_train]
        y_train_onehot = np.array(y_train_onehot).T

        # loss history.
        self.loss_hist_ = []

        xshape = X_train.shape
        n_samples = xshape[0]

        # Checking input
        assert X_train.shape[0] == y_train_onehot.shape[1]
        assert kfold >= 1
        assert X_train.shape[0] >= 1 # at least one sample
        assert X_train.shape[0] >= kfold


        kfold_data = {}

        for k in range(kfold):

            n_samples_per_fold = n_samples // kfold

            assert n_samples_per_fold >= 1

            kfold_data[k] = {}
            kfold_data[k]["X_train"] = []
            kfold_data[k]["y_train"] = []

            kfold_data[k]["X_validation"] = []
            kfold_data[k]["y_validation"] = []

        # mini batch SGD implementation
        #for epoch in range(self.n_epochs):
        for epoch in range(self.model['n_epochs']):
 

            #if self.model['n_samples_per_epoch'] == -1:
            #    n_samples = xshape[0]
            #else:
            #    n_samples = self.model['n_samples_per_epoch']


            # Check batch size
            self.batch_size = min(self.model['batch_size'], n_samples)

            assert self.model['batch_size'] >= 1

            indices = np.arange(n_samples)

            # reshuffle samples
            np.random.shuffle(indices)

            #debug
            #print("indices=", indices)

            grads = []

            loss = 0
            loss_hist = []

            # Train with a batch size
            for start_idx in range(0, n_samples - self.model['batch_size'] + 1, self.model['batch_size']):
                end_idx = min(start_idx + self.model['batch_size'], xshape[0])

                sel = indices[start_idx:end_idx]
            
                # select a batch of samples
                X = X_train[sel]
                #y_truth = y_train.T[sel].T
                y_truth_onehot = y_train_onehot.T[sel].T

                #print(y_truth_onehot.T)

                # Compute forward propagation data
                y_predicted_onehot = self.forward_propagation(X)
                #print(y_predicted_onehot)
                
                #y_predicted = self.predict_prob(X)

                #print("y_predicted.shape=", y_predicted.shape)

                # Compute loss
                #loss = self.MSELoss(y_predicted, y_truth)
                loss = self.loss(y_predicted_onehot, y_truth_onehot)
                #print("loss=", loss)
                
                loss_hist.append(loss)

                # Compute the loss derivative value
                #dloss = self.dMSELoss(y_predicted, y_truth)
                dloss = self.dloss(y_predicted_onehot, y_truth_onehot) #+ self.regularization()
                
                # Do backward propagation
                self.backward_propagation(dloss)


            # Compute average loss for each epoch
            avg_loss = np.mean(loss_hist)
            self.loss_hist_.append(avg_loss)

            # self.model['debug'] = True


            # Print accuracy for each 10 epochs
            if epoch % self.model['print_per_epoch'] == 0 and self.model['debug'] == True:
                y_predicted = self.predict(X_train)
                accuracy = self.accuracy(y_predicted, y_train)
                print(f"epoch={epoch} loss={loss} accuracy={accuracy}")
    """
 
    def fit(self, X_train, y_train):
        
        # Re-initialize weights again.
        #self.init_weights()

        # Create labels
        self.model['sorted_labels'] = self.get_labels(y_train)

        if self.model['n_output'] > len(self.model['sorted_labels']):
            self.model['sorted_labels'] += [None] * (self.model['n_output'] - len(self.model['sorted_labels']))


        # label-to-onehot
        n_labels = len(self.model['sorted_labels'])
        onehot_I = np.eye(n_labels, dtype = np.double)
        self.model['label_to_onehot'] = {k:tuple(onehot_I[i]) for i, k in enumerate(self.model['sorted_labels'])}
        #print(self.model['label_to_onehot'].items())

        # onehot-to-label
        self.model['onehot_to_label'] = {v:k for k,v in self.model['label_to_onehot'].items()}
        #print(self.model['onehot_to_label'])

        # one-hot encoding for labels
        y_train_onehot = [list(self.model['label_to_onehot'][y]) for y in y_train]
        y_train_onehot = np.array(y_train_onehot).T

        # loss history.
        self.loss_hist_ = []
        
        # accuracy history.
        self.accuracy_hist_ = []

        # Checking input
        assert X_train.shape[0] == y_train_onehot.shape[1]

        # mini batch SGD implementation
        #for epoch in range(self.n_epochs):
        for epoch in range(self.model['n_epochs']):
 
            xshape = X_train.shape

            #if self.model['n_samples_per_epoch'] == -1:
            #    n_samples = xshape[0]
            #else:
            #    n_samples = self.model['n_samples_per_epoch']

            n_samples = xshape[0]

            # Check batch size
            self.batch_size = min(self.model['batch_size'], n_samples)

            assert self.model['batch_size'] >= 1

            indices = np.arange(n_samples)

            # reshuffle samples
            np.random.shuffle(indices)

            #debug
            #print("indices=", indices)

            grads = []

            loss = 0
            loss_hist = []

            # Train with a batch size
            for start_idx in range(0, n_samples - self.model['batch_size'] + 1, self.model['batch_size']):
                end_idx = min(start_idx + self.model['batch_size'], xshape[0])

                sel = indices[start_idx:end_idx]
            
                # select a batch of samples
                X = X_train[sel]
                #y_truth = y_train.T[sel].T
                y_truth_onehot = y_train_onehot.T[sel].T

                #print(y_truth_onehot.T)

                # Compute forward propagation data
                y_predicted_onehot = self.forward_propagation(X)
                #print(y_predicted_onehot)
                
                #y_predicted = self.predict_prob(X)

                #print("y_predicted.shape=", y_predicted.shape)

                # Compute loss
                #loss = self.MSELoss(y_predicted, y_truth)
                loss = self.loss(y_predicted_onehot, y_truth_onehot)
                #print("loss=", loss)
                
                loss_hist.append(loss)

                # Compute the loss derivative value
                #dloss = self.dMSELoss(y_predicted, y_truth)
                dloss = self.dloss(y_predicted_onehot, y_truth_onehot) #+ self.regularization()
                
                # Do backward propagation
                self.backward_propagation(dloss)


            # Compute average loss for each epoch
            avg_loss = np.mean(loss_hist)
            self.loss_hist_.append(avg_loss)

            # self.model['debug'] = True


            # Print accuracy for each 10 epochs
            if epoch % self.model['print_per_epoch'] == 0 and self.model['debug'] == True:
                y_predicted = self.predict(X_train)
                accuracy = self.accuracy(y_predicted, y_train)
                print(f"epoch={epoch} loss={loss} accuracy={accuracy}")
 
                self.accuracy_hist_.append(accuracy)


            """ Old implementation with batch size 1
            idx = np.random.permutation(n_samples)

            for i in range(n_samples):

                sel = idx[i]

                X = X_train[sel]
                y_truth = y_train.T[sel]

                # Compute forward propagation data
                y_predicted = self.forward_propagation(X)

                # Compute loss
                loss = self.MSELoss(y_predicted, y_truth)

                loss_hist.append(loss)

                # Compute the loss derivative value
                dloss = self.dMSELoss(y_predicted, y_truth)

                # Do backward propagation
                self.backward_propagation(dloss)

            # Compute average loss for each epoch
            avg_loss = np.mean(loss_hist)

            self.loss_hist.append(avg_loss)

            # Print accuracy for each 10 epochs
            if epoch % 10 == 0 and self.debug == True:
                y_predicted = self.predict(X_train)
                accuracy = self.accuracy(y_predicted, y_train)
                print(f"epoch={epoch} loss={loss} accuracy={accuracy}")
            """

    """ Predict in the form of onehot vector.
        
        X can have multiple rows, each row is a x.

    """

    def predict(self, X):

        y_probs = self.predict_prob(X)

        y_probs = y_probs.T

        y = np.where(y_probs >= self.model['threshold'], 1, 0)

        y_labels = []

        for yi in y:
            #decode onehot to label
            if tuple(yi) in self.model['onehot_to_label']:
                label = self.model['onehot_to_label'][tuple(yi)]
            else:
                label = random.choice(list(self.model['onehot_to_label'].values()))
            y_labels.append(label)

        y_labels = np.array(y_labels)
        y_labels = y_labels.flatten()

        return y_labels


    """ Return the probabilities/scores of X

    """

    def predict_prob(self, X):

        y_probs = []

        xshape = X.shape

        for i in range(xshape[0]):
            xi = X[i]
            xi = xi.reshape(1, -1)
            yi = self.predict_prob_one(xi)
            yi = yi.flatten()
            y_probs.append(list(yi))

        y_probs = np.array(y_probs)

        y_probs = y_probs.T

        return y_probs


    """ Predict the raw score from forward propagation
    """

    def predict_prob_one(self, X):
        y = self.forward_propagation(X)
        return y

    """ Return the loss history
    """

    def loss_history(self):
        return np.array(self.loss_hist_)


    """ Save model to a file
    """

    def save(self, modelfile):

        print("Saving model: ", modelfile)
        
        self.model['weights'] = []

        for layer in self.net:
            weights = (layer.W, layer.b)
            self.model['weights'].append(weights)

        pickle.dump(self.model, open(modelfile, "wb" ))
        #json.dump(self.model, open(modelfile, "wt" ), indent = 4)
        


    """ Load model from a file
    """

    def load(self, modelfile):
        
        print("Loading model: ", modelfile)

        self.model = pickle.load(open(modelfile, "rb" ))
        #self.model = json.load(open(modelfile, "rt" ))


