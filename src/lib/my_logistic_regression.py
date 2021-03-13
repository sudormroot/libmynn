#!env python3

""" Despite I have implemented a full-feature NN as a part of the assignment II of the module machine learning.

    I won't use it for Part-I and Part-II, I will totally write a Logistic Regression implementation here.

    But I will re-use some ideas I used before to make the project elegant.

    Our algorithms will always use matrix operations instead of using loops.

"""

import numpy as np
import random
import pickle

"""
Course:        CT5133 Deep Learning (2020-2021)
Student Name:  Jiaolin Luo
Student ID:    20230436
Student email: j.luo2@nuigalway.ie

This a fancy implementation a logistic regression, we want the
APIs can be compatible with scikit-learn.

"""

class MyFancyLogisticRegression:

    def __init__(    self,
                     *,   # this indicate for later parameter assignments,
                         # we need to use a named parameter assignments.

                     learning_rate = 0.01, # Learning rate
                     max_iters = 100, # max iterations
                     n_input = None, # input
                     print_per_iter = 10, # print per iteration

                     modelfile = None # Loading existing model
                    ):

        self.model = {}

        # If the model is empty, we initialise a new model
        # otherwise we will load a model from file.
        if modelfile is None:

            # We save our parameters to a dictionary
            # in order to conveniently save/load to/from a disk file.
            self.model["learning_rate"] = learning_rate
            self.model["max_iters"] = max_iters
            self.model["n_input"] = n_input
            self.model['print_per_iter'] = 10

            # We hope our weights are small to avoid the saturation of sigmoid activation function.
            sigma = 0.001

            # Initialise weights
            self.w = np.random.uniform(-sigma, sigma, (1, n_input)).flatten()
            self.b = np.random.uniform(-sigma, sigma)

        # If we assign the modelfile, we will load the model from a file
        # instead of initialising it.
        elif os.path.exists(modelfile):
            print("Loading model: ", modelfile)
            self.model = pickle.load(open(modelfile, "rb" ))

            self.w = self.model["w"]
            self.b = self.model["b"]

        else:
            print("Model file does not exist: ", modelfile)
            exit()


    """ We can save our model into a file

    """

    def save(self, modelfile):

        print("Saving model: ", modelfile)

        self.model["w"] = self.w
        self.model["b"] = self.b

        pickle.dump(self.model, open(modelfile, "wb" ))


    """ The Sigmoid activation function

    """

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))



    """ We do forward computation here.
        This function will be used for training API fit()
        the predict() API, etc.

    """

    def forward(self, x):

        x = np.array(x.copy()).flatten()

        # We store x for the use of backward().
        self.x = x

        # Calculating the z
        z = self.w.dot(x) + self.b

        # Calculating the prediction
        return self.sigmoid(z)



    """ We implement our backward propogation here.
        This function will be used for training API fit()

    """

    def backward(self, grad):

        # Calculating the gradients wrt w and b.
        dw = grad * self.x
        db = grad

        # Updating the weights and bias now!
        self.w = self.w - self.model["learning_rate"] * dw
        self.b = self.b - self.model["learning_rate"] * db



    """ Calculating the loss value

    """

    def loss(self, y_hat, y):
        loss = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        return loss



    """ Calculating the accuracy

    """

    def accuracy(self, y_pred, y_truth):
        return np.mean(y_pred == y_truth)


    """ We may need to return the history loss values
        for users to plot the learning curve.

    """

    def hist_loss(self):
        return self.hist_loss_



    """ A SGD implementation

    """

    def fit(self, X_train, y_train):

        xshape = X_train.shape
        n_samples = xshape[0]

        indices = np.arange(n_samples)


        self.hist_loss_ = []

        # Each epoch/iteration will call the forward()
        # to calculate the prediction, and then the predicted
        # value will be used to calculated the error. Finally the
        # error will be used to adjust the weights.

        for it in range(self.model["max_iters"]):

            # We randomly sample a data point.
            sel = np.random.choice(indices)

            x = X_train[sel]
            y = y_train[sel]

            # We calculate the prediction first.
            y_hat = self.forward(x)

            # Calculate the gradient
            grad = y_hat - y

            # We feed the graident backwards.
            self.backward(grad)

            # Calculating the loss
            loss = np.abs(self.loss(y_hat, y))

            self.hist_loss_.append(loss)

            # We print the result for each print_per_iter.
            if it % self.model['print_per_iter'] == 0 and it > 0:

                y_pred = self.predict(X_train)

                accuracy = self.accuracy(y_pred, y_train)

                print(f"#{it} accuracy = {accuracy:.5f} loss = {loss:.4f}")



    """ We do prediction here.

    """

    def predict(self, X):

        y_hats = np.array([self.predict_prob(xi) for xi in X])

        y_hats[y_hats >= 0.5] = 1
        y_hats[y_hats <  0.5] = 0

        return y_hats



    """ This API returns the probability of a x.

    """

    def predict_prob(self, x):
        y_hat = self.forward(x)
        return y_hat












