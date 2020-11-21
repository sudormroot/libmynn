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


""" We implement a NN layer here.

"""

class MyMLPCNNLayer:

    """ Activation functions and their derivative forms.
        We allow users to choose either 'sigmoid' or 'tanh'
    """

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        y = self.sigmoid(x)
        return y * (1 - y)


    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        y = self.tanh(x)
        return 1.0 - y ** 2

    def relu(self, x):
        y = x.copy()
        y[y < 0] = 0
        return y

    def drelu(self, x):
        return 1. * (x > 0)

    def __init__(   self, 
                    *, 
                    name, # the name of this layer
                    n_input, # the dimension of inputs
                    n_neurons = 11, # the number of neurons
                    random_seed = 0,  # we enable to configure the random seed
                    learning_rate = 0.5, # learning rate
                    batch_size = 1, # batch size used for mini batch training
                    activation = 'sigmoid', # activation function
                    W = None, # Used for loading module from file
                    b = None, # Used for loading module from file
                    debug = False # debug flag
                    ):

        # We keep all parameters here for later use.
        self.learning_rate = learning_rate

        activations = { 'sigmoid':  (self.sigmoid,  self.dsigmoid),
                        'tanh':     (self.tanh,     self.dtanh),
                        'relu':     (self.relu,     self.drelu)
                        }

        # Check
        assert activation in activations

        # set activation function
        self.f = activations[activation][0]
        self.df = activations[activation][1]

        self.name = name
        self.batch_size = batch_size
        self.n_input = n_input
        self.n_neurons = n_neurons
        
        if random_seed > 0:
            np.random.seed(random_seed)
 
        self.debug = debug

        """ Notice!
            We should use Gaussian distribution to initialise the W matrix and b vector,
            otherwise the algorithm isn't stable of coverging.
        """

        self.W = np.random.uniform(-1, 1, (self.n_neurons, self.n_input))
        self.b = np.random.uniform(-1, 1, (self.n_neurons, 1))

        #self.W = np.random.normal(-0.5, 0.5, (self.n_neurons, self.n_input))
        #self.b = np.random.normal(0.5, 0.5, self.n_neurons)

        # x is the input from prior layer.
        # y = wx + b
        # z = f(y)
        self.x = None 
        self.y = None
        self.z = None

        # The gradients of W and b
        self.dW = None
        self.db = None



    """ forward propagation implementation for one layer.
        x is the inputs from prior layer.

        z = f(w*x + b)

    """

    def forward(self, x):

        # Keep a private copy
        self.x = x.copy()

        # Compute y = w*x + b
        self.y = self.W.dot(self.x) + self.b

        # Compute z = f(y)
        self.z = self.f(self.y)
        
        return self.z

    """ backward propagation implementation for one layer

        grad is the gradient from next layer.

    """

    def backward(self, grad):

        # Keep a private copy of dL / dz
        dLdz = grad.copy()


        # We compute the value of the derivative on z.
        # The value is dz / dy
        dzdy = self.df(self.y)


        # Compute (dL / dz) * (dz / dy)
        dLdy = dLdz * dzdy 
        
        # Compute the gradients of W and b
        #self.db = dLdy
        #self.dW = dLdy.reshape(-1, 1).dot(self.x.reshape(1, -1))

        self.dW = dLdy.dot(self.x.T) / self.batch_size

        self.db = np.mean(dLdy, axis=1).reshape(-1, 1)

        # Compute the output gradients for prior layer.
        #grad_next = dLdy.dot(self.W)
        grad_next = dLdy.T.dot(self.W)


        grad_next = np.sum(grad_next, axis = 0) / self.batch_size

        grad_next = grad_next.reshape(-1, 1)


        # We can adjust weights now.
        self.W = self.W - self.learning_rate * self.dW
        self.b = self.b - self.learning_rate * self.db

        return grad_next




""" The class of the implementation of a simple Multiple Layer Perceptron Classifier

"""

class MyMLPClassifier:

    # '*' indicates keyword only parameters
    def __init__(   self, 
                    *, 
                    modelfile = None, # Load a module from given filename
                    n_input, # The dimension of inputs
                    n_output, # The dimension of output
                    n_neurons = 7, # The number of neurons
                    n_hiddens = 3, # The number of hidden layers
                    learning_rate = 0.5, # The learning rate
                    batch_size = 200, # The batch size for mini batch training
                    n_epochs = 30,  # The number of epochs
                    threshold = 0.5, # The threshold for prediction
                    activation = 'relu', # activation function for input and hidden layers
                    random_seed = 0, # random seed
                    debug = False
                    ):


        """ Checking if we need to load a module from file
        """

        if modelfile:
            # We load existing module from a file here.
            raise NotImplemented
            return


        """ Otherwise, we create a new module from parameters

        """

        # Check parameters
        assert n_epochs >= 1
        assert threshold > 0.
        assert n_input >= 1
        assert n_output >= 1
        assert n_hiddens >= 1
        assert learning_rate >= 0.
        assert batch_size >= 1

        # We keep the parameters here for later uses.
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.n_input = n_input
        self.n_output = n_output
        self.n_neurons = n_neurons
        self.n_hiddens = n_hiddens
        self.learning_rate = learning_rate
        self.activation = activation
        self.random_seed = random_seed
        self.debug = debug
        

        self.loss_hist = []

        """ We define the network structure by using our MyMLPCNNLayer class as building blocks.
            
        """

        # Keep network structure as a list
        self.net = []

        # The input layer
        layer_input = MyMLPCNNLayer( 
                                name = "input", 
                                n_input = n_input, 
                                n_neurons = n_neurons, 
                                batch_size = batch_size,
                                random_seed = random_seed,
                                activation = activation,
                                debug = debug
                                )

        self.net.append(layer_input)
        
        # Hidden layers
        for i in range(n_hiddens):
            layer_hidden = MyMLPCNNLayer(    
                                        name = f"hidden_{i}", 
                                        n_input = n_neurons, 
                                        n_neurons = n_neurons, 
                                        batch_size = batch_size,
                                        random_seed = random_seed,
                                        activation = activation,
                                        debug = debug
                                        )
            self.net.append(layer_hidden)


        # output layer
        # We use sigmoid activation for last layer to score into [0, 1]
        layer_output = MyMLPCNNLayer(    
                                    name = "output", 
                                    n_input = n_neurons, 
                                    n_neurons = n_output, 
                                    batch_size = batch_size,
                                    random_seed = random_seed,
                                    activation = 'sigmoid', 
                                    debug = debug
                                    )

        self.net.append(layer_output)


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

    def MSEdLoss(self, y_predicted, y_truth):
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

    def fit(self, X_train, y_train):

        self.loss_hist = []

        # Checking input
        assert X_train.shape[0] == y_train.shape[1]

        # mini batch SGD implementation
        for epoch in range(self.n_epochs):
 
            xshape = X_train.shape

            n_samples = xshape[0]

            # Check batch size
            self.batch_size = min(self.batch_size, n_samples)
            assert self.batch_size >= 1

            indices = np.arange(n_samples)

            # reshuffle samples
            np.random.shuffle(indices)

            #debug
            #print("indices=", indices)

            grads = []

            loss = 0
            loss_hist = []

            # Train with a batch size
            for start_idx in range(0, n_samples - self.batch_size + 1, self.batch_size):
                end_idx = min(start_idx + self.batch_size, xshape[0])

                sel = indices[start_idx:end_idx]
            
                # select a batch of samples
                X = X_train[sel]
                y_truth = y_train.T[sel].T

                # Compute forward propagation data
                y_predicted = self.forward_propagation(X)
                #y_predicted = self.predict_prob(X)

                #print("y_predicted.shape=", y_predicted.shape)

                # Compute loss
                loss = self.MSELoss(y_predicted, y_truth)
                #print("loss=", loss)
                loss_hist.append(loss)

                # Compute the loss derivative value
                dloss = self.MSEdLoss(y_predicted, y_truth)
                
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
                dloss = self.MSEdLoss(y_predicted, y_truth)

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

        y = np.where(y_probs >= self.threshold, 1, 0)

        y = y.T

        return y


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
        return np.array(self.loss_hist)


    """ Save model to a file
    """

    def save(self, modelfile):
        raise NotImplemented
        


    """ Load model from a file
    """

    def load(self, modelfile):
        raise NotImplemented



