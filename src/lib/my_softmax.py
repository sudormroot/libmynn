#!env python3


import numpy as np


class MySoftMaxLayer:
    
    def __init__(self):
        self.x = None
    
    def init_weights(self):
        pass

    def forward(self, x):
        # Adding a constant number doesn't change the derivative form.
        # This constant number can make the exp() stable.

        #z = x.copy()
        z = x - np.max(x)


        p = np.exp(z)

        s = np.sum(p)

        y = p / s
    
        self.x = z
        self.y = y

        #print("softmax: ", y.T)

        return self.y

    def backward(self, grad):

        """ We first compute the gradient matrix (Jacobian matrix)
            for the given input vector where y is the input vector.

        """

        # last y
        y = self.y.reshape(1, -1)

        # identity vector
        e = np.ones(y.shape).reshape(1, -1)

        # Computing the Jacobian matrix with the Kronecker delta matrix.
        J = np.diagflat(e) - y.copy().T.reshape(-1, 1) @ e

        J = J * y.reshape(-1, 1)

        """ We then compute the output gradient vector according to
            the Jacobian matrix

        """

        # Computing the outputing gradients by multiplying 
        # the Jacobian matrix with inputing gradients
        #print("J.shape=", J.shape)
        #print("grad.shape=", grad.shape)
        grad_out = J @ grad
        
        return grad_out



