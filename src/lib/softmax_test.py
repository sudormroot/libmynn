#!/usr/bin/env python3

import numpy as np

x = np.array([1, 2, 3])

def softmax(x):
    z = x - np.max(x)
    y = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    return y

print("x=", x)
y = softmax(x)
print("softmax(x)=", y)

def dsoftmax(y):
    s = y.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

y = dsoftmax(y)

print("dsoftmax(y)=", y)

z = np.sum(y, axis = 1)

print("z=", z)

print("type(z)=", type(z))
