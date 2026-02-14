import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

#X, Y = "data for breast cancer"  

# the function to access tanh function easily
def tanh(z):
    return np.tanh(z)

# the function to access sigmoid function easily
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# getting the layer sizes for later setting shapes for parameters
def layer_size(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h = 8  # there are 8 nodes in the hidden layer of the 2-layer neural network
    return n_x, n_y, n_h

# initializing the parameters with the shapes we got from the layers
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    W2 = np.random.randn(n_y, n_h) * 0.01
    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }
    return parameters

# simple forward propagation of logistic regression implemented for each layer
def forward_prop(parameters, X):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "Z2": Z2,
        "A1": A1,
        "A2": A2
    }

    return cache

# computing cost function using the formulas used in logistic regression
def compute_cost(cache, Y):
    m = Y.shape[1]
    A2 = cache["A2"]
    A2 = np.clip(A2, 1e-15, 1 - 1e-15)

    loss = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -((1 / m) * np.sum(loss))
    cost = float(np.squeeze(cost))

    return cost

# simple backward propagation for logistic regression implemented for each layer
def back_prop(parameters, cache, X, Y):
    m = X.shape[1]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        "dZ2": dZ2,
        "dW2": dW2,
        "db2": db2,
        "dZ1": dZ1,
        "dW1": dW1,
        "db1": db1
    }
    return gradients

# updating the parameters (this will be iterated for gradient descent)
def parameter_update(parameters, gradients, learning_rate=0.01):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    W1 = W1 - (learning_rate * dW1)
    b1 = b1 - (learning_rate * db1)
    W2 = W2 - (learning_rate * dW2)
    b2 = b2 - (learning_rate * db2)

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters
