import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# input_size es el número de columnas de X
# hidden_size es el número de nodos en la capa del medio
# num_labels es el tamaño de la solución

def backward_propagate(a1, a2, h, z2, theta2, Y, delta1, delta2):
    d3 = h - Y
    a2_ = a2 * (1 - a2) # a2 (5000,26)
                        # a2_ (5000, 26)

    d2 = np.dot(d3, theta2) * a2_
    d2 = np.delete(d2, 0, 1)

    delta1 = delta1 + np.dot(a1.T, d2).T
    delta2 = delta2 + np.dot(a2.T, d3).T

def testClassificator(h, Y):
    labels = (h >= 0.5) * 1
    precision = np.mean(labels == Y) * 100
    print("La precisión es del", precision)


def forward_propagate(X, theta1, theta2):
    m = X.shape[0] 

    a1 = np.hstack([np.ones([m, 1]), X])    # (5000, 401)
    z2 = np.dot(a1, theta1.T)   # (5000, 25)

    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])  # (5000, 26)
    z3 = np.dot(a2, theta2.T)   # (5000, 10)

    h = sigmoid(z3) # (5000, 10)

    return a1, z2, a2, z3, h


def main():
    data = loadmat("ex3data1.mat")
    Y = data["y"] # Y (5000, 1)
    X = data["X"] # X (5000, 400)

    weights = loadmat("ex3weights.mat")
    theta1, theta2 = weights["Theta1"], weights["Theta2"]
    # Theta1 es de dimensión 25 x 401
    # Theta2 es de dimensión 10 x 26

    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    testClassificator(h, Y)

    # backward_propagate(a1, a2, h, z2, theta2, Y, delta1, delta2)
    

    #Theta1 es de dimensión 25 x 401
    #Theta2 es de dimensión 10 x 26

main()