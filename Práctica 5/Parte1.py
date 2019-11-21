import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

def h(X, Theta):
    return np.dot(X, Theta)

def coste_reg(X, y, Theta, reg):
    m = len(y)
    return (1 / (2 * m)) * (np.sum((h(X, Theta[:, None]) - y) ** 2)) \
        + (reg / (2 * m)) * (np.sum(Theta[1:] ** 2))


def main():
    data = loadmat("ex5data1.mat")

    y = data["y"]
    X = data["X"]

    yval = data["yval"]
    Xval = data["Xval"]

    ytest = data["ytest"]
    Xtest = data["Xtest"]

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    n = X.shape[1]
    Theta = np.array([1, 1])
    reg = 1

    coste = coste_reg(X, y, Theta, reg)
    print(coste)


main()