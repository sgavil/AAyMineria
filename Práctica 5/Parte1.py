import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

def dibuja_grafica(Theta, X, y):
    xx = np.linspace(np.amin(X), np.amax(X), 256)
    plt.scatter(X, y, marker='x', c='red')

    xx = xx[:, None]
    xx_ones = np.hstack((np.ones((xx.shape[0], 1)), xx))
    plt.plot(xx, h(xx_ones, Theta[:, None]))

    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')

    plt.show()


def h(X, Theta):
    return np.dot(X, Theta)

def f_coste(Theta, X, y, reg):
    m = len(y)
    return (1 / (2 * m)) * (np.sum((h(X, Theta[:, None]) - y) ** 2)) \
        + (reg / (2 * m)) * (np.sum(Theta[1:] ** 2))

def f_gradiente(Theta, X, y, reg):
    m = len(y)

    return (1 / m) * (np.sum(np.dot((h(X, Theta[:, None]) - y).T, X), axis=0)) \
        + (reg / m) * Theta[1:]

def f_optimizacion(Theta, X, y, reg):
    return f_coste(Theta, X, y, reg), f_gradiente(Theta, X, y, reg)

def main():
    data = loadmat("ex5data1.mat")

    y = data["y"]
    X = data["X"]

    yval = data["yval"]
    Xval = data["Xval"]

    ytest = data["ytest"]
    Xtest = data["Xtest"]

    X_ones = np.hstack((np.ones((X.shape[0], 1)), X))
    n = X_ones.shape[1]
    Theta = np.array([1, 1])
    reg = 0
    
    optTheta = opt.minimize(fun=f_optimizacion, x0=Theta, 
            args=(X_ones, y, reg), method='TNC', jac=True,
            options={'maxiter': 70})

    dibuja_grafica(optTheta.x, X, y)


main()