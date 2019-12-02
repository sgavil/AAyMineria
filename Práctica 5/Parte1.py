import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

########################################################################
############################                ############################
############################    DIBUJADO    ############################
############################                ############################
########################################################################

def dibuja_grafica_inicial(Theta, X, y):
    xx = np.linspace(np.amin(X), np.amax(X))
    plt.scatter(X, y, marker='x', c='red', s=100, linewidths=0.5)

    xx = xx[:, None]
    xx_ones = np.hstack((np.ones((xx.shape[0], 1)), xx))
    plt.plot(xx, h(xx_ones, Theta[:, None]))

    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')

    plt.show()



########################################################################
################                                        ################
################  CALCULOS DE COSTE, GRADIENTE Y THETA  ################
################                                        ################
########################################################################

def h(X, Theta):
    return np.dot(X, Theta)


def f_coste(Theta, X, y, reg):
    m = len(X)
    Theta = Theta[:, None]
    return (1 / (2 * m)) * np.sum(np.square(h(X, Theta) - y)) \
        + (reg / (2 * m)) * np.sum(np.square(Theta[1:]))


def f_gradiente(Theta, X, y, reg):
    m = len(X)
    return (1 / m) * (np.sum(np.dot((h(X, Theta[:, None]) - y).T, X), axis=0)) \
        + (reg / m) * Theta


def f_optimizacion(Theta, X, y, reg):
    return f_coste(Theta, X, y, reg), f_gradiente(Theta, X, y, reg)



########################################################################
################################        ################################
################################  MAIN  ################################
################################        ################################
########################################################################

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
            options={'maxiter': 200})

    dibuja_grafica_inicial(optTheta.x, X, y)


main()