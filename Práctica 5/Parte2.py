import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

def dibuja_grafica(Theta, X, error, error_val):
    xx = np.linspace(0, 12, 12)
    plt.plot(xx, error, label='Train')
    plt.plot(xx, error_val, label='Cross Validation')

    plt.title("Learning curve for linear regression")
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')

    plt.legend()

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
    Xval_ones = np.hstack((np.ones((Xval.shape[0], 1)), Xval))

    ytest = data["ytest"]
    Xtest = data["Xtest"]

    X_ones = np.hstack((np.ones((X.shape[0], 1)), X))
    n = X_ones.shape[1]
    Theta = np.array([1, 1])
    reg = 0
    
    error = np.zeros(len(y))
    error_val = np.zeros(len(y))

    for i in range(len(y)):
        optTheta = opt.minimize(fun=f_optimizacion, x0=Theta, 
                args=(X_ones[0 : i + 1], y[0 : i + 1], reg), method='TNC', jac=True,
                options={'maxiter': 70})
        
        Theta = optTheta.x
        error[i] = f_coste(Theta, X_ones[0 : i + 1], y[0 : i + 1], reg)
        error_val[i] = f_coste(Theta, Xval_ones, yval, reg)

    dibuja_grafica(optTheta.x, X, error, error_val)


main()