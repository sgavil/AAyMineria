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


def dibuja_learning_curve(error, error_val):
    xx = np.linspace(0, 12, 12)
    plt.plot(xx, error, label='Train')
    plt.plot(xx, error_val, label='Cross Validation')

    plt.title("Learning curve for linear regression")
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')

    plt.legend()

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
        + (reg / m) * Theta[1:]


def f_optimizacion(Theta, X, y, reg):
    return f_coste(Theta, X, y, reg), f_gradiente(Theta, X, y, reg)


def get_optimize_theta(X, y, reg):
    initial_theta = np.zeros((X.shape[1], 1))

    optTheta = opt.minimize(fun=f_optimizacion, x0=initial_theta, 
            args=(X, y, reg), method='TNC', jac=True,
            options={'maxiter': 200})

    return optTheta.x



########################################################################
######################                            ######################
######################  APARTADOS DE LA PRACTICA  ######################
######################                            ######################
########################################################################

def learning_curve(X, y, Xval, yval, reg):
    m = len(X)

    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    for i in range(1, m + 1):
        Theta = get_optimize_theta(X[: i], y[: i], reg)

        error_train[i - 1] = f_optimizacion(Theta, X[: i], y[: i], 0)[0]
        error_val[i - 1] = f_optimizacion(Theta, Xval, yval, 0)[0]

    dibuja_learning_curve(error_train, error_val)



########################################################################
################################        ################################
################################  MAIN  ################################
################################        ################################
########################################################################

def main():
    data = loadmat("ex5data1.mat")

    y = data["y"]
    X = data["X"]
    X_ones = np.hstack((np.ones((X.shape[0], 1)), X))

    yval = data["yval"]
    Xval = data["Xval"]
    Xval_ones = np.hstack((np.ones((Xval.shape[0], 1)), Xval))

    ytest = data["ytest"]
    Xtest = data["Xtest"]

    reg = 0
    
    learning_curve(X_ones, y, Xval_ones, yval, reg)

main()