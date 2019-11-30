import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat


def h(X, Theta):
    return np.dot(X, Theta)


def f_coste(Theta, X, y, reg):
    m = len(y)
    return (1 / (2 * m)) * (np.sum((h(X, Theta[:, None]) - y) ** 2)) \
        + (reg / (2 * m)) * (np.sum(Theta[1:] ** 2))


def f_gradiente(Theta, X, y, reg):
    m = len(y)
    return (1 / m) * \
        (np.sum(np.dot((h(X, Theta[:, None]) - y).T, X), axis=0))\
        + (reg / m) * Theta


def f_optimizacion(Theta, X, y, reg):
    return f_coste(Theta, X, y, reg), f_gradiente(Theta, X, y, reg)


def h_Polinomial(X, p):  # X[12,1] p(8)
    newTrainingSet = np.zeros((X.shape[0], p))  # [12,8]

    for i in range(1, X.shape[0]+1):
        new_col = np.zeros((1, p))
        new_col = ((X[:p])**i).T
        newTrainingSet[i-1] = new_col

    return newTrainingSet  # Shape [12,p]


def normalizacion_pol(hPol):  # Se normaliza con media 0 y desv. Estandar 1
    cols = hPol.shape[1]
    mu = np.zeros(cols)
    sigma = np.zeros(cols)

    h_norm = np.zeros((hPol.shape[0], cols))

    for i in range(cols):
        mu[i] = np.mean(hPol[:, i])
        sigma[i] = np.std(hPol[:, i])

    for i in range(cols):
        h_norm[:, i] = (hPol[:, i] - 0) / 1

    mu = mu[:, None]
    sigma = sigma[:, None]

    # return [m,p] normalizado , 1xp media de cada columna, 1xp desviacion de cada columna
    return h_norm, mu.T, sigma.T


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
    reg = 0
    p = 8

    newSet = h_Polinomial(X, p)
    newSet_norm, mu, sigma = normalizacion_pol(newSet)

    newSet_norm = np.hstack((np.ones((newSet_norm.shape[0], 1)), newSet_norm))

    Theta = np.zeros((newSet_norm.shape[1], 1))

    # Theta Shape 9,1
    # NewSet Shape 12,9

    optTheta = opt.minimize(fun=f_optimizacion, x0=Theta,
                            args=(newSet_norm, y, reg), method='TNC', jac=True,
                            options={'maxiter': 70})

    xx = np.arange(np.amin(X), np.amax(X), 0.05)
    xx = xx[:, None]
    xx = h_Polinomial(xx, p)

    xx_norm = np.zeros((xx.shape[0], xx.shape[1]))  # 1712,8

    for i in range(xx.shape[1]):
        xx_norm[:, i] = (xx[:, i] - mu[0][i]) / sigma[0][i]

    xx_norm = np.hstack((np.ones((xx_norm.shape[0], 1)), xx_norm))

    newY = h(xx_norm, optTheta.x)

    dibuja_grafica(X, y, xx_norm, newY)


def dibuja_grafica(X, y, newX, newY):

    plt.scatter(X, y, marker='x', c='red')

    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')

    plt.show()


main()
