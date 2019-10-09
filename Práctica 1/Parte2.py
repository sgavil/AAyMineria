from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from pandas.io.parsers import read_csv


def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))


def make_data(t0_range, t1_range, X, Y):
    """Genera las matrices X, Y, Z para generar un plot en 3D"""
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)

    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    """Theta0 y Theta1 tienen las mismas dimensiones, de forma que
    cogiendo un elemento de cada uno se generan las coordenadas x, y
    de todos los puntos de la rejilla"""

    Coste = np.empty_like(Theta0)
    for ix, iy, in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])

    return [Theta0, Theta1, Coste]


def sumatory(m, X, Y, j, thetha):
    sum = 0
    for i in range(m):
        sum = sum + (h(np.array([X[i, 0], X[i, 1], X[i, 2]]),
                       thetha) - Y[i]) * X[i, j]
    return sum


def norm_matrix(X):
    fils = len(X)
    cols = len(X[0])

    onesColumn = np.ones((fils, 1))
    X_norm = np.zeros((fils, cols))

    mu = np.zeros(cols)
    sigma = np.zeros(cols)

    for i in range(cols):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])

    for i in range(cols):
        X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]

    X_norm = np.hstack((onesColumn, X_norm))

    return X_norm, mu, sigma


def descenso_gradiente(X, Y, alpha=0.1, iter=1500):
    # Inicialización de los valores de theta a 0, con cada iteración su valor irá cambiando y siempre para mejor, en el caso de que vaya a peor es que esta mal hecho
    n = X.shape[1]
    theta = np.zeros(n)

    m = X.shape[0]
    costes = []
    for i in range(iter):
        for j in range(n):
            theta[j] = theta[j] - alpha * (1/m) * sumatory(m, X, Y, j, theta)
            #theta[j] = theta[j] - alpha * (1 / m) * np.sum((h(np.array([X[:, 0], X[:, 1], X[:, 2]]), theta) - Y) * X[:, j], axis=1)
        costes.append(coste(X, Y, theta))
    return theta


def ecuacion_normal(X, Y):
    a = np.linalg.inv((np.dot(X.T, X)))
    b = np.dot(X.T, Y)
    return np.dot(a, b)


def h(X, Theta):
    return np.dot(X, Theta)


def main(file_name):
    file = carga_csv(file_name)
    X = np.delete(file, np.shape(file)[1]-1, axis=1)
    Y = file[:, file.shape[1]-1]
    X_norm, mu, sigma = norm_matrix(X)

    t_grad = descenso_gradiente(X_norm, Y)
    t_ecnormal = ecuacion_normal(X, Y)
    print("Thetha Grad:", t_grad)
    print("Thetha Ec.Normal:", t_ecnormal)

    x1 = (1650 - mu[0]) / sigma[0]
    x2 = (3 - mu[1]) / sigma[1]

    print("Test Gradiente", np.dot(np.array([1, x1, x2]), t_grad.T))
    print("Test Ecnormal", np.dot(np.array([1650, 3]), t_ecnormal.T))


main("ex1data2.csv")
