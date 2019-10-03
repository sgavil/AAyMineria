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

def norm_matrix(X):
    fils = len(X)
    cols = len(X[0])

    X_norm = np.zeros((fils, cols))

    mu = np.zeros(cols)
    sigma = np.zeros(cols)

    for i in range(cols):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])

    for i in range(cols):
        X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]

    return X_norm, mu, sigma


def descenso_gradiente(X, alpha=0.01, iter=1500):
    # Inicialización de los valores de theta a 0, con cada iteración su valor irá cambiando y siempre para mejor, en el caso de que vaya a peor es que esta mal hecho
    theta = np.zeros(len(X[0]))

    # Inizialización de un array que guarda el historial de los costes
    costeArray = np.zeros(iter)

    m = len(X)

    for i in range(iter):
        temp0 = theta[0] - alpha * (1 / m) * np.sum(h(X, Theta) - casos[:,1], axis=0)
        temp1 = theta[1] - alpha * (1 / m) * np.sum((h(X[:,0], theta) - casos[:,1]) * casos[:, 0], axis=0) 
        theta[0] = temp0
        theta[1] = temp1

        funH = h(casos[:,0], theta)
        costeArray[i] = (1 / (2 * m)) * np.sum(np.square(h(casos[:,0], theta) - casos[:,1]), axis=0)
        plt.clf()
        print(costeArray[i])

    #fig, subPlot = plt.subplots(1, 2, figsize=(12, 5))

    #dibuja_grafica(subPlot[0], casos, funH, theta)
    #dibuja_costes(subPlot[1], iter, costeArray)
    #plt.show()

def h(X, Theta):
   return np.dot(X, Theta)


def main(file_name):
    X = carga_csv(file_name)
    X_norm, mu, sigma = norm_matrix(X)
    #descenso_gradiente(casos=a)


main("ex1data2.csv")