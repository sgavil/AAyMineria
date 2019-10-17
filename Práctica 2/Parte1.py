import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Carga el fichero csv especificado y lo devuelve en un array de numpy


def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


def dibuja_grafica(X, Y):
    admitted = np.where(Y == 1)
    notAdmitted = np.where(Y == 0)

    plt.scatter(X[admitted, 0], X[admitted, 1],
                marker='+', c='k', label='Admitted')
    plt.scatter(X[notAdmitted, 0], X[notAdmitted, 1], marker='o',
                c='yellowgreen', label='Not Admitted')

    plt.legend()

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    return plt


def dibuja_h(Theta, X, Y, plt):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))

    h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)),
                       xx1.ravel(), xx2.ravel()].dot(Theta))
    h = h.reshape(xx1.shape)

    #  el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.show()
    plt.close()


def funcion_coste(Theta, X, Y):
    m = len(X)
    print(Theta.shape, X.shape, Y.shape)
    H = sigmoide(np.matmul(X, Theta))
    return ((-1/m) * (np.dot(np.log(sigmoide(np.dot(X, Theta))).T, Y) +
                      (np.dot(np.log(1 - sigmoide(np.dot(X, Theta))).T, 1 - Y))))


def funcion_gradiente(Theta, X, Y):
    m = X.shape[0]
    return (1 / m) * (np.dot(X.T, sigmoide(np.dot(X, Theta)) - Y))


def h(X, Theta):
    return (1 / (1 + np.exp(np.dot(X, -Theta.T))))


def sigmoide(z):
    return (1 / (1 + np.exp(-z)))  # z = theta.T * x


def regresion_logistica(Theta, X, Y):
    gradiente = funcion_gradiente(Theta, X, Y)
    coste = funcion_coste(Theta, X, Y)

    print("Función gradiente:", gradiente)
    print("Función coste:", coste)

    result = opt.fmin_tnc(func=funcion_coste, x0=Theta,
                          fprime=funcion_gradiente, args=(X, Y))
    Theta_Opt = result[0]

    return Theta_Opt


def main():
    datos = carga_csv("ex2data1.csv")
    X = np.delete(datos, np.shape(datos)[1]-1, axis=1)
    Y = datos[:, datos.shape[1]-1]

    plt = dibuja_grafica(X, Y)

    onesColumn = np.ones((X.shape[0], 1))
    X = np.hstack((onesColumn, X))

    Theta = np.zeros(X.shape[1])

    Theta = regresion_logistica(Theta, X, Y)
    dibuja_h(Theta, np.delete(X, 0, axis=1), Y, plt)


main()
