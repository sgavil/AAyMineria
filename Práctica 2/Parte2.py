import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

# Carga el fichero csv especificado y lo devuelve en un array de numpy


def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


def dibuja_grafica(X, Y):
    admitted = np.where(Y == 1)
    notAdmitted = np.where(Y == 0)

    plt.scatter(X[admitted, 0], X[admitted, 1],
                marker='+', c='k', label='y = 1')
    plt.scatter(X[notAdmitted, 0], X[notAdmitted, 1], marker='o',
                c='yellowgreen', label='y = 0')

    plt.legend()

    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')

    return plt


def dibuja_h(Theta, X, Y, plt, poly):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))

    h = sigmoide(poly.fit_transform(
        np.c_[xx1.ravel(), xx2.ravel()]).dot(Theta))
    h = h.reshape(xx1.shape)

    #  el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.show()
    plt.close()


def f_gradiente(Theta, X, Y, lam):
    m = len(X)
    tempTheta = np.r_[[0], Theta[1:]]
    return (((1 / m) * np.dot(X.T, sigmoide(np.dot(X, Theta)) - Y))
            + ((lam / m) * tempTheta))


def f_coste(Theta, X, Y, lam):
    m = len(X)
    return (((-1 / m) * (np.dot(np.log(sigmoide(np.dot(X, Theta))).T, Y)
                         + np.dot(np.log(1 - sigmoide(np.dot(X, Theta))).T, (1 - Y))))
            + ((lam / (2 * m)) * np.sum(Theta**2, initial=1)))


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def regresion_logistica_regularizada(X, Y, Theta, lam):
    poly = PolynomialFeatures(6)
    X_poly = poly.fit_transform(X)

    grad = f_gradiente(Theta, X_poly, Y, lam)
    coste = f_coste(Theta, X_poly, Y, lam)

    result = opt.fmin_tnc(func=f_coste, x0=Theta,
                          fprime=f_gradiente, args=(X_poly, Y, lam))
    Theta_Opt = result[0]
    return poly, Theta_Opt


def main():
    datos = carga_csv("ex2data2.csv")
    X = np.delete(datos, np.shape(datos)[1]-1, axis=1)
    Y = datos[:, datos.shape[1]-1]

    Theta = np.zeros(28)
    lam = 1

    plt = dibuja_grafica(X, Y)
    poly, Theta_Opt = regresion_logistica_regularizada(X, Y, Theta, lam)

    dibuja_h(Theta_Opt, X, Y, plt, poly)


main()
