import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Carga el fichero csv especificado y lo devuelve en un array de numpy
def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


def dibuja_grafica(Theta, X, Y):
    pointsX = np.linspace(26, 104, len(X), endpoint=True)

    admitted = np.where(Y == 1)
    notAdmitted = np.where(Y == 0)

    plt.scatter(X[admitted, 0], X[admitted, 1], marker='+', c='k', label='Admitted')
    plt.scatter(X[notAdmitted, 0], X[notAdmitted, 1], marker='o', c='yellowgreen', label='Not Admitted')

    funH = sigmoide(np.dot(X, Theta))
    #funH = funH.reshape(X[:, 0].shape)
    plt.contour(X[:, 0], X[:, 1], funH, [0.5], linewidth=1, c='b')
    plt.legend()

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    plt.show()

def funcion_coste(Theta, X, Y):
    m = len(X)
    return ((-1/m) * (np.dot(np.log(sigmoide(np.dot(X, Theta))).T, Y) + 
    (np.dot(np.log(1 - sigmoide(np.dot(X, Theta))).T, 1- Y))))


def funcion_gradiente(Theta, X, Y):
    m = len(X)
    return (1 / m) * (np.dot(X.T, sigmoide(np.dot(X, Theta)) - Y))


def h(X, Theta):
    return (1 / (1 + np.exp(np.dot(X, -Theta.T))))


def sigmoide(z):
    return (1 / (1 + np.exp(-z)))

def regresion_logistica(Theta, X, Y):
    gradiente = funcion_gradiente(Theta, X, Y)
    coste = funcion_coste(Theta, X, Y)

    print ("Función gradiente:", gradiente)
    print ("Función coste:", coste)
    
    result = opt.fmin_tnc(func=funcion_coste, x0=Theta, fprime=funcion_gradiente, args=(X, Y))
    Theta_Opt = result[0]

    return Theta_Opt


def main():
    datos = carga_csv("ex2data1.csv")
    X = np.delete(datos, np.shape(datos)[1]-1, axis=1)
    Y = datos[:, datos.shape[1]-1]
    Theta = np.zeros(X.shape[1])

    Theta = regresion_logistica(Theta, X, Y)
    dibuja_grafica(Theta, X, Y)



main()