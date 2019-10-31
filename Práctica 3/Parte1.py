import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat


def mostrar_numeros_ejemplo(X):
    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis("off")

    plt.show()

    return plt


def f_gradiente(Theta, X, Y, lam):
    m = len(X)
    tempTheta = np.r_[[0], Theta[1:]]
    return (((1 / m) * np.dot(X.T, sigmoide(np.dot(X, Theta)) - np.ravel(Y)))
            + ((lam / m) * tempTheta))


def f_coste(Theta, X, Y, lam):
    m = len(X)
    return (((-1 / m) * (np.dot(np.log(sigmoide(np.dot(X, Theta))).T, Y)
                         + np.dot(np.log(1 - sigmoide(np.dot(X, Theta))).T, (1 - Y))))
            + ((lam / (2 * m)) * np.sum(Theta**2, initial=1)))


def sigmoide(z):
    return 1 / (1 + np.exp(-z))  # z = theta.T * x


def main():
    data = loadmat("ex3data1.mat")

    Y = data["y"]
    X = data["X"]

    # plt = mostrar_numeros_ejemplo(X)

    oneVsAll(X, Y, 10, 0.1)


def testClassificator(Theta, X, Y):
    aciertos = 0
    #resultadosCorrectos = np.zeros(X.shape[0])
    for m in range(X.shape[0]):  # Para cada ejemplo de entrenamiento
        mejorClassificator = -1
        index = 0
        for j in range(Theta.shape[0]):  # Ponemos a prueba cada clasificador
            res = sigmoide(np.dot(Theta[j], X[m]))
            if(res > mejorClassificator):
                mejorClassificator = res
                index = j+1
        if(index == Y[m]):
            aciertos += 1

        #resultadosCorrectos[m] = (index == Y[m])

    print("Porcentaje:", aciertos / X.shape[0])


def oneVsAll(X, y, num_etiquetas, reg):

    X = np.hstack((np.ones((X.shape[0], 1)), X))  # (5000,401)

    matResult = np.zeros((num_etiquetas, X.shape[1]))  # ()
    Theta = np.zeros(X.shape[1])  # (401,)
    m = X.shape[1]

    for n in range(9):
        nuevaY = np.array((y == n+1)*1)
        result = opt.fmin_tnc(func=f_coste, x0=Theta,
                              fprime=f_gradiente, args=(X, nuevaY, reg))

        matResult[n] = result[0]  # (10,401)

    nuevaY = np.array((y == 10)*1)
    result = opt.fmin_tnc(func=f_coste, x0=Theta,
                          fprime=f_gradiente, args=(X, nuevaY, reg))
    matResult[9] = result[0]  # (10,401)

    resCorrectos = testClassificator(matResult, X, y)


main()
