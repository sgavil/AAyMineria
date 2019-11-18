import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from checkNNGradients import checkNNGradients
from displayData import displayData


# Cáculo del coste no regularizado
def coste_no_reg(m, K, h, Y):
    return ((1 / m) * np.sum((-Y * np.log(h)) - 
    ((1 - Y) * np.log(1 - h)), initial=1))


# Cálculo del coste regularizado
def coste_reg(m, K, h, Y, reg, theta1, theta2):
    return (coste_no_reg(m, K, h, Y) +
            ((reg / (2 * m)) * (np.sum(theta1, initial=1) ** 2 + 
            np.sum(theta2, initial=1) ** 2)))


# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Cáculo de la derivada de la función sigmoide
def der_sigmoid(z):
    return (z * (1.0 - z))


# Inicializa una matriz de pesos aleatorios
def pesosAleatorios(L_in, L_out):
    ini = 0.12
    theta = np.random.uniform(low=-ini, high=ini, size=(L_out, L_in))

    theta = np.hstack((np.ones((theta.shape[0], 1)), theta))

    return theta



def forward_propagate(X, theta1, theta2):
    m = X.shape[0] 

    a1 = np.hstack([np.ones([m, 1]), X])    # (5000, 401)
    z2 = np.dot(a1, theta1.T)   # (5000, 25)

    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])  # (5000, 26)
    z3 = np.dot(a2, theta2.T)   # (5000, 10)

    h = sigmoid(z3) # (5000, 10)

    return a1, z2, a2, z3, h


# Devuelve el coste y el gradiente de una red neuronal de dos capas
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    
    # Despliegue de params_rn para sacar las Thetas
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)],
            (num_ocultas, (num_entradas + 1)))

    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1): ], 
        (num_etiquetas, (num_ocultas + 1)))

    # Thetha 3 es el resultado de la h.

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
        
    Y = np.zeros(h.shape)
    for i in range (h.shape[0]):
        max = np.argmax(h[i])
        Y[i, max] = 1
    

    costeReg = coste_reg(X.shape[0], num_etiquetas, h, Y, reg, theta1, theta2)

    d3 = h - Y

    d2 = np.dot(d3, theta2) * der_sigmoid(a2)
    d2 = np.delete(d2, 0, axis=1)

    delta1 = np.zeros_like(theta1)
    delta2 = np.zeros_like(theta2)

    delta1 = delta1 + np.dot(d2.T, a1)
    delta2 = delta2 + np.dot(d3.T, a2)

    grad1 = (1 / X.shape[0]) * delta1
    grad2 = (1 / X.shape[0]) * delta2

    grad = np.concatenate((np.ravel(grad1), np.ravel(grad2)))

    return costeReg, grad
    


def main():
    data = loadmat("ex4data1.mat")

    num_ocultas = 25
    num_etiquetas = 10

    Y = data["y"]
    X = data["X"]

    X_show = np.zeros((100, X.shape[1]))
    for i in range(100):
        random = np.random.randint(low=0, high=X.shape[0])
        X_show[i] = X[random]
        
    #displayData(X_show)
    #plt.show()

    #weights = loadmat("ex4weights.mat")
    #theta1, theta2 = weights["Theta1"], weights["Theta2"]
    theta1 = pesosAleatorios(400, 25)
    theta2 = pesosAleatorios(25, 10)

    # Theta1 es de dimensión 25 x 401
    # Theta2 es de dimensión 10 x 26

    thetaVec = np.concatenate((np.ravel(theta1), np.ravel(theta2)))


    checkNNGradients(backprop, 0.1)
    #backprop(thetaVec, X.shape[1], num_ocultas, num_etiquetas, X, Y, 0.1)


main()
