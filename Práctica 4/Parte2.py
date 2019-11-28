import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from checkNNGradients import checkNNGradients


# Cáculo del coste no regularizado
def coste_no_reg(m, h, y):
    J = 0
    for i in range(m):
        J += np.sum(-y[i] * np.log(h[i]) - (1-y[i]) * np.log(1-h[i]))
    return (J / m)


# Cálculo del coste regularizado
def coste_reg(m, h, Y, reg, theta1, theta2):
    return (coste_no_reg(m, h, Y) + 
        ((reg / (2 * m)) * 
        (np.sum(theta1[1:] ** 2) + 
        np.sum(theta2[1:] ** 2))))


# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Cálculo de la derivada de la función sigmoide
def der_sigmoid(z):
    return (z * (1.0 - z))


# Inicializa una matriz de pesos aleatorios
def pesosAleatorios(L_in, L_out):
    ini = 0.12
    theta = np.random.uniform(low=-ini, high=ini, size=(L_out, L_in))

    theta = np.hstack((np.ones((theta.shape[0], 1)), theta))

    return theta


# Devuelve "Y" a partir de una X y no unos pesos determinados
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
    m = X.shape[0]

    # Despliegue de params_rn para sacar las Thetas
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)],
            (num_ocultas, (num_entradas + 1)))

    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1): ], 
        (num_etiquetas, (num_ocultas + 1)))

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)  

    coste = coste_no_reg(m, h, y) # Coste sin regularizar
    print(coste)
    costeReg = coste_reg(m, h, y, reg, theta1, theta2) # Coste regularizado
    print(costeReg)

    # Inicialización de dos matrices "delta" a 0 con el tamaño de los thethas respectivos
    delta1 = np.zeros_like(theta1)
    delta2 = np.zeros_like(theta2)

    # Por cada ejemplo
    for t in range(m):
        a1t = a1[t, :] # (1, 401)
        a2t = a2[t, :] # (1, 26)
        ht = h[t, :] # (1, 10)
        yt = y[t]

        d3t = ht - yt
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) # (1, 26)

        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    delta1 = delta1 / m
    delta2 = delta2 / m

    # Gradiente perteneciente a cada delta
    delta1[:, 1:] = delta1[:, 1:] + (reg * theta1[:, 1:]) / m
    delta2[:, 1:] = delta2[:, 1:] + (reg * theta2[:, 1:]) / m
    
    # Concatenación de los gradientes
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return costeReg, grad
    


def main():
    data = loadmat("ex4data1.mat")

    y = data["y"].ravel()
    X = data["X"]

    num_entradas = X.shape[1]
    num_ocultas = 25
    num_etiquetas = 10

    # Transforma Y en una matriz de vectores, donde cada vector está formado por todo 
    # 0s excepto el valor marcado en Y, que se pone a 1
    # 3 ---> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    lenY = len(y)
    y = (y - 1)
    y_onehot = np.zeros((lenY, num_etiquetas))
    for i in range(lenY):
        y_onehot[i][y[i]] = 1

    # Inicialización de dos matrices de pesos de manera aleatoria
    #theta1 = pesosAleatorios(400, 25) # (25, 401)
    #theta2 = pesosAleatorios(25, 10) # (10, 26)

    # Lectura de los pesos del archivo
    weights = loadmat("ex4weights.mat")
    theta1 = weights["Theta1"] # (25, 401)
    theta2 = weights["Theta2"] # (10, 26)

    # Concatenación de las matrices de pesos en un solo vector
    thetaVec = np.concatenate((np.ravel(theta1), np.ravel(theta2)))

    # Chequeo del gradiente
    checkNNGradients(backprop, 0.1)
    #backprop(thetaVec, X.shape[1], num_ocultas, num_etiquetas, X, y_onehot, 0.1)


main()
