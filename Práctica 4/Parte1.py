import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat


def displayData(X):
    num_plots = int(np.size(X, 0)**.5)
    fig, ax = plt.subplots(num_plots, num_plots, sharex=True, sharey=True)
    plt.subplots_adjust(left=0, wspace=0, hspace=0)
    img_num = 0
    for i in range(num_plots):
        for j in range(num_plots):
            # Convert column vector into 20x20 pixel matrix
            # transpose
            img = X[img_num, :].reshape(20, 20).T
            ax[i][j].imshow(img, cmap='Greys')
            ax[i][j].set_axis_off()
            img_num += 1

    plt.show()
    return (fig, ax)


def displayImage(im):
    fig2, ax2 = plt.subplots()
    image = im.reshape(20, 20).T
    ax2.imshow(image, cmap='gray')
    return (fig2, ax2)


# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Cálculo del coste no regularizado
def coste_no_reg(m, K, h, Y):
    return ((1 / m) * np.sum((-Y * np.log(h)) - 
    ((1 - Y) * np.log(1 - h)), initial=1))


# Cálculo del coste regularizado
def coste_reg(m, K, h, Y, reg, theta1, theta2):
    return (coste_no_reg(m, K, h, Y) +
            ((reg / (2 * m)) * (np.sum(theta1, initial=1) ** 2 + 
            np.sum(theta2, initial=1) ** 2)))


def forward_propagate(X, theta1, theta2):
    m = X.shape[0] 

    a1 = np.hstack([np.ones([m, 1]), X])    # (5000, 401)
    z2 = np.dot(a1, theta1.T)   # (5000, 25)

    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])  # (5000, 26)
    z3 = np.dot(a2, theta2.T)   # (5000, 10)

    h = sigmoid(z3) # (5000, 10)

    return a1, z2, a2, z3, h


# Devuelve el coste
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
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
    
    coste = coste_no_reg(X.shape[0], num_etiquetas, h, Y)
    costeReg = coste_reg(X.shape[0], num_etiquetas, h, Y, reg, theta1, theta2)
    print(coste)
    print(costeReg)   


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
        
    weights = loadmat("ex4weights.mat")
    theta1, theta2 = weights["Theta1"], weights["Theta2"]
    # Theta1 es de dimensión 25 x 401
    # Theta2 es de dimensión 10 x 26

    thetaVec = np.concatenate((np.ravel(theta1), np.ravel(theta2)))

    backprop(thetaVec, X.shape[1], num_ocultas, num_etiquetas, X, Y, 0.1)


main()
