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

def f_coste(Theta, X, Y):
    m = len(X)

def main():
    data = loadmat("ex3data1.mat")

    Y = data["y"]
    X = data["X"]

    plt = mostrar_numeros_ejemplo(X)

main()

