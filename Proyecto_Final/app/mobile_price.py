import numpy as np
from pandas.io.parsers import read_csv
from scipy.io import loadmat

def load_data(file_name):
    values = read_csv(file_name, header=None).values
    return values.astype(float)

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def normalize_matrix(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm / sigma

    return X_norm


def get_data_matrix(data, X_toAdd):
    X = np.delete(data, data.shape[1] - 1, axis=1) # (1200, 20)
    X = np.vstack((X, X_toAdd))
    X = normalize_matrix(X)

    return X

# Cálculo de la precisión
def testClassificator(h, Y):
    aciertos = 0
    for i in range (h.shape[0]):
        max = np.argmax(h[i])

        if max == Y[i]:
            aciertos += 1

    precision = round((aciertos / h.shape[0]) * 100, 1)
    return precision

# Devuelve "Y" a partir de una X y no unos pesos determinados
def forward_propagate(X, theta1, theta2):
    m = X.shape[0] 

    a1 = np.hstack([np.ones([m, 1]), X])    # (5000, 401)
    z2 = np.dot(a1, theta1.T)   # (5000, 25)

    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])  # (5000, 26)
    z3 = np.dot(a2, theta2.T)   # (5000, 10)

    h = sigmoid(z3) # (5000, 10)

    return a1, z2, a2, z3, h

def calculate_range_price():

    values = read_csv("../ProcessedDataSet/user.csv", header=None).values
    X_user = values.astype(float)

    data = load_data("../ProcessedDataSet/test.csv")
    X = get_data_matrix(data, X_user)

    weights = loadmat("weights.mat")
    theta1 = weights["Theta1"] # (8 x 21)
    theta2 = weights["Theta2"] # (4 x 9)

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    res = np.argmax(h[h.shape[0] - 1])
    return res