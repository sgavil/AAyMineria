from colorama import init
from termcolor import colored

import warnings
import numpy as np
from pandas.io.parsers import read_csv
import OneVsAll

'''
    Theta: (n + 1, 1)
    X: (m, n + 1)
    Y: (m, 1)
'''

# Carga un fichero ".csv" y devuelve los datos
def loadData(file_name):
    values = read_csv(file_name, header=None).values
    return values.astype(float)

def normalize_matrix(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm / sigma

    return X_norm

def get_data_matrix(data):
    X = np.delete(data, data.shape[1] - 1, axis=1) # (1200, 20)
    X = normalize_matrix(X)
    X = np.insert(X, 0, 1, axis=1) # (1200, 21)
    Y = data[:, data.shape[1] - 1] # (1200,)

    return X, Y
    

def main():
    train_data = loadData("../ProcessedDataSet/train.csv")
    validation_data = loadData("../ProcessedDataSet/validation.csv")
    test_data = loadData("../ProcessedDataSet/test.csv")

    X, Y = get_data_matrix(train_data)
    X_val, Y_val = get_data_matrix(validation_data)
    X_test, Y_test = get_data_matrix(test_data)

    best_lambda = OneVsAll.lambda_term_selection(X, Y, X_val, Y_val)

    OneVsAll.test_error(X, Y, X_test, Y_test, best_lambda)

    tnc_theta = OneVsAll.oneVsAll(X, Y, 4, best_lambda)
    OneVsAll.testClassificator(tnc_theta, X_test, Y_test)


init()
warnings.filterwarnings("ignore")
main()
