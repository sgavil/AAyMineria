from colorama import init
from termcolor import colored

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

import warnings
import numpy as np
from pandas.io.parsers import read_csv
import reg_logistica

'''
    Theta: (n + 1, 1)
    X: (m, n + 1)
    Y: (m, 1)
'''

def draw_precission(precission):
    plt.figure(figsize=(14, 6))
    plt.title('Regularized Logistic Regression Precission')
    plt.xlabel('Algorithm method')
    plt.ylabel('Precission')

    plt.ylim(0, 100)

    x = np.arange(len(precission))
    rects = plt.bar(x, precission)
    plt.xticks(x, ('Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP'))
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.show()

# Carga un fichero ".csv" y devuelve los datos
def load_data(file_name):
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
    train_data = load_data("../ProcessedDataSet/train.csv")
    validation_data = load_data("../ProcessedDataSet/validation.csv")
    test_data = load_data("../ProcessedDataSet/test.csv")

    X, Y = get_data_matrix(train_data)
    X_val, Y_val = get_data_matrix(validation_data)
    X_test, Y_test = get_data_matrix(test_data)


    nelder_precission = reg_logistica.logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, 'Nelder-Mead', False)
    powell_precission = reg_logistica.logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, 'Powell', False)
    cg_precission = reg_logistica.logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, 'CG', True)
    bfgs_precission = reg_logistica.logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, 'BFGS', True)
    newton_precission = reg_logistica.logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, 'Newton-CG', True)
    l_bfgs_b_precission = reg_logistica.logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, 'L-BFGS-B', True)
    tnc_precission = reg_logistica.logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, 'TNC', True)
    cobyla_precission = reg_logistica.logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, 'COBYLA', False)
    slsqp_precission = reg_logistica.logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, 'SLSQP', True)

    precission = [nelder_precission, powell_precission, cg_precission, bfgs_precission, newton_precission, \
        l_bfgs_b_precission, tnc_precission, cobyla_precission, slsqp_precission]

    draw_precission(precission)



init()
warnings.filterwarnings("ignore")
main()
