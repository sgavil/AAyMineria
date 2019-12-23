from colorama import init
from termcolor import colored

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

########################################################################
###############             FUNCIONES BASICAS          #################
########################################################################

""" Sigmoide """
def sigmoid(z):
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid


""" Calcula el coste de un determinado conjunto de ejemplos """
def f_cost(Theta, X, Y, reg):
    m = X.shape[0]
    # Calculo del coste sin el termino de regularizacion
    h = sigmoid(np.dot(X, Theta))
    term1 = np.dot(-Y.T, np.log(h))
    term2 = np.dot((1 - Y).T, np.log(1 - h))

    # Termino de regularizacion
    reg_term = (reg / (2 * m)) * np.sum(np.square(Theta[1:]))

    # Calculo del coste con termino de regularizacion
    cost = (np.sum(term1 - term2) / m) + reg_term

    print(term2, "---------->", cost)

    return cost


""" Calcula el gradiente de un determinado conjunto de ejemplos """
def f_gradient(Theta, X, Y, reg):
    m = X.shape[0]

    # Calculo del gradiente sin el termino de regularizacion
    h = sigmoid(np.dot(X, Theta))
    gradient = (1 / m) * np.dot(X.T, (h - Y))
    
    # Termino de regularizacion
    reg_term = (reg / m) * (Theta[1:])

    # Calculo del gradiente con termino de regularizacion
    gradient[1:] = gradient[1:] + reg_term

    return gradient


""" Devuelve el coste y el gradiente """
def f_opt(Theta, X, Y, reg):
    return f_cost(Theta, X, Y, reg), f_gradient(Theta, X, Y, reg)

########################################################################
#############   FUNCIONES USADAS PARA EL ENTRENAMIENTO   ###############
########################################################################

""" Calcula el Theta optimo """
def get_optimize_theta(X, Y, reg):
    initial_theta = np.zeros((X.shape[1], 1))

    optTheta = minimize(fun=f_opt, x0=initial_theta,
                            args=(X, Y, reg), method='SLSQP', jac=True,
                            options={'maxiter': 200})

    return optTheta.x


""" Selecciona el mejor termino de regularizacion de una tupla de posibles valores """
def lambda_term_selection(X, Y, X_val, Y_val):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))

    for i in range(len(lambda_vec)):
        reg = lambda_vec[i]

        Theta = get_optimize_theta(X, Y, reg)

        error_train[i] = f_opt(Theta, X, Y, 0)[0]
        error_val[i] = f_opt(Theta, X_val, Y_val, 0)[0]

    print('lambda\tTrain Error\tValidation Error\n')
    for i in range(len(lambda_vec)):
        print('{}\t{}\t{}\n'.format(
            lambda_vec[i], error_train[i], error_val[i]))

    print("Best lambda:", lambda_vec[np.argmin(error_val)])
    return lambda_vec[np.argmin(error_val)]


""" Calcula el error sobre el dataSet de test"""
def test_error(X, Y, X_test, Y_test, reg):
    Theta = get_optimize_theta(X, Y, reg)
    error_test = f_opt(Theta, X_test, Y_test, 0)[0]

    print("Test error for the best lambda: {0:.4f}".format(error_test))


""" Entrena los clasificadores de cada clase """
def oneVsAll(X, Y, num_of_price_range, reg):
    # Numero de propiedades de los ejemplos
    n = X.shape[1]

    matResult = np.zeros((num_of_price_range, n)) # (4, 21)

    for i in range(num_of_price_range):    
        # Se obtiene una nueva "y" donde se indica si el ejemplo
        # j-esimo pertence a dicha clase o no.
        newY = np.array((Y == i) * 1)
        newY = newY[:None]

        matResult[i] = get_optimize_theta(X, newY, reg)

    return matResult


""" Calcula la precision """
def testClassificator(Theta, X, Y):
    aciertos = 0
    for m in range(X.shape[0]):  # Para cada ejemplo de entrenamiento
        bestClassificator = -1
        index = 0
        for j in range(Theta.shape[0]):  # Ponemos a prueba cada clasificador
            result = sigmoid(np.dot(Theta[j], X[m]))
            if(result > bestClassificator):
                bestClassificator = result
                index = j

        if(index == Y[m]):
            aciertos += 1
        
    precission = round((aciertos / X.shape[0]) * 100, 1)
    print(colored('Precision: {}%'.format(precission), 'green'))

    return precission


init()
warnings.filterwarnings("ignore")