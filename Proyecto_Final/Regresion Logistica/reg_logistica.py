import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


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
    h_theta = sigmoid(np.dot(X, Theta))

    # Calculo del coste sin el termino de regularizacion
    term1 = np.dot(-Y.T, np.log(h_theta))
    term2 = np.dot((1 - Y).T, np.log(1 - h_theta))

    # Calculo del termino de regularizacion
    reg_term = (reg / (2 * m)) * np.sum(np.square(Theta[1:]))

    # Calculo del coste
    cost = (np.sum(term1 - term2) / m) + reg_term

    return cost


""" Calcula el gradiente de un determinado conjunto de ejemplos """

def f_gradient(Theta, X, Y, reg):
    m = X.shape[0]
    h_theta = sigmoid(np.dot(X, Theta))

    # Calculo del gradiente sin el termino de regularizacion
    reg_term = (reg / m) * (Theta[1:])

    # Calculo del termino de regularizacion
    gradient = (1 / m) * np.dot(X.T, (h_theta - Y))

    # Calculo del gradiente
    gradient[1:] = gradient[1:] + reg_term

    return gradient


""" Devuelve el coste y el gradiente """

def f_opt(Theta, X, Y, reg):
    return f_cost(Theta, X, Y, reg), f_gradient(Theta, X, Y, reg)


########################################################################
#############   FUNCIONES USADAS PARA EL ENTRENAMIENTO   ###############
########################################################################

def num_to_vector(Y, n):
    newY = np.array((Y == n) * 1)
    newY = newY[:None]

    return newY

""" Calcula el Theta optimo """

def get_optimize_theta(X, Y, reg, comp_method, use_jac):
    initial_theta = np.zeros((X.shape[1], 1))

    if use_jac:
        optTheta = minimize(fun=f_cost, x0=initial_theta,
                            args=(X, Y, reg), method=comp_method, jac=f_gradient)
    else:
        optTheta = minimize(fun=f_cost, x0=initial_theta,
                            args=(X, Y, reg), method=comp_method)

    return optTheta.x


""" Selecciona el mejor termino de regularizacion de una tupla de posibles valores """

def lambda_term_selection(X, Y, X_val, Y_val, comp_method, use_jac):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3])

    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))

    for i in range(len(lambda_vec)):
        reg = lambda_vec[i]

        for n in range(4):
            newY = num_to_vector(Y, n)
            newY_val = num_to_vector(Y_val, n)

            Theta = get_optimize_theta(X, newY, reg, comp_method, use_jac)

            error_train[i] += f_opt(Theta, X, newY, reg)[0]
            error_val[i] += f_opt(Theta, X_val, newY_val, reg)[0]

    draw_lambda_values(lambda_vec, error_train, error_val, method=comp_method)
    best_lambda = 0
    min_error = float("inf")

    for i in range(len(lambda_vec)):
        if not math.isnan(error_val[i]) and error_val[i] < min_error:
            min_error = error_val[i]
            best_lambda = lambda_vec[i]

    return best_lambda


""" Entrena los clasificadores de cada clase """

def oneVsAll(X, Y, num_of_price_range, reg, comp_method, use_jac):
    # Numero de propiedades de los ejemplos
    n = X.shape[1]

    matResult = np.zeros((num_of_price_range, n))  # (4, 21)

    for i in range(num_of_price_range):
        # Se obtiene una nueva "y" donde se indica si el ejemplo
        # j-esimo pertence a dicha clase o no.
        newY = num_to_vector(Y, i)

        matResult[i] = get_optimize_theta(
            X, newY, reg, comp_method, use_jac).ravel()

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

    return precission


def draw_lambda_values(lambda_values, error_train, error_val, method):
    plt.figure(figsize=(8, 5))
    plt.plot(lambda_values, error_val, 'or--', label='Validation Set Error')
    plt.plot(lambda_values, error_train, 'bo--', label='Training Set Error')
    plt.xlabel('$\lambda$ value', fontsize=16)
    plt.ylabel('Classification Error [%]', fontsize=14)
    plt.title(f'Finding Best $\lambda$ value for method {method}', fontsize=18)
    plt.xscale('log')
    plt.legend()
    plt.show()


def logistic_regression(X, Y, X_val, Y_val, X_test, Y_test, method, jac):
    best_lambda = lambda_term_selection(X, Y, X_val, Y_val, method, jac)
    optTheta = oneVsAll(X, Y, 4, best_lambda, method, jac)
    print(f'método {method} terminado con éxito!')
    return testClassificator(optTheta, X_test, Y_test)

warnings.filterwarnings("ignore")
