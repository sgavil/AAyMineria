import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import math
from scipy.io import savemat

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cálculo de la derivada de la función sigmoide
def der_sigmoid(z):
    return (sigmoid(z) * (1.0 - sigmoid(z)))

# Cáculo del coste no regularizado
def coste_no_reg(m, h, y):
    J = 0
    for i in range(m):
        J += np.sum(-y[i] * np.log(h[i]) \
             - (1 - y[i]) * np.log(1 - h[i]))
    return (J / m)


# Cálculo del coste regularizado
def f_cost(m, h, Y, reg, theta1, theta2):
    return (coste_no_reg(m, h, Y) + 
        ((reg / (2 * m)) * 
        (np.sum(np.square(theta1[:, 1:])) + 
        np.sum(np.square(theta2[:, 1:])))))


# Inicializa una matriz de pesos aleatorios
def random_weight(L_in, L_out):
    ini = 0.12
    theta = np.random.uniform(low=-ini, high=ini, size=(L_out, L_in))

    theta = np.hstack((np.ones((theta.shape[0], 1)), theta))

    return theta

def num_to_vector(n, output_layer):
    lenN = len(n)
    n = n.ravel()
    n_onehot = np.zeros((lenN, output_layer))
    for i in range(lenN):
        n_onehot[i][int(n[i])] = 1

    return n_onehot


""" Selecciona el mejor termino de regularizacion de una tupla de posibles valores """
def lambda_term_selection(nn_params, input_layer, hidden_layer , output_layer, X, Y_onehot, \
    X_val, Y_val_onehot, comp_method, use_jac):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))

    for i in range(len(lambda_vec)):
        reg = lambda_vec[i]

        Theta = get_optimize_theta(nn_params, input_layer, hidden_layer, \
        output_layer, X, Y_onehot, reg, comp_method, use_jac)

        # Despliegue de params_rn para sacar las Thetas
        theta1 = np.reshape(Theta.x[:hidden_layer * (input_layer + 1)],
                (hidden_layer, (input_layer + 1)))

        theta2 = np.reshape(Theta.x[hidden_layer * (input_layer + 1): ], 
            (output_layer, (hidden_layer + 1)))

        a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
        error_train[i] = f_cost(X.shape[0], h, Y_onehot, reg, theta1, theta2)

        a1, z2, a2, z3, h_val = forward_propagate(X_val, theta1, theta2) 
        error_val[i] = f_cost(X_val.shape[0], h_val, Y_val_onehot, reg, theta1, theta2)

    #draw_lambda_values(lambda_vec, error_train, error_val, method=comp_method)
    best_lambda = 0
    min_error = float("inf")

    for i in range(len(lambda_vec)):
        if not math.isnan(error_val[i]) and error_val[i] < min_error:
            min_error = error_val[i]
            best_lambda = lambda_vec[i]

    return best_lambda


def get_optimize_theta(nn_params, input_layer, hidden_layer , output_layer, X, Y_onehot, reg, comp_method, use_jac):
    initial_theta = np.zeros((X.shape[1], 1))

    # Obtención de los pesos óptimos entrenando una red con los pesos aleatorios
    if use_jac:
        optTheta = opt.minimize(
            fun=backprop,
            x0=nn_params, 
            args=(input_layer, hidden_layer, output_layer, X, Y_onehot, reg), 
            method=comp_method, 
            jac=True,
            options={'maxiter': 70})
    else:
        optTheta = opt.minimize(
            fun=backprop, 
            x0=nn_params, 
            args=(input_layer, hidden_layer, output_layer, X, Y_onehot, reg), 
            method=comp_method,
            options={'maxiter': 70})

    return optTheta

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
def backprop(params_rn, input_layer, hidden_layer, output_layer, X, y, reg):    
    m = X.shape[0]
    
    # Despliegue de params_rn para sacar las Thetas
    theta1 = np.reshape(params_rn[:hidden_layer * (input_layer + 1)],
            (hidden_layer, (input_layer + 1)))

    theta2 = np.reshape(params_rn[hidden_layer * (input_layer + 1): ], 
        (output_layer, (hidden_layer + 1)))

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)  

    coste = f_cost(m, h, y, reg, theta1, theta2) # Coste regularizado

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

    return coste, grad

# Cálculo de la precisión
def testClassificator(h, Y):
    aciertos = 0
    for i in range (h.shape[0]):
        max = np.argmax(h[i])

        if max == Y[i]:
            aciertos += 1

    precision = round((aciertos / h.shape[0]) * 100, 1)
    return precision

def training_neural_network(X, Y, X_val, Y_val, X_test, Y_test, input_layer, hidden_layer, output_layer, \
    comp_method, use_jac):
    # Transforma Y en un vector
    Y_onehot = num_to_vector(Y, output_layer)
    Y_val_onehot = num_to_vector(Y_val, output_layer)

    # Inicialización de dos matrices de pesos de manera aleatoria
    Theta1 = random_weight(input_layer, hidden_layer)
    Theta2 = random_weight(hidden_layer, output_layer)

    # Crea una lista de Thetas
    Thetas = [Theta1, Theta2]

    # Concatenación de las matrices de pesos en un solo vector
    unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]
    nn_params = np.concatenate(unrolled_Thetas)

    reg = lambda_term_selection(nn_params, input_layer, hidden_layer, output_layer, \
        X, Y_onehot, X_val, Y_val_onehot, comp_method, use_jac)

    optTheta = get_optimize_theta(nn_params, input_layer, hidden_layer, \
        output_layer, X, Y_onehot, reg, comp_method, use_jac)

    # Desglose de los pesos óptimos en dos matrices
    newTheta1 = np.reshape(optTheta.x[:hidden_layer * (input_layer + 1)],
        (hidden_layer, (input_layer + 1)))

    newTheta2 = np.reshape(optTheta.x[hidden_layer * (input_layer + 1): ], 
        (output_layer, (hidden_layer + 1)))

    #savemat("weights.mat", {'Theta1': newTheta1, 'Theta2': newTheta2})

    # H, resultado de la red al usar los pesos óptimos
    a1, z2, a2, z3, h = forward_propagate(X_test, newTheta1, newTheta2) 
    
    # Cálculo de la precisión
    return testClassificator(h, Y_test)

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
