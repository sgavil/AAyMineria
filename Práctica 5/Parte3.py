import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures

########################################################################
############################                ############################
############################    DIBUJADO    ############################
############################                ############################
########################################################################

def dibuja_grafica_inicial(Theta, X, y):
    xx = np.linspace(np.amin(X), np.amax(X))
    plt.scatter(X, y, marker='x', c='red', s=100, linewidths=0.5)

    xx = xx[:, None]
    xx_ones = np.hstack((np.ones((xx.shape[0], 1)), xx))
    plt.plot(xx, h(xx_ones, Theta[:, None]))

    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')


def dibuja_learning_curve(error_train, error_val, reg, axs):
    m = len(error_train)

    axs[1].plot(range(1, m + 1), error_train, label='Train')
    axs[1].plot(range(1, m + 1), error_val, label='Cross Validation')

    axs[1].title.set_text("Learning curve for linear regression " + r'$(\lambda = {})$'.format(reg))
    axs[1].set_xlabel('Number of training examples')
    axs[1].set_ylabel('Error')

    axs[1].legend()


def dibuja_polynomial_regression(Theta, X, y, mu, sigma, reg, p, axs):
    axs[0].scatter(X, y, marker='x', c='red', linewidths=0.5, s = 100)

    axs[0].title.set_text("Polinomial regression " r'$(\lambda = {})$'.format(reg))
    axs[0].set_xlabel('Change in water level (x)')
    axs[0].set_ylabel('Water flowing out of the dam (y)')

    x = np.array(np.arange(min(X) - 5, max(X) + 5, 0.05))
    X_pol = polinomial_matrix(x, p)
    X_pol = (X_pol - mu) / sigma
    X_pol = np.insert(X_pol, 0, 1, axis=1)
    axs[0].plot(x, np.dot(X_pol, Theta))



########################################################################
################                                        ################
################  CALCULOS DE COSTE, GRADIENTE Y THETA  ################
################                                        ################
########################################################################

def h(X, Theta):
    return np.dot(X, Theta)


def f_coste(Theta, X, y, reg):
    m = len(y)
    return (1 / (2 * m)) * (np.sum((h(X, Theta[:, None]) - y) ** 2)) \
        + (reg / (2 * m)) * (np.sum(Theta[1:] ** 2))


def f_gradiente(Theta, X, y, reg):
    m = len(y)
    return (1 / m) * (np.sum(np.dot((h(X, Theta[:, None]) - y).T, X), axis=0)) \
        + (reg / m) * Theta


def f_optimizacion(Theta, X, y, reg):
    return f_coste(Theta, X, y, reg), f_gradiente(Theta, X, y, reg)


def get_optimize_theta(X, y, reg):
    initial_theta = np.zeros((X.shape[1], 1))

    optTheta = opt.minimize(fun=f_optimizacion, x0=initial_theta, 
            args=(X, y, reg), method='TNC', jac=True,
            options={'maxiter': 200})

    return optTheta.x



########################################################################
###############                                        #################
###############  NORMALIZACION DE MATRICES POLINOMICAS #################
###############                                        #################
########################################################################

def get_polynomial_matrix(X, Xval, Xtest, p):
    # X
    X_pol = polinomial_matrix(X, p)
    X_pol, mu, sigma = normalize_matrix(X_pol)
    X_pol = np.hstack((np.ones((X_pol.shape[0], 1)), X_pol))

    # Xval
    Xval_pol = polinomial_matrix(Xval, p)
    Xval_pol = (Xval_pol - mu) / sigma
    Xval_pol = np.hstack((np.ones((Xval_pol.shape[0], 1)), Xval_pol))

    # Xtest
    Xtest_pol = polinomial_matrix(Xtest, p)
    Xtest_pol = (Xtest_pol - mu) / sigma
    Xtest_pol = np.hstack((np.ones((Xtest_pol.shape[0], 1)), Xtest_pol))

    return X_pol, Xval_pol, Xtest_pol, mu, sigma


def polinomial_matrix(X, p):
    X_poly = X

    for i in range(1, p):
        X_poly = np.column_stack((X_poly, np.power(X, i+1)))   
    
    return X_poly


def normalize_matrix(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm / sigma
    
    return X_norm, mu, sigma



########################################################################
######################                            ######################
######################  APARTADOS DE LA PRACTICA  ######################
######################                            ######################
########################################################################

def polynomial_regression(X, y, X_pol, mu, sigma, reg, p, axs):      
    Theta = get_optimize_theta(X_pol, y, reg)
    dibuja_polynomial_regression(Theta, X, y, mu, sigma, reg, p, axs)


def learning_curve(X, y, Xval, yval, reg, axs):
    m = len(X)

    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    for i in range(1, m + 1):
        Theta = get_optimize_theta(X[: i], y[: i], reg)

        error_train[i - 1] = f_optimizacion(Theta, X[: i], y[: i], 0)[0]
        error_val[i - 1] = f_optimizacion(Theta, Xval, yval, 0)[0]

    dibuja_learning_curve(error_train, error_val, reg, axs)


def regression(X, y, X_pol, Xval_pol, yval, mu, sigma, reg, p, axs):
    polynomial_regression(X, y, X_pol, mu, sigma, reg, p, axs)
    learning_curve(X_pol, y, Xval_pol, yval, reg, axs)
    
    

########################################################################
################################        ################################
################################  MAIN  ################################
################################        ################################
########################################################################

def main():
    data = loadmat("ex5data1.mat")

    y = data["y"]
    X = data["X"]

    yval = data["yval"]
    Xval = data["Xval"]

    ytest = data["ytest"]
    Xtest = data["Xtest"]
    
    p = 8

    X_pol, Xval_pol, Xtest_pol, mu, sigma = get_polynomial_matrix(X, Xval, Xtest, p)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    regression(X=X, y=y, X_pol=X_pol, Xval_pol=Xval_pol, yval=yval, mu=mu, sigma=sigma, reg=0, p=p, axs=axs)

    plt.show()

main()