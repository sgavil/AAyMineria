import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures

# DIBUJADO
def dibuja_grafica(Theta, X, y, reg, axs, plotCol):
    axs[0, plotCol].scatter(X, y, marker='x', c='red', linewidths=0.5, s = 100)

    axs[0, plotCol].title.set_text("Polinomial regression " r'$(\lambda = {})$'.format(reg))
    axs[0, plotCol].set_xlabel('Change in water level (x)')
    axs[0, plotCol].set_ylabel('Water flowing out of the dam (y)')

def dibuja_polynomial_regression(Theta, X, mu, sigma, p, axs, plotCol):
    x = np.array(np.arange(min(X) - 5, max(X) + 5, 0.05))

    X_poly = polinomial_matrix(x, p)
    X_poly = (X_poly - mu) / sigma

    X_poly = np.insert(X_poly, 0, 1, axis=1)

    axs[0, plotCol].plot(x, np.dot(X_poly, Theta))


def dibuja_learning_curve(error, error_val, reg, axs, plotCol):
    m = len(error)

    axs[1, plotCol].plot(range(1, m + 1), error, label='Train')
    axs[1, plotCol].plot(range(1, m + 1), error_val, label='Cross Validation')

    axs[1, plotCol].title.set_text("Learning curve for linear regression " + r'$(\lambda = {})$'.format(reg))
    axs[1, plotCol].set_xlabel('Number of training examples')
    axs[1, plotCol].set_ylabel('Error')

    axs[1, plotCol].legend()

def dibuja_lambda_selection(lambda_vec, error, error_val):
    plt.figure(figsize=(8, 6))
    plt.xlabel('$\lambda$')
    plt.ylabel('Error')
    plt.title('Selecting $\lambda$ using a cross validation set')
    plt.plot(lambda_vec, error, 'b', label='Train')
    plt.plot(lambda_vec, error_val, 'g', label='Cross Validation')
    plt.legend()


# FUNCIONES
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


# NORMALIZACION DE MATRICES POLINOMICAS
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


# METODOS DE LA PRACTICA
def polynomial_regression(X, y, X_pol, mu, sigma, p, reg, axs, plotCol):      
    Theta = get_optimize_theta(X_pol, y, reg)

    dibuja_grafica(Theta, X, y, reg, axs, plotCol)
    dibuja_polynomial_regression(Theta, X, mu, sigma, p, axs, plotCol)


def learning_curve(X, y, Xval, yval, reg, axs, plotCol):
    m = len(X)

    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    for i in range(1, m + 1):
        Theta = get_optimize_theta(X[: i], y[: i], reg)

        error_train[i - 1] = f_optimizacion(Theta, X[: i], y[: i], 0)[0]
        error_val[i - 1] = f_optimizacion(Theta, Xval, yval, 0)[0]

    dibuja_learning_curve(error_train, error_val, reg, axs, plotCol)

def regression(reg, p, X, y, X_pol, mu, sigma, Xval_pol, yval, axs, plotCol):
    polynomial_regression(X, y, X_pol, mu, sigma, p, reg, axs, plotCol)
    learning_curve(X_pol, y, Xval_pol, yval, reg, axs, plotCol)

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


def main():
    data = loadmat("ex5data1.mat")

    y = data["y"]
    X = data["X"]

    yval = data["yval"]
    Xval = data["Xval"]

    ytest = data["ytest"]
    Xtest = data["Xtest"]
    
    p = 8

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    X_pol, Xval_pol, Xtest_pol, mu, sigma = get_polynomial_matrix(X, Xval, Xtest, p)

    regression(reg=0, p=p, X=X, y=y, X_pol=X_pol, mu=mu, sigma=sigma, Xval_pol=Xval_pol,
            yval=yval, axs=axs, plotCol=0)
    regression(reg=1, p=p, X=X, y=y, X_pol=X_pol, mu=mu, sigma=sigma, Xval_pol=Xval_pol,
            yval=yval, axs=axs, plotCol=1)
    regression(reg=100, p=p, X=X, y=y, X_pol=X_pol, mu=mu, sigma=sigma, Xval_pol=Xval_pol,
            yval=yval, axs=axs, plotCol=2)


    plt.show()

main()