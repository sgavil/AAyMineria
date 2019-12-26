
from pandas.io.parsers import read_csv
import numpy as np
import scipy.optimize as opt


def sigmoid(X):
    return np.power(1 + np.exp(-X), -1)


def h(X, theta):
    return sigmoid(np.dot(X, theta))


def loadData(file_name):
    values = read_csv(file_name, header=None).values
    return values.astype(float)


def f_cost(theta, X, y, lamda=None):
    m = X.shape[0]
    theta[0] = 0
    if lamda:
        return (-(1/m) * (np.dot(y.T, np.log(h(X, theta)))
                          + np.dot((1-y).T, np.log(1 - h(X, theta))))
                + (lamda/(2*m))*np.dot(theta.T, theta))
    return -(1/m) * (np.dot(y.T, np.log(h(X, theta))) +
                     np.dot((1-y).T, np.log(1 - h(X, theta))))


def f_gradient(theta, X, y, lamda=None):
    m = X.shape[0]
    if lamda:
        return (1/m) * np.dot(X.T, (h(X, theta) - y)) + (lamda/m) * theta
    return (1/m) * np.dot(X.T, (h(X, theta) - y))


def f_opt(theta, X, y, lamda=None):
    return f_cost(theta, X, y, lamda), f_gradient(theta, X, y, lamda)


def normalize_matrix(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm / sigma

    return X_norm, mu, sigma


def get_optimize_theta(X, y, lmbda):
    initial_theta = np.zeros((X.shape[1], 1))

    output = opt.fmin_tnc(func=f_cost, x0=initial_theta,
                          fprime=f_gradient, args=(X, y, lmbda))

    print(output[0])
    return output[0]


def lambda_term_selection(X, Y, X_val, Y_val):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))

    for i in range(len(lambda_vec)):
        reg = lambda_vec[i]

        Theta = get_optimize_theta(X, Y, reg)[0]

        trainPredictionValue = sigmoid(
            np.dot(X, Theta)).reshape((Y.shape[0], 1))
        trainError = 100. * \
            float(sum(trainPredictionValue != Y))/Y.shape[0]
        print(trainError)
        validationPredictionValue = h(X_val, Theta)

    return lambda_vec[np.argmin(error_val)]


def main():
    train_data = loadData("../ProcessedDataSet/train.csv")
    validation_data = loadData("../ProcessedDataSet/validation.csv")
    test_data = loadData("../ProcessedDataSet/test.csv")

    X = np.delete(train_data, train_data.shape[1] - 1, axis=1)  # (1200, 20)
    y = train_data[:, train_data.shape[1] - 1]  # (1200,)
    X, mu, sigma = normalize_matrix(X)
    X_ones = np.hstack((np.ones((X.shape[0], 1)), X))

    Xv = np.delete(validation_data,
                   validation_data.shape[1] - 1, axis=1)  # (1200, 20)
    yv = train_data[:, validation_data.shape[1] - 1]  # (1200,)
    XvNorm = (Xv - mu)/sigma
    XvOnes = np.hstack((np.ones((XvNorm.shape[0], 1)), XvNorm))

    best_lambda = lambda_term_selection(X_ones, y, XvOnes, yv)
    #grad = f_gradient(theta, X_ones, y, lmbda)
    #cost = f_cost(theta, X_ones, y, lmbda)


main()
