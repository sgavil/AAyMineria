import numpy as np
from pandas.io.parsers import read_csv
import svm as ourSVM


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
    X = np.delete(data, data.shape[1] - 1, axis=1)
    X = normalize_matrix(X)
    X = np.insert(X, 0, 1, axis=1)
    Y = data[:, data.shape[1] - 1]

    return X, Y


def main():
    train_data = load_data("../ProcessedDataSet/train.csv")
    validation_data = load_data("../ProcessedDataSet/validation.csv")
    test_data = load_data("../ProcessedDataSet/test.csv")

    X, Y = get_data_matrix(train_data)
    X_val, Y_val = get_data_matrix(validation_data)
    X_test, Y_test = get_data_matrix(test_data)

    ourSVM.trainSVMForLinear(X, Y, X_test, Y_test, Y_val, X_val)


main()
