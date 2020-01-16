import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
import neural_network

'''

3 capas:
    + 20 en la primera capa (la primera siempre fijada +1)
    + 8 en la capa oculta
    + 4 en la de salida

Theta1 de dimension (8 x 21)
Theta2 de dimension (4 x 9)

'''

def draw_precission(precission):
    plt.figure(figsize=(14, 6))
    plt.title('Neural Network Precission with best $\lambda$ for each method')
    plt.xlabel('Algorithm method')
    plt.ylabel('Precission')

    plt.ylim(0, 100)

    x = np.arange(len(precission))
    rects = plt.bar(x, precission)
    plt.xticks(x, ('CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP'))
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, 
                    height / 2),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color=(0.0, 0.0, 0.0, 1.0),
                    fontsize=20, weight='bold')
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

    input_layer = X.shape[1]
    hidden_layer = 32
    output_layer = 4

    cg_precission = neural_network.training_neural_network(X, Y, X_val, Y_val, X_test, Y_test, input_layer, \
        hidden_layer, output_layer, 'CG', True)
    bfgs_precission = neural_network.training_neural_network(X, Y, X_val, Y_val, X_test, Y_test, input_layer, \
        hidden_layer, output_layer, 'BFGS', True)
    l_bfgs_b_precission = neural_network.training_neural_network(X, Y, X_val, Y_val, X_test, Y_test, input_layer, \
        hidden_layer, output_layer, 'L-BFGS-B', True)
    tnc_precission = neural_network.training_neural_network(X, Y, X_val, Y_val, X_test, Y_test, input_layer, \
        hidden_layer, output_layer, 'TNC', True)
    slsqp_precission = neural_network.training_neural_network(X, Y, X_val, Y_val, X_test, Y_test, input_layer, \
        hidden_layer, output_layer, 'SLSQP', True)

    precission = [cg_precission, bfgs_precission, l_bfgs_b_precission, tnc_precission, slsqp_precission]
    draw_precission(precission)

main()