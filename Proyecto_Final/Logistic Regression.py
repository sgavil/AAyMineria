import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Carga un fichero ".csv" y devuelve los datos
def loadData(file_name):
    values = read_csv(file_name, header=None).values
    return values.astype(float)


########################################################################
################                                        ################
################      CALCULOS DE COSTE Y GRADIENTE     ################
################                                        ################
########################################################################

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def f_coste(Theta, X, Y, reg):
    m = len(X)
    Theta = Theta[:None]
    return ((1 / m) * np.sum(np.dot(-Y, np.log(sigmoid(np.dot(Theta, X.T)))) - \
        np.dot(1 - Y, np.log(1 - sigmoid(np.dot(Theta, X.T)))))) + \
        ((reg / (2 * m)) * np.sum(np.square(Theta[1:])))


def f_gradiente(Theta, X, Y, reg):
    m = len(X)
    tempTheta = np.r_[[0], Theta[1:]]
    Theta = Theta[:None]
    '''return ((1 / m) * np.sum(np.dot(sigmoid(np.dot(Theta, X.T)) - np.ravel(Y), X))) + \
        ((reg / m) * tempTheta)'''
    
    return ((1 / m) * np.dot(X.T, sigmoid(np.dot(Theta, X.T)) - np.ravel(Y))) + \
        ((reg / m) * tempTheta)



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

    print("Porcentaje:", (aciertos / X.shape[0]) * 100, "%")

def oneVsAll(X, y, num_of_price_range, reg_term):
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # (1200, 21)

    matResult = np.zeros((num_of_price_range, X.shape[1])) # (4, 21)
    Theta = np.zeros(X.shape[1]) # (21,)

    for n in range(num_of_price_range):
        # Se obtiene una nueva "y" donde se indica si el ejemplo
        # j-esimo pertence a dicha clase o no.
        newY = np.array((y == n) * 1)
        result = opt.fmin_tnc(func=f_coste, 
                              x0=Theta,
                              fprime=f_gradiente, 
                              args=(X, newY, reg_term))

        matResult[n] = result[0]

    testClassificator(matResult, X, y)

def main():
    train_data = loadData("ProcessedDataSet/train.csv")
    validation_data = loadData("ProcessedDataSet/validation.csv")
    test_data = loadData("ProcessedDataSet/test.csv")

    X = np.delete(train_data, train_data.shape[1] - 1, axis=1) # (1200, 20)
    Y = train_data[:, train_data.shape[1] - 1] # (1200,)

    X_val = np.delete(validation_data, validation_data.shape[1] - 1, axis=1)
    Y_val = validation_data[:, validation_data.shape[1] - 1]

    X_test = np.delete(test_data, test_data.shape[1] - 1, axis=1)
    Y_test = test_data[:, test_data.shape[1] - 1]

    oneVsAll(X, Y, 4, 0.1)


main()
