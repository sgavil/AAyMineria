from pandas.io.parsers import read_csv
import csv
import numpy as np


def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores[1:]


def main():
    datos = carga_csv("train.csv")
    # Eliminamos la primera fila con los atributos del dataset

    datosLen = len(datos)  # 2001 casos de entrenamiento

    # Cogemos el 60% de los datos set de entrenamiento
    n_training = int(datosLen*0.6)

    training_set = datos[:n_training]

    # Por otro lado el 20% para el set de validacion
    n_validation = int(datosLen*0.2)

    validation_set = datos[n_training:n_training+n_validation]

    # Por ultimo el 20% restante para el conjunto de prueba
    n_test = datosLen - n_training - n_validation

    test_set = datos[-n_test:]

    with open('ProcessedDataSet/train.csv', mode='w', newline='') as processedTraining:
        processedTrainingWriter = csv.writer(
            processedTraining, delimiter=',')
        processedTrainingWriter.writerows(training_set)

    with open('ProcessedDataSet/validation.csv', mode='w', newline='') as processedValidation:
        processedValidationWriter = csv.writer(
            processedValidation, delimiter=',')
        processedValidationWriter.writerows(validation_set)

    with open('ProcessedDataSet/test.csv', mode='w', newline='') as processedTest:
        processedTestWriter = csv.writer(processedTest, delimiter=',')
        processedTestWriter.writerows(test_set)


main()
