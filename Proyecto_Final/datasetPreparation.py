from pandas.io.parsers import read_csv
import csv
import numpy as np


def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores


def main():
    trainingSet = carga_csv("train.csv")
    trainingSetLen = len(trainingSet)  # 2001 casos de entrenamiento

    # Utilizamos el 30% de los ejemplos de entrenamiento como conjunto de validación
    n_cv_set = int(trainingSetLen*0.3)
    n_training_set = int(trainingSetLen*0.7)

    processedTrainingSet = trainingSet[:n_training_set]
    processedValidationSet = trainingSet[-n_cv_set:]

    # Tamaño del set de entrenamiento a utilizar: 1400 casos
    print(len(processedTrainingSet))
    # Tamaño del set de validación a utilizar: 600 casos
    print(len(processedValidationSet))

    with open('ProcessedDataSet/train.csv', mode='w', newline='') as processedTraining:
        processedTrainingWriter = csv.writer(
            processedTraining, delimiter=',')
        processedTrainingWriter.writerows(processedTrainingSet)

    with open('ProcessedDataSet/validation.csv', mode='w', newline='') as processedValidation:
        processedValidationWriter = csv.writer(
            processedValidation, delimiter=',')
        processedValidationWriter.writerows(processedValidationSet)


main()
