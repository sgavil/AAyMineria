from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf


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
    X = np.delete(data, data.shape[1] - 1, axis=1)  # (1200, 20)
    X = normalize_matrix(X)
    X = np.insert(X, 0, 1, axis=1)  # (1200, 21)
    Y = data[:, data.shape[1] - 1]  # (1200,)

    return X, Y

def main():
    train_data = load_data("../ProcessedDataSet/train.csv")
    X, Y = get_data_matrix(train_data)

    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03]

    for i in range(len(lambda_vec)):
        model = Sequential()
        model.add(Dense(units=4, input_shape=(21,), activation='sigmoid'))

        model.compile(optimizer=optimizers.Adam(lr=lambda_vec[i]), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Plot training & validation accuracy values
        history = model.fit(x=X, y=Y, validation_split=0.25, batch_size=16, verbose=1, epochs=100)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy with $\lambda$ = {}'.format(lambda_vec[i]))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss with $\lambda$ = {}'.format(lambda_vec[i]))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

main()