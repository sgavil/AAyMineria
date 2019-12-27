import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.metrics import precision_score
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve


def trainSigmoidSVM(X_train, y_train, X_test, y_test):
    svclassifier = svm.SVC(kernel='sigmoid')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def trainGaussianSVM(X_train, y_train, X_test, y_test):
    svclassifier = svm.SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def trainPolynomialSVM(X_train, y_train, X_test, y_test):
    svclassifier = svm.SVC(kernel='poly', degree=8)
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def draw_C_values(C_test_values, error_train, error_val):
    plt.figure(figsize=(8, 5))
    plt.plot(C_test_values, error_val, 'or--', label='Validation Set Error')
    plt.plot(C_test_values, error_train, 'bo--', label='Training Set Error')
    plt.xlabel('$C$ Value', fontsize=16)
    plt.ylabel('Classification Error [%]', fontsize=14)
    plt.title('Finding Best C Value', fontsize=18)
    plt.legend()
    plt.show()


def findBetterCForLinear(Xtrain, ytrain, Xval, yval):
    C_test_values = [0.0001, 0.001, 0.01, 0.03, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                     5.0, 7.0, 10.0, 15.0, 18.0, 21.0, 25.0, 27.0, 30.0, 35.0, 37.0, 40.0]
    error_train = []
    error_val = []
    print('C\tTrain Error\tValidation Error\n')

    for testing_c in C_test_values:

        linear_svm = svm.SVC(C=testing_c, kernel='linear')

        # Ajustamos el kernel a los ejemplos de entrenamiento
        linear_svm.fit(Xtrain, ytrain)

        # calculamos el error con los ejemplos de validacion
        predictedValY = linear_svm.predict(Xval)
        validation_error = 100.0 * \
            float(sum(predictedValY != yval))/yval.shape[0]

        error_val.append(validation_error)

        # calculamos el error con los ejemplos de entrenamiento
        predictedTrainY = linear_svm.predict(Xtrain)
        train_error = 100.0 * \
            float(sum(predictedTrainY != ytrain))/ytrain.shape[0]
        error_train.append(train_error)

        print('{}\t{}\t{}\n'.format(testing_c, train_error, validation_error))

    draw_C_values(C_test_values, error_train, error_val)
    # De la grafica podemos observar que los mejores valores se encuentrar alrededor de utilizar una C con un
    # valor de  37.0


def trainSVMForLinear(X_train, y_train, X_test, y_test, yval, Xval):
    findBetterCForLinear(X_train, y_train, Xval, yval)
    svclassifier = svm.SVC(kernel='linear', C=37.0)
    svclassifier.fit(X_train, y_train)

    # Ahora que hemos encontrado un buen valor de C, lo comprobamos contra los ejemplos
    # de prueba
    y_pred = svclassifier.predict(X_test)
    linearSVMAccuracy = precision_score(y_test, y_pred, average='micro')*100.

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print(
        f"La precisión del kernel lineal con un valor C = 37.0 es del {linearSVMAccuracy}%")
