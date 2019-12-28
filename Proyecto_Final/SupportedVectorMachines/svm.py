import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.metrics import precision_score
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif


def getSampleError(Xval, yval, Xtrain, ytrain, classificator, testingValue):
    '''
    Funcion para calcular el error dados unos ejemplos de entrenamiento, ejemplos de validadion, un clasificador 
    y unos valores a probar
    '''

    # calculamos el error con los ejemplos de validacion
    predictedValY = classificator.predict(Xval)
    validation_error = 100.0 * \
        float(sum(predictedValY != yval))/yval.shape[0]

    # calculamos el error con los ejemplos de entrenamiento
    predictedTrainY = classificator.predict(Xtrain)
    train_error = 100.0 * \
        float(sum(predictedTrainY != ytrain))/ytrain.shape[0]

   # print('{}\t{}\t{}\n'.format(testingValue, train_error, validation_error))

    return train_error, validation_error


def trainSigmoidSVM(X_train, y_train, Xval, yval, X_test, y_test):
     # Despues de haber calculado los errores con distintos grados, hemos visto que el resultado variaba en funcion
    # del valor de C, y no el grado, asi que calculamos directamente el mejor C

    C_test_values = [0.0001, 0.001, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0,
                     5.0, 7.0, 10.0, 15.0, 18.0, 21.0, 25.0, 27.0, 30.0, 35.0, 37.0, 40.0]

    error_train = []
    error_val = []

    for testing_c in C_test_values:

        svclassifier = svm.SVC(
            kernel='sigmoid', C=testing_c)
        svclassifier.fit(X_train, y_train)

        train_error, val_error = getSampleError(
            Xval, yval, X_train, y_train, svclassifier, testing_c)

        error_train.append(train_error)
        error_val.append(val_error)

        print(
            f'C{testing_c}:\t {train_error}\t {val_error}')

    # Vemos que el mejor valor es con C = 0.6


def trainGaussianSVM(X_train, y_train, Xval, yval, X_test, y_test):
    # Despues de haber calculado los errores con distintos grados, hemos visto que el resultado variaba en funcion
    # del valor de C, y no el grado, asi que calculamos directamente el mejor C

    C_test_values = [0.0001, 0.001, 0.01, 0.03, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                     5.0, 7.0, 10.0, 15.0, 18.0, 21.0, 25.0, 27.0, 30.0, 35.0, 37.0, 40.0]

    error_train = []
    error_val = []

    print('degree and C \tTrain Error\tValidation Error\n')

    for testing_c in C_test_values:
        svclassifier = svm.SVC(
            kernel='rbf', C=float(testing_c))
        svclassifier.fit(X_train, y_train)

        train_error, val_error = getSampleError(
            Xval, yval, X_train, y_train, svclassifier, testing_c)

        error_train.append(train_error)
        error_val.append(val_error)

        print(f'C{testing_c}:\t {train_error}\t {val_error}')
    draw_findingBestValue(C_test_values, error_val, error_train,
                          'Finding Best C for gaussian kernel', 'C')


def draw_findingBestValue(test_values, error_val, error_train, title, xLabelText):
    plt.figure(figsize=(8, 5))
    plt.plot(test_values, error_val, 'or--', label='Validation Set Error')
    plt.plot(test_values, error_train, 'bo--', label='Training Set Error')
    plt.xlabel(f'${xLabelText}$ Value', fontsize=16)
    plt.ylabel('Classification Error [%]', fontsize=14)
    plt.title(title, fontsize=18)
    plt.legend()
    plt.show()


def trainPolynomialSVM(X_train, y_train, Xval, yval, X_test, y_test):
    test_degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    error_train = []
    error_val = []

    print('Degree\tTrain Error\tValidation Error\n')
    for tDegree in test_degrees:
        svclassifier = svm.SVC(kernel='poly', degree=tDegree)
        svclassifier.fit(X_train, y_train)

        train_error, val_error = getSampleError(
            Xval, yval, X_train, y_train, svclassifier, tDegree)

        error_train.append(train_error)
        error_val.append(val_error)

    draw_findingBestValue(test_degrees, error_val, error_train,
                          'Finding Best polynomial degree value', 'Degree')
    # Parece que el mejor grado es un grado 3


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

        train_error, val_error = getSampleError(
            Xval, yval, Xtrain, ytrain, linear_svm, testing_c)

        error_train.append(train_error)
        error_val.append(val_error)

    #draw_findingBestValue(C_test_values, error_train,error_val, 'Finding Best C Value', 'C')
    # De la grafica podemos observar que los mejores valores se encuentrar alrededor de utilizar una C con un
    # valor de  37.0


def trainSVMForLinear(X_train, y_train, X_test, y_test, yval, Xval):
    findBetterCForLinear(X_train, y_train, Xval, yval)


def testLinearSVM(X_train, y_train, X_test, y_test):
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

    plotConfusionMatrix(svclassifier, X_test,
                        y_test, 'true', "Linear")

    return linearSVMAccuracy, svclassifier


def testSigmoidSVM(X_train, y_train, X_test, y_test):
    bestC = 0.6
    finalSigmoidClassifier = svm.SVC(kernel='sigmoid', C=bestC)
    finalSigmoidClassifier.fit(X_train, y_train)

    y_pred = finalSigmoidClassifier.predict(X_test)
    sigmoidSVMAccuracy = precision_score(y_test, y_pred, average='micro')*100.

    print(
        f"La precisión del kernel sigmoide con un valor de C = {bestC} es del {sigmoidSVMAccuracy}%")

    plotConfusionMatrix(finalSigmoidClassifier, X_test,
                        y_test, 'true', "Sigmoid")
    return sigmoidSVMAccuracy, finalSigmoidClassifier


def testGaussianSVM(X_train, y_train, X_test, y_test):
    bestC = 2.8
    finalGaussianClassifier = svm.SVC(kernel='rbf', C=bestC)
    finalGaussianClassifier.fit(X_train, y_train)

    y_pred = finalGaussianClassifier.predict(X_test)
    gaussianSVMAccuracy = precision_score(y_test, y_pred, average='micro')*100.

    print(
        f"La precisión del kernel gaussiano con un valor de C = {bestC} es del {gaussianSVMAccuracy}%")

    plotConfusionMatrix(finalGaussianClassifier, X_test,
                        y_test, 'true', "Gaussian")

    return gaussianSVMAccuracy, finalGaussianClassifier


def testPolynomialSVM(X_train, y_train, X_test, y_test):
    finalPolyClassifier = svm.SVC(kernel='poly', degree=3)
    finalPolyClassifier.fit(X_train, y_train)

    y_pred = finalPolyClassifier.predict(X_test)
    polySVMAccuracy = precision_score(y_test, y_pred, average='micro')*100.

    print(
        f"La precisión del kernel polinómico con un grado = 3 es del {polySVMAccuracy}%")

    plotConfusionMatrix(finalPolyClassifier, X_test,
                        y_test, 'true', "Polynomial")
    return polySVMAccuracy, finalPolyClassifier


def show_global_accuracy(X_train, y_train, X_test, y_test):
    preccision = [testSigmoidSVM(X_train, y_train, X_test, y_test), testGaussianSVM(X_train, y_train, X_test, y_test),
                  testPolynomialSVM(X_train, y_train, X_test, y_test),  testLinearSVM(X_train, y_train, X_test, y_test)]

    draw_different_kernels_accuracy(preccision)


def draw_different_kernels_accuracy(precission):
    plt.figure(figsize=(30, 20))
    plt.title('SVM kernels accuracy ')
    plt.xlabel('Kernel')
    plt.ylabel('Precission[%]')

    plt.ylim(0, 110)

    x = np.arange(len(precission))
    rects = plt.bar(x, precission, color='red')
    plt.xticks(x, ('Sigmoid', 'Gaussian', 'Polynomial', 'Linear',))
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}%'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
    plt.show()


def plotConfusionMatrix(classifier, X_test, y_test, normalize, classifierName):
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=[
                                     'low cost', 'medium cost', 'high cost', 'very high cost'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(f'Normalized confusion matrix for {classifierName}')

    plt.show()


'''
    Univariate feature selection works by selecting the best features based on 
    univariate statistical tests. It can be seen as a preprocessing step to an estimator.
    Scikit-learn exposes feature selection routines as objects that implement the transform method:

    Para clasificacion se puede usar:
       chi2, f_classif, mutual_info_classif
'''


def univariate_feature_selection(X_train, y_train):

    selector = SelectKBest(f_classif,
                           k=2).fit_transform(X_train, y_train)

    return selector


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_sets(X_train, y_train, classifier, name):

    X = univariate_feature_selection(X_train, y_train)
    clf = classifier.fit(X, y_train)
    fig, ax = plt.subplots()
    title = (f'Decision surface for kernel {name}')

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('Mobile spectrum')
    ax.set_xlabel('2 Most significant attributes using f_classif method')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()
