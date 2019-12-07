import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.io import loadmat

def drawData(X, y, title):
    y = y.ravel()
    pos = y==1
    neg = y==0

    plt.title(title)
    plt.scatter(X[:,0][pos], X[:,1][pos], c="k", marker="+")
    plt.scatter(X[:,0][neg], X[:,1][neg], c="y", marker="o")
    plt.show()

def drawSVM_Linear(X, y, model, title):
    # Get the separating hyperplane.
    w = model.coef_[0]
    a = -w[0] / w[1]
    # Only 2 points are required to define a line, e.g. min and max.
    xx = np.array([X[:,0].min(), X[:,0].max()])
    yy = a * xx - (model.intercept_[0]) / w[1]
    # Plot the separating line.
    plt.plot(xx, yy, 'b-')
    # Plot the training data.
    drawData(X, y, title)

def drawSVM_Gaussian(X, y, sigma, model, title):
    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = model.predict(gaussian_kernel(this_X, X, sigma))

    # Plot the SVM boundary
    plt.contour(X1, X2, vals, colors="b", levels=[0,0])
    # Plot the training data.
    drawData(X, y, title)

def linear_svm(X, y, C, tol, iter):
    y = y.ravel()
    clf = svm.SVC(C=C, kernel="linear", tol=tol, max_iter=iter)
    return clf.fit(X, y)

def gaussian_kernel(X1, X2, sigma):
    Gram = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.ravel()
            x2 = x2.ravel()
            Gram[i, j] = np.exp(-np.sum(np.square(x1 - x2)) / (2 * (sigma**2)))
    return Gram

def gaussian_svm(X, y, C, tol, iter, sigma):
    y = y.ravel()
    clf = svm.SVC(C=C, kernel="rbf", gamma=1 / ( 2 * sigma **2))
    return clf.fit(gaussian_kernel(X, X, sigma), y)

def select_params(X, y, Xval, yval):
    vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    predictions = dict()
    for C in vec:
        for sigma in vec:
            model = gaussian_svm(X, y, C, 0.001, -1, sigma)
            # Perform classification on samples in Xval.
            # For precomputed kernels, the expected shape of
            # X is [n_samples_validation, n_samples_train]
            prediction = model.predict(gaussian_kernel(Xval, X, sigma))
            # Compute the prediction errors.
            predictions[(C, sigma)] = np.mean((prediction != yval).astype(int))
    C, sigma = min(predictions, key=predictions.get)
    return C, sigma

def main():
    # Datos 1
    '''
    data = loadmat("ex6data1.mat")

    y = data["y"] # (51, 1)
    X = data["X"] # (51, 2)

    drawData(X, y, "Conjunto de datos 1")

    model = linear_svm(X, y, 1, "linear", 0.001, -1)
    drawSVM(X, y, model, "SVM con C = 1")

    model = linear_svm(X, y, 100, "linear", 0.001, -1)
    drawSVM(X, y, model, "SVM con C = 100")
    '''

    # Datos 2
    '''
    data = loadmat("ex6data2.mat")

    y = data["y"] # (863, 1)
    X = data["X"] # (863, 2)

    #drawData(X, y, "Conjunto de datos 2")
    C = 1
    sigma = 0.1
    model = gaussian_svm(X, y, C, 0.001, 100, sigma)
    drawSVM_Gaussian(X, y, sigma, model, "SVM con kernel gaussiano")
    '''

    #Datos 3
    data = loadmat("ex6data3.mat")

    y = data["y"] # (211, 1)
    X = data["X"] # (211, 2)

    yval = data["yval"]
    Xval = data["Xval"]
    #drawData(X, y, "Conjunto de datos 3")

    C, sigma = select_params(X, y, Xval, yval)
    model = gaussian_svm(X, y, C, 0.001, -1, sigma)
    drawSVM_Gaussian(X, y, sigma, model, "Kernel Gaussiano")

main()
