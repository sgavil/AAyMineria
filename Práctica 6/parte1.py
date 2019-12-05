from sklearn import svm
from scipy.io import loadmat


def main():
    data = loadmat("ex6data1.mat")

    y = data["y"]
    X = data["X"]

    clf = svm.SVC(C=1.0, kernel='rbf', tol=0.001, max_iter=1)
    clf.fit(X, y)


main()
