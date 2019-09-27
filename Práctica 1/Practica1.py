import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def axis_lim(minX, maxX, minY, maxY):
    plt.xlabel('Población de la ciudad en 10.000s')
    plt.ylabel('Ingresos en $10.000s')

    plt.xlim([minX - 1, maxX + 1])
    plt.ylim([minY - 4, maxY + 0.3])

def dibuja_grafica(fArray, funH, theta):
    X = np.linspace(5.0, 22.5, len(fArray), endpoint=True)
    plt.scatter(fArray[:, 0], fArray[:, 1], s=50, c='red', marker="x")

    axis_lim(5.0, 22.5, 0, 25)

    plt.plot(fArray[:, 0], funH)

    max_value = np.amax(fArray[:, 0])
    t0 = theta[0]
    t1 = theta[1]
    plt.annotate(r'$h(x)={}+{}x$'.format(t0, t1) ,
        xy=(max_value, h(max_value, theta)), xycoords='data',
        xytext=(5, 26.5), fontsize=10,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"))

    #plt.show()
    plt.draw()
    plt.pause(1e-17)


def descenso_gradiente(casos, alpha=0.01, iter=1500):
    # Inicialización de los valores de theta a 0, con cada iteración su valor irá cambiando y siempre para mejor, en el caso de que vaya a peor es que esta mal hecho
    theta = np.zeros(2)

    # Inizialización de un array que guarda el historial de los costes
    costeArray = np.zeros(iter)

    m = len(casos)

    plt.ion()

    for i in range(iter):
        temp0 = theta[0] - alpha * (1 / m) * np.sum(h(casos[:,0], theta) - casos[:,1], axis=0)
        temp1 = theta[1] - alpha * (1 / m) * np.sum((h(casos[:,0], theta) - casos[:,1]) * casos[:, 0], axis=0) 
        theta[0] = temp0
        theta[1] = temp1

        funH = h(casos[:,0], theta)
        plt.clf()
        dibuja_grafica(casos, funH, theta)
    plt.show()
    return theta



def h(x, theta):
   return theta[0] + x * theta[1]

def main(file_name):
    a = carga_csv(file_name)
    
    theta = descenso_gradiente(casos=a)

    print("TERMINADO")
    """funH = h(a[:, 0], theta)
    dibuja_grafica(a, funH)"""


main("ex1data1.csv")