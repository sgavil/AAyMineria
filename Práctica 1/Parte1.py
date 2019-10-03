import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def axis_lim_grafica(subPlt, minX, maxX, minY, maxY):
    subPlt.set_xlabel('Población de la ciudad en 10.000s')
    subPlt.set_ylabel('Ingresos en $10.000s')

    subPlt.set_xlim([minX - 1, maxX + 1])
    subPlt.set_ylim([minY - 4, maxY + 0.3])

def dibuja_grafica(subPlt, fArray, funH, theta):
    X = np.linspace(5.0, 22.5, len(fArray), endpoint=True)
    subPlt.scatter(fArray[:, 0], fArray[:, 1], s=50, c='red', marker="x")

    axis_lim_grafica(subPlt, 5.0, 22.5, 0, 25)

    subPlt.plot(fArray[:, 0], funH)

    max_value = np.amax(fArray[:, 0])
    t0 = theta[0]
    t1 = theta[1]
    subPlt.annotate(r'$h(x)={}+{}x$'.format(t0, t1) ,
        xy=(max_value, h(max_value, theta)), xycoords='data',
        xytext=(5, 26.5), fontsize=10,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"))

def axis_lim_costes(subPlt, minX, maxX, minY, maxY):
    subPlt.set_xlabel('Número de iteraciones')
    subPlt.set_ylabel('Coste')

    subPlt.set_xlim([minX - 50, maxX + 200])
    subPlt.set_ylim([minY - 0.5, maxY + 0.5])

def dibuja_costes(subPlt, numCasos, costeArray):
    X = np.linspace(0, numCasos, numCasos, endpoint=True)
    axis_lim_costes(subPlt, 0, numCasos, costeArray[len(costeArray) - 1], costeArray[0])
    subPlt.plot(range(numCasos), costeArray)

def descenso_gradiente(casos, alpha=0.01, iter=1500):
    # Inicialización de los valores de theta a 0, con cada iteración su valor irá cambiando y siempre para mejor, en el caso de que vaya a peor es que esta mal hecho
    theta = np.zeros(2)

    # Inizialización de un array que guarda el historial de los costes
    costeArray = np.zeros(iter)

    m = len(casos)

    #plt.ion()

    for i in range(iter):
        temp0 = theta[0] - alpha * (1 / m) * np.sum(h(casos[:,0], theta) - casos[:,1], axis=0)
        temp1 = theta[1] - alpha * (1 / m) * np.sum((h(casos[:,0], theta) - casos[:,1]) * casos[:, 0], axis=0) 
        theta[0] = temp0
        theta[1] = temp1

        funH = h(casos[:,0], theta)
        costeArray[i] = (1 / (2 * m)) * np.sum(np.square(h(casos[:,0], theta) - casos[:,1]), axis=0)
        plt.clf()
        print(costeArray[i])

    fig, subPlot = plt.subplots(1, 2, figsize=(12, 5))

    dibuja_grafica(subPlot[0], casos, funH, theta)
    dibuja_costes(subPlot[1], iter, costeArray)
    plt.show()



def h(x, theta):
   return theta[0] + x * theta[1]

def main(file_name):
    a = carga_csv(file_name)
    descenso_gradiente(casos=a)


main("ex1data1.csv")