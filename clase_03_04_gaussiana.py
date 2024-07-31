"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Gaussiana 3D
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

def gaussiana_3D(t, sigma):

    ventana = np.linspace(-t/2, t/2, t)
    u,v = np.meshgrid(ventana, ventana)

    G = (1/sqrt((sigma**2*2*pi)))*np.exp(-(u**2+v**2)/(2*sigma**2))
    N = G/np.sum(G.flatten())  #normalizamos

    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.plot_surface(u, v, N, cmap='Spectral')
    plt.show()

    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.contour3D(u, v, N, 50, cmap='binary')
    plt.show()

#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

t= 100
sigma = 20
gaussiana_3D(t, sigma)