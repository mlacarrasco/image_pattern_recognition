"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Laplaciano 3D
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""


import numpy as np
import matplotlib.pyplot as plt
from math import pi


def laplaciano_3D(t, sigma):

    ventana = np.linspace(-t/2, t/2, t)
    u,v = np.meshgrid(ventana, ventana)

    mat = (u**2 + v**2) /(2*sigma**2)
    L = (-1/(pi*sigma**4)) * (1-mat)*np.exp(-mat)

    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.plot_surface(u, v, L, cmap='Spectral')
    plt.show()

    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.contour3D(u, v, L, 50, cmap='binary')
    plt.show()

    return L
#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

t= 100
sigma = 20
L = laplaciano_3D(t, sigma)