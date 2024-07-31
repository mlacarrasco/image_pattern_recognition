"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación del filtro laplaciano
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from math import pi, sqrt


def filtro_laplaciano(A, L):
    S = np.sum(A.flatten()*L.flatten())
    return S

def laplaciano(t, sigma):
    ventana = np.linspace(-t/2, t/2, t)
    u,v = np.meshgrid(ventana, ventana)

    mat = (u**2 + v**2) /(2*sigma**2)
    L = (-1/(pi*sigma**4)) * (1-mat)*np.exp(-mat)
    
    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.plot_surface(u, v, L, cmap='Spectral')
    plt.show()
    return L

#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

img = cv2.imread('data/imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#construccion filtro
t = 19
sigma = sqrt(t)/2
L = laplaciano(t, sigma)

#aplicamos el filtro punto_medio
filtro= ndi.generic_filter(gray, filtro_laplaciano, [t,t], extra_keywords={'L':L})

#desplegamos los resultados
plt.figure()
plt.imshow(filtro, cmap='gray')
plt.show()