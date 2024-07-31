"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Detección de esquinas con algoritmo de Harris
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""

import cv2
from sklearn import preprocessing
import numpy as np 

from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max


def fspecial_gauss(size, sigma):

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def harris(im, umbral, sze, sigma):
    # Harris. Calcula esquinas segun un umbral
    # v.1.0

    dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # Mascara X
    dy = np.array([[ 1, 2, 1], [ 0, 0, 0],[-1, -2, -1]]) # % Mascara Y

    # derivadas direccionales
    Ix = convolve2d(im, dx, mode='same')   
    Iy = convolve2d(im, dy, mode='same')

    # Convolucion de derivada con filtro gaussiano
    g = fspecial_gauss(sze, sigma)

    A = convolve2d(Ix**2, g, mode ='same')
    B = convolve2d(Ix*Iy, g, mode= 'same')
    C = convolve2d(Iy**2, g, mode= 'same')


    # gradiente(im, A,C)    
    # Medición de respuesta de la esquina
    k = 0.04      #%% Parametro de Harris
    H = (A*C - B**2) - k*(A + C)**2
    xy = peak_local_max(H, min_distance=3, threshold_abs=umbral)

    return H, xy

#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

img= cv2.imread('data/imagenes/cameraman.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new_scale = (0,1)
new_gray = preprocessing.MinMaxScaler(new_scale).fit_transform(gray)


#% parametros de Harris
umbral = 1.5  # umbral de seleccion
sze = 7        # tamano de mascara
sigma = 2      # desviacion estandar de gausiana

# funcion Harris
H, xy = harris(new_gray, umbral, sze, sigma)

#% Buscamos los puntos que cumplen criterio
X = xy[:,0]; Y = xy[:,1]
plt.figure()
plt.subplot(1,2,1)
plt.imshow(H, cmap='hot')
plt.scatter(Y,X,marker='x')

plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.scatter(Y,X,marker='x', color='red')

plt.show()