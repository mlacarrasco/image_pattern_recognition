"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Extracción de características de textura
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""

import cv2
from numpy.ma import arange
from skimage.feature import graycomatrix
import numpy as np
from math import radians
from sklearn import preprocessing

def texture_correlation(P):
    #cálculo de la correlación de la matriz de Co-ocurrencia
    P = P/ np.sum(P)
    # Calculamos las posiciones i j
    fil, col= P.shape
    vfil = np.arange(0,fil)
    vcol = np.arange(0,col)
    c,r = np.meshgrid(vfil, vcol)
 
    # Media en direccion del pixel
    mr = np.sum(r*P)
    mc = np.sum(c*P)

    # Desviacion estandar 
    Sr = np.sqrt(np.sum((r-mr)**2*P))
    Sc = np.sqrt(np.sum((c-mc)**2*P))

    c = np.sum((r-mr)*(c-mc)*P)/(Sr*Sc)
    return c

#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

img = cv2.imread('data/imagenes/textura_1.tif')
gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

new_scale = (0,11)
new_gray = preprocessing.MinMaxScaler(new_scale).fit_transform(gray).astype(int)

# algoritmo graycomatrix P01
l= 12  #numero de niveles de la imagen
P_1_0 = graycomatrix(new_gray, distances=[1], angles=[radians(90)], levels=l, symmetric=False, normed=False)
P = P_1_0.reshape(l,l)
print('Correlación:',texture_correlation(P))









