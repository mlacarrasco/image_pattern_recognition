"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación del filtro laplaciano
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
 rev.1.1 (squeeze along vector)
"""

import cv2
from numpy.ma import arange
from skimage.feature import graycomatrix, graycoprops
from sklearn import preprocessing
import numpy as np


img= cv2.imread('data/imagenes/textura_1.tif')
gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

new_scale = (0,11)
new_gray = preprocessing.MinMaxScaler(new_scale).fit_transform(gray).astype(int)

# algoritmo graycomatrix P01
l = 12  #numero de niveles de la imagen
P_1_0 = graycomatrix(new_gray, distances=[1], angles=[0], levels=l, symmetric=False, normed=False)


# extracción de caracteristicas a traves de greycomatrix
features = ['contrast','correlation', 'dissimilarity','homogeneity','ASM','energy']
for ft in features:
    
    sts = graycoprops(P_1_0, ft).squeeze()
    print(f'{ft}: {float(sts)}')
