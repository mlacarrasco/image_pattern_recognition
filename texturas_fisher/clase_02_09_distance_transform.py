"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Uso de función integrada de Distance Transform
 Autor:. Miguel Carrasco (14-08-2021)
 rev.1.0
"""

import cv2
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt

def cambio_escala (binary, factor):

    factor_times = factor
    height = int(binary.shape[0] * factor_times)
    width  = int(binary.shape[1] * factor_times)
    dim    = (width, height) 
    #cambio de escala 
    binary = cv2.resize(binary, dim, interpolation=cv2.INTER_AREA)
    return binary


#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

#imagen inventada
bw=[[0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]]

#transformamos los datos a uint8
bw = np.array(bw, dtype='uint8')

#cambiamos la escala de la imagen
binary = cambio_escala (bw, 1)


#funcion integrada de distancia : {cv2.DIST_L1, cv2.DIST_L2, cv2.DIST_C}
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

plt.figure()
plt.imshow(dist, cmap="gray")
plt.show()
