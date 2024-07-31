"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Etiquetado de regiones de una imagen binaria
 
 Autor:. Miguel Carrasco (14-08-2021)
 rev.1.0
 rev.1.1 (cambios de imagen binaria, 30/07/2024)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label

# Lectura de la imagen
im = cv2.imread('data/imagenes/dibujo.png')

#transormación a escala de grises
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#imagen binaria
ret, bw = cv2.threshold(gray, 50, 256, cv2.THRESH_BINARY)

labels = label(bw)

#desplegar imagen
plt.figure()
plt.imshow(labels, cmap='jet')
plt.show()
