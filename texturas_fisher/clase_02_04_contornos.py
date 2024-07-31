"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Extraccion de contornos
 Autor:. Miguel Carrasco (26-08-2021)
 rev.1.0. Version inicial

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#imagen inventada
bw=[[0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]]

#transformamos los datos a uint8
binary = np.array(bw, dtype='uint8')

#buscamos las coordenadas de los contornos 
contours, hierarchy = cv2.findContours(binary, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_NONE)

#generamos una matriz (fila/columna) con los contornos
contornos = np.vstack(contours[0])

#graficamos
plt.figure(dpi=150)
plt.imshow(binary, cmap='gray')
plt.scatter(contornos[:,0], contornos[:,1], color='red', marker='x')
plt.show()