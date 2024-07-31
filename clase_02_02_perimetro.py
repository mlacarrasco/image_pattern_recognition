"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Extraccion de perimetro
 Autor:. Miguel Carrasco (26-08-2021)
 rev.1.0. Version inicial
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#******************************
#      PROGRAMA PRINCIPAL     *
#******************************
img = cv2.imread('data/imagenes/Imagen_1.png', cv2.IMREAD_GRAYSCALE)

#generamos una imagen binaria de tipo uint8
binaria = (img>1*1).astype('uint8')

#extraemos los contornos de una region
contours, hierarchy = cv2.findContours(binaria, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_NONE)

# generamos una matriz (fila/columna) con los contornos
contornos = np.vstack(contours[0])

 # extraemos las coordenadas x,y 
x , y = zip(*contornos)
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

plt.figure()
plt.imshow(binaria, cmap='gray')
plt.plot(x, y, color='green')
plt.scatter(x, y, marker='o', s=30, color='red')
plt.grid(alpha=0.2)
plt.show()

