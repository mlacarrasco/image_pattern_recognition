"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Extracción de los 7 momentos de HU
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt
from math import copysign, log10

def cambio_escala (image, factor):

    factor_times = factor
    height = int(image.shape[0] * factor_times)
    width  = int(image.shape[1] * factor_times)
    dim    = (width, height) 
    #cambio de escala 
    output = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    return output


#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

im=[[2,  1,  2,  3,  2,  1,  4,  0],
    [4,  3,  4,  5,  1,  4,  3,  0],
    [3,  4, 23, 19, 13,  1,  3,  1],
    [0,  2, 18, 12, 17, 18,  4,  2],
    [1,  1, 22, 45, 45, 23,  5,  2],
    [0,  1,  1, 31, 21, 12,  6,  3],
    [0,  2,  4,  2,  3,  5,  7,  4],
    [0,  0,  5,  1,  1,  3,  3,  6]]

#transformamos los datos a uint8
image = np.array(im, dtype='uint8')

#cambiamos la escala de la imagen
scaled_img = cambio_escala (image, 10)


#extracción de los siete momentos de HU
huMoments = cv2.HuMoments(cv2.moments(image))
huMoments_Scaled = cv2.HuMoments(cv2.moments(scaled_img))

# Log scale hu moments
for i in range(0,7):
    huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(np.abs(huMoments[i]))
    huMoments_Scaled[i] = -1* copysign(1.0, huMoments_Scaled[i]) * log10(np.abs(huMoments_Scaled[i]))
    print(f'Momento {i}\t Original:{huMoments[i].round(7)}\t Modificada:{ huMoments_Scaled[i].round(7)}')


plt.figure(dpi=100)
plt.subplot(121)
plt.imshow(image, cmap='gray')

plt.subplot(122)
plt.imshow(scaled_img, cmap='gray')
plt.show()
