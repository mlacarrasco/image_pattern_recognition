"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Cálculo de contraste
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""


import cv2
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt

def contraste(im, level):

    #imagen binaria umbralizada
    bw = im>level

    # calculamos el entorno y aplicamos 
    # una negación a la región
    ebw = np.logical_not(bw)

    # calculamos el promedio del entorno
    Ge = np.sum(im[ebw])/np.sum(ebw)
    Gr = np.sum(im[bw])/np.sum(bw)

    K1 = (Gr-Ge)/Ge
    K2 = (Gr-Ge)/(Gr+Ge)
    K3 = np.log(Gr/Ge)
    K = K1, K2, K3
    return K


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

K = contraste(image, 7)

# Resultados del contraste
print(f'Contraste K1: {K[0]}')
print(f'Contraste K2: {K[1]}')
print(f'Contraste K3: {K[2]}')

plt.figure()
plt.imshow(image) 
plt.show()



