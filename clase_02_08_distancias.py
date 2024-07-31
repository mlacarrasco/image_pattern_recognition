"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Distancia Normalizada del centro de una región a su frontera
 Autor:. Miguel Carrasco (02-08-2021)
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

def mrs(r, s, I, J):
    i = I**r
    j = J**s
    return np.sum(i*j)

def centro_masa(bw):
    coords = np.argwhere(bw==1)
    m00 = mrs(0,0,coords[:,0], coords[:,1])
    m10 = mrs(1,0,coords[:,0], coords[:,1])
    m01 = mrs(0,1,coords[:,0], coords[:,1])
    ci = m10/m00
    cj = m01/m00
    return ci, cj

def distancia(image):
 
    #%determinamos el borde 
    #buscamos las coordenadas de los contornos 
    contours, hierarchy = cv2.findContours(image, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_NONE)

    # generamos una matriz (fila/columna) con los contornos
    contornos = np.vstack(contours[0])

    # extraemos las coordenadas x,y 
    x , y = zip(*contornos)
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)

    # calculo del centro de masa
    [ci, cj] = centro_masa(image)
    
    # distancia euclidiana
    dd=np.sqrt((x-ci)**2+(y-cj)**2)

    #descriptores normalizados 
    dmean = np.mean(dd)
    dmax  = np.max(dd)/dmean
    dmin  = np.min(dd)/dmean
    delta = dmax/dmin
    features = dmax, dmin, delta, ci, cj, x, y

    return features

#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

#imagen inventada
bw=[[0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]]

#transformamos los datos a uint8
bw = np.array(bw, dtype='uint8')

#cambiamos la escala de la imagen
binary = cambio_escala (bw, 10)

dmax, dmin, delta, ci, cj, x, y  = distancia(binary)
print(f'Distancia máxima: {dmax}')
print(f'Distancia minima: {dmin}')

plt.figure()
plt.imshow(binary, cmap="gray")
plt.scatter(x, y, marker='.', s=100, c='red')
plt.plot(cj, ci, marker = 'o', color ='blue')
plt.show()
