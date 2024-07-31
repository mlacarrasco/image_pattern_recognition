"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 FFT de contornos
 Autor:. Miguel Carrasco (26-08-2021)
 rev.1.0. Version inicial
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def cambio_escala (binary, factor):
    factor_times = factor
    height = int(binary.shape[0] * factor_times)
    width  = int(binary.shape[1] * factor_times)
    dim    = (width, height) 

    #cambio de escala 
    binary= cv2.resize(binary, dim, interpolation=cv2.INTER_AREA)
    return binary

#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

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

#cambiamos la escala de la imagen
binary = cambio_escala (binary, 10)

#buscamos las coordenadas de los contornos 
contours, hierarchy = cv2.findContours(binary, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_NONE)

#generamos una matriz (fila/columna) con los contornos
contornos = np.vstack(contours[0])

# extraemos las coordenadas x,y 
x , y = zip(*contornos)

# dejamos las coordenadas en forma compleja
CX = np.array(x) + 1j*np.array(y)
CX = np.append(CX, CX[0:1])

print(CX)
FF = np.fft.fft(CX)

#Energía
E = np.abs(FF)/np.sum(np.abs(FF))
acum = np.cumsum(E)

#ordenamos de menor a mayor
idx = np.argsort(E)
idx = idx[::-1] #orden inverso

#graficamos
plt.figure()
plt.bar(np.arange(len(E)), E[idx])
plt.show()

#Seleccionamos las columnas según un umbral o valor fijo
cols = 20
sel = idx[0:cols]
iFF = np.zeros_like(FF)
iFF[sel] = FF[sel]
rF = np.fft.ifft(iFF)

# obtejemos los nuevos valores de coordenadas
new_x, new_y = np.real(rF), np.imag(rF)

plt.figure(dpi=150)
plt.imshow(binary, cmap='gray')
plt.plot(new_x, new_y, color='red')
plt.show()

