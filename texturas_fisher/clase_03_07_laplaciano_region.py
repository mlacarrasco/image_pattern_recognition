"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación del filtro laplaciano
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from math import pi

def filtro_laplaciano(A, L):
    S = np.sum(A.flatten()*L.flatten())
    return S

def laplaciano(t, s):

    ventana = np.linspace(-t/2, t/2, t)
    u,v = np.meshgrid(ventana, ventana)

    mat = (u**2 + v**2) /(2*s**2)
    L = (-1/(pi*sigma**4)) * (1-mat)*np.exp(-mat)

    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.plot_surface(u, v, L, cmap='Spectral')
    plt.show()
    return L


#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

im=[[ 2,  1,  2,  3,  2,  1,  4,  0],
    [4, 29, 23, 25, 26, 24,  3,  0],
    [3, 28, 23, 19, 13, 27,  3,  4],
    [0, 28, 18, 12, 17, 18, 25,  2],
    [1, 27, 22, 45, 45, 23, 29,  2],
    [0,  1, 32, 31, 21, 12, 25,  3],
    [0,  2,  4, 31, 28, 29, 30,  4],
    [0,  0,  5,  1,  1,  3,  3,  6]]

#transformamos los datos a uint8
image = np.array(im, dtype='uint8')

# generamos una imagen binaria
bw = (image>7)

#construccion filtro
t= 7
sigma = 0.5
L = laplaciano(t, sigma)

# aplicamos el filtro laplaciano sobre toda la imagen
filtro= ndi.generic_filter(image, filtro_laplaciano, [t,t], extra_keywords={'L':L})

#aplicamos el filtro de la región sobre la imagen filtrada
region_filter = filtro*bw
promedio_region = np.mean(region_filter)

outer_filter = filtro * np.logical_not(bw)
promedio_outer = np.mean(outer_filter)

print(f'Promedio Laplaciano pixeles region {promedio_region}')
print(f'Promedio Laplaciano area exterior {promedio_outer}')

plt.figure()
plt.imshow(region_filter, cmap='gray') 
plt.title('Filtro aplicado solo a una región')
plt.show()




