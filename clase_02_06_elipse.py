"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo que interpola una elipse en la region
 Autor:. Miguel Carrasco (02-08-2021)
 rev.1.0

 >> metodo implementado
 Fitzgibbon, A. W., Pilu, M and Fischer, R. B.: Direct least squares
 fitting of ellipses. In Proc. of the 13th International Conference on 
 Pattern Recognition, pages 253–257, Vienna, September 1996
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

import numpy.matlib

from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse

def cambio_escala (binary, factor):
    factor_times = factor
    height = int(binary.shape[0] * factor_times)
    width  = int(binary.shape[1] * factor_times)
    dim    = (width, height) 

    #cambio de escala 
    binary= cv2.resize(binary, dim, interpolation=cv2.INTER_AREA)
    return binary

def fitzgibbon(x, y):
    # algoritmo de Fitzgibbon (1996) para la estimación de una elipse
    # a partir de las coordenadas de la frontera de una región
    
    n = x.shape[0]                                  #puntos de la frontera
    D = [x**2, x*y, y**2, x, y, np.ones((n,1))]     # matriz X
    D = np.hstack(D)
    S = np.matmul(D.T,D)                            # matriz dispersión
    C = np.zeros((6,6))
    C[0,2]=2; C[1,1]=-1; C[2,0]=2;                  # matriz restricción

    # Problema: S·a = lambda·C·a 
    # --> Solución a traves de Eigenvalues
    E,V = np.linalg.eig(np.dot(np.linalg.inv(S),C))
    idx = np.argwhere(E>0 & ~np.isinf(E))
    params = V[:,idx]                                    # seleccionamos el valor propio 
    return params

def parametros(x, y):
    # Funcion que busca los parametros parametricos
    # de la elipse segun las coordendas de los bordes.

    # parametros de fitzgibbon
    a,b,c,d,e,f = fitzgibbon(x, y)

    #estimados de angulos y factores de Fitzgibbon
    alpha = np.arctan2(b,a-c)/2

    ct = np.cos(alpha)
    st = np.sin(alpha)
    ap = a*ct*ct + b*ct*st + c*st*st
    cp = a*st*st - b*ct*st + c*ct*ct

    #% get translations
    T = np.array([[a,  b/2],
                  [b/2, c]])
    T = T.reshape(2,2)
    t = -1* np.matmul(np.linalg.inv(2*T),np.array([d, e]).reshape(2,1))
    cx = t[0]
    cy = t[1]

    #% factor de escala
    val = np.matmul(t.T,np.matmul(T,t))
    scale = 1 / (val- f)

    #% parametros
    r1 = 1/np.sqrt(scale*ap)
    r2 = 1/np.sqrt(scale*cp)
    v = np.array([r1.ravel(), r2.ravel(), cx.ravel(), cy.ravel(), alpha.ravel()])
    return v


def draw_ellipse(binary, v, N=100):
# funcion auxiliar que permite dibujar una ellipse
    ae, be, x0, y0, alfa = v
    #angulo de rotación interno
    theta = np.linspace(0, 2*pi, N) 
    
    #parametro estimado anteriormente
    R = np.array([[np.cos(alfa), -np.sin(alfa)],
                  [np.sin(alfa), np.cos(alfa)]])
    R = R.reshape(2,2)

    X = ae*np.cos(theta)
    Y = be*np.sin(theta)
    M = np.array([X,Y]).reshape(2,-1)
    CC=np.matmul(R,M) + np.matlib.repmat(np.array([x0,y0]).reshape(2,1),1,N)

    plt.figure()
    plt.imshow(binary, cmap="gray")
    plt.scatter(CC[0,:], CC[1,:], marker='x', s=100, c='red')
    plt.plot(CC[0,:], CC[1,:])
    plt.show()

    return CC

#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

#puntos de la elipse
N= 20 

#imagen inventada
bw=[[0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]]

#transformamos los datos a uint8
binary = np.array(bw, dtype='uint8')

#cambiamos la escala de la imagen
binary = cambio_escala (binary, 5)

#buscamos las coordenadas de los contornos 
contours, hierarchy = cv2.findContours(binary, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_NONE)

#generamos una matriz (fila/columna) con los contornos
contornos = np.vstack(contours[0])

#extraemos las coordenadas x,y 
x , y = zip(*contornos)
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

#extraemos los parámetros de la elipse
# xc, yc, a, b, theta
v = parametros(x,  y) 

print(f'parametros fitzgibon: {v}')

#comparativa de parametros con modelo implementado en Skimage
ell = EllipseModel()
ell.estimate(contornos)
xc, yc, a, b, theta = ell.params
print(f'parametros Skimage: {xc, yc, a, b, theta}')

#extraemos las coodenadas de la elipse
CC = draw_ellipse(binary, v, N)

#EOF