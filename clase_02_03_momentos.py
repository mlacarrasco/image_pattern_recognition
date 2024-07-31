"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Extraccion de momementos de HU
 Autor:. Miguel Carrasco (26-08-2021)
 rev.1.0. Version inicial
 rev.1.1 (modificado valores en log)

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import copysign, log10

def mrs(r, s, I, J):
    i=I**r
    j=J**s
    return np.sum(i*j)

# Funcion de momento central
def  m_central(r, s, I, J):
    m00 = mrs(0,0,I,J)
    m10 = mrs(1,0,I,J)
    m01 = mrs(0,1,I,J)

    ci = m10/m00
    cj = m01/m00
    i = (I-ci)**r
    j = (J-cj)**s

    return sum(i*j)

# Funcion de momentos de Hu
def hu(I, J):
    H= np.zeros(7)
    eta11 = eta(1,1,I,J)
    eta12 = eta(1,2,I,J)
    eta20 = eta(2,0,I,J)
    eta21 = eta(2,1,I,J)
    eta02 = eta(0,2,I,J)
    eta03 = eta(0,3,I,J)
    eta30 = eta(3,0,I,J)

    H[0] = eta20+eta02
    H[1] = (eta20-eta02)**2 + 4*eta11**2
    H[2] = (eta30-3*eta12)**2+(3*eta21-eta03)**2
    H[3] = (eta30+eta12)**2+(eta21+eta03)**2
    H[4] =(eta30-3*eta12)*(eta30+eta12)*( (eta30+eta12)**2-3*(eta21+eta03)**2)+(3*eta21-eta03)*(eta21+eta03)*(3* (eta30+eta12)**2- (eta21+eta03)**2 )
    H[5] = (eta20-eta02)*((eta30+eta12)**2-(eta21+eta03)**2+ 4*eta11*(eta30+eta12)*(eta21+eta03))
    H[6] = (3*eta21-eta03)*(eta30+eta12)*((eta30+eta12)**2-3*(eta21+eta03)**2)+(eta30-3*eta12)*(eta21+eta03)* (3*(eta30+eta12)**2-(eta21+eta03)**2)
    return H

# Funcion eta (necesaria para momentos de Hu
def eta(r, s, I, J):

    t = (r+s)/2 + 1
    a =  m_central(r,s,I,J)
    b =  m_central(0,0,I,J)

    return a/(b**t)

#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

bin1 = cv2.imread('data/imagenes/Imagen_2.png', cv2.IMREAD_GRAYSCALE)

dim =  bin1.shape[0]*20,bin1.shape[1]*20
bin1 = cv2.resize(bin1,dim, interpolation =cv2.INTER_AREA )

plt.figure()
plt.imshow(bin1, cmap='gray')
plt.show()

cij = np.argwhere(bin1==255)
huMoments = hu(cij[:,0], cij[:,1])

#normalizamos los momentos
for i in range(0,7):
    huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(np.abs(huMoments[i]))
    print(huMoments[i])