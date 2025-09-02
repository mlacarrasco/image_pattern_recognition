"""
 Universidad Diego Portales
 Facultad de Ingeniería y Ciencias
 Reconocimiento de Patrones en imágenes

 Aplicación de analisis de Texturas en un grupo de imágenes
 Autor:. Miguel Carrasco (28-08-2025)
 rev.1.0
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np

im =[[2,  1,  2,  3,  2,  1,  4,  0],
    [4,  3,  4,  5,  1,  4,  3,  0],
    [3,  4, 23, 19, 13,  1,  3,  1],
    [0,  2, 18, 12, 17, 18,  4,  2],
    [1,  1, 22, 45, 45, 23,  5,  2],
    [0,  1,  1, 31, 21, 12,  6,  3],
    [0,  2,  4,  2,  3,  5,  7,  4],
    [0,  0,  5,  1,  1,  3,  3,  6]]

#transformamos los datos a uint8
im = np.array(im, dtype='uint8')

fil, col = im.shape
scale =  20
dsize = (fil*scale,col*scale)

output_image_linear = cv2.resize(im,dsize,None, interpolation =cv2.INTER_LINEAR)
output_image_cubic = cv2.resize(im,dsize,None, interpolation =cv2.INTER_CUBIC)
output_image_area = cv2.resize(im,dsize,None, interpolation =cv2.INTER_AREA)

#ploteamos las figuras
fig, ax= plt.subplots(nrows=2, ncols=2)
fig.set_figwidth(10)
ax[0,0].imshow(im, cmap='gray'); ax[0,1].set_title('Original')
ax[0,1].imshow(output_image_linear,cmap='gray'); ax[0,1].set_title('Inter Linear')
ax[1,0].imshow(output_image_cubic,cmap='gray'); ax[1,0].set_title('Inter Cubic')
ax[1,1].imshow(output_image_area,cmap='gray'); ax[1,1].set_title('Inter Area')
plt.tight_layout()
plt.show()


def mrs(imagen,r,s,umbral):
    R = np.argwhere(imagen>umbral)
    i = R[:,0]
    j = R[:,1]
    return np.sum((i**r) * (j**s)* imagen[i,j])


def mu_rs(imagen,r,s, umbral):
    R = np.argwhere(imagen>umbral)
    i = R[:,0] #pixeles que son las coordenadas "i" pertenecen a R
    j = R[:,1] #pixeles que son las coordenadas "j"

    m_00 = mrs(imagen, 0, 0, umbral)
    m_10 = mrs(imagen, 1, 0, umbral)
    m_01 = mrs(imagen, 0, 1, umbral)

    i_barra = m_10/m_00
    j_barra = m_01/m_00

    return np.sum((i-i_barra)**r * (j-j_barra)**s * imagen[i,j])

def eta_rs(image, r, s, umbral):

    t = (r+s)/2+1
    return mu_rs(image, r,s, umbral)/(mu_rs(image, 0, 0, umbral)**t)

def hu_test(im, umbral):

    eta20 = eta_rs(im, 2, 0, umbral)
    eta02 = eta_rs(im, 0, 2, umbral)
    eta11 = eta_rs(im, 1, 1, umbral)

    phi_1 = eta20+eta02
    phi_2 = (eta20-eta02)**2 +4*eta11**2

    return phi_1, phi_2

umbral =  10

print(hu_test(output_image_linear, umbral))
print(hu_test(output_image_cubic, umbral))
print(hu_test(output_image_area, umbral))
