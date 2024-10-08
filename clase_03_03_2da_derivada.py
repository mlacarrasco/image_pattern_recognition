import cv2
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt

def derivada_region(I, J, im):

    n = len(I)
    out = 0
    for i in range(0, n):
        out = out + segunda_derivada(I[i],J[i],im);   

    return float(out/n)

def segunda_derivada(i, j, L):
    #% funcion que calcula el laplaciano
    #% en la posicion posL de la imagen L
    Lxy= L[i-1,j]+L[i+1,j]+ L[i,j-1]+L[i,j+1]-4*L[i,j]
    return Lxy


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
bw = image>7
bw = np.array(bw, 'uint8')

#buscamos las coordenadas de los contornos 
contours, hierarchy = cv2.findContours(bw, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_NONE)


#generamos una matriz (fila/columna) con los contornos
contornos = np.vstack(contours[0])

#extraemos las coordenadas x,y 
x , y = zip(*contornos)
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

# calculamos la segunda derivada del perímetro
dprom = derivada_region(x,y, image)
print(f'Segunda derivada promedio: {dprom:2.2f}')

plt.figure()
plt.imshow(image) 
plt.scatter(x,y,marker='x', c='red')
plt.show()