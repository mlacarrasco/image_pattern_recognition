import cv2
import numpy as np 
import matplotlib.pyplot as plt


def color_promedio(im):
    b, g, r = cv2.split(im)
    area = r.shape[0]*r.shape[1]
    prom_r = np.sum(r)/area 
    prom_g = np.sum(g)/area
    prom_b = np.sum(b)/area
    return   prom_r, prom_g, prom_b 


#******************************
#      PROGRAMA PRINCIPAL     *
#******************************
im = cv2.imread('data/imagenes/figura.jpg')

pr,pg,pb = color_promedio(im)

print(f'Color promedio Red: {pr}')
print(f'Color promedio Greeb: {pg}')
print(f'Color promedio Blue: {pb}')





