import cv2
import matplotlib.pyplot as plt

#lectura de la imagen
img = cv2.imread('data/imagenes/lena.png')

#acceso a posiciones y canales
roi = img[255:281,314:348,2]

#sub acceso
subroi = roi[10:20, 15:25]
print(subroi)
