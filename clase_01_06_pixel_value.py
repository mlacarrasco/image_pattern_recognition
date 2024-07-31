import cv2

#lectura de imagen
img = cv2.imread('data/imagenes/lena.png')

#acceso a nivel de pixel
pixel = img[265,327,:]
print(pixel)
