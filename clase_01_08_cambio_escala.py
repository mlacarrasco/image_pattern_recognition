import cv2
import matplotlib.pyplot as plt

#lectura de la imagen
img = cv2.imread('data/imagenes/lena.png')
roi = img[255:281,314:348,[2,1,0]]


factor_times = 10
height = int(roi.shape[0] * factor_times)
width  = int(roi.shape[1] * factor_times)
dim    = (width, height) 

#cambio de escala 
resized= cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)

plt.figure()
plt.imshow(resized)
plt.show()
