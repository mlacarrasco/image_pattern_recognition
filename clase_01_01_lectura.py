import cv2
import matplotlib.pyplot as plt

#lectura de imagen
img = cv2.imread('data/imagenes/lena.png')

plt.figure()
plt.imshow(img)
plt.show()
