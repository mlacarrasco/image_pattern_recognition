import cv2
import matplotlib.pyplot as plt

#lectura de imagen
img = cv2.imread('data/imagenes/lena.png')

#separación de canales
blue, green, red = cv2.split(img)

plt.figure()
plt.imshow(red, cmap="gray")
plt.show()
