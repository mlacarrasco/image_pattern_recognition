import cv2
import matplotlib.pyplot as plt

#lectura de imagen
img = cv2.imread('data/imagenes/lena.png')

#transformación a niveles de gris
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img_rgb)
plt.show()
