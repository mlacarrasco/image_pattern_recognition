import cv2
import matplotlib.pyplot as plt

#lectura de una imagen
img = cv2.imread('data/imagenes/lena.png')
roi = img[255:281,314:348,2]
subroi = roi[10:20, 15:25]

plt.figure()
plt.imshow(subroi, cmap="gray")
plt.show()
