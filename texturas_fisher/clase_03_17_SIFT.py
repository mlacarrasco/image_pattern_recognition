#importante: Instalar paquetes adicionales en opencv
# Install opencv contrib for non-free modules `xfeatures2d`
# 'pip install opencv-contrib-python'
# docs: https://www.kite.com/python/docs/cv2.xfeatures2d.SIFT_create

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('data/imagenes/lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Empleamos modulo externo para extraer 
# descriptores con algoritmo SIFT
sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)


# En colab > !pip install opencv-contrib-python==4.4.0.44
# sift  = cv2.SIFT_create()

# Extracción de descriptores
key_query,desc_query = sift.detectAndCompute(img,None)

# Marcamos los puntos de salida
plt.imshow(img)
for key in key_query:
    plt.scatter(key.pt[0], key.pt[1], marker='+', s=100)
    
plt.show()