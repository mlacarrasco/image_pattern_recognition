#importante: Instalar paquetes adicionales en opencv
# Install opencv contrib for non-free modules `xfeatures2d`
# 'pip install opencv-contrib-python'
# docs: https://www.kite.com/python/docs/cv2.xfeatures2d.SURF_create

# OJO! no funciona en Colab

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('data/imagenes/lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Empleamos modulo externo para extraer 
# descriptores con algoritmo SURF
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000)

# Extracción de descriptores
key_query,desc_query = surf.detectAndCompute(img,None)

# Marcamos los puntos de salida
plt.imshow(img)
for key in key_query:
    plt.scatter(key.pt[0], key.pt[1], marker='+')
    
plt.show()