#importante: Instalar paquetes adicionales en opencv
# Install opencv contrib for non-free modules `xfeatures2d`
# 'pip install opencv-contrib-python'

# doc knn match:https://www.kite.com/python/docs/cv2.BFMatcher.knnMatch
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import imutils

img = cv2.imread('data/imagenes/lena.png')

#generamos dos imágenes
imgA = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgB = cv2.resize(imgA, (100,100), cv2.INTER_AREA)
imgB = imutils.rotate(imgB, angle= 42)

# Empleamos modulo externo para extraer 
# descriptores con algoritmo SIFT
sift = cv2.xfeatures2d.SIFT_create()

# Extracción de descriptores en dos imagenes
kpA,desA = sift.detectAndCompute(imgA,None)
kpB,desB = sift.detectAndCompute(imgB,None)

#analiza la comparación de dos descriptores
#emplaar NORM_L1 para SIFT o SURF
bf = cv2.BFMatcher(cv2.NORM_L1)
matches = bf.knnMatch(desA, desB, k=2)

#determina las relaciones más cercanas
good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]

img3 = cv2.drawMatchesKnn(imgA, kpA, imgB, kpB, good, None, flags=2)
plt.imshow(img3)
plt.show()

