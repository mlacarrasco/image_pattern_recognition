#importante: Instalar paquetes adicionales en opencv
# Install opencv contrib for non-free modules `xfeatures2d`
# 'pip install opencv-contrib-python'

# doc knn match:https://www.kite.com/python/docs/cv2.BFMatcher.knnMatch
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import imutils

imgA = cv2.imread('estereo/foto_A.jpg')
imgB = cv2.imread('estereo/foto_B.jpg')

#generamos dos imágenes
imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

# Empleamos modulo externo para extraer 
# descriptores con algoritmo SIFT
sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)

# Extracción de descriptores en dos imagenes
kpA,desA = sift.detectAndCompute(imgA,None)
kpB,desB = sift.detectAndCompute(imgB,None)

#analiza la comparación de dos descriptores
#emplaar NORM_L1 para SIFT o SURF
bf = cv2.BFMatcher(cv2.NORM_L1)
matches = bf.knnMatch(desA, desB, k=2)

#determina las relaciones más cercanas
good = [[m] for m, n in matches if m.distance < 0.6 * n.distance]

img3 = cv2.drawMatchesKnn(imgA, kpA, imgB, kpB, good, None, flags=0)
plt.imshow(img3)
plt.show()

