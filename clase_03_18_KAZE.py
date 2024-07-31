import cv2 
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('data/imagenes/lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Empleamos modulo externo para extraer 
# descriptores con algoritmo KAZE
kaze = cv2.KAZE_create()

# Extracción de descriptores
key_query,desc_query = kaze.detectAndCompute(img,None)

# Marcamos los puntos de salida
plt.imshow(img)
for key in key_query:
    plt.scatter(key.pt[0], key.pt[1], marker='+', s=100)
    
plt.show()
