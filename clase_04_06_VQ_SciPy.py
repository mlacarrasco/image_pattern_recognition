"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Implementación de algoritmo VQ en múltiples dimensiones
 Autor: Miguel Carrasco (28-09-2022)
 rev.1.0
 
 Referencia:  Linde, Buzo, y Gray (LBG) 1980. 
 https://ieeexplore.ieee.org/document/1094577
 https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.74.1995&rep=rep1&type=pdf


"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from scipy.cluster import vq
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

#***********************************
#      PROGRAMA PRINCIPAL          *
#***********************************

r_seed = 29
features = 2 # numero de caracteristicas del problemas


# Generamos tres clusters 2D con  1000 puntos
X, _ = make_blobs(  n_samples=1000, 
                    centers=5, 
                    n_features=features)

new_X = StandardScaler().fit_transform(X)

plt.figure(figsize=(8,8))
plt.scatter(new_X[:,0], new_X[:,1], c='red')

#normalizamos los datos
obs = vq.whiten(new_X)
plt.scatter(obs[:,0], obs[:,1], c='blue')

#implementación de kmeans para VectorQuantization
codebook = vq.kmeans(obs, 128, iter=10)
bk = codebook[0]
plt.scatter(bk[:,0], bk[:,1], c='red')
plt.show()