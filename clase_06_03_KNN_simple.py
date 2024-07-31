"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Algoritmo KNN en biblioteca SKLEARN
 Autor:. Miguel Carrasco (16-08-2021)
 rev.1.0
"""

import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.datasets import make_blobs
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors


r_seed = 42

# Generamos tres clusters 2D con  1000 puntos
X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=r_seed)
random.seed(r_seed)
random_pts = []

# Generamos puntos aleatorios que se agregan a los datos
for i in range(50):
    random_pts.append([random.randint(-10, 10), random.randint(-10, 10)])


X = np.append(X, random_pts, axis=0)

#generamos un dataframe
df = DataFrame(dict(x=X[:,0], y=X[:,1]))
df.plot(kind='scatter', x='x', y='y')
plt.show()


#aplicamos el algoritmo KNN
k = 5 #número de vecinos
knn = NearestNeighbors(n_neighbors=k)

knn.fit(X)
# Calcula la distancia a los k vecinos
neighbors_and_distances = knn.kneighbors(X)
knn_distances = neighbors_and_distances[0]


# distancia promedio a los k vecinos más cercanos
tnn_distance = np.mean(knn_distances, axis=1)


plt.figure()
plt.plot(np.sort(tnn_distance))
plt.title('Punto codo para seleccionar el umbral (Threshold)')
plt.ylabel('Distancias entre vecinos por punto')
plt.show()

#definir un umbral (el umbral se define en función del punto codo)
umbral = 1
indices = tnn_distance > umbral


print(knn_distances)

#PCM = df.plot(kind='scatter', x='x', y='y', c=tnn_distance, colormap='viridis')
plt.scatter(X[:, 0], X[:,1], color='red')
plt.scatter(X[indices, 0], X[indices,1], color='blue')
plt.xlabel('X')
plt.xlabel('Y')
plt.show()


