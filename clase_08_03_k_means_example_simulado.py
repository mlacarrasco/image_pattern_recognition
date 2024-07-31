"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (27/08/2019)

 Objetivo:
 
 1) Generar distribuciones de datos aleatorios
 2) Aplicar un modelo descriptivo sobre los datos
 3) Graficar los centros de masa de cada cluster
    
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


import numpy as np
from sklearn.preprocessing import StandardScaler

#////////////////////////////////
#Creación de los datos
m=400  #puntos para 1ra distribución
n=500  #puntos para 2da distribución

# Creamos puntos según m y n
data1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], m)
data2 = np.random.multivariate_normal([15, 15], [[1, 0.5], [0.5, 3]], n)

#juntamos los datos
data = np.row_stack([data1, data2])

 
#////////////////////////////////
# Aplicamos modelo descriptivo
clus=2                        # numuero de clusters
km = KMeans(n_clusters=clus)  # Creamos un objeto con k clusters 
km=km.fit(data)               # Ajustamos los datos al modelo creado
print(km.inertia_)            # Imprime la suma de las distancias 
                              #  al cuadrado al centro de cluster más cercano

#////////////////////////////////
# Graficamos los datos
#c=color segun niveles de atributo, s = tamaño punto, alpha= nivel transparencia
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
ax[0].scatter(data[:,0], data[:,1], c=km.labels_, s=5, alpha=0.7)
ax[1].scatter(data[:,0], data[:,1], c=km.labels_, s=5, alpha=0.3)
ax[1].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', s=50)
print("centroides:", km.cluster_centers_)
plt.show()

newData = StandardScaler().fit_transform(data) # Normalizamos los datos

"""
# EJERCICIO

# Objetivo: Utilice los datos de newDate y realice las siguientes tareas:

 1) grafique los datos 
 2) determina los valores de estadística básica (media y std)
 3) determine la distancia entre los centroides WD (km.cluster_center)
 4) determine la distancia entre cada punto y su centroide (BC)
    Hint: aplica una busqueda lógica (index = clase==0)
 5) Modifique los valores de los centros de tal forma que la distancia entre
    ambos sea menor a 2 y explique qué ocurre con el algoritmo k-Means
"""
    