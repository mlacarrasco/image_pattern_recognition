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
import scipy.spatial as ss
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
print(km.inertia_)            # Imprime la suma de las distancias al cuadrado al centro de cluster más cercano
labels_old = km.labels_       # Almacenemos las clases de kmeans

#////////////////////////////////
# Graficamos los datos
#c=color segun niveles de atributo, s = tamaño punto, alpha= nivel transparencia
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
ax[0].scatter(data[:,0], data[:,1], c=labels_old, s=5, alpha=0.7)
ax[1].scatter(data[:,0], data[:,1], c=labels_old, s=5, alpha=0.7)
ax[1].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', s=50)
print("centroides:", km.cluster_centers_)
plt.show()

newData = StandardScaler().fit_transform(data) # Normalizamos los datos



### EJERCICIO
# Objetivo: Utilice los datos de newDate y realice las siguientes tareas:

#////////////////////////////////
# 1) grafique los datos 

fig, ax=plt.subplots()
ax.scatter(newData[:,0], newData[:,1],c=labels_old, s=5, alpha=0.7)
ax.set_title("Puntos normalizados")

 
#////////////////////////////////
# 2) determina los valores de estadística básica (media y std)

print("promedio:",round(newData.mean(),8))
print("Desviación estandar:",round(newData.std(),8))


 
#////////////////////////////////
# 3) determine la distancia entre los centroides WD (km.cluster_center)

km      = KMeans(n_clusters=clus)  # Creamos un objeto con k clusters 
km      = km.fit(newData)      # Calculamos los nuevos clusters

# Ajustamos los datos a coordenadas con dimensiones (1,2) (1 fila, 2 columnas)
center1 = np.reshape(km.cluster_centers_[0, :], (1,2))   
center2 = np.reshape(km.cluster_centers_[1, :], (1,2))

# calculamos la distancia entre dos puntos
distancia = ss.distance_matrix(center1,center2,p=2)
print("La distancia entre los centros es:", distancia)


 
#////////////////////////////////
#4) determine la distancia entre cada punto y su centroide (BC)

index    = km.labels_==0       # buscamos los puntos que complen una condicion
puntos_0 = newData[index,:]  # seleccionamos de la matrix los datos de index
puntos_1 = newData[~index,:] # seleccionamos de la matrix los datos de not(index)

mat1     = ss.distance_matrix(puntos_0,center1,p=2)
mat2     = ss.distance_matrix(puntos_1,center2,p=2)
print("la distancia del centro 0 a sus puntos es: ", mat1.mean())
print("la distancia del centro 1 a sus puntos es: ", mat2.mean())


#////////////////////////////////
# 5) Modifique los valores de los centros de tal forma que la distancia entre
#    ambos sea menor a 2 y explique qué ocurre con el algoritmo k-Means


#inventamos una nueva distribucion
data1   = np.random.multivariate_normal([5, 5], [[1, 0.5], [0.5, 3]], m)
data2   = np.random.multivariate_normal([6, 6], [[1, 0.5], [0.5, 3]], n)

#juntamos los datos y normalizamos
data    = np.row_stack([data1, data2])
newData = StandardScaler().fit_transform(data) 

#calculamos un nuevo objeto kmeans
km_new = KMeans(n_clusters=clus)  # Creamos un objeto con k clusters 
km_new = km.fit(newData)          # Ajustamos los datos al modelo creado
labels_new = km_new.labels_       # Almacenmos las clases de kmeans

center1   = np.reshape(km_new.cluster_centers_[0, :], (1,2))   
center2   = np.reshape(km_new.cluster_centers_[1, :], (1,2))
distancia = ss.distance_matrix(center1,center2,p=2)
print("La distancia entre los centros es:", distancia)


fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15,5))
ax[0].scatter(newData[:,0], newData[:,1], s=10, alpha=0.6)
ax[0].set_title("[GRA #1].Puntos de la nueva distribución")
ax[1].scatter(newData[:,0], newData[:,1],c=labels_old, s=10, alpha=0.6)
ax[1].set_title("[GRA #2].Labels de primer k-means")
ax[2].scatter(newData[:,0], newData[:,1],c=labels_new, s=5, alpha=0.9)
ax[2].set_title("[GRA #3].Con labels de k-mean actual")
plt.show()
"""
A medida que los centroides se acercan el algoritmo Kmeans solo 
puede dividir el conjunto de puntos (GRA#3), ya que algoritmo
no puede lidiar con dos distribuciones que se superponen una con otra.
En el gráfico #2, se observa como en la realidad corresponde
cada  punto en la distribución (algo que no puede determinarse con k-means
"""