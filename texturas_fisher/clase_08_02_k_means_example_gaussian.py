"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (27/08/2019)


 Objetivo:
 
 1) Generar distribuciones de datos aleatorios
 2) Aplicar un modelo descriptivo sobre los datos
 3) Mostrar una distribución gaussiana sobre el modelo de puntos
    
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import kde

import numpy as np
from sklearn.preprocessing import StandardScaler

#////////////////////////////////
#Creación de los datos
m=400  #puntos para 1ra distribución
n=5000  #puntos para 2da distribución

# Creamos puntos según m y n
data1 = np.random.multivariate_normal([0, 0], [[4, 2.5], [2.5, 4]], m)
data2 = np.random.multivariate_normal([15, 15], [[8, 0.5], [0.5, 8]], n)

#juntamos los datos
data = np.row_stack([data1, data2])

 #estima la funcion de densidad a partir de los datos
nbins=20
xi, yi = np.mgrid[data[:,0].min():data[:,0].max():nbins*1j, data[:,1].min():data[:,1].max():nbins*1j] # crea una malla de puntos  
k1 = kde.gaussian_kde(data1.T)                       
k2 = kde.gaussian_kde(data2.T)                       
zi1 = k1(np.vstack([xi.flatten(), yi.flatten()]))  
zi2 = k2(np.vstack([xi.flatten(), yi.flatten()]))  
Zi = zi1+ zi2
#////////////////////////////////
# Aplicamos modelo descriptivo

clus=3
km = KMeans(n_clusters=clus)     # Creamos un objeto con k clusters 
km=km.fit(data)               # Ajustamos los datos al modelo creado


#////////////////////////////////
# Graficamos los datos
#c=color segun niveles de atributo, s = tamaño punto, alpha= nivel transparencia
fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10,5))

ax[0].pcolormesh(xi, yi, Zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.gnuplot)
ax[1].scatter(data[:,0], data[:,1], c=km.labels_, s=5, alpha=0.3)
ax[2].scatter(data[:,0], data[:,1], c=km.labels_, s=5, alpha=0.3)
ax[2].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', s=50)
ax[2].contour(xi, yi, Zi.reshape(xi.shape),colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')

plt.show()


