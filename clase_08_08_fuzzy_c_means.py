"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (10/09/2019)

 Objetivo:
 
 1) Aplicar el algoritmo Fuzzy C-Means sobre un conjunto de datos
 2) Obtener la matriz de probabilidades y aplicar un color

  #code from: https://codereview.stackexchange.com/questions/188455/fuzzy-c-means-in-python    
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


rgb2hex = lambda r,g,b: f"#{r:02x}{g:02x}{b:02x}"

def fcm(data, n_clusters=1, n_init=30, m=2, max_iter=300, tol=1e-16):

    min_cost = np.inf
    for iter_init in range(n_init):

        # Randomly initialize centers
        centers = data[np.random.choice(
            data.shape[0], size=n_clusters, replace=False
            ), :]

        # Compute initial distances
        # Zeros are replaced by eps to avoid division issues
        dist = np.fmax(
            cdist(centers, data, metric='sqeuclidean'),
            np.finfo(np.float64).eps
        )

        for iter1 in range(max_iter):

            # Compute memberships       
            u = (1 / dist) ** (1 / (m-1))
            um = (u / u.sum(axis=0))**m

            # Recompute centers
            prev_centers = centers
            centers = um.dot(data) / um.sum(axis=1)[:, None]
            dist = cdist(centers, data, metric='sqeuclidean')

            if np.linalg.norm(centers - prev_centers) < tol:
                break

        # Compute cost
        cost = np.sum(um * dist)
        
        if cost < min_cost:
            min_cost = cost
            min_centers = centers
            mem = um.argmax(axis=0)

    return min_centers, mem,um, dist

def color_to_hex(um):
    table=[]
    for i in range(0,um.shape[1]):
        r= (um[0,i]*255//1).astype(int)
        g= (um[1,i]*255//1).astype(int)
        b= 125
        table.append(rgb2hex(r,g,b))
    
    return table


# INICIO DEL PROGRAMA ->
#PARAMETROS
#////////////////////////////////
#Creación de los datos
m       = 600  #puntos para 1ra distribución
n       = 600  #puntos para 2da distribución
eps_sel = 0.22  #valor de eps seleccionado

#DATOS ALEATORIOS
# Creamos puntos según m y n
data1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], m)
data2 = np.random.multivariate_normal([10, 10], [[1, 0.5], [0.5, 3]], n)

#juntamos los datos
data = np.row_stack([data1, data2])
#vat(data, figuresize=(5,5))
centers, mem , um, dist= fcm(data, n_clusters=2, n_init=10, m=3, max_iter=300, tol=1e-16)
table_col = color_to_hex(um)

#////////////////////////////////
# Desplegamos los resultados 
fig, ax= plt.subplots(nrows=1, ncols=2,figsize=(10,5))
ax[0].scatter(data[:,0], data[:,1], c='blue')
ax[0].title.set_text("Datos originales")
ax[1].scatter(data[:,0], data[:,1], c=table_col)
ax[1].scatter(centers[:,0], centers[:,1], c='blue', s=80, alpha=0.5)
ax[1].title.set_text("Clustering Fuzzy C Means")
plt.show()

Fu= np.trace(np.matmul(um,um.T))/ (n+m)
print("Fuzzy partition coeficient ", round(Fu,2))


