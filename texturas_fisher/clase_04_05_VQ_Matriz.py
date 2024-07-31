"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Implementación de algoritmo VQ en múltiples dimensiones
 Autor: Miguel Carrasco (07-10-2021)
 rev.1.0
 
 Referencia:  Linde, Buzo, y Gray (LBG) 1980. 
 https://ieeexplore.ieee.org/document/1094577
 https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.74.1995&rep=rep1&type=pdf


"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from sklearn import neighbors
from sklearn.datasets import make_blobs
from itertools import combinations, tee

from sklearn.preprocessing import StandardScaler

def count_iterable(it):
    # genera una lista a partir de un iterable
    cont = 0
    list_combs=[]
    for comb in it:
        list_combs.append(comb)
    return list_combs, len(list_combs)

def VQ(v, no_centroids, e):
    #% v:  Cada columna es un dato. Los datos están transpuestos 
    #% b:  Codebook generated
    neighbors_number = 2  # se ocupa cuando no existen datos del vecino más cercano
    #% Primera fase: Generacion de dos centroides
    num_feat, *_ = v.shape
    no_update = 5                       # no de actualizacion
    c = np.mean(v, axis=1).reshape(num_feat,1)   # media inicial
    
    center = np.zeros((num_feat,2))
    center[:,0] = (1+e)*c.T       # % separacion de las medias
    center[:,1] = (1-e)*c.T

    for up1 in range(no_update):
        d = euclid_dist(v,center)      #% calculo de la distancia euclidiana
        id = np.argmin(d,axis=0)    #% busqueda de los mas cercanos
        rows, cols = center.shape
        
        #% busqueda de los clusters
        for j in range (cols):
            center[:,j] = np.mean(v[:,np.argwhere(id==j).flatten()],axis=1) 
        
    n=1
    n=n*2

    # Segunda fase: generacion de nuevos centroides
    while cols <= no_centroids:  #% creacion de nuevos centroides
                                #% hasta alcanzar el numero requerido    
        c = center
        center = np.zeros((num_feat,n*2))
        for i in range(cols):            # % numero de centroides
            center[:,i] = (1+e)*c[:,i]   # % separacion de las medias
            center[:,i+n] = (1-e)*c[:,i]
        
        
        for up2 in range(no_update):     # update2
            d = euclid_dist(v,center)    # calculo de la distancia euclidiana
            
            i = np.argmin(d, axis=0)     # busqueda de los mas cercanos
            
            rows, cols = center.shape
            
            #% busqueda de los clusters
            for j in range(cols):
                #si no hay vecinos cercanos del punto escojemos el más cercano
                if len(np.argwhere(i==j).flatten())==0:
                    id_closest = np.argsort(d[j,:])
                    center[:,j] = np.mean(v[:,id_closest[:neighbors_number].flatten()], axis=1)
                else:
                    center[:,j] = np.mean(v[:,np.argwhere(i==j).flatten()], axis=1)
            
        
        n=n*2

    return center

def euclid_dist(x,y):
    #%Calcula la distancia eucliciana entre dos matrices

    #% x: Matriz datos-columna
    #% y: centroides datos-columna

    n_feat = x.shape[0]
    N  = x.shape[1]
    P = y.shape[1]
    d = np.zeros((P, N))
    for i in range(P):
        d[i,:] = np.sum(( x - repmat(y[:,i].reshape(n_feat,1), 1,N))**2, axis=0)
    
    d = np.sqrt(d)
    return d

#***********************************
#      PROGRAMA PRINCIPAL          *
#***********************************

r_seed = 29
features = 5 # numero de caracteristicas del problemas

# Generamos tres clusters 2D con  1000 puntos
X, _ = make_blobs(  n_samples=1000, 
                    centers=8, 
                    n_features=features)

new_X = StandardScaler().fit_transform(X)

# transponemos los datos de forma que sean empleados por el algoritmo
new_X = new_X.T  


# Parametros de algoritmo VQ
e = 0.1   #% parametro de division
center = VQ(new_X, 128, e)

plt.figure(dpi=100)
plt.scatter(new_X[0,:], new_X[1,:], s=1, c='blue')
plt.scatter(center[0,:], center[1,:], s=10, c='red',marker='x')
plt.show()
print(center)


