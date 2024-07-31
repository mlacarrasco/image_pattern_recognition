"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación de PCA sobre data
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""


import numpy as np
from numpy.matlib import repmat
from numpy.linalg import eig
import matplotlib.pyplot as plt

def pca(x, pct):

    #% n: filas 
    #% m: Características o columnas
    n = x.shape[0]
    m = x.shape[1]

    # 0o. Calcular la media por columnas
    mu= np.mean(x, axis=0)           
    
    # 1o.  Restar la media los datos (centrado)
    B = x - repmat(mu,n,1)          
    
    # 2o.  Calcular la covarianza
    C = 1/(n-1) * np.matmul(B.T, B)  

    # 3o. Calcular los valores y vectores propios
    D, V = eig(C)                  

    # 4to. Tomar los valores de la diagonal y ordernarlos en 
    # orden reverso (de mayor a menor)
    idx = np.argsort(D)            
    idx = idx[::-1]                

    #5to. Calcular la energia total y acumulada de los valores propios
    e_total = np.sum(D)                             
    cumE = np.cumsum(D[idx])/e_total
    
    # 6to. Buscar un numero de columnas de acuerdo al criterio de la energia
    sel = np.argwhere(cumE<=pct).flatten()   
    W = V[:,idx[sel]]                        

    #7mo. Terminamos. El resultados es la transformacion Karhunen-Loeve
    KLT = np.matmul(B,W)              

    #Final: Datos transformados con pérdida de información
    LOSS = np.matmul(KLT,W.T)+repmat(mu,n,1)

    msg = f'Reducción de datos: {(1-len(sel)/m)*100:2.1f}%'
    return KLT, LOSS,B,idx, V, msg

#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

# data inventada
X= [[2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9], 
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]]

X = np.array(X)
e = 0.99
KLT, LOSS, B, idx, V, msg = pca(X, e)
print('KLT transformation')
print (KLT.round(4))


plt.figure()
plt.scatter(LOSS[:,0], LOSS[:,1], marker='o', color='red')
plt.scatter(X[:,0], X[:,1], marker='o', color='blue')
plt.title(msg)
plt.show()
