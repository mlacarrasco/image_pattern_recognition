"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Autor:. Miguel Carrasco (19-08-2021)
 rev.1.0

 Objetivo:
 1) Emplear un clasificador Supervisado no parametrico Mahalanobis
 2) Medir el rendimiento del clasificador
"""

from sklearn import datasets
import scipy as sp
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


iris = datasets.load_iris()
X = iris.data[:, 0:3]
y = iris.target
h = 1  # step size in the mesh

xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size= 0.5)

clases= list(set(tuple(sorted(y))))


""""recorremos los datos para cada clase del conjunto de 
    entrenamiento y calculamos la distancia de Mahalanobis
    entre cada punto y la clase_i
"""
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h),
                     np.arange(z_min, z_max, h))

x_test_c= np.c_[xx.ravel(), yy.ravel(),zz.ravel()]
mahal_c = np.empty([np.size(x_test_c,0),np.size(clases)])

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                            
for class_i in clases:
    #extraemos solo un conjunto por clase
    xtrain_pos = xtrain[ytrain == class_i, :]  
    
    mu = np.mean(xtrain_pos, axis=0)
    dss = x_test_c-mu
    
    cov = np.cov(xtrain_pos.T)
    inv_covmat = sp.linalg.inv(cov)
    
    left_term=np.dot(dss,inv_covmat)
    temp = np.dot(left_term,dss.T)
    mahal_c[:,class_i]= np.diag(temp)

# buscamos en cada columna el menor valor. 
# Su posición indica la mejor clase
predicted= np.argmin(mahal_c, axis=1)
expected= ytest

# Plot the surface.
Z= predicted
Z = Z.reshape(xx.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')


ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cmap_bold, edgecolor='k', s=20)
ax.scatter(xx, yy, zz,c=predicted, cmap=cmap_bold, s=45)
plt.show()
