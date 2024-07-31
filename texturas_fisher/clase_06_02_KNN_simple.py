"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Implementación del algoritmo KNN (vista en clases)
 Autor:. Miguel Carrasco (17-08-2021)
 rev.1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

data = [[0,0],[2,1], [2,2], [3,1], [4,2],[8,1]]
data = np.array(data)
print(data) 

d= np.zeros( (data.shape[0], data.shape[0])) 


#a cada punto calculamos la distancia a todos sus vecinos
for i in range(0,data.shape[0]):
  for j in range(0, data.shape[0]):
    if (i!=j):
      d[i,j] = round(np.sqrt((data[i,0]-data[j,0])**2 + (data[i,1]-data[j,1])**2 ),2)
    else:
      d[i,j]= np.infty

print(d)

k=3
#buscamos los k vecinos máás cercanos a cada punto
average_mat = np.zeros((data.shape[0],1))
for i in range(0,data.shape[0]):
  val_sort= np.sort(d[i,:])
  index_sort = np.argsort(d[i,:])
  print(d[i,index_sort[:3]])
  #print(np.mean(val_sort[index_sort[:3]]))
  average_mat[i] = round(np.mean(d[i,index_sort[:3]]),2)


print(average_mat)

fig = plt.figure()
plt.scatter(data[: , 0], data[ : , 1])
plt.show()