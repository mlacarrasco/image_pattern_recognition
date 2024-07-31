"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo de Distancia de Mahalanobis para clasificación
 Autor:. Miguel Carrasco (16-08-2021)
 rev.1.0
"""
import numpy as np
import matplotlib.pyplot as plt

#punto que deseamos clasificar
test_point = np.array([1,5])

#data
D=[ [4,1,1],
    [5,1,1],
    [5,2,1],
    [6,7,2],
    [7,6,2],
    [7,7,2],
    [1,8,3],
    [2,7,3],
    [2,8,3]]
    
data = np.array(D)

#buscamos el numero de clases del problema (en la última columna)
no_class = np.max(data[:,-1])

#extraemos la última columna
id_class = data[:,-1]

#quitamos la clase de los datos
data = data[:,:2]
distance = np.zeros((no_class,1))

plt.figure(dpi=100)
plt.scatter(test_point[0], test_point[1], marker='x', color='red', s=80)

#recorremos cada una de las clase
for ic in range(0,no_class): 

    #seleccionamos la clase según el indice
    idx = np.where(id_class==ic+1)
    cluster = data[idx]
    
    #graficamos cada subgrupo de datos
    plt.scatter(cluster[:,0],cluster[:,1], marker='o', label=f'Clase {ic+1}')
    
    #centramos los datos
    mean_data = (test_point - np.mean(cluster, axis=0)).reshape(1,-1)
    
    #calculamos la matriz de covarianza inversa
    invCov = np.linalg.inv(np.cov(cluster,rowvar=False))
    
    #calculamos la metrica de distancia de Mahalanobis
    distance[ic] = np.matmul(np.matmul(mean_data,invCov),mean_data.T)

    print(f'Distancia punto {test_point} y data: {distance[ic]}')

plt.grid(alpha=0.2)
plt.legend()
plt.show()


# Clasificación final
id_min = np.argmin(distance)
print(f'Punto {test_point} es clasificado como clase {id_min+1}')