"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo de LDA
 Autor:. Miguel Carrasco (16-08-2021)
 rev.1.0
"""
import numpy as np
import matplotlib.pyplot as plt


#data original
D=[ [4,3,1],
    [5,3,1],
    [5,4,1],
    [5,5,1],
    [6,5,1],
    [2,4,2],
    [3,4,2],
    [3,5,2],
    [4,6,2]]
    
data = np.array(D)

#buscamos el numero de clases del problema (en la última columna)
no_class = int(np.max(data[:,-1]))

#extraemos la última columna
id_class = data[:,-1]

#quitamos la clase de los datos
data = data[:,:2]
rows = data.shape[0]

#inicializamos las matrices
apriori =  np.zeros((no_class,1))
sigma = np.zeros((no_class,no_class))
mean_data = []
invCov = []

plt.figure(dpi=100)

#recorremos cada una de las clase
for ic in range(0,no_class): 
    #seleccionamos la clase según el indice
    idx = np.where(id_class==ic+1)
    cluster = data[idx]

    #calculamos la probabilidad a priori de cada cluster
    apriori[ic] = len(idx[0])/len(data) 
    
    #graficamos cada subgrupo de datos
    plt.scatter(cluster[:,0],cluster[:,1], marker='o', label=f'Clase {ic+1}')
    
    #centramos los datos
    mean_data.append(np.mean(cluster, axis=0).reshape(1,-1))

    #calculamos la matriz de covarianza inversa
    sigma += np.cov(cluster,rowvar=False)*apriori[ic]

plt.grid(alpha=0.2)
plt.legend()
plt.show()


sigma_inv = np.linalg.inv(sigma)
f_class = np.zeros((rows,2))

#recorremos cada uno de los datos para clasificar los puntos
for i, point in enumerate(data):
    point = point.reshape(1,-1)
    for i_class in range(no_class):
        mean_class = mean_data[i_class]
        f_class[i,i_class] = np.matmul(np.matmul(point, sigma_inv), mean_class.T) - \
                             0.5*np.matmul(np.matmul(mean_class, sigma_inv), mean_class.T)+ \
                             np.log(apriori[i_class]) 


#resultado del clasificador
output = np.argmin(f_class, axis=1)
print('Oputput Clasificador LDA')
print(output)