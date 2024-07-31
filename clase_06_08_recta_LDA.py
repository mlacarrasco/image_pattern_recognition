"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo de clasificación con LDA (recta)
 Autor:. Miguel Carrasco (16-08-2021)
 rev.1.0
"""
import numpy as np
import matplotlib.pyplot as plt


#data original
D=[ [2.5,3,1],
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

data_centered = data-np.mean(data, axis=0)
plt.figure(dpi=100)
#recorremos cada una de las clase
for ic in range(0,no_class): 
    #seleccionamos la clase según el indice
    idx = np.where(id_class==ic+1)
    cluster = data[idx]

    #calculamos la probabilidad a priori de cada cluster
    apriori[ic] = len(idx[0])/len(data) 
    

    #centramos los datos
    center = np.mean(cluster, axis=0).reshape(1,-1)
    mean_data.append(center)

     #graficamos cada subgrupo de datos
    plt.scatter(cluster[:,0],cluster[:,1], marker='o', label=f'Clase {ic+1}')

    #calculamos la matriz de covarianza inversa
    sigma += np.cov(cluster,rowvar=False)*apriori[ic]



sigma_inv = np.linalg.inv(sigma)
f_class = np.zeros((rows,2))

#recorremos cada uno de los datos para clasificar los puntos
diff_mean = (mean_data[0]-mean_data[1]).reshape(1,-1)
sum_mean = (mean_data[0]+mean_data[1]).reshape(1,-1)

W = np.matmul(sigma_inv,diff_mean.T)
w0 = -0.5*np.matmul(np.matmul(sum_mean, sigma_inv), diff_mean.T)+ np.log(apriori[0]/apriori[1])

#parametros de la recta
m = -W[0]/W[1]
intercepto = -w0/W[1]

min_data = np.min(data[:,0]-1) 
max_data = np.max(data[:,0]+1)
x_eval = np.arange(min_data,max_data).reshape(-1,1)
fx =  x_eval*m + intercepto

plt.plot(x_eval, fx ,color='red')
plt.grid(alpha=0.2)
plt.axis('equal')
plt.legend()
plt.show()