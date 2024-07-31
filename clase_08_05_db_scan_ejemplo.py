"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (27/08/2019)

 Semana 5 
 
 Objetivo:
 
 1) Crear una distrubucion de datos con una clase
 2) Generar un grafico que muestre los cambios de los valores al modificar el eps
    
    
    * DISCLAIMER *
    Este codigo solo tiene el proposito de experimentar con 
    datos con clases (Clasificación supervisada)
    
"""


import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import mode


#////////////////////////////////
#Creación de los datos
m=500        #puntos para 1ra distribución
n=500        #puntos para 2da distribución
lim_x = 100   #limite de espacio de busqueda

# Creamos puntos según m y n
data1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], m)
data2 = np.random.multivariate_normal([8, 8], [[1, 0.5], [0.1, 3]], n)
clase1= np.ones([m,])
clase2= np.zeros([n,])

# juntamos los datos
data = np.row_stack([data1, data2])
clase =np.concatenate((clase1, clase2))


#////////////////////////////////
# Creamos un tipo especial de grafico que sea interactivo
plt.ion()
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
line1 = []

x_vect = range(0,lim_x)
y_out = np.zeros([lim_x,])


#////////////////////////////////
# Aplicamos modelo descriptivo y variamos el valor de eps
for eps_int in range(1,lim_x):

    val_eps = eps_int/lim_x
    db= DBSCAN(eps=val_eps, min_samples=5).fit(data)
    clusters = np.array(db.labels_)
 
    # vamos a corregir el índice de las etiquetas predichas para 
    # que correspondan con las clases reales
    labels_predicted = np.zeros_like(clusters)
    for i in range(2):
        mask = (clusters == i)
        labels_predicted[mask] = mode(clase[mask])[0]
    
    y_out[eps_int,]= accuracy_score(clase, labels_predicted)
    print ('eps:', round(val_eps,3),'accuracy:',round(accuracy_score(clase, labels_predicted),3))
    
    #si no hay datos, entonces define los parametros del grafico
    if line1==[]:    
        line1, = ax.plot(x_vect, y_out,'-o',alpha=0.8)
        plt.ylim([0,1])
        plt.xlim([1,lim_x])
        plt.ylabel('Accuracy')
        plt.xlabel('Eps index')
        plt.show()
    
    line1.set_ydata(y_out)            
    plt.title("eps:"+str(round(val_eps,3)))
    plt.pause(0.05)
plt.show()

#////////////////////////////////
# Graficamos los datos
#c=color segun niveles de atributo, s = tamaño punto, alpha= nivel transparencia
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(13,5))
ax[0].scatter(data1[:,0], data1[:,1], c='red', s=5, alpha=0.7)
ax[0].scatter(data2[:,0], data2[:,1], c='blue', s=5, alpha=0.7)
ax[0].title.set_text('Datos originales con su clase')

#buscamos el mejor del proceso anterior
index_max= np.argmax(y_out)
print("mejor eps: ", index_max/lim_x)


#////////////////////////////////
# aplicamos DBSCAN con el mejor indice
db= DBSCAN(eps=index_max/lim_x, min_samples=5).fit(data)
clusters = np.array(db.labels_)+1

# corregimos los indices de las clases
labels_predicted = np.zeros_like(clusters)
for i in range(2):
    mask = (clusters == i)
    labels_predicted[mask] = mode(clase[mask])[0]


#////////////////////////////////
# Graficamos los datos   
for i in set(labels_predicted):
    index= labels_predicted==i
    ax[1].scatter(data[index,0], data[index,1],s=5, cmap='jet',alpha=1)
ax[1].title.set_text('Datos con clasificación DBSCAN')

plt.show()