"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
  
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (10/09/2019)
 version 1.1 (25/07/2022)

 Objetivo:
 
 1) Aplicar las medida de Hopkins y VAT
 2) Evaluar la métrica davies- bouldin para buscar el número óptimo de clusters


 Requiere:
 1) Instalar Pip en windows 
    https://www.liquidweb.com/kb/install-pip-windows/
    
    
 2) instalarp yclustertend 
    Más info en https://pypi.org/project/pyclustertend/ 

 3) pip install pyclustertend

"""

from sklearn import datasets
from pyclustertend import hopkins, vat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans


#////////////////////////////////
#Creación de los datos
m = 400  #puntos para 1ra distribución
n = 500  #puntos para 2da distribución
r = 200  #puntos para 3ra distribución

# Creamos puntos según m y n
data1 = np.random.multivariate_normal([0, 0], [[4, 2.5], [2.5, 4]], m)
data2 = np.random.multivariate_normal([15, 15], [[8, 0.5], [0.5, 8]], n)
data3 = np.random.multivariate_normal([35, 35], [[2, 1.5], [1.5, 12]], r)
data4 = np.random.multivariate_normal([45, 45], [[2, 1.5], [1.5, 5]], r)

#juntamos los datos
data = np.row_stack([data1, data2, data3, data4])
#more info at https://pyclustertend.readthedocs.io/en/latest/
#vat(data)
#plt.show()

# mientras más cercano a cero, datos más agrupados
print(f'hopkins distance: {hopkins(data,50)}')

kmeans = KMeans(n_clusters=3, random_state=30)
labels = kmeans.fit_predict(data)

# mientras más cercano a cero, datos más agrupados
print(f'davies- bouldin score con 3 clusters: {davies_bouldin_score(data,labels)}')

# ejemplo que busca el mejor indices de DBS a medida que aumenta el número
# de clusters
results = {}
for i in range(2,11):
    kmeans = KMeans(n_clusters=i, random_state=30)
    labels = kmeans.fit_predict(data)
    db_index = davies_bouldin_score(data, labels)
    results.update({i: db_index})

plt.plot(list(results.keys()), list(results.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Boulding Index")
plt.show()