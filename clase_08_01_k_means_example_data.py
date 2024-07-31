"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (27/08/2019)

 Objetivo:
 
 1) Generar distribuciones de datos aleatorios
 
    
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd  



#////////////////////////////////
# Lectura de los dartos
data = pd.read_csv('data/week3.csv')
print(data.columns)

#limpieza de los datos
data.dropna(subset=['op','co','ag','ne','categoria'], inplace=True)

#escogemos un subconjunto de datos
subdata= data[['ag','ex']]

#////////////////////////////////
# Aplicamos modelo descriptivo

k=2
km = KMeans(n_clusters=k)

km=km.fit(subdata)

plt.scatter(subdata['ag'], subdata['ex'], c=km.labels_, s=50, alpha=0.5)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', s=50)
plt.show()