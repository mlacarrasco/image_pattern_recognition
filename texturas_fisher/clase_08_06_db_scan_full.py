"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (02/09/2019)

 Ayudantía 05: 02-Septiembre-2019
 
 Objetivo:
 1) Clustering de datos con DBSCAN
 2) Desplegar el resultado en un plot
 
    
"""
import pandas as pd
import sklearn.utils
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

df=pd.read_csv('data/week5.csv')   # lee los datos del archivo csv
df.dropna(subset=['op','ag'], inplace=True)
data_df = df[["op","ag"]]

data_df= StandardScaler().fit_transform(data_df)
db= DBSCAN(eps=0.4, min_samples=7).fit(data_df)
labels = np.array(db.labels_)

colores=['red', 'blue', 'green', 'black']
fig, ax = plt.subplots()
for i in set(labels):
    index= labels==i
    ax.scatter(data_df[index,1], data_df[index,0], cmap='jet')
    

plt.show()
