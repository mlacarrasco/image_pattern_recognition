"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (02/09/2019)

 
 Objetivo:
 1) Clustering de datos con clustering Jerarquico (Dendograma)
 2) Desplegar el resultado en un plot
 
    
"""
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
import seaborn as sns
import numpy as np

clusNum = 3


#////////////////////////////////
#carga de datos
iris = sns.load_dataset("iris")
# características del problema
X = iris[['petal_length', 'petal_width']]       

# columna contiene la clase (no aplica a todos los problemas)
tipos = list(iris['species'])

linked = hierarchy.linkage(X, method='complete')    
plt.figure(figsize=(10, 5))  
hierarchy.dendrogram(linked,        
           labels= tipos,
           leaf_rotation=90,
           leaf_font_size=6,
           )
plt.show() 


clusters = AgglomerativeClustering(n_clusters=clusNum,  compute_full_tree='auto', affinity='euclidean', linkage='ward')  
clusters.fit_predict(X) 
plt.figure(figsize=(10, 5))
plt.scatter(iris['sepal_length'],iris['petal_length'], c=clusters.labels_, cmap='rainbow')  




    
