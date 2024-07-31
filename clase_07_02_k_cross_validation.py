"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (08/11/2019)

 Objetivo:
 
 1) Emplear la técnica K-cross validation para separar conjunto de datos

 Ejemplo tomado desde:
 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    
"""

import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8],[9, 10]])
y = np.array([1, 2, 3, 4, 5])
kf = KFold(n_splits=3)
kf.get_n_splits(X)

print(kf)  

for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
