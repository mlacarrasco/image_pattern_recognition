"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo de clasificacion con NaiveBayes 
 
 Características: 
    + Dataset: Iris
    + Validación Cruzada
    
 Autor:. Miguel Carrasco (09-11-2021)
 rev.1.0
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from seaborn import pairplot
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

colormap = np.array(['r', 'g', 'b'])

# Empleamos un dataset con clase
X, y = load_iris(return_X_y=True)
df = pd.DataFrame(np.hstack((X,y.reshape(-1,1))), columns=['LSepal', 'WSepal', 'LPetal', 'WPetal', 'target'])

# Graficamos todas las combinaciones. 
# En la columna target se encuentra la clase
pairplot(data = df, hue= 'target', kind='scatter',palette='tab10')
plt.show()

# modelo de clasificación
model = GaussianNB()

# Validación cruzada
k_fold = KFold(n_splits=5 , shuffle=True, random_state=None)

#no necesario..obtener indices
indices = k_fold.split(X)

# evaluación de  validación cruzada
scores =  cross_val_score(model, X, y, cv=k_fold, n_jobs=1)
print(scores)

# predicción de validación cruzada
y_pred =  cross_val_predict(model, X, y, cv=k_fold, n_jobs=1)

accuracy = metrics.accuracy_score(y, y_pred)
print(f'Accuracy promedio {accuracy}')

plt.figure()
plt.bar(x=np.arange(len(scores)), height=scores)
plt.title('Accuracy')
plt.show()