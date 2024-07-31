"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo de Clasificación con algoritmo Naive Bayes
  
Características: 
    + Dataset: Iris
    + Test/Training: 0.2/0.8

 Autor:. Miguel Carrasco (06-11-2021)
 rev.1.0
"""
from os import replace
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from seaborn import pairplot

colormap = np.array(['r', 'g', 'b'])

# Empleamos un dataset con clase
X, y = load_iris(return_X_y=True)
df = pd.DataFrame(np.hstack((X,y.reshape(-1,1))), columns=['LSepal', 'WSepal', 'LPetal', 'WPetal', 'target'])

# Graficamos todas las combinaciones. 
# En la columna target se encuentra la clase
pairplot(data = df, hue= 'target', kind='scatter',palette='tab10')
plt.show()

# Separación de grupos de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# modelo de clasificación
model = GaussianNB()
model.fit(X_train, y_train)

# predicción
y_pred = model.predict(X_test)

# evaluación
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Rendimiento Accuracy: {accuracy:2.3f}')