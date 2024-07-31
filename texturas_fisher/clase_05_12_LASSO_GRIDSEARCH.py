"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación del algoritmo LASSO para selección de características
 Búsqueda de parámetro Alpha con searchgrid
 
 Autor:. Miguel Carrasco (06-08-2023)
 rev.1.1
 
 basado en ejemplo https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a
"""


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

from sklearn.datasets import load_diabetes
X,y = load_diabetes(return_X_y=True)
features = load_diabetes()['feature_names']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('modelo',Lasso())
])

search = GridSearchCV(pipeline,
                      {'modelo__alpha':np.arange(0,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )

search.fit(X_train,y_train)
print(search.best_params_)

coefficients = search.best_estimator_.named_steps['modelo'].coef_

importance = np.abs(coefficients)
print(importance)

print(np.array(features)[importance > 0])