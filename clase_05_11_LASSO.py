"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación del LASSO para seleccion de características
 Autor:. Miguel Carrasco (06-08-2023)
 rev.1.0
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#cargamos dataset de ejempo
X,y = load_diabetes(return_X_y=True)

#seleccionamos las caracteristicas
features = load_diabetes()['feature_names']

#separamos el dataset en entrenamiento y testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# normalizamos los datos de test y entrenamiento
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#definimos el modelo de regresion con penalizacion LASSO
model = Lasso(alpha=1.2)
model.fit(X_train,y_train)

#extraemos los coeficientes
coefficients= model.coef_
importance = np.abs(coefficients)
print(importance)

#detereminamos las características relevantes
print(np.array(features)[importance > 0])