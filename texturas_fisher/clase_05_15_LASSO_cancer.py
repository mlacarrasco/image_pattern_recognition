"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo de LASSO con dataset de cancer
 
 Basado en ejemplo https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a
 Modificado por Miguel Carrasco
 rev.1.1
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer

X,y = load_breast_cancer(return_X_y=True)
features = load_breast_cancer()['feature_names']
clases= y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('modelo',Lasso())
])

search = GridSearchCV(pipeline,
                      {'modelo__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )

search.fit(X_train,y_train)
print(search.best_params_)

coefficients = search.best_estimator_.named_steps['modelo'].coef_

importance = np.abs(coefficients)
print(importance)

print(np.array(features)[importance > 0])

data_selection = X[:,importance > 0]
xx = data_selection[:,0]
yy = data_selection[:,1]
zz = data_selection[:,2]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xx,yy,zz, c=clases, edgecolor='k', s=20)


plt.show()
