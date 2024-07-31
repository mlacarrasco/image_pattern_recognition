"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 
 Basado  en https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
 Modificado por Miguel Carrasco
 rev.1.1
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso


# Construimos el dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)

print(f'Tamaño del dataset:{X.shape}')

#separamos el conjunto de datos train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#generamos nombres para las características
feature_names = [f"feature {i}" for i in range(X.shape[1])]


pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('modelo',Lasso())
])

search = GridSearchCV(pipeline,
                      {'modelo__alpha':np.arange(0.01,1,0.001)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )

search.fit(X_train,y_train)
print(search.best_params_)

coefficients = search.best_estimator_.named_steps['modelo'].coef_

importance = np.abs(coefficients)
print(importance)

print(np.array(feature_names)[importance > 0])