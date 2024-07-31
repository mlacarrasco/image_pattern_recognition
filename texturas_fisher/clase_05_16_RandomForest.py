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
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

#empleamos un clasificador y realizamos la clasificación
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

#extraemos la importancia de cada variable
importances = forest.feature_importances_


df = pd.DataFrame()

for i, tree in enumerate(forest.estimators_):
    aux = np.array(tree.feature_importances_)
    df.loc[:,i] = pd.DataFrame(aux)
df.index = feature_names
print(df)

# calculamos el promedio de una list-of-comprehension
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

#generamos una serie con los resultados
forest_importances = pd.Series(importances, index=feature_names)

#desplegamos los resultados
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title('Importancia de características según MDI')
ax.set_ylabel('Mean decrease in impurity (MDI)')
fig.tight_layout()
plt.show()