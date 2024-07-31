"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo de clasificación con Árbol de Decisión
 Autor:. Miguel Carrasco (16-08-2021)
 rev.1.0
"""

from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz
from subprocess import call
import matplotlib.pyplot as plt
from sklearn import tree

iris = load_iris()
model = tree.DecisionTreeClassifier(random_state=0)
# Train
model.fit(iris.data, iris.target)

# Exporta el modelo a un formato de árbol
export_graphviz(model, out_file='tree.dot', 
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Covierte el modelo (tree.dot) en un grafico
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

plt.figure(figsize = (10, 5))
plt.imshow(plt.imread('tree.png'))
plt.axis('off');
plt.show();