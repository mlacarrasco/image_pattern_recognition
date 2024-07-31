"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (12/11/2019)

 Objetivo:
 
 1) Generar una distribución con múltiples clases
 2) Transformar los datos de clases desbalaceadas en clases balanceadas
 2) Plotear la nueva distribución balanceada

 Instrucciones:
   >Instalar pip install -U imbalanced-learn

%leer info desde:
https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
X, y = make_classification(
    n_classes=2, class_sep=1, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=10, n_clusters_per_class=1,
    n_samples=100, random_state=10
)

df = pd.DataFrame(X)
df['target'] = y
plt.figure(dpi=100)
df.target.value_counts().plot(kind='bar', title='Count (target)')


pca = PCA(n_components=2)
Z = pca.fit_transform(X)

plt.figure(dpi=100)
plot_2d_space(Z, y, 'clases no balanceadas')


smote = SMOTE() 
X_sm, y_sm =   smote.fit_resample(Z,y)
plt.figure(dpi=100)
plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')


plt.figure(dpi=100)
df = pd.DataFrame(X_sm)
df['target'] = y_sm
df.target.value_counts().plot(kind='bar', title='Count (target)');
plt.show()